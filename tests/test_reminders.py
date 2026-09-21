"""End-to-end reminder behaviour: created, stored, delivered, not repeated.

Before this the assistant said "I will remind you at 3:45" and wrote a sentence
into memory. There was no scheduler anywhere in the project, so the reminder
never arrived.
"""

from datetime import timedelta

import pytest
from asgiref.sync import sync_to_async
from django.utils import timezone

from pipeline.reminder_scheduler import (
    MAX_LATE_DELIVERY,
    ReminderScheduler,
    create_reminder,
    pending_reminders,
)
from voice.models import Reminder

# transaction=True is required, not incidental: the async tests reach the ORM
# through sync_to_async, which runs on a separate connection that does not take
# part in the usual rollback, so rows leaked between tests without it.
pytestmark = pytest.mark.django_db(transaction=True)


class RecordingConsumer:
    """Stands in for the websocket; records what it was asked to speak."""

    def __init__(self, fail=False):
        self.delivered = []
        self.fail = fail

    async def send_reminder(self, text, spoken_time=''):
        if self.fail:
            raise ConnectionError('socket closed')
        self.delivered.append((text, spoken_time))


def make_reminder(user_key='u1', text='take your tablets', minutes_ago=1, **kwargs):
    return Reminder.objects.create(
        user_key=user_key,
        text=text,
        spoken_time='3:45 PM',
        due_at=timezone.now() - timedelta(minutes=minutes_ago),
        **kwargs,
    )


# The ORM is synchronous; async tests reach it through these.
amake_reminder = sync_to_async(make_reminder)
acreate = sync_to_async(Reminder.objects.create)
acount = sync_to_async(lambda **f: Reminder.objects.filter(**f).count())


# ------------------------------------------------------------------ model

async def test_create_reminder_persists_a_row():
    await create_reminder('u1', 'take your tablets', timezone.now() + timedelta(minutes=10), '3:45 PM')

    reminder = await Reminder.objects.aget()
    assert reminder.text == 'take your tablets'
    assert reminder.is_pending


async def test_pending_excludes_delivered_and_cancelled():
    await amake_reminder(text='due now')
    await amake_reminder(text='already done', delivered_at=timezone.now())
    await amake_reminder(text='called off', cancelled=True)

    assert await acount(delivered_at__isnull=True, cancelled=False) == 1


async def test_pending_reminders_is_scoped_by_user():
    await amake_reminder(user_key='alice', text='alice thing')
    await amake_reminder(user_key='bob', text='bob thing')

    alice = await pending_reminders('alice')
    assert [r.text for r in alice] == ['alice thing']


# -------------------------------------------------------------- delivery

async def test_due_reminder_is_delivered_once():
    await amake_reminder(text='take your tablets')

    scheduler = ReminderScheduler()
    consumer = RecordingConsumer()
    scheduler._listeners['u1'] = {consumer}

    await scheduler._deliver_due()
    assert consumer.delivered == [('take your tablets', '3:45 PM')]

    # A second pass must not repeat it.
    await scheduler._deliver_due()
    assert len(consumer.delivered) == 1

    reminder = await Reminder.objects.aget()
    assert reminder.delivered_at is not None


async def test_future_reminder_is_not_delivered_early():
    await acreate(user_key='u1', text='later', due_at=timezone.now() + timedelta(hours=1))

    scheduler = ReminderScheduler()
    consumer = RecordingConsumer()
    scheduler._listeners['u1'] = {consumer}

    await scheduler._deliver_due()
    assert consumer.delivered == []


async def test_reminder_missed_while_offline_is_delivered_on_return():
    """The point of storing them: a reminder outlives the connection."""
    await amake_reminder(text='take your tablets', minutes_ago=30)

    scheduler = ReminderScheduler()
    consumer = RecordingConsumer()
    scheduler._listeners['u1'] = {consumer}

    await scheduler._deliver_due()
    assert consumer.delivered == [('take your tablets', '3:45 PM')]


async def test_very_stale_reminder_is_not_delivered():
    """A reminder from last week is noise, not help."""
    await acreate(
        user_key='u1',
        text='ancient',
        due_at=timezone.now() - MAX_LATE_DELIVERY - timedelta(hours=1),
    )

    scheduler = ReminderScheduler()
    consumer = RecordingConsumer()
    scheduler._listeners['u1'] = {consumer}

    await scheduler._deliver_due()
    assert consumer.delivered == []


async def test_reminder_is_not_marked_delivered_when_sending_fails():
    await amake_reminder(text='take your tablets')

    scheduler = ReminderScheduler()
    scheduler._listeners['u1'] = {RecordingConsumer(fail=True)}

    await scheduler._deliver_due()

    reminder = await Reminder.objects.aget()
    assert reminder.delivered_at is None, 'must stay pending so it can be retried'


async def test_reminders_are_not_delivered_to_another_user():
    await amake_reminder(user_key='alice', text='alice private thing')

    scheduler = ReminderScheduler()
    bob = RecordingConsumer()
    scheduler._listeners['bob'] = {bob}

    await scheduler._deliver_due()
    assert bob.delivered == []


# ----------------------------------------------------------- registration

def test_register_and_unregister_tracks_listeners():
    scheduler = ReminderScheduler()
    consumer = RecordingConsumer()

    scheduler._listeners.setdefault('u1', set()).add(consumer)
    assert scheduler._listeners['u1'] == {consumer}

    scheduler.unregister('u1', consumer)
    assert 'u1' not in scheduler._listeners


def test_unregister_an_unknown_listener_is_harmless():
    ReminderScheduler().unregister('nobody', RecordingConsumer())
