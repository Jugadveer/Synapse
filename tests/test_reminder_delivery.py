"""Reminder delivery under awkward conditions.

test_reminders.py covers the ordinary path. These cover what happens with two
devices open, a listener that fails mid-delivery, a backlog larger than one
poll, and the boundary where a late reminder stops being worth delivering.
"""

from datetime import timedelta

import pytest
from asgiref.sync import sync_to_async
from django.utils import timezone

from pipeline.reminder_scheduler import (
    MAX_LATE_DELIVERY, ReminderScheduler, pending_reminders,
)
from voice.models import Reminder

pytestmark = pytest.mark.django_db(transaction=True)


class Listener:
    """Stands in for a connected websocket."""

    def __init__(self, fail=False, label=''):
        self.delivered = []
        self.fail = fail
        self.label = label

    async def send_reminder(self, text, spoken_time=''):
        if self.fail:
            raise ConnectionError('socket closed')
        self.delivered.append(text)


def make(user_key='u1', text='take your tablets', minutes_ago=1, **kwargs):
    return Reminder.objects.create(
        user_key=user_key,
        text=text,
        spoken_time='3:45 PM',
        due_at=timezone.now() - timedelta(minutes=minutes_ago),
        **kwargs,
    )


amake = sync_to_async(make)
acreate = sync_to_async(Reminder.objects.create)
areminders = sync_to_async(lambda **f: list(Reminder.objects.filter(**f)))


def scheduler_with(listeners):
    scheduler = ReminderScheduler()
    for user_key, people in listeners.items():
        scheduler._listeners[user_key] = set(people)
    return scheduler


# --------------------------------------------------------- two devices

async def test_a_reminder_reaches_every_open_device():
    """Someone may have the app open on a phone and a tablet."""
    await amake(text='take your tablets')

    phone, tablet = Listener(label='phone'), Listener(label='tablet')
    await scheduler_with({'u1': [phone, tablet]})._deliver_due()

    assert phone.delivered == ['take your tablets']
    assert tablet.delivered == ['take your tablets']


async def test_a_reminder_is_marked_delivered_only_once():
    await amake(text='take your tablets')

    scheduler = scheduler_with({'u1': [Listener(), Listener()]})
    await scheduler._deliver_due()
    await scheduler._deliver_due()

    delivered = await areminders(delivered_at__isnull=False)
    assert len(delivered) == 1


async def test_one_failing_device_does_not_block_the_other():
    """A stale socket must not cost the person their reminder."""
    await amake(text='take your tablets')

    broken, working = Listener(fail=True), Listener()
    await scheduler_with({'u1': [broken, working]})._deliver_due()

    assert working.delivered == ['take your tablets']
    delivered = await areminders(delivered_at__isnull=False)
    assert len(delivered) == 1, 'a reminder one device received is delivered'


async def test_all_devices_failing_leaves_it_pending():
    await amake(text='take your tablets')

    await scheduler_with({'u1': [Listener(fail=True), Listener(fail=True)]})._deliver_due()

    assert len(await areminders(delivered_at__isnull=True)) == 1


# ------------------------------------------------------------- backlog

async def test_a_backlog_is_delivered_oldest_first():
    now = timezone.now()
    for minutes, text in ((30, 'first'), (20, 'second'), (10, 'third')):
        await acreate(user_key='u1', text=text, due_at=now - timedelta(minutes=minutes))

    listener = Listener()
    await scheduler_with({'u1': [listener]})._deliver_due()

    assert listener.delivered == ['first', 'second', 'third']


async def test_a_large_backlog_drains_over_several_passes():
    """One pass takes at most five, so nobody is buried under an avalanche."""
    now = timezone.now()
    for i in range(12):
        await acreate(user_key='u1', text=f'thing {i}', due_at=now - timedelta(minutes=30 - i))

    listener = Listener()
    scheduler = scheduler_with({'u1': [listener]})

    await scheduler._deliver_due()
    assert len(listener.delivered) == 5, 'one pass should be bounded'

    for _ in range(3):
        await scheduler._deliver_due()
    assert len(listener.delivered) == 12, 'the backlog still drains'


# ------------------------------------------------------- lateness edge

async def test_a_reminder_just_inside_the_window_is_delivered():
    await acreate(
        user_key='u1', text='still worth hearing',
        due_at=timezone.now() - MAX_LATE_DELIVERY + timedelta(minutes=5),
    )

    listener = Listener()
    await scheduler_with({'u1': [listener]})._deliver_due()
    assert listener.delivered == ['still worth hearing']


async def test_a_reminder_just_outside_the_window_is_not():
    await acreate(
        user_key='u1', text='too old to matter',
        due_at=timezone.now() - MAX_LATE_DELIVERY - timedelta(minutes=5),
    )

    listener = Listener()
    await scheduler_with({'u1': [listener]})._deliver_due()
    assert listener.delivered == []


async def test_a_stale_reminder_stays_pending_rather_than_vanishing():
    """It is skipped, not marked delivered - the row is still there to inspect."""
    await acreate(
        user_key='u1', text='too old',
        due_at=timezone.now() - MAX_LATE_DELIVERY - timedelta(hours=1),
    )

    await scheduler_with({'u1': [Listener()]})._deliver_due()
    assert len(await areminders(delivered_at__isnull=True)) == 1


# --------------------------------------------------------- registration

def test_registering_the_same_device_twice_is_not_a_duplicate():
    scheduler = ReminderScheduler()
    listener = Listener()

    scheduler._listeners.setdefault('u1', set()).add(listener)
    scheduler._listeners['u1'].add(listener)

    assert len(scheduler._listeners['u1']) == 1


def test_unregistering_one_device_leaves_the_other():
    scheduler = ReminderScheduler()
    phone, tablet = Listener(), Listener()
    scheduler._listeners['u1'] = {phone, tablet}

    scheduler.unregister('u1', phone)
    assert scheduler._listeners['u1'] == {tablet}

    scheduler.unregister('u1', tablet)
    assert 'u1' not in scheduler._listeners


async def test_delivery_to_a_user_who_just_disconnected():
    """The listener set can empty between the query and the send."""
    await amake(text='take your tablets')

    scheduler = scheduler_with({'u1': []})
    await scheduler._deliver_due()

    assert len(await areminders(delivered_at__isnull=True)) == 1


async def test_cancelled_reminders_are_never_delivered():
    await amake(text='called off', cancelled=True)

    listener = Listener()
    await scheduler_with({'u1': [listener]})._deliver_due()
    assert listener.delivered == []


async def test_pending_list_is_capped():
    now = timezone.now()
    for i in range(9):
        await acreate(user_key='u1', text=f'thing {i}', due_at=now + timedelta(hours=i + 1))

    assert len(await pending_reminders('u1')) == 5
    assert len(await pending_reminders('u1', limit=2)) == 2
