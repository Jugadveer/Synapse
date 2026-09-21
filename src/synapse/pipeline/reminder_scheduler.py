"""Delivery of scheduled reminders.

The assistant previously replied "I will remind you at 3:45" and wrote a
sentence into memory. Nothing in the project ever looked at it again - there
was no scheduler of any kind - so no reminder was ever delivered. For a
memory aid that promise is the whole feature.

One polling task runs per process. Reminders live in the database, so a
reminder survives the websocket that created it, a disconnect, and a restart;
anything that came due while nobody was listening is delivered as soon as the
person reconnects.
"""

import asyncio
import logging
from datetime import timedelta

from channels.db import database_sync_to_async
from django.utils import timezone

logger = logging.getLogger(__name__)

POLL_SECONDS = 15
#: How far back to still deliver something that came due while offline. A
#: reminder to take medicine is worth hearing late; one from last week is not.
MAX_LATE_DELIVERY = timedelta(hours=12)


class ReminderScheduler:
    """Process-wide poller that delivers due reminders to connected people."""

    _instance = None

    def __init__(self):
        # user_key -> set of consumers currently listening for that person.
        self._listeners = {}
        self._task = None
        self._wake = asyncio.Event()

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = ReminderScheduler()
        return cls._instance

    # ------------------------------------------------------------------
    # registration
    # ------------------------------------------------------------------

    def register(self, user_key, consumer):
        self._listeners.setdefault(user_key, set()).add(consumer)
        self.start()
        # Deliver anything that fell due while they were away.
        self._wake.set()

    def unregister(self, user_key, consumer):
        listeners = self._listeners.get(user_key)
        if not listeners:
            return
        listeners.discard(consumer)
        if not listeners:
            self._listeners.pop(user_key, None)

    def start(self):
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())
            logger.info("Reminder scheduler started")

    async def stop(self):
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    # ------------------------------------------------------------------
    # polling
    # ------------------------------------------------------------------

    async def _run(self):
        while True:
            try:
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=POLL_SECONDS)
                except asyncio.TimeoutError:
                    pass
                self._wake.clear()

                if self._listeners:
                    await self._deliver_due()
            except asyncio.CancelledError:
                logger.info("Reminder scheduler stopped")
                raise
            except Exception as e:
                # A failure here must not end reminder delivery for the process.
                logger.exception(f"Reminder poll failed: {e}")
                await asyncio.sleep(POLL_SECONDS)

    async def _deliver_due(self):
        for user_key in list(self._listeners):
            due = await self._due_reminders(user_key)
            for reminder in due:
                consumers = list(self._listeners.get(user_key, ()))
                if not consumers:
                    break

                delivered = False
                for consumer in consumers:
                    try:
                        await consumer.send_reminder(reminder.text, reminder.spoken_time)
                        delivered = True
                    except Exception as e:
                        logger.warning(f"Could not deliver reminder to a listener: {e}")

                if delivered:
                    await self._mark_delivered(reminder)
                    logger.info(f"Delivered reminder {reminder.pk}: {reminder.text}")

    @database_sync_to_async
    def _due_reminders(self, user_key):
        from voice.models import Reminder

        now = timezone.now()
        return list(
            Reminder.objects.filter(
                user_key=user_key,
                cancelled=False,
                delivered_at__isnull=True,
                due_at__lte=now,
                due_at__gte=now - MAX_LATE_DELIVERY,
            ).order_by('due_at')[:5]
        )

    @database_sync_to_async
    def _mark_delivered(self, reminder):
        reminder.mark_delivered()


@database_sync_to_async
def create_reminder(user_key, text, due_at, spoken_time='', session_id=None):
    """Persist a reminder so it can be delivered after this connection ends."""
    from voice.models import ConversationSession, Reminder

    session = None
    if session_id:
        session = ConversationSession.objects.filter(session_id=session_id).first()

    return Reminder.objects.create(
        user_key=user_key,
        session=session,
        text=text,
        spoken_time=spoken_time,
        due_at=due_at,
    )


@database_sync_to_async
def pending_reminders(user_key, limit=5):
    from voice.models import Reminder

    return list(
        Reminder.objects.filter(
            user_key=user_key, cancelled=False, delivered_at__isnull=True
        ).order_by('due_at')[:limit]
    )
