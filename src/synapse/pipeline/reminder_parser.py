"""Turn a spoken reminder request into a time and a task.

Deliberately conservative: when the time is missing or ambiguous the parser
says so instead of guessing, and the assistant asks one short question. For a
memory aid a reminder at the wrong hour is worse than one more question.
"""

import re
from datetime import datetime, timedelta, timezone

TRIGGERS = (
    'remind me', 'set a reminder', 'set reminder', 'remember to remind',
    'give me a reminder', 'wake me', 'alert me',
)

SECONDS_PER_UNIT = {
    'second': 1, 'seconds': 1, 'sec': 1, 'secs': 1,
    'minute': 60, 'minutes': 60, 'min': 60, 'mins': 60,
    'hour': 3600, 'hours': 3600, 'hr': 3600, 'hrs': 3600,
    'day': 86400, 'days': 86400,
    'week': 604800, 'weeks': 604800,
}

WORD_NUMBERS = {
    'a': 1, 'an': 1, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11,
    'twelve': 12, 'fifteen': 15, 'twenty': 20, 'thirty': 30, 'forty': 40,
    'forty-five': 45, 'fortyfive': 45, 'sixty': 60, 'half': 30,
}

_DURATION = re.compile(
    r'\b(?:in|after)\s+(?P<amount>\d+|' + '|'.join(WORD_NUMBERS) + r')\s*'
    r'(?P<unit>' + '|'.join(SECONDS_PER_UNIT) + r')\b'
)

_CLOCK = re.compile(
    r'\b(?:at|by)\s+(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<meridiem>am|pm|a\.m\.|p\.m\.)?\b'
)

_BARE_UNIT = re.compile(r'\b(?:in|after)\s+(?P<amount>\d+)\b(?!\s*(?:' + '|'.join(SECONDS_PER_UNIT) + r'))')

_TASK_PATTERNS = (
    re.compile(r'\bremind me\s+(?:to|about|that)\s+(?P<task>.+?)(?=\s+(?:in|at|by|after|tomorrow|tonight)\b|$)'),
    re.compile(r'\bset (?:a )?reminder\s+(?:to|for|about)\s+(?P<task>.+?)(?=\s+(?:in|at|by|after|tomorrow|tonight)\b|$)'),
    re.compile(r'\b(?:wake|alert) me\s+(?:to|about|for)\s+(?P<task>.+?)(?=\s+(?:in|at|by|after|tomorrow|tonight)\b|$)'),
)


class ReminderRequest:
    """A parsed reminder. `due_at` is None when the time could not be read."""

    __slots__ = ('due_at', 'task', 'spoken_time', 'needs_time', 'needs_task')

    def __init__(self, due_at=None, task='', spoken_time='', needs_time=False, needs_task=False):
        self.due_at = due_at
        self.task = task
        self.spoken_time = spoken_time
        self.needs_time = needs_time
        self.needs_task = needs_task

    @property
    def is_complete(self):
        return self.due_at is not None and bool(self.task)

    def __repr__(self):  # pragma: no cover - debugging aid
        return (f"ReminderRequest(due_at={self.due_at!r}, task={self.task!r}, "
                f"needs_time={self.needs_time}, needs_task={self.needs_task})")


def looks_like_reminder(text):
    lowered = (text or '').lower()
    return any(trigger in lowered for trigger in TRIGGERS)


def _to_amount(token):
    token = token.strip().lower()
    if token.isdigit():
        return int(token)
    return WORD_NUMBERS.get(token)


def _format_time(moment, now):
    stamp = moment.strftime('%I:%M %p').lstrip('0')
    if moment.date() == now.date():
        return stamp
    if moment.date() == (now + timedelta(days=1)).date():
        return f"{stamp} tomorrow"
    return f"{stamp} on {moment.strftime('%b %d')}"


def _parse_duration(lowered, now):
    match = _DURATION.search(lowered)
    if not match:
        return None
    amount = _to_amount(match.group('amount'))
    if amount is None:
        return None
    return now + timedelta(seconds=amount * SECONDS_PER_UNIT[match.group('unit')])


def _parse_clock(lowered, now):
    match = _CLOCK.search(lowered)
    if not match:
        return None

    hour = int(match.group('hour'))
    minute = int(match.group('minute') or 0)
    meridiem = (match.group('meridiem') or '').replace('.', '')

    if hour > 23 or minute > 59:
        return None

    if meridiem == 'pm' and hour < 12:
        hour += 12
    elif meridiem == 'am' and hour == 12:
        hour = 0
    elif not meridiem and hour <= 12:
        # No am/pm given. Choose the next occurrence rather than assuming,
        # so "at 9" in the evening means tomorrow morning, not hours ago.
        candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if candidate <= now:
            candidate = now.replace(hour=(hour + 12) % 24, minute=minute, second=0, microsecond=0)
        if candidate <= now:
            candidate += timedelta(days=1)
        return candidate

    due = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if due <= now:
        due += timedelta(days=1)
    return due


def _parse_task(lowered):
    for pattern in _TASK_PATTERNS:
        match = pattern.search(lowered)
        if match:
            task = match.group('task').strip(' .,!?')
            if task:
                return task
    return ''


def parse_reminder(text, now=None):
    """Parse `text` into a ReminderRequest, or return None if not a reminder.

    `now` should be timezone-aware; the result inherits its tzinfo. A naive
    value produces a naive due_at, which Django stores as though it were UTC -
    a reminder set for four in the afternoon would then fire at whatever local
    time four PM UTC happens to be.
    """
    if not looks_like_reminder(text):
        return None

    now = now or datetime.now(timezone.utc).astimezone()
    lowered = (text or '').lower().strip()

    task = _parse_task(lowered)

    due_at = _parse_duration(lowered, now)
    if due_at is None:
        if 'tomorrow' in lowered:
            clock = _parse_clock(lowered, now)
            due_at = clock if clock else now.replace(
                hour=9, minute=0, second=0, microsecond=0
            ) + timedelta(days=1)
            if due_at <= now:
                due_at += timedelta(days=1)
        else:
            due_at = _parse_clock(lowered, now)

    # "remind me in 10" - a number with no unit is not a time.
    if due_at is None and _BARE_UNIT.search(lowered):
        return ReminderRequest(task=task, needs_time=True, needs_task=not task)

    if due_at is None:
        return ReminderRequest(task=task, needs_time=True, needs_task=not task)

    if not task:
        return ReminderRequest(
            due_at=due_at, spoken_time=_format_time(due_at, now), needs_task=True
        )

    return ReminderRequest(due_at=due_at, task=task, spoken_time=_format_time(due_at, now))
