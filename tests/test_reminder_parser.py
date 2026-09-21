"""Reminder parsing tests.

The assistant used to answer "Reminder set for 10 minutes, I will remind you
at 3:45 PM" and then write a sentence into memory. There was no scheduler in
the project, so the reminder never arrived. Parsing is the first half of
actually keeping that promise; these tests pin the cases it must get right and,
just as importantly, the ones where it must ask instead of guessing.
"""

from datetime import datetime, timedelta

import pytest

from pipeline.reminder_parser import looks_like_reminder, parse_reminder

NOW = datetime(2026, 9, 21, 14, 30, 0)  # a Monday, 2:30 pm


def parse(text, now=NOW):
    return parse_reminder(text, now=now)


# ---------------------------------------------------------------- detection

@pytest.mark.parametrize('text', [
    'remind me to take my medicine in 10 minutes',
    'set a reminder for my appointment at 4pm',
    'Remind me about the doctor tomorrow',
])
def test_detects_reminder_requests(text):
    assert looks_like_reminder(text)


@pytest.mark.parametrize('text', [
    'what is the weather today',
    'i left my keys on the table',
    'where did i put my glasses',
])
def test_ignores_non_reminders(text):
    assert not looks_like_reminder(text)
    assert parse(text) is None


# ----------------------------------------------------------------- duration

@pytest.mark.parametrize('text,delta', [
    ('remind me to call my son in 10 minutes', timedelta(minutes=10)),
    ('remind me to rest in 2 hours', timedelta(hours=2)),
    ('remind me to take my pills in 30 seconds', timedelta(seconds=30)),
    ('remind me about the bins in 3 days', timedelta(days=3)),
    ('remind me to stretch in five minutes', timedelta(minutes=5)),
    ('remind me to drink water in an hour', timedelta(hours=1)),
])
def test_relative_times(text, delta):
    result = parse(text)
    assert result.is_complete
    assert result.due_at == NOW + delta


def test_extracts_the_task_not_the_whole_sentence():
    result = parse('remind me to take my heart medicine in 20 minutes')
    assert result.task == 'take my heart medicine'


# -------------------------------------------------------------- clock times

def test_afternoon_clock_time_today():
    result = parse('remind me to call the doctor at 4pm')
    assert result.due_at == NOW.replace(hour=16, minute=0)
    assert result.task == 'call the doctor'


def test_clock_time_with_minutes():
    result = parse('remind me to take my tablets at 9:15 pm')
    assert result.due_at == NOW.replace(hour=21, minute=15)


def test_time_already_past_rolls_to_tomorrow():
    result = parse('remind me to take my pills at 9am')
    assert result.due_at == (NOW + timedelta(days=1)).replace(hour=9, minute=0)


def test_ambiguous_hour_picks_the_next_occurrence():
    """"at 9" at 2:30pm means 9pm today, not 9am this morning."""
    result = parse('remind me to lock the door at 9')
    assert result.due_at == NOW.replace(hour=21, minute=0)


def test_tomorrow_without_a_time_defaults_to_morning():
    result = parse('remind me about my appointment tomorrow')
    assert result.due_at == (NOW + timedelta(days=1)).replace(hour=9, minute=0)


def test_tomorrow_with_a_time():
    result = parse('remind me to see the nurse tomorrow at 11am')
    assert result.due_at == (NOW + timedelta(days=1)).replace(hour=11, minute=0)


# ------------------------------------------------------- asking, not guessing

def test_missing_time_asks_rather_than_guessing():
    result = parse('remind me to take my medicine')
    assert not result.is_complete
    assert result.needs_time
    assert result.task == 'take my medicine'
    assert result.due_at is None


def test_number_with_no_unit_is_not_a_time():
    """"remind me in 10" used to be answered as though it were minutes."""
    result = parse('remind me to call in 10')
    assert result.needs_time
    assert result.due_at is None


def test_missing_task_is_flagged():
    result = parse('remind me in 10 minutes')
    assert result.due_at == NOW + timedelta(minutes=10)
    assert result.needs_task
    assert not result.is_complete


def test_bare_request_needs_both():
    result = parse('set a reminder')
    assert result.needs_time
    assert result.needs_task


def test_impossible_clock_time_is_rejected():
    result = parse('remind me to eat at 45:99')
    assert result.needs_time


# ------------------------------------------------------------------ display

def test_spoken_time_is_read_back_naturally():
    assert parse('remind me to call at 4pm').spoken_time == '4:00 PM'
    assert 'tomorrow' in parse('remind me to call at 9am').spoken_time


def test_due_time_is_never_in_the_past():
    for text in ('remind me to x at 1am', 'remind me to x at 2:29 pm', 'remind me to x at 11pm'):
        result = parse(text)
        if result.due_at:
            assert result.due_at > NOW, text


# ------------------------------------------------------------ edge cases

def test_zero_duration_is_not_a_time():
    """"in 0 minutes" would fire at the moment it was asked for."""
    result = parse('remind me to rest in 0 minutes')
    assert result.needs_time
    assert result.due_at is None


def test_negative_duration_is_not_a_time():
    result = parse('remind me to rest in -5 minutes')
    assert result.due_at is None


def test_twenty_four_hour_readings_are_not_guessed():
    """"at 00:30" was read as half past twelve in the afternoon."""
    assert parse('remind me to call at 00:30').due_at == (
        NOW + timedelta(days=1)).replace(hour=0, minute=30)
    assert parse('remind me to call at 19:00').due_at == NOW.replace(hour=19, minute=0)
    assert parse('remind me to call at 06:15').due_at == (
        NOW + timedelta(days=1)).replace(hour=6, minute=15)


def test_noon_and_midnight():
    assert parse('remind me to eat at 12pm').due_at.hour == 12
    assert parse('remind me to sleep at 12am').due_at.hour == 0


def test_task_after_the_time_is_still_found():
    """"remind me at 7 to take my pills" puts the task last."""
    for text, task in (
        ('remind me at 7 to take my pills', 'take my pills'),
        ('remind me in 10 minutes to call my son', 'call my son'),
        ('remind me tomorrow to see the nurse', 'see the nurse'),
    ):
        assert parse(text).task == task, text


def test_case_is_ignored():
    result = parse('REMIND ME TO TAKE MY PILLS AT 4PM')
    assert result.due_at == NOW.replace(hour=16, minute=0)
    assert result.task


def test_accented_text_survives():
    result = parse('remind me to call José at 4pm')
    assert 'josé' in result.task.lower()


def test_a_very_long_task_is_kept_whole():
    """Trimming for speech happens later, in the safety rules."""
    task = 'do ' + 'something ' * 60
    result = parse(f'remind me to {task.strip()} in 5 minutes')
    assert result.is_complete
    assert len(result.task) > 200


def test_no_reminder_is_ever_scheduled_in_the_past():
    for text in ('remind me to x at 1am', 'remind me to x at 00:30',
                 'remind me to x at 2:29 pm', 'remind me to x in 1 minute',
                 'remind me to x at 11pm', 'remind me to x tomorrow'):
        result = parse(text)
        if result and result.due_at:
            assert result.due_at > NOW, text
