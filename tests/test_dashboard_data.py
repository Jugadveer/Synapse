"""Dashboard statistics.

Streaks and the weekly chart are date arithmetic over timezone-aware
timestamps, which is where this kind of code usually goes wrong.
"""

from datetime import timedelta

import pytest
from django.contrib.auth.models import User
from django.utils import timezone

from synapse.models import ScanResult

pytestmark = pytest.mark.django_db


@pytest.fixture
def user():
    return User.objects.create_user(username='dash', password='a-good-password-42')


def add_scan(user, when, confidence=0.8, risk='LOW'):
    scan = ScanResult.objects.create(
        user=user, scan_type='AUDIO', result='No Dementia',
        confidence=confidence, risk_level=risk,
    )
    # created_at is auto_now_add, so it has to be moved afterwards.
    ScanResult.objects.filter(pk=scan.pk).update(created_at=when)
    return scan


def data(client, user):
    client.force_login(user)
    return client.get('/dashboard-data/').json()


# ----------------------------------------------------------------- empty

def test_no_scans_yet(client, user):
    body = data(client, user)
    assert body['sessions'] == 0
    assert body['streak'] == 0
    assert body['risk'] == 'LOW'
    assert len(body['weekly_scores']) == 7
    assert len(body['labels']) == 7
    assert all(score == 0 for score in body['weekly_scores'])


# ---------------------------------------------------------------- streak

def test_a_run_ending_today(client, user):
    today = timezone.localtime()
    for days in range(3):
        add_scan(user, today - timedelta(days=days))

    assert data(client, user)['streak'] == 3


def test_a_run_ending_yesterday_still_counts(client, user):
    """Someone who has not scanned yet today has not lost their streak."""
    today = timezone.localtime()
    for days in range(1, 4):
        add_scan(user, today - timedelta(days=days))

    assert data(client, user)['streak'] == 3


def test_a_gap_ends_the_streak(client, user):
    today = timezone.localtime()
    add_scan(user, today)
    add_scan(user, today - timedelta(days=1))
    add_scan(user, today - timedelta(days=4))    # after a gap

    assert data(client, user)['streak'] == 2


def test_several_scans_in_one_day_count_once(client, user):
    today = timezone.localtime()
    for hour in (2, 9, 21):
        add_scan(user, today.replace(hour=hour, minute=0))

    body = data(client, user)
    assert body['streak'] == 1
    assert body['sessions'] == 3


def test_a_scan_late_in_the_local_day_lands_on_that_day(client, user):
    """The bug this guards: .date() on a UTC timestamp is the UTC date.

    East of Greenwich, a scan at one in the morning local time is the previous
    day in UTC, so it was counted against the wrong day and broke the streak.
    """
    local_now = timezone.localtime()
    one_am = local_now.replace(hour=1, minute=30, second=0, microsecond=0)
    if one_am > local_now:
        one_am -= timedelta(days=1)

    add_scan(user, one_am)
    add_scan(user, one_am - timedelta(days=1))

    assert data(client, user)['streak'] >= 2


# ----------------------------------------------------------- weekly chart

def test_weekly_scores_are_percentages(client, user):
    add_scan(user, timezone.localtime(), confidence=0.75)

    body = data(client, user)
    assert body['weekly_scores'][-1] == pytest.approx(75.0)


def test_weekly_scores_average_within_a_day(client, user):
    today = timezone.localtime()
    add_scan(user, today, confidence=0.6)
    add_scan(user, today, confidence=0.8)

    assert data(client, user)['weekly_scores'][-1] == pytest.approx(70.0)


def test_days_with_no_scan_read_zero(client, user):
    add_scan(user, timezone.localtime(), confidence=0.9)

    scores = data(client, user)['weekly_scores']
    assert scores[-1] == pytest.approx(90.0)
    assert all(score == 0 for score in scores[:-1])


def test_scans_older_than_a_week_are_not_charted(client, user):
    add_scan(user, timezone.localtime() - timedelta(days=30), confidence=0.9)

    body = data(client, user)
    assert all(score == 0 for score in body['weekly_scores'])
    assert body['sessions'] == 1, 'still counted in the total'


# ------------------------------------------------------------------ risk

def test_risk_comes_from_the_most_recent_scan(client, user):
    today = timezone.localtime()
    add_scan(user, today - timedelta(days=2), risk='HIGH')
    add_scan(user, today, risk='INCONCLUSIVE')

    assert data(client, user)['risk'] == 'INCONCLUSIVE'


def test_another_users_scans_are_not_counted(client, user):
    other = User.objects.create_user(username='other', password='a-good-password-42')
    add_scan(other, timezone.localtime())

    body = data(client, user)
    assert body['sessions'] == 0
    assert body['streak'] == 0


# ------------------------------------------------- timezone, east of UTC

@pytest.mark.parametrize('zone', ['Asia/Kolkata', 'Pacific/Auckland'])
def test_early_morning_scans_bucket_on_the_local_day(client, user, settings, zone):
    """A scan at half past one in the morning belongs to that local day.

    created_at is stored in UTC. East of Greenwich its .date() is the previous
    day, so taking the date without converting to local time put the scan on
    the wrong day: it broke streaks and moved points on the weekly chart.
    """
    settings.TIME_ZONE = zone
    timezone.activate(zone)
    try:
        local_now = timezone.localtime()
        early = local_now.replace(hour=1, minute=30, second=0, microsecond=0)
        if early > local_now:
            early -= timedelta(days=1)

        add_scan(user, early, confidence=0.9)
        add_scan(user, early - timedelta(days=1), confidence=0.9)
        add_scan(user, early - timedelta(days=2), confidence=0.9)

        body = data(client, user)
        assert body['streak'] == 3, f'{zone}: early-morning scans fell on the wrong day'

        charted = [score for score in body['weekly_scores'] if score > 0]
        assert len(charted) == 3, f'{zone}: weekly chart bucketed them wrongly'
    finally:
        timezone.deactivate()


def test_two_local_days_are_not_collapsed_into_one(client, user, settings):
    """The shift only shows when it merges two local days.

    In IST, 01:30 today and 23:00 yesterday are both *yesterday* in UTC. Taking
    the date without converting to local time counts them as a single day, so a
    two-day streak reads as one and the weekly chart loses a bar.
    """
    settings.TIME_ZONE = 'Asia/Kolkata'
    timezone.activate('Asia/Kolkata')
    try:
        local_now = timezone.localtime()
        today_early = local_now.replace(hour=1, minute=30, second=0, microsecond=0)
        if today_early > local_now:
            today_early -= timedelta(days=1)
        yesterday_late = today_early - timedelta(hours=2, minutes=30)   # 23:00 the day before

        assert today_early.date() != yesterday_late.date(), 'setup: two local days'

        add_scan(user, today_early, confidence=0.9)
        add_scan(user, yesterday_late, confidence=0.9)

        body = data(client, user)
        assert body['streak'] == 2, 'two local days were counted as one'
        assert len([s for s in body['weekly_scores'] if s > 0]) == 2
    finally:
        timezone.deactivate()
