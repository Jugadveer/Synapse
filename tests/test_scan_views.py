"""Scan endpoint tests.

audio_scan and mri_scan had no login requirement, wrote uploads under the
client-supplied filename, and only deleted the file on the happy path.
"""

import io
import os

import pytest
from django.contrib.auth.models import User

pytestmark = pytest.mark.django_db


@pytest.fixture
def user():
    return User.objects.create_user(username='tester', password='a-good-password-42')


def post_audio(client, name='clip.wav', data=b'RIFF0000WAVEfmt '):
    upload = io.BytesIO(data)
    upload.name = name
    return client.post('/audio-scan/', {'audio': upload})


def media_files(settings):
    root = settings.MEDIA_ROOT
    return set(os.listdir(root)) if os.path.isdir(root) else set()


# ------------------------------------------------------------------- auth

def test_audio_scan_requires_login(client):
    assert post_audio(client).status_code in (302, 401, 403)


def test_mri_scan_requires_login(client):
    upload = io.BytesIO(b'\x89PNG\r\n')
    upload.name = 'scan.png'
    assert client.post('/mri-scan/', {'mri': upload}).status_code in (302, 401, 403)


def test_dashboard_data_requires_login(client):
    assert client.get('/dashboard-data/').status_code in (302, 401, 403)


def test_anonymous_scan_creates_no_record(client):
    from synapse.models import ScanResult

    post_audio(client)
    assert ScanResult.objects.count() == 0


# ------------------------------------------------------------ validation

def test_rejects_unsupported_extension(client, user):
    client.force_login(user)
    upload = io.BytesIO(b'#!/bin/sh\n')
    upload.name = 'payload.sh'
    response = client.post('/audio-scan/', {'audio': upload})
    assert response.status_code == 400
    assert 'Unsupported' in response.json()['error']


def test_rejects_missing_file(client, user):
    client.force_login(user)
    assert client.post('/audio-scan/', {}).status_code == 400


def test_rejects_get(client, user):
    client.force_login(user)
    assert client.get('/audio-scan/').status_code == 405


def test_upload_is_not_stored_under_the_client_filename(client, user, settings, audio_predictor):
    """Two people uploading 'scan.wav' used to collide in MEDIA_ROOT."""
    captured = {}

    def record(path):
        captured['path'] = path
        return ('No Dementia', 0.8)

    audio_predictor(record)
    client.force_login(user)
    post_audio(client, name='scan.wav')

    assert captured['path']
    assert not captured['path'].endswith('scan.wav')
    assert captured['path'].endswith('.wav')


# --------------------------------------------------------------- cleanup

def test_upload_is_removed_after_a_successful_scan(client, user, settings, audio_predictor):
    before = media_files(settings)
    client.force_login(user)
    assert post_audio(client).status_code == 200
    assert media_files(settings) == before


def test_upload_is_removed_when_inference_fails(client, user, settings, audio_predictor):
    """A failed scan used to leave its upload behind."""
    def boom(path):
        raise RuntimeError('model exploded')

    audio_predictor(boom)
    before = media_files(settings)
    client.force_login(user)

    assert post_audio(client).status_code == 502
    assert media_files(settings) == before, 'temporary upload was not cleaned up'


def test_unreadable_audio_returns_an_error_not_a_crash(client, user, audio_predictor):
    """predict_audio returns (None, None); rounding that used to raise."""
    audio_predictor(lambda path: (None, None))
    client.force_login(user)

    response = post_audio(client)
    assert response.status_code == 422
    assert 'error' in response.json()


# ---------------------------------------------------------------- results

def test_clean_result_is_low_risk(client, user, audio_predictor):
    from synapse.models import ScanResult

    audio_predictor(lambda path: ('No Dementia', 0.91))
    client.force_login(user)

    body = post_audio(client).json()
    assert body['result'] == 'No Dementia'
    assert body['risk'] == 'LOW', 'a clean result must not be reported as high risk'
    assert body['confidence'] == 91.0

    record = ScanResult.objects.get()
    assert record.user == user
    assert record.risk_level == 'LOW'
    assert record.confidence == pytest.approx(0.91)


def test_positive_result_is_high_risk(client, user, audio_predictor):
    audio_predictor(lambda path: ('Dementia', 0.77))
    client.force_login(user)
    assert post_audio(client).json()['risk'] == 'HIGH'


def test_unmapped_label_is_refused(client, user, audio_predictor):
    from synapse.models import ScanResult

    audio_predictor(lambda path: ('Something Unexpected', 0.5))
    client.force_login(user)

    assert post_audio(client).status_code == 502
    assert ScanResult.objects.count() == 0


def test_scans_are_scoped_to_their_owner(client, user, audio_predictor):
    other = User.objects.create_user(username='someone-else', password='another-password-42')
    audio_predictor(lambda path: ('No Dementia', 0.9))

    client.force_login(user)
    post_audio(client)

    client.force_login(other)
    assert client.get('/dashboard-data/').json()['sessions'] == 0


# ----------------------------------------------------------------- logout

def test_logout_rejects_get(client, user):
    client.force_login(user)
    # Used to fall off the end and return None, which Django turns into a 500.
    assert client.get('/logout/').status_code == 405


def test_logout_works_on_post(client, user):
    client.force_login(user)
    assert client.post('/logout/').status_code == 302
