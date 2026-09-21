"""Risk mapping tests.

get_risk_level was written against the MRI class names and then reused for the
audio model, whose labels share no vocabulary with them. Both audio labels fell
through to the final else, so a clean screening result was reported as HIGH.
"""

import pytest

from synapse.utils import get_risk_level


@pytest.mark.parametrize('label,expected', [
    ('No Impairment', 'LOW'),
    ('Very Mild Impairment', 'MEDIUM'),
    ('Mild Impairment', 'HIGH'),
    ('Moderate Impairment', 'HIGH'),
])
def test_mri_labels(label, expected):
    assert get_risk_level(label, 'MRI') == expected


@pytest.mark.parametrize('label,expected', [
    ('No Dementia', 'LOW'),
    ('Dementia', 'HIGH'),
])
def test_audio_labels(label, expected):
    assert get_risk_level(label, 'AUDIO') == expected


def test_clean_audio_result_is_not_high_risk():
    """The regression: a healthy patient was told they were high risk."""
    assert get_risk_level('No Dementia', 'AUDIO') == 'LOW'


def test_audio_and_mri_vocabularies_do_not_leak():
    """An MRI label is not valid for the audio model, and vice versa."""
    assert get_risk_level('No Impairment', 'AUDIO') is None
    assert get_risk_level('No Dementia', 'MRI') is None


def test_unrecognised_label_returns_none_rather_than_guessing():
    assert get_risk_level('something unexpected', 'AUDIO') is None
    assert get_risk_level(None, 'MRI') is None
    assert get_risk_level('', 'AUDIO') is None


def test_unknown_scan_type_returns_none():
    assert get_risk_level('Dementia', 'ULTRASOUND') is None


def test_labels_are_matched_case_and_space_insensitively():
    assert get_risk_level('  no dementia  ', 'AUDIO') == 'LOW'
    assert get_risk_level('VERY MILD IMPAIRMENT', 'mri') == 'MEDIUM'
