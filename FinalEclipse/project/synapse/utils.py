"""Mapping from a model's class label to a displayed risk level.

The audio and MRI models emit different vocabularies. This used to be a chain
of substring tests written against the MRI class names only, so every audio
result fell through to the final else and was reported as HIGH - including
"No Dementia". Each model now gets an explicit table, and an unrecognised
label yields None rather than a fabricated risk.
"""

import logging

logger = logging.getLogger(__name__)

RISK_LOW = 'LOW'
RISK_MEDIUM = 'MEDIUM'
RISK_HIGH = 'HIGH'
#: The model declines to answer. At this level of separation a binary verdict
#: is misleading in both directions, so most recordings land here honestly.
RISK_INCONCLUSIVE = 'INCONCLUSIVE'

# synapse/app/data/predict.py - binary classifier over acoustic features.
AUDIO_RISK_BY_LABEL = {
    'no dementia': RISK_LOW,
    'inconclusive': RISK_INCONCLUSIVE,
    'dementia': RISK_HIGH,
}

# synapse/predict.py - four-class CNN over MRI slices.
MRI_RISK_BY_LABEL = {
    'no impairment': RISK_LOW,
    'very mild impairment': RISK_MEDIUM,
    'mild impairment': RISK_HIGH,
    'moderate impairment': RISK_HIGH,
}

RISK_BY_SCAN_TYPE = {
    'AUDIO': AUDIO_RISK_BY_LABEL,
    'MRI': MRI_RISK_BY_LABEL,
}


def get_risk_level(result, scan_type):
    """Return LOW/MEDIUM/HIGH for a model label, or None if unrecognised.

    Args:
        result: the class label the model produced.
        scan_type: 'AUDIO' or 'MRI'.
    """
    if not result:
        return None

    table = RISK_BY_SCAN_TYPE.get((scan_type or '').upper())
    if table is None:
        logger.error(f"Unknown scan type {scan_type!r}; cannot map risk")
        return None

    risk = table.get(result.strip().lower())
    if risk is None:
        logger.error(f"Unmapped {scan_type} label {result!r}; refusing to guess a risk level")
    return risk
