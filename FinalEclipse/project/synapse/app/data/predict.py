"""Audio dementia indicator.

The model is a calibrated pipeline: the scaler lives inside it, so there is no
separate scaler artifact to fall out of step with the classifier.

The decision threshold comes from the model card written at training time, not
from the default 0.5. At 0.5 this model answers "No Dementia" to nearly
everyone - it scored 7.7% sensitivity, telling 24 of 26 people who had
dementia that they were clear. The card's threshold trades specificity for
sensitivity, which is the right direction for screening.
"""

import json
import logging
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[1] / 'models'
MODEL_PATH = MODELS_DIR / 'dementia_model.pkl'
CARD_PATH = MODELS_DIR / 'audio_model_card.json'

DEMENTIA = 'Dementia'
NO_DEMENTIA = 'No Dementia'
INCONCLUSIVE = 'Inconclusive'


@lru_cache(maxsize=1)
def get_model():
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def get_model_card():
    """Measured performance and the chosen operating point.

    Returned to the caller so the interface can state what the number is
    worth instead of presenting it as a finding.
    """
    try:
        return json.loads(CARD_PATH.read_text(encoding='utf-8'))
    except (OSError, ValueError) as e:
        logger.warning(f'No usable model card at {CARD_PATH}: {e}')
        return {}


def decision_threshold():
    threshold = get_model_card().get('decision_threshold')
    try:
        return float(threshold)
    except (TypeError, ValueError):
        return 0.5


def band_cuts():
    """The two cuts bounding the inconclusive band.

    Separation on this corpus does not support a binary verdict: forced to
    choose, the model either misses cases or flags most healthy people. With
    two cuts it answers for about a third of recordings and is right roughly
    four times in five when it does, instead of being right three times in
    five always.

    Falling back to (t, t) reproduces the old binary behaviour if a model card
    predates the bands.
    """
    card = get_model_card()
    try:
        low = float(card['band_low'])
        high = float(card['band_high'])
    except (KeyError, TypeError, ValueError):
        threshold = decision_threshold()
        return threshold, threshold
    return (low, high) if low < high else (decision_threshold(),) * 2


#: Averaging several windows of one recording was tried and dropped: on
#: held-out speakers it scored no better (auc 0.736 against 0.732) for four
#: times the inference cost. The per-speaker gain that motivated it came from
#: averaging across different recordings of a person, which is not what a
#: single upload provides. One window also matches how training rows are made.
MAX_WINDOWS = 1


def classify_probability(probability):
    """Turn a calibrated probability into one of the three outcomes.

    Confidence is always reported for the label returned, so a "No Dementia"
    answer carries the probability that it is right, not the probability of
    the thing it ruled out.
    """
    low, high = band_cuts()
    if probability >= high:
        return DEMENTIA, probability
    if probability < low:
        return NO_DEMENTIA, 1.0 - probability
    # Neither end of the range: say so rather than guess.
    return INCONCLUSIVE, probability


def predict_audio(audio_path):
    """Return (label, confidence_in_that_label), or (None, None) if unreadable."""
    from .extract_features import extract_feature_windows

    windows = extract_feature_windows(audio_path, max_windows=MAX_WINDOWS)
    if not windows:
        return None, None

    try:
        probabilities = get_model().predict_proba(np.vstack(windows))[:, 1]
    except Exception as e:
        logger.exception(f'Audio inference failed: {e}')
        return None, None

    probability = float(np.mean(probabilities))
    if np.isnan(probability):
        return None, None

    return classify_probability(probability)
