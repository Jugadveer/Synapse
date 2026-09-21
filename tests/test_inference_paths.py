"""The real model inference paths, against the real wheels and real inputs.

Everything else in the suite substitutes the models. These load the actual
artifacts committed to the repo and run real recordings and real MRI slices
through them. They skip when the inference wheels are not installed.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import REAL_AUDIO, REAL_ML, REAL_STT, REAL_VISION

REPO_ROOT = Path(__file__).resolve().parents[1]
RESOURCES = REPO_ROOT / 'FinalEclipse' / 'project' / 'synapse' / 'resources'
MODELS_DIR = REPO_ROOT / 'FinalEclipse' / 'project' / 'synapse' / 'app' / 'models'

#: wav2vec2 hidden states pooled to mean+std (1536) followed by the
#: hand-crafted summary: MFCC(40) + chroma(12) + spectral contrast(7) + ZCR(1).
EMBEDDING_DIMS = 1536
HANDCRAFTED_DIMS = 60
EXPECTED_FEATURES = EMBEDDING_DIMS + HANDCRAFTED_DIMS

AUDIO_LABELS = {'Dementia', 'No Dementia', 'Inconclusive'}
DEMENTIA_LABEL = 'Dementia'
MRI_LABELS = {'Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment'}


def sample_wavs(group, limit=3):
    root = RESOURCES / group
    if not root.is_dir():
        return []
    return sorted(root.rglob('*.wav'))[:limit]


def mri_images(limit=4):
    """MRI slices live outside the repo; see the scratchpad note in the fixture."""
    scratch = Path(os.environ.get('SYNAPSE_MRI_SAMPLES', ''))
    if scratch.is_dir():
        return sorted(scratch.glob('*.jpg'))[:limit]
    return []


@pytest.fixture(autouse=True)
def real_predictors():
    """Import the genuine inference modules, not the conftest stand-ins.

    conftest registers stubs for these under the same names so the view tests
    stay fast. Left in place they would shadow the real modules here, and a
    test asserting "the model returned a valid label" would pass against a
    hardcoded one.
    """
    import sys

    from tests import conftest

    names = ('synapse.app.data.predict', 'synapse.predict')
    for name in names:
        sys.modules.pop(name, None)
    try:
        yield
    finally:
        for name in names:
            sys.modules.pop(name, None)
        conftest._install_predictor_stubs()


audio_only = pytest.mark.skipif(not REAL_AUDIO, reason='librosa/scikit-learn/joblib not installed')
vision_only = pytest.mark.skipif(not REAL_VISION, reason='tensorflow not installed')
ml_only = pytest.mark.skipif(not REAL_ML, reason='sentence-transformers/faiss not installed')
stt_only = pytest.mark.skipif(not REAL_STT, reason='faster-whisper not installed')


# ------------------------------------------------------- audio: features

@audio_only
def test_there_are_recordings_to_test_with():
    assert sample_wavs('Dementia'), 'no Dementia samples under resources/'
    assert sample_wavs('NoDementia'), 'no NoDementia samples under resources/'


@audio_only
def test_feature_extraction_returns_the_expected_width():
    """The classifier was fitted on exactly this many features."""
    from synapse.app.data.extract_features import extract_features

    wav = sample_wavs('Dementia', 1)[0]
    features = extract_features(str(wav))

    assert features is not None, f'extraction returned None for {wav.name}'
    assert features.shape == (EXPECTED_FEATURES,), f'got shape {features.shape}'
    assert np.isfinite(features).all(), 'feature vector contains NaN or inf'


@audio_only
def test_windows_share_the_training_feature_shape():
    """Inference averages over windows; each must look like a training row."""
    from synapse.app.data.extract_features import extract_feature_windows

    windows = extract_feature_windows(str(sample_wavs('Dementia', 1)[0]), max_windows=3)
    assert windows, 'no windows extracted'
    for window in windows:
        assert window.shape == (EXPECTED_FEATURES,)


@audio_only
def test_unreadable_audio_yields_no_windows():
    from synapse.app.data.extract_features import extract_feature_windows

    import tempfile

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as handle:
        handle.write(b'not audio')
        path = handle.name
    assert extract_feature_windows(path) == []


@audio_only
def test_feature_extraction_is_deterministic():
    from synapse.app.data.extract_features import extract_features

    wav = str(sample_wavs('Dementia', 1)[0])
    assert np.allclose(extract_features(wav), extract_features(wav))


@audio_only
def test_feature_extraction_rejects_a_non_audio_file(tmp_path):
    from synapse.app.data.extract_features import extract_features

    junk = tmp_path / 'not-audio.wav'
    junk.write_bytes(b'this is not a wav file at all')
    assert extract_features(str(junk)) is None


# --------------------------------------------------- audio: scaler/model

@audio_only
def test_classifier_expects_the_features_we_produce():
    """A mismatch here means every prediction is made from the wrong basis."""
    import joblib

    model = joblib.load(MODELS_DIR / 'dementia_model.pkl')
    assert model.n_features_in_ == EXPECTED_FEATURES
    assert set(model.classes_.tolist()) <= {0, 1}, f'unexpected classes {model.classes_}'


@audio_only
def test_scaling_lives_inside_the_pipeline():
    """One artifact, so the scaler cannot drift out of step with the model."""
    assert not (MODELS_DIR / 'scaler.pkl').exists(), 'stale standalone scaler'


@audio_only
def test_model_card_records_how_it_was_measured():
    from synapse.app.data.predict import get_model_card

    card = get_model_card()
    assert card, 'no model card shipped alongside the model'
    assert 'speaker-disjoint' in card['evaluation'], (
        'metrics must come from a speaker-disjoint split; a split that shares '
        'speakers measures voice recognition'
    )
    assert card['n_features'] == EXPECTED_FEATURES
    assert 0.0 < card['decision_threshold'] < 1.0


@audio_only
def test_shipped_model_catches_most_cases():
    """The regression that matters.

    The previous model scored 7.7% sensitivity: it answered "No Dementia" to
    almost everyone and told 24 of 26 people who had dementia they were clear.
    False reassurance is the worst failure mode for a screening tool.
    """
    from synapse.app.data.predict import get_model_card

    metrics = get_model_card()['metrics_at_threshold']
    assert metrics['sensitivity'] >= 0.75, (
        f'sensitivity fell to {metrics["sensitivity"]:.1%}'
    )


@audio_only
def test_decision_threshold_is_taken_from_the_card():
    """Leaving it at 0.5 is what produced the 7.7% sensitivity."""
    from synapse.app.data.predict import decision_threshold, get_model_card

    assert decision_threshold() == get_model_card()['decision_threshold']


@audio_only
def test_probabilities_are_calibrated():
    import joblib
    from sklearn.calibration import CalibratedClassifierCV

    model = joblib.load(MODELS_DIR / 'dementia_model.pkl')
    assert isinstance(model, CalibratedClassifierCV), (
        'an uncalibrated score presented as a confidence is not a probability'
    )


@audio_only
def test_confidence_is_reported_for_the_label_that_was_returned():
    """A "No Dementia" answer reports how sure it is of that, not of the opposite."""
    from synapse.app.data.predict import NO_DEMENTIA, classify_probability

    label, confidence = classify_probability(0.02)
    assert label == NO_DEMENTIA
    assert confidence == pytest.approx(0.98)


@audio_only
def test_real_recordings_produce_a_valid_outcome():
    from synapse.app.data.predict import predict_audio

    for wav in sample_wavs('NoDementia', 2) + sample_wavs('Dementia', 2):
        label, confidence = predict_audio(str(wav))
        assert label in AUDIO_LABELS, f'{wav.name}: {label!r}'
        assert 0.0 <= confidence <= 1.0


# ------------------------------------------------------------------- MRI

@vision_only
def test_mri_model_loads_and_has_the_expected_shape():
    from synapse.predict import get_model

    model = get_model()
    assert model.input_shape[1:3] == (128, 128), f'input shape {model.input_shape}'
    assert model.output_shape[-1] == len(MRI_LABELS), f'output shape {model.output_shape}'


@vision_only
def test_predict_mri_on_real_slices():
    images = mri_images()
    if not images:
        pytest.skip('no MRI samples available (set SYNAPSE_MRI_SAMPLES)')

    from synapse.predict import predict_mri

    for image in images:
        label, confidence = predict_mri(str(image))
        assert label in MRI_LABELS, f'{image.name}: unexpected label {label!r}'
        assert 0.0 <= confidence <= 1.0, f'{image.name}: confidence {confidence}'


@vision_only
def test_every_mri_label_maps_to_a_risk_level():
    from synapse.utils import get_risk_level

    for label in MRI_LABELS:
        assert get_risk_level(label, 'MRI') is not None, f'{label!r} is unmapped'


# -------------------------------------------------- semantic memory (real)

@ml_only
def test_real_embeddings_store_and_recall(tmp_path):
    """The whole memory suite runs on stand-ins; this one uses the real model."""
    from models_wrapper.faiss_memory import FAISSMemory

    store = FAISSMemory(memory_dir=tmp_path / 'faiss_memory')
    assert store.enabled, 'embedder failed to load'

    store.store('I left my keys on the kitchen table', 'keys', 'location', user_key='u1')
    store.store('My daughter Priya visits on Sundays', 'daughter', 'relationship', user_key='u1')

    results = store.search('where are my keys', user_key='u1')
    assert results, 'nothing recalled'
    assert 'keys' in results[0]['text'].lower(), f'wrong memory came back: {results[0]["text"]!r}'


@ml_only
def test_real_embeddings_survive_a_restart(tmp_path):
    from models_wrapper.faiss_memory import FAISSMemory

    directory = tmp_path / 'faiss_memory'
    FAISSMemory(memory_dir=directory).store(
        'I put my glasses in the bedside drawer', 'glasses', 'location', user_key='u1'
    )

    reopened = FAISSMemory(memory_dir=directory)
    assert len(reopened.metadata) == 1
    assert reopened.index.ntotal == 1
    assert reopened.search('glasses', user_key='u1')


@ml_only
def test_real_embedding_dimension_matches_the_index(tmp_path):
    from models_wrapper.faiss_memory import FAISSMemory

    store = FAISSMemory(memory_dir=tmp_path / 'faiss_memory')
    vector = store._encode(['a sentence'])
    assert vector.shape == (1, store.dimension), f'got {vector.shape}'


# ------------------------------------------------------------------- STT

@stt_only
@pytest.mark.slow
def test_whisper_transcribes_a_real_recording():
    """Downloads the model on first run."""
    from faster_whisper import WhisperModel

    wav = sample_wavs('NoDementia', 1)
    if not wav:
        pytest.skip('no recordings available')

    model = WhisperModel('tiny', device='cpu', compute_type='int8')
    segments, info = model.transcribe(str(wav[0]), beam_size=1, vad_filter=True)
    text = ' '.join(s.text for s in segments).strip()

    assert info.language, 'no language detected'
    assert text, 'transcription was empty'


# ------------------------------------------------- malformed model output

@audio_only
def test_a_malformed_score_produces_no_verdict():
    """NaN compares false against everything, so it reached Inconclusive."""
    from synapse.app.data.predict import classify_probability

    assert classify_probability(float('nan')) == (None, None)
    assert classify_probability(None) == (None, None)
    assert classify_probability('not a number') == (None, None)


@audio_only
def test_confidence_can_never_exceed_certainty():
    """A probability outside [0, 1] used to be shown as 150% confidence."""
    from synapse.app.data.predict import classify_probability

    for probability in (-0.5, 1.5, -100, 100):
        label, confidence = classify_probability(probability)
        assert label is not None
        assert 0.0 <= confidence <= 1.0, f'{probability} -> {confidence}'


@audio_only
def test_band_edges_fall_on_the_documented_side():
    from synapse.app.data.predict import (
        DEMENTIA, INCONCLUSIVE, NO_DEMENTIA, band_cuts, classify_probability,
    )

    low, high = band_cuts()
    assert classify_probability(low - 1e-6)[0] == NO_DEMENTIA
    assert classify_probability(low)[0] == INCONCLUSIVE
    assert classify_probability(high - 1e-6)[0] == INCONCLUSIVE
    assert classify_probability(high)[0] == DEMENTIA
