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

#: The feature vector the scaler and classifier were fitted on:
#: MFCC(40) + chroma(12) + spectral contrast(7) + ZCR(1).
EXPECTED_FEATURES = 60

AUDIO_LABELS = {'Dementia', 'No Dementia'}
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
    """The classifier and scaler were fitted on exactly 60 features."""
    from synapse.app.data.extract_features import extract_features

    wav = sample_wavs('Dementia', 1)[0]
    features = extract_features(str(wav))

    assert features is not None, f'extraction returned None for {wav.name}'
    assert features.shape == (EXPECTED_FEATURES,), f'got shape {features.shape}'
    assert np.isfinite(features).all(), 'feature vector contains NaN or inf'


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
def test_scaler_expects_the_features_we_produce():
    """A mismatch here means every prediction is scaled from the wrong basis."""
    import joblib

    scaler = joblib.load(MODELS_DIR / 'scaler.pkl')
    assert scaler.n_features_in_ == EXPECTED_FEATURES


@audio_only
def test_classifier_expects_the_features_we_produce():
    import joblib

    model = joblib.load(MODELS_DIR / 'dementia_model.pkl')
    assert model.n_features_in_ == EXPECTED_FEATURES
    assert set(model.classes_.tolist()) <= {0, 1}, f'unexpected classes {model.classes_}'


@audio_only
def test_predict_audio_on_real_recordings():
    from synapse.app.data.predict import predict_audio

    for wav in sample_wavs('Dementia', 2) + sample_wavs('NoDementia', 2):
        label, confidence = predict_audio(str(wav))

        assert label in AUDIO_LABELS, f'{wav.name}: unexpected label {label!r}'
        assert 0.0 <= confidence <= 1.0, f'{wav.name}: confidence {confidence} out of range'


@audio_only
def test_predict_audio_returns_none_pair_for_unreadable_input(tmp_path):
    """The contract the view relies on to return 422 instead of crashing."""
    from synapse.app.data.predict import predict_audio

    junk = tmp_path / 'broken.wav'
    junk.write_bytes(b'nope')
    assert predict_audio(str(junk)) == (None, None)


@audio_only
def test_every_audio_label_maps_to_a_risk_level():
    """Ties the model's real vocabulary to the risk table."""
    from synapse.utils import get_risk_level

    for label in AUDIO_LABELS:
        assert get_risk_level(label, 'AUDIO') is not None, f'{label!r} is unmapped'
    assert get_risk_level('No Dementia', 'AUDIO') == 'LOW'


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
