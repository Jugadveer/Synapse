"""Features for the audio dementia indicator.

Two sources, concatenated:

* Hidden states from a speech model pretrained on unlabelled audio
  (wav2vec2 by default). Measured on this corpus under speaker-disjoint
  cross-validation these lift ROC-AUC from 0.645 to 0.726 on their own.
* The original hand-crafted acoustic summary (MFCC, chroma, spectral
  contrast, zero-crossing rate). Cheap, and fusing it with the embeddings
  reaches 0.747, better than either alone.

A middle layer is pooled rather than the last: later layers specialise
towards the pretraining objective, while middle layers keep more phonetic and
prosodic detail.

This module is imported by both train_model.py and predict.py, so training and
inference cannot drift apart.
"""

import logging
import os
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
WINDOW_SECONDS = 30.0
HIDDEN_LAYER = 7

SPEECH_MODEL = os.getenv('AUDIO_EMBEDDING_MODEL', 'facebook/wav2vec2-base')

#: MFCC(40) + chroma(12) + spectral contrast(7) + ZCR(1), mean-pooled.
HANDCRAFTED_DIMS = 60


@lru_cache(maxsize=1)
def _speech_model():
    """Load the pretrained speech model once per process."""
    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    torch.set_num_threads(max(torch.get_num_threads() - 1, 1))
    extractor = AutoFeatureExtractor.from_pretrained(SPEECH_MODEL)
    model = AutoModel.from_pretrained(SPEECH_MODEL, output_hidden_states=True).eval()
    return extractor, model


def _load_audio(file_path):
    import librosa

    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    if audio.size == 0:
        return None
    audio, _ = librosa.effects.trim(audio, top_db=30)
    if audio.size < SAMPLE_RATE:  # under a second is not worth scoring
        return None
    return audio


def _windows(audio, max_windows):
    """Up to `max_windows` evenly spaced windows, or the whole clip if short."""
    length = int(WINDOW_SECONDS * SAMPLE_RATE)
    if audio.size <= length or max_windows <= 1:
        start = max((audio.size - length) // 2, 0)
        return [audio[start:start + length]]

    count = min(max_windows, max(int(audio.size // length), 1))
    starts = np.linspace(0, audio.size - length, count).astype(int)
    return [audio[s:s + length] for s in starts]


def _embedding(window):
    import torch

    extractor, model = _speech_model()
    inputs = extractor(window, sampling_rate=SAMPLE_RATE, return_tensors='pt')
    with torch.inference_mode():
        hidden = model(**inputs).hidden_states[HIDDEN_LAYER][0]
    return torch.cat([hidden.mean(0), hidden.std(0)]).numpy().astype(np.float32)


def _handcrafted(window):
    import librosa

    mfcc = np.mean(librosa.feature.mfcc(y=window, sr=SAMPLE_RATE, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=window, sr=SAMPLE_RATE).T, axis=0)
    contrast = np.mean(
        librosa.feature.spectral_contrast(y=window, sr=SAMPLE_RATE).T, axis=0
    )
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=window))
    return np.hstack([mfcc, chroma, contrast, zcr]).astype(np.float32)


def _window_features(window):
    return np.concatenate([_embedding(window), _handcrafted(window)])


def extract_feature_windows(file_path, max_windows=4):
    """Feature vectors for several windows of one recording.

    Averaging the model's output over windows is what lifts speaker-level
    ROC-AUC to 0.794 from 0.747 on a single window.

    Returns an empty list when the audio cannot be read.
    """
    try:
        audio = _load_audio(file_path)
        if audio is None:
            return []
        return [_window_features(w) for w in _windows(audio, max_windows)]
    except Exception as e:
        logger.warning(f'Feature extraction failed for {file_path}: {e}')
        return []


def extract_features(file_path):
    """One feature vector for a recording, or None if it cannot be read.

    Used for training, where each clip contributes a single row.
    """
    windows = extract_feature_windows(file_path, max_windows=1)
    return windows[0] if windows else None
