"""Test fixtures.

The heavy ML wheels (torch, faiss, sentence-transformers, tensorflow) are not
needed to exercise the application logic, and pulling them in would make the
suite unrunnable on most machines. They are replaced here with small
deterministic stand-ins that honour the same contracts the real libraries do.
"""

import hashlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_APP_DIR = REPO_ROOT / 'src' / 'synapse'
FINAL_ECLIPSE_DIR = REPO_ROOT / 'FinalEclipse' / 'project'

for path in (SRC_APP_DIR, FINAL_ECLIPSE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


DIMENSION = 384


def _deterministic_embedding(text, dimension=DIMENSION):
    """Stable pseudo-embedding: same text always maps to the same vector."""
    digest = hashlib.sha256(text.encode('utf-8')).digest()
    seed = int.from_bytes(digest[:8], 'big') % (2 ** 32)
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(dimension).astype(np.float32)
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


class _FakeSentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return _deterministic_embedding(texts)
        return np.stack([_deterministic_embedding(t) for t in texts])


class _FakeIndexFlatL2:
    """Exact L2 index over a numpy matrix - same semantics as IndexFlatL2."""

    def __init__(self, dimension):
        self.d = dimension
        self._vectors = np.zeros((0, dimension), dtype=np.float32)

    @property
    def ntotal(self):
        return int(self._vectors.shape[0])

    def add(self, vectors):
        vectors = np.asarray(vectors, dtype=np.float32)
        self._vectors = np.vstack([self._vectors, vectors]) if self.ntotal else vectors

    def search(self, queries, k):
        queries = np.asarray(queries, dtype=np.float32)
        if self.ntotal == 0:
            empty = np.full((queries.shape[0], k), -1, dtype=np.int64)
            return np.full((queries.shape[0], k), np.inf, dtype=np.float32), empty

        k = min(k, self.ntotal)
        distances = ((queries[:, None, :] - self._vectors[None, :, :]) ** 2).sum(axis=2)
        order = np.argsort(distances, axis=1)[:, :k]
        best = np.take_along_axis(distances, order, axis=1)
        return best.astype(np.float32), order.astype(np.int64)


def _install_ml_stubs():
    if 'sentence_transformers' not in sys.modules:
        module = types.ModuleType('sentence_transformers')
        module.SentenceTransformer = _FakeSentenceTransformer
        sys.modules['sentence_transformers'] = module

    if 'faiss' not in sys.modules:
        module = types.ModuleType('faiss')
        module.IndexFlatL2 = _FakeIndexFlatL2
        sys.modules['faiss'] = module


def _install_predictor_stubs():
    """Stand in for the two inference modules.

    Importing them for real pulls in joblib, librosa and tensorflow. The views
    import them lazily inside the request, so a stub module here is enough and
    each test can set the return value it needs.
    """
    for name in ('synapse.app.data.predict', 'synapse.predict'):
        module = types.ModuleType(name)
        module.predict_audio = lambda path: ('No Dementia', 0.9)
        module.predict_mri = lambda path: ('No Impairment', 0.9)
        sys.modules[name] = module


_install_ml_stubs()
_install_predictor_stubs()


@pytest.fixture
def audio_predictor():
    """Set the audio model's return value (or raise) for one test."""
    module = sys.modules['synapse.app.data.predict']
    original = module.predict_audio

    def _set(fn):
        module.predict_audio = fn

    yield _set
    module.predict_audio = original


@pytest.fixture
def memory_store(tmp_path):
    """A FAISSMemory rooted in a throwaway directory."""
    from models_wrapper.faiss_memory import FAISSMemory

    return FAISSMemory(dimension=DIMENSION, memory_dir=tmp_path / 'faiss_memory')
