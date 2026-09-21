"""Regression tests for the semantic memory store.

The store previously wrote a datetime into a JSON document. json.dump streams,
so it emitted every field up to "timestamp": and then raised, leaving a
truncated file behind that the next startup could not parse. Every write
destroyed the store; the repository still carries nine quarantined files from
that failure. These tests pin the behaviour that must not regress.
"""

import json

import numpy as np
import pytest


def test_store_round_trips_to_disk(memory_store):
    assert memory_store.store('i kept my key on a table', 'keys', 'fact') is True

    raw = memory_store._metadata_file.read_text(encoding='utf-8')
    records = json.loads(raw)  # would raise on a truncated document

    assert len(records) == 1
    assert records[0]['text'] == 'i kept my key on a table'
    assert isinstance(records[0]['timestamp'], str)


def test_timestamp_is_json_serialisable(memory_store):
    """The exact failure: a datetime timestamp aborts the write mid-file."""
    memory_store.store('remember the appointment', 'memory', 'fact')
    record = memory_store.metadata[0]

    # Serialising the record on its own must not raise.
    json.dumps(record)


def test_memories_survive_a_restart(memory_store):
    from models_wrapper.faiss_memory import FAISSMemory

    memory_store.store('i left my wallet on the shelf', 'wallet', 'fact')
    memory_store.store('my daughter is called Priya', 'daughter', 'relationship')

    reopened = FAISSMemory(dimension=memory_store.dimension, memory_dir=memory_store.memory_dir)

    assert len(reopened.metadata) == 2
    assert {r['text'] for r in reopened.metadata} == {
        'i left my wallet on the shelf',
        'my daughter is called Priya',
    }
    # The index is rebuilt from the embeddings, so search works immediately.
    assert reopened.index.ntotal == 2
    assert reopened.search('where is my wallet')


def test_non_ascii_text_survives_a_restart(memory_store):
    """Hindi is a supported input language; cp1252 would mangle it."""
    from models_wrapper.faiss_memory import FAISSMemory

    memory_store.store('मैंने चाबी मेज़ पर रखी', 'keys', 'fact')

    reopened = FAISSMemory(dimension=memory_store.dimension, memory_dir=memory_store.memory_dir)
    assert reopened.metadata[0]['text'] == 'मैंने चाबी मेज़ पर रखी'


def test_updating_a_memory_reembeds_it(memory_store):
    """A conflict update used to change the text but leave the old vector."""
    memory_store.store('my keys are on the kitchen table', 'keys', 'fact')
    original = memory_store.embeddings[0].copy()

    memory_store.store('my keys are in the bedroom drawer', 'keys', 'fact')

    assert len(memory_store.metadata) == 1
    assert memory_store.metadata[0]['text'] == 'my keys are in the bedroom drawer'
    assert memory_store.metadata[0]['previous_value'] == 'my keys are on the kitchen table'
    assert not np.allclose(memory_store.embeddings[0], original), 'vector must follow the text'

    top = memory_store.search('where are my keys')[0]
    assert top['text'] == 'my keys are in the bedroom drawer'


def test_index_and_metadata_stay_aligned(memory_store):
    for i in range(5):
        memory_store.store(f'memory number {i}', f'entity{i}', 'fact')

    assert memory_store.index.ntotal == len(memory_store.metadata)
    assert memory_store.embeddings.shape[0] == len(memory_store.metadata)


def test_corrupt_metadata_is_quarantined_not_fatal(memory_store):
    from models_wrapper.faiss_memory import FAISSMemory

    memory_store.store('something worth keeping', 'thing', 'fact')
    memory_store._metadata_file.write_text('[{"id": 0, "timestamp": ', encoding='utf-8')

    reopened = FAISSMemory(dimension=memory_store.dimension, memory_dir=memory_store.memory_dir)

    assert reopened.metadata == []
    assert list(memory_store.memory_dir.glob('metadata.json.bad_*'))


def test_embeddings_are_rebuilt_when_missing(memory_store):
    """Losing the vectors must not lose the text - it can be re-embedded."""
    from models_wrapper.faiss_memory import FAISSMemory

    memory_store.store('the spare key is under the mat', 'keys', 'fact')
    memory_store._embeddings_file.unlink()

    reopened = FAISSMemory(dimension=memory_store.dimension, memory_dir=memory_store.memory_dir)

    assert len(reopened.metadata) == 1
    assert reopened.index.ntotal == 1
    assert reopened.search('spare key')


def test_search_is_scoped_by_user(memory_store):
    memory_store.store('my keys are on my own desk', 'keys', 'fact', user_key='alice')
    memory_store.store('my keys are in a locker', 'keys', 'fact', user_key='bob')

    alice = memory_store.search('where are my keys', user_key='alice')
    bob = memory_store.search('where are my keys', user_key='bob')

    assert [r['text'] for r in alice] == ['my keys are on my own desk']
    assert [r['text'] for r in bob] == ['my keys are in a locker']


def test_empty_text_is_rejected(memory_store):
    assert memory_store.store('   ', 'keys', 'fact') is False
    assert memory_store.metadata == []
