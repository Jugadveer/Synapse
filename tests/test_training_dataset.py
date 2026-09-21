"""Train/serve parity for the intent router.

The fine-tune taught the model a three-field answer while the router asked for
an eight-field one, under different instructions. `confidence` was never
produced, the pipeline substituted 0.7, and that default sat below its own 0.8
clarification threshold - so every reasoning turn was diverted into a question
and the Mistral layer was unreachable. These tests fail if the two drift apart
again.
"""

import json

import pytest

from pipeline.prompts import (
    CLASSIFY_SCHEMA,
    MEMORY_SCHEMA,
    SYSTEM_PROMPT,
    classify_prompt,
    memory_analyst_prompt,
)
from training.build_intent_dataset import build


@pytest.fixture(scope='module')
def rows():
    return build(400, seed=11)


def answers(rows):
    return [json.loads(r['messages'][2]['content']) for r in rows]


def test_dataset_is_chat_formatted(rows):
    for row in rows:
        roles = [m['role'] for m in row['messages']]
        assert roles == ['system', 'user', 'assistant']
        assert row['messages'][0]['content'] == SYSTEM_PROMPT


def test_every_answer_matches_a_runtime_schema(rows):
    classify_keys = set(CLASSIFY_SCHEMA)
    memory_keys = set(MEMORY_SCHEMA)

    for answer in answers(rows):
        keys = set(answer)
        assert keys in (classify_keys, memory_keys), f"unexpected fields: {sorted(keys)}"


def test_confidence_is_always_present_and_in_range(rows):
    """The field whose absence broke the clarification gate."""
    for answer in answers(rows):
        assert 'confidence' in answer
        assert 0.0 <= answer['confidence'] <= 1.0


def test_user_turn_is_the_exact_runtime_prompt(rows):
    """Training on different text than the router sends is the skew itself."""
    classify_head = classify_prompt('x').split('\n')[0]
    memory_head = memory_analyst_prompt('x').split('\n')[0]

    for row in rows:
        first_line = row['messages'][1]['content'].split('\n')[0]
        assert first_line in (classify_head, memory_head)


def test_memory_answers_carry_completeness(rows):
    memory_answers = [a for a in answers(rows) if 'information_completeness' in a]
    assert memory_answers, 'no memory-analyst examples were generated'

    for answer in memory_answers:
        completeness = answer['information_completeness']
        assert set(completeness) == {'is_complete', 'missing_fields', 'should_ask'}
        # An incomplete memory must ask, and a complete one must not.
        assert completeness['should_ask'] == answer['needs_clarification']
        if not completeness['is_complete']:
            assert answer['clarification_question'], 'incomplete memory with no question'


def test_incomplete_memories_are_represented(rows):
    """Asking rather than assuming is the behaviour being trained."""
    memory_answers = [a for a in answers(rows) if 'information_completeness' in a]
    asking = [a for a in memory_answers if a['needs_clarification']]
    assert len(asking) > len(memory_answers) * 0.2


def test_low_confidence_accompanies_unclear_turns(rows):
    unclear = [a for a in answers(rows) if a['intent'] == 'unclear']
    assert unclear
    assert all(a['confidence'] < 0.6 for a in unclear)


def test_clarification_threshold_is_reachable(rows):
    """Confident turns must clear the gate, or reasoning stays unreachable."""
    from pipeline.clarification_worker import CONFIDENCE_THRESHOLD

    confident = [a for a in answers(rows) if a['confidence'] >= CONFIDENCE_THRESHOLD]
    assert confident, 'no example clears the clarification threshold'


def test_dataset_covers_every_intent(rows):
    seen = {a['intent'] for a in answers(rows)}
    assert {'command', 'memory_store', 'memory_retrieve', 'unclear', 'casual', 'question'} <= seen


def test_build_is_deterministic():
    assert build(50, seed=3) == build(50, seed=3)
