"""Decision shaping and the clarification gate.

The router's output is whatever a small model produced, so normalisation has
to cope with missing fields, wrong types and values out of range. What it
decides then determines whether a turn is acted on or asked about.
"""

import asyncio

import pytest

from pipeline.clarification_worker import (
    CONFIDENCE_THRESHOLD, ClarificationWorker, mentions_any,
)
from pipeline.qwen_router import QwenRouter


class FakePipeline:
    def __init__(self):
        self.shutdown_event = asyncio.Event()
        self.intent_queue = asyncio.Queue()
        self.gpt_input_queue = asyncio.Queue()
        self.response_queue = asyncio.Queue()
        self.generation = 0
        self.user_key = 'u1'
        self.conversation_state = {
            'last_intent': None, 'pending_slots': {}, 'user_profile': {},
            'context_window': [], 'last_confirmation': None,
        }
        self.consumer = self
        self.decisions = []

    def is_stale(self, generation):
        return generation is not None and generation != self.generation

    async def send_decision(self, decision):
        self.decisions.append(decision)

    def update_conversation_context(self, user_text, decision):
        pass


@pytest.fixture
def router():
    return QwenRouter(FakePipeline())


@pytest.fixture
def gate():
    pipeline = FakePipeline()
    return ClarificationWorker(pipeline), pipeline


# --------------------------------------------------------- normalisation

def test_an_empty_decision_still_has_every_field(router):
    normalised = router._normalize({})

    for field in ('intent', 'is_fast_response', 'needs_reasoning', 'needs_gpt',
                  'needs_memory_storage', 'needs_memory_retrieval', 'confidence'):
        assert field in normalised
    assert normalised['intent'] == 'unclear'


def test_none_normalises_without_raising(router):
    assert router._normalize(None)['intent'] == 'unclear'


def test_a_missing_confidence_stays_missing(router):
    """Inventing 0.7 put every turn under the clarification threshold."""
    assert router._normalize({'intent': 'casual'})['confidence'] is None


@pytest.mark.parametrize('value,expected', [
    (0.9, 0.9),
    ('0.85', 0.85),
    (1, 1.0),
    ('nonsense', None),
    (None, None),
    ([], None),
    ({}, None),
])
def test_confidence_is_coerced_or_dropped(router, value, expected):
    assert router._normalize({'confidence': value})['confidence'] == expected


def test_intent_implies_the_memory_flags(router):
    assert router._normalize({'intent': 'memory_store'})['needs_memory_storage']
    assert router._normalize({'intent': 'memory_retrieve'})['needs_memory_retrieval']


def test_explicit_flags_win_over_the_intent(router):
    normalised = router._normalize(
        {'intent': 'memory_store', 'needs_memory_storage': False}
    )
    assert normalised['needs_memory_storage'] is False


def test_a_fast_response_always_has_something_to_say(router):
    assert router._normalize({'is_fast': True})['fast_response']


def test_needs_gpt_mirrors_needs_reasoning(router):
    assert router._normalize({'needs_reasoning': True})['needs_gpt'] is True
    assert router._normalize({'needs_reasoning': False})['needs_gpt'] is False


# ----------------------------------------------------- memory clarifying

def test_a_complete_memory_is_not_queried(router):
    decision = router._normalize({
        'intent': 'memory_store', 'confidence': 0.95,
        'information_completeness': {'is_complete': True, 'missing_fields': [], 'should_ask': False},
    })
    assert not router._needs_clarification(decision)


@pytest.mark.parametrize('completeness', [
    {'is_complete': False, 'missing_fields': [], 'should_ask': False},
    {'is_complete': True, 'missing_fields': ['location'], 'should_ask': False},
    {'is_complete': True, 'missing_fields': [], 'should_ask': True},
])
def test_any_sign_of_incompleteness_asks(router, completeness):
    decision = router._normalize({
        'intent': 'memory_store', 'confidence': 0.95,
        'information_completeness': completeness,
    })
    assert router._needs_clarification(decision)


def test_low_confidence_asks_even_when_complete(router):
    decision = router._normalize({'intent': 'memory_store', 'confidence': 0.4})
    assert router._needs_clarification(decision)


def test_a_missing_confidence_does_not_force_a_question(router):
    """Otherwise no memory could ever be stored on first mention."""
    decision = router._normalize({
        'intent': 'memory_store',
        'information_completeness': {'is_complete': True, 'missing_fields': [], 'should_ask': False},
    })
    assert not router._needs_clarification(decision)


def test_other_intents_are_never_memory_clarified(router):
    for intent in ('casual', 'question', 'command', 'memory_retrieve'):
        assert not router._needs_clarification(router._normalize({'intent': intent}))


# ------------------------------------------------------------- entities

@pytest.mark.parametrize('text,entity', [
    ('I left my keys on the table', 'keys'),
    ('my wallet is in the drawer', 'wallet'),
    ('where are my glasses', 'glasses'),
    ('the pills are in the cupboard', 'medicine'),
    ('my hearing aid is by the bed', 'hearing aid'),
])
def test_entities_are_recognised(router, text, entity):
    assert router._entity_for(text, {'intent': 'memory_store'}) == entity


def test_a_word_containing_an_entity_does_not_match(router):
    """"monkeys" contains "keys"; "turnips" contains no entity at all."""
    assert router._entity_for('the monkeys were loud', {'intent': 'casual'}) == 'user'


def test_an_unrecognised_store_still_gets_a_label(router):
    assert router._entity_for('something else entirely',
                              {'intent': 'memory_store'}) == 'memory'


# ---------------------------------------------------------- word matching

def test_terms_match_on_whole_words_only():
    """A substring test matched 'in' inside 'remind'."""
    assert mentions_any('I left it on the desk', ['desk'])
    assert not mentions_any('I went to the deskbound office', ['desk'])
    assert not mentions_any('reminder', ['in'])
    assert mentions_any('the key is here', ['key'])
    assert not mentions_any('the keyboard is here', ['key'])


def test_matching_ignores_case_and_handles_nothing():
    assert mentions_any('MY KEYS ARE HERE', ['keys'])
    assert not mentions_any('', ['keys'])
    assert not mentions_any(None, ['keys'])


# ------------------------------------------------------------ slot gate

def test_an_object_with_no_place_is_asked_about(gate):
    worker, _ = gate
    missing = worker._check_required_slots('memory_store', {}, 'I put my keys somewhere')
    assert 'location' in missing


def test_a_place_with_no_object_is_asked_about(gate):
    worker, _ = gate
    missing = worker._check_required_slots('memory_store', {}, 'I left it on the table')
    assert 'object' in missing


def test_a_complete_statement_needs_no_slots(gate):
    worker, _ = gate
    assert worker._check_required_slots(
        'memory_store', {}, 'I left my keys on the table'
    ) == {}


def test_only_one_question_is_ever_asked(gate):
    worker, _ = gate
    question = worker._slot_question({'location': 'Where?', 'object': 'What?'})
    assert question in ('Where?', 'What?')
    assert isinstance(question, str)


async def test_a_confident_turn_reaches_the_reasoning_layer(gate):
    worker, pipeline = gate
    await worker.handle({
        'user_text': 'Tell me about my appointment',
        'decision': {'intent': 'question', 'confidence': 0.95},
        'generation': 0,
    })
    assert pipeline.gpt_input_queue.qsize() == 1
    assert pipeline.response_queue.qsize() == 0


async def test_an_unsure_turn_is_asked_about_instead(gate):
    worker, pipeline = gate
    await worker.handle({
        'user_text': 'the thing, you know',
        'decision': {'intent': 'unclear', 'confidence': 0.3},
        'generation': 0,
    })
    assert pipeline.response_queue.qsize() == 1
    assert pipeline.gpt_input_queue.qsize() == 0


async def test_a_turn_with_no_confidence_is_not_blocked(gate):
    """A missing confidence is not evidence of doubt."""
    worker, pipeline = gate
    await worker.handle({
        'user_text': 'Tell me about my appointment',
        'decision': {'intent': 'question'},
        'generation': 0,
    })
    assert pipeline.gpt_input_queue.qsize() == 1


async def test_stale_work_is_dropped(gate):
    worker, pipeline = gate
    pipeline.generation = 2
    await worker.handle({
        'user_text': 'Hello there',
        'decision': {'intent': 'casual', 'confidence': 0.95},
        'generation': 1,
    })
    assert pipeline.gpt_input_queue.qsize() == 0
    assert pipeline.response_queue.qsize() == 0


def test_the_threshold_leaves_room_to_act():
    """A gate at 1.0 would divert everything."""
    assert 0.5 < CONFIDENCE_THRESHOLD < 1.0
