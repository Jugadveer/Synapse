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


# ------------------------------------------------- retrieval is built, not asked

def test_a_recalled_memory_is_phrased_without_the_model(router):
    """Asking the model to rewrite the memory ignored the context it was given.

    After the router was fine-tuned for JSON, "where did I leave my keys" came
    back as "Could you tell me where they were last?" while the answer sat in
    the context all along. It is built now, and costs no round trip.
    """
    answer = router._compose_retrieve_response(
        'where did I leave my keys', {}, 'I left my keys on the kitchen table'
    )
    assert answer == 'You left your keys on the kitchen table.'


def test_retrieval_phrasing_is_not_a_coroutine(router):
    """It makes no model call, so it must not need awaiting."""
    import inspect

    assert not inspect.iscoroutinefunction(router._compose_retrieve_response)


def test_nothing_recalled_says_so(router):
    for context in ('', '   ', None):
        assert router._compose_retrieve_response('where are my keys', {}, context) == (
            "I don't have that written down yet."
        )


def test_only_the_first_memory_is_read_back(router):
    """Several matches would be a mouthful; the closest one is the answer."""
    answer = router._compose_retrieve_response(
        'where are my keys', {},
        'I left my keys on the kitchen table\nI put my keys in the drawer',
    )
    assert answer == 'You left your keys on the kitchen table.'


# ------------------------------------------- spending a model call or not

@pytest.mark.parametrize('text', [
    'Hello there', 'Good morning', 'hi', 'thanks very much', 'thank you',
    'how are you going today', 'okay', 'yes', 'no', 'bye', 'sorry',
])
def test_courtesies_are_recognised_without_the_model(text):
    from pipeline.qwen_router import is_small_talk, needs_memory_analysis

    assert is_small_talk(text)
    assert not needs_memory_analysis(text)


@pytest.mark.parametrize('text', [
    'Where did I leave my keys',
    'where are my glasses',
    'I put my glasses somewhere',
    'I read up to page 78',
    'my daughter is called Priya',
    "what's the weather like",
    'tell me about my appointment',
    'sorry I cannot find my keys',
])
def test_a_turn_with_content_still_reaches_the_model(text):
    """The gate must not swallow anything that carries a noun.

    "sorry I cannot find my keys" opens with a courtesy and is not one.
    """
    from pipeline.qwen_router import is_small_talk, needs_memory_analysis

    assert not is_small_talk(text)
    assert needs_memory_analysis(text)


@pytest.mark.parametrize('text', ['', '   ', None])
def test_nothing_is_neither_small_talk_nor_analysed(text):
    from pipeline.qwen_router import is_small_talk, needs_memory_analysis

    assert not is_small_talk(text)
    assert not needs_memory_analysis(text)


# ------------------------------------------------- answering a courtesy

@pytest.mark.parametrize('text,expected', [
    ('Hello there', 'Hello. It is good to hear from you.'),
    ('Good morning', 'Good morning.'),
    ('good evening', 'Good evening.'),
    ('good night', 'Goodbye. Take care.'),
    ('bye', 'Goodbye. Take care.'),
    ('thanks very much', "You're welcome."),
    ('thank you', "You're welcome."),
    ('how are you going today', "I'm well, thank you. How are you?"),
    ('sorry', 'That is quite all right.'),
    ('okay', 'All right.'),
])
def test_each_courtesy_gets_its_own_reply(text, expected):
    """One canned greeting for all of them answered "thanks" with "Hello"."""
    from pipeline.phrasing import small_talk_reply

    assert small_talk_reply(text) == expected


def test_a_greeting_mirrors_the_time_of_day():
    """Answering "good evening" with "good morning" reads as inattentive."""
    from pipeline.phrasing import small_talk_reply

    assert small_talk_reply('good afternoon') == 'Good afternoon.'
    assert small_talk_reply('good evening') != small_talk_reply('good morning')


@pytest.mark.parametrize('text', [
    'I left my keys on the table', 'where are my keys', '', None,
])
def test_anything_with_substance_has_no_canned_reply(text):
    from pipeline.phrasing import small_talk_reply

    assert small_talk_reply(text) is None


# --------------------------------------- finishing a clarified memory

@pytest.mark.parametrize('original,answer,expected', [
    ('I put my glasses somewhere', 'on the bookshelf',
     'I put my glasses on the bookshelf'),
    ('I left it somewhere safe', 'in the top drawer',
     'I left it in the top drawer'),
    ('I put my glasses somewhere.', 'On the bookshelf.',
     'I put my glasses on the bookshelf'),
    ('I put my wallet down', 'by the front door',
     'I put my wallet down by the front door'),
])
def test_a_place_is_spliced_in_without_the_model(original, answer, expected):
    """This was the whole 14-field analysis prompt to join two strings."""
    from pipeline.phrasing import complete_with_location

    assert complete_with_location(original, answer) == expected


@pytest.mark.parametrize('answer', [
    'I think I moved them yesterday',   # not a slot fill
    'the bookshelf',                    # no preposition
    'in a very very very very very very very very very very long place',
    '',
])
def test_an_answer_that_is_not_a_place_is_left_to_the_model(answer):
    from pipeline.phrasing import complete_with_location

    assert complete_with_location('I put my glasses somewhere', answer) is None


@pytest.mark.parametrize('original', ['', None, 'somewhere'])
def test_nothing_to_splice_into_returns_nothing(original):
    from pipeline.phrasing import complete_with_location

    assert complete_with_location(original, 'on the shelf') is None


# ------------------------------------- recognising a memory turn outright

@pytest.mark.parametrize('text', [
    'Where did I leave my keys', 'where are my glasses', "where's my wallet",
    'what did I do with my phone', 'have you seen my hearing aid',
    'do you know where the tablets are',
])
def test_a_question_about_a_known_thing_is_a_retrieval(text):
    from pipeline.qwen_router import looks_like_memory_question

    assert looks_like_memory_question(text)


@pytest.mark.parametrize('text', [
    'where is the nearest chemist',      # nothing stored could answer it
    'where am I',
    'what is the weather like',
    'I left my keys on the table',       # a statement
    'tell me about my keys',             # not a where-question
    '', None,
])
def test_anything_else_is_left_to_the_model(text):
    """Answering "where is the nearest chemist" from an empty memory store
    with "I don't have that written down" would be worse than thinking."""
    from pipeline.qwen_router import looks_like_memory_question

    assert not looks_like_memory_question(text)


@pytest.mark.parametrize('text', [
    'I put my glasses somewhere', 'I left it somewhere safe',
    'I put my wallet down', 'I moved my keys',
])
def test_a_memory_with_no_place_is_recognised(text):
    from pipeline.qwen_router import looks_like_incomplete_memory

    assert looks_like_incomplete_memory(text)


@pytest.mark.parametrize('text', [
    'I left my keys on the kitchen table',       # complete
    'I put the tablets in the bedside drawer',
    'Where did I put my glasses',                # a question
    'Hello there',
    'my daughter is called Priya',               # not a placement
    '', None,
])
def test_a_complete_or_unrelated_turn_is_not_incomplete(text):
    from pipeline.qwen_router import looks_like_incomplete_memory

    assert not looks_like_incomplete_memory(text)


@pytest.mark.parametrize('text', [
    'I put my glasses somewhere', 'I left my keys on the kitchen table',
    'Where did I leave my keys', 'where are my glasses', 'Hello there',
    'I put the tablets in the bedside drawer', 'I moved my keys',
    'my daughter is called Priya', 'what is the weather like',
    'remind me to take my tablets in 10 minutes',
])
def test_the_deterministic_gates_do_not_overlap(text):
    """Each turn takes exactly one path, or none and goes to the model."""
    from pipeline.qwen_router import (
        looks_like_incomplete_memory, looks_like_memory_question,
        looks_like_memory_statement, is_small_talk,
    )

    hits = sum(bool(gate(text)) for gate in (
        looks_like_incomplete_memory, looks_like_memory_question,
        looks_like_memory_statement, is_small_talk,
    ))
    assert hits <= 1, f'{text!r} was claimed by {hits} gates'


@pytest.mark.parametrize('text,expected', [
    ('I put my glasses somewhere', 'Where did you put your glasses?'),
    ('I left it somewhere safe', 'Where did you leave it?'),
    ('I moved my keys', 'Where did you move your keys?'),
    ('I hid my documents', 'Where did you hide your documents?'),
    ('I kept my phone somewhere', 'Where did you keep your phone?'),
])
def test_the_question_echoes_the_verb_in_the_base_form(text, expected):
    """"Where did you left it?" - "did" takes the base form."""
    from pipeline.phrasing import ask_where
    from pipeline.qwen_router import mentions_entity, placement_verb

    assert ask_where(mentions_entity(text), placement_verb(text)) == expected


def test_the_question_names_the_thing_rather_than_saying_them():
    """Pronouns are what the person is having trouble with."""
    from pipeline.phrasing import ask_where

    assert ask_where('glasses') == 'Where did you put your glasses?'
    assert ask_where(None) == 'Where did you put it?'
    for placeholder in ('user', 'memory', 'fact'):
        assert ask_where(placeholder) == 'Where did you put it?'


def test_an_unknown_verb_falls_back_rather_than_mangling_the_sentence():
    from pipeline.phrasing import ask_where

    assert ask_where('keys', 'yeeted') == 'Where did you put your keys?'
    assert ask_where('keys', None) == 'Where did you put your keys?'
