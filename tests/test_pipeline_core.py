"""Pipeline behaviour: interruption, worker resilience and safety rules."""

import asyncio

import pytest

from pipeline.safety import apply_safety_rules, keep_single_question, shorten
from pipeline.worker import PipelineWorker


class FakePipeline:
    """Minimal stand-in exposing what PipelineWorker touches."""

    def __init__(self):
        self.shutdown_event = asyncio.Event()
        self.work_queue = asyncio.Queue()
        self.generation = 0

    def is_stale(self, generation):
        return generation is not None and generation != self.generation


class CountingWorker(PipelineWorker):
    name = 'counting'
    input_queue_name = 'work_queue'

    def __init__(self, pipeline):
        super().__init__(pipeline)
        self.handled = []
        self.errors = []

    async def handle(self, item):
        if item == 'boom':
            raise ValueError('bad turn')
        self.handled.append(item)

    async def on_error(self, item, exc):
        self.errors.append(item)


# ------------------------------------------------------------- resilience

async def test_worker_survives_a_failing_item():
    """A single bad turn used to end the worker for the whole connection."""
    pipeline = FakePipeline()
    worker = CountingWorker(pipeline)
    task = asyncio.create_task(worker.run())

    for item in ('one', 'boom', 'two', 'three'):
        await pipeline.work_queue.put(item)

    await asyncio.sleep(0.3)
    pipeline.shutdown_event.set()
    await asyncio.wait_for(task, timeout=2)

    assert worker.handled == ['one', 'two', 'three'], 'worker kept going after the failure'
    assert worker.errors == ['boom']


async def test_worker_stops_on_shutdown():
    pipeline = FakePipeline()
    worker = CountingWorker(pipeline)
    task = asyncio.create_task(worker.run())

    await asyncio.sleep(0.05)
    pipeline.shutdown_event.set()
    await asyncio.wait_for(task, timeout=2)
    assert task.done()


# ------------------------------------------------------------- interruption

def test_generation_marks_superseded_work():
    pipeline = FakePipeline()
    assert not pipeline.is_stale(0)

    pipeline.generation = 1
    assert pipeline.is_stale(0), 'work from the previous turn must be dropped'
    assert not pipeline.is_stale(1)


def test_missing_generation_is_never_stale():
    pipeline = FakePipeline()
    pipeline.generation = 5
    assert not pipeline.is_stale(None)


# ----------------------------------------------------------- safety rules

def test_only_one_question_survives():
    """Two questions in a turn are hard to hold in working memory."""
    text = 'Where did you leave it? Was it the kitchen? Or the bedroom?'
    assert keep_single_question(text) == 'Where did you leave it?'


def test_single_question_is_left_alone():
    text = 'Where did you leave it?'
    assert keep_single_question(text) == text


def test_statement_with_one_question_is_kept_whole():
    text = 'I saved that for you. Where did you leave it?'
    assert keep_single_question(text) == text


def test_long_replies_are_trimmed_to_whole_sentences():
    text = ' '.join(f'This is sentence number {i}.' for i in range(30))
    result = shorten(text, limit=100)
    assert len(result) <= 100
    assert result.endswith('.')


def test_short_replies_are_untouched():
    text = 'Your keys are on the kitchen table.'
    assert apply_safety_rules(text) == text


def test_not_found_is_softened():
    result = apply_safety_rules('I could not find that in memory yet.')
    assert 'could not find' not in result.lower()
    assert result


def test_safety_rules_handle_empty_input():
    assert apply_safety_rules('') == ''
    assert apply_safety_rules(None) is None


# --------------------------------------------------- small-talk guardrail

def test_greetings_are_not_stored_as_memories():
    """A small router calls almost everything memory_store.

    The 0.5b model answers memory_store for "Hello there" at confidence 1.0,
    and the slot checks do not catch it: a greeting is missing both an object
    and a location rather than exactly one of them.
    """
    from pipeline.qwen_router import is_storable

    for text in ('Hello there', 'Good morning, how are you?', 'Thanks very much',
                 'yes', 'ok', 'Bye', 'Thank you'):
        assert not is_storable(text), f'{text!r} would be stored as a memory'


def test_real_memories_are_still_stored():
    from pipeline.qwen_router import is_storable

    for text in ('I left my keys on the kitchen table',
                 'My daughter is called Priya',
                 'I put the tablets in the bedside drawer',
                 'I read up to page 78'):
        assert is_storable(text), f'{text!r} should be storable'


def test_empty_input_is_not_storable():
    from pipeline.qwen_router import is_storable

    assert not is_storable('')
    assert not is_storable(None)
    assert not is_storable('   ')


# ------------------------------------------- deterministic memory statements

def test_clear_memory_statements_are_recognised_without_the_model():
    """A 1.5b router calls "I left my keys on the table" a retrieval.

    Nothing then gets stored and the person is told "Okay." Reminders avoid
    that by being handled in Python; the unambiguous memory statements are
    settled the same way, and the rest still go to the model.
    """
    from pipeline.qwen_router import looks_like_memory_statement

    for text in ('I left my keys on the kitchen table',
                 'I put the tablets in the bedside drawer',
                 'I kept my wallet in the blue bowl',
                 'Please remember that my appointment is Tuesday',
                 'My daughter is called Priya'):
        assert looks_like_memory_statement(text), f'{text!r} should be stored'


def test_questions_are_never_treated_as_statements():
    """"Where did I put my glasses" shares its verb with the statement form."""
    from pipeline.qwen_router import looks_like_memory_statement

    for text in ('Where did I put my glasses?',
                 'Where did I leave my keys',
                 'what did i put there',
                 'Do you remember where I put my wallet',
                 'Can you remember where I left it',
                 'Is my wallet on the table'):
        assert not looks_like_memory_statement(text), f'{text!r} is a question'


def test_small_talk_is_not_a_memory_statement():
    from pipeline.qwen_router import looks_like_memory_statement

    for text in ('Hello there', 'Thanks very much', '', None):
        assert not looks_like_memory_statement(text)


def test_a_memory_with_a_hole_in_it_is_left_to_the_clarifier():
    """"I put my glasses somewhere" has no place, so it must be asked about.

    Storing it as it stands would record a memory that cannot answer the
    question it exists to answer.
    """
    from pipeline.qwen_router import looks_like_memory_statement

    for text in ('I put my glasses somewhere', 'I left it there', 'I kept them safe'):
        assert not looks_like_memory_statement(text), f'{text!r} needs a question first'


# ------------------------------------------------------------- phrasing

def test_quoted_text_is_moved_into_the_second_person():
    """"remind me to take my tablets" must not become "take my tablets".

    Echoed verbatim, the assistant ends up talking about its own tablets.
    """
    from pipeline.phrasing import to_second_person

    assert to_second_person('take my tablets') == 'take your tablets'
    assert to_second_person('I left my keys on the table') == 'You left your keys on the table'
    assert to_second_person('call my daughter') == 'call your daughter'
    assert to_second_person('I am cold') == 'You are cold'
    assert to_second_person("I'm tired") == "You're tired"


def test_phrasing_leaves_other_words_alone():
    from pipeline.phrasing import to_second_person

    assert to_second_person('the milk is in the fridge') == 'the milk is in the fridge'
    # "my" inside a longer word must not be touched.
    assert to_second_person('mystery novel') == 'mystery novel'


def test_reminder_confirmation_uses_the_persons_terms():
    from pipeline.phrasing import confirm_reminder

    assert confirm_reminder('take my tablets', '7:37 PM') == (
        "I'll remind you to take your tablets at 7:37 PM."
    )
    assert 'my tablets' not in confirm_reminder('take my tablets', '7:37 PM')


def test_memory_confirmation_is_not_a_thank_you():
    """At 1.5b the model answered "Thank you for remembering to save that".

    The assistant was thanking the person for doing its own job, so the
    confirmation is built rather than generated.
    """
    from pipeline.phrasing import acknowledge_memory

    reply = acknowledge_memory('I left my keys on the kitchen table')
    assert reply == "I'll remember that you left your keys on the kitchen table."
    assert 'thank' not in reply.lower()


def test_recall_reads_a_memory_back_correctly():
    from pipeline.phrasing import answer_from_memory

    # The old fallback swapped only a leading "I", leaving "your" as "my".
    assert answer_from_memory('I left my keys on the kitchen table') == (
        'You left your keys on the kitchen table.'
    )
    assert answer_from_memory('') == "I don't have that written down yet."


def test_phrasing_handles_missing_parts():
    from pipeline.phrasing import confirm_reminder, describe_reminder

    assert confirm_reminder('', '') == "I'll remind you."
    assert confirm_reminder('take your pills', '') == "I'll remind you to take your pills."
    assert describe_reminder('', '') == 'You have a reminder set.'
