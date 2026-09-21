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
