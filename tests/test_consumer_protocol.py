"""Websocket protocol edges.

The consumer sits between an open socket and the pipeline, so it has to
survive whatever arrives: frames in the wrong order, frames of the wrong
shape, a flood of audio, and a client that disappears mid-turn.
"""

import asyncio
import json
from contextlib import asynccontextmanager

import pytest
from channels.testing import WebsocketCommunicator
from django.contrib.auth.models import User

from project.asgi import application
from tests.fake_ollama import FakeOllama

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.fixture(autouse=True)
def quiet_tts(monkeypatch):
    """Capture synthesis rather than calling out to Google on every turn."""
    from pipeline.tts_worker import TTSWorker

    monkeypatch.setattr(TTSWorker, '_synthesize_gtts', lambda self, text: b'ID3')


@pytest.fixture(autouse=True)
def ollama(monkeypatch):
    """Point the router at a stand-in, so these tests do not need a real model."""
    with FakeOllama() as server:
        monkeypatch.setenv('OLLAMA_URL', server.url)
        yield server


@asynccontextmanager
async def open_socket(username='protocol'):
    """An authenticated socket, closed on the way out.

    Deliberately not a fixture. An async generator fixture is finalised after
    the test's event loop has moved on, and the disconnect then fails against
    an application task that has already been cancelled.
    """
    from asgiref.sync import sync_to_async

    from tests.test_voice_agent_e2e import _session_cookie

    user = await sync_to_async(User.objects.create_user)(
        username=username, password='a-good-password-42'
    )
    cookie = await sync_to_async(_session_cookie)(user)
    communicator = WebsocketCommunicator(application, '/ws/voice/', headers=[
        (b'origin', b'http://localhost:8000'),
        (b'host', b'localhost:8000'),
        (b'cookie', cookie),
    ])
    connected, _ = await asyncio.wait_for(communicator.connect(), 15)
    assert connected, 'websocket refused an authenticated user'
    await communicator.receive_json_from(timeout=15)
    try:
        yield communicator
    finally:
        # A receive timeout during the test may already have cancelled the
        # application task; disconnecting then is not an error.
        try:
            await communicator.disconnect()
        except asyncio.CancelledError:
            pass


async def wait_for_reply(communicator, timeout=25):
    """True as soon as a spoken reply arrives.

    Deliberately returns on the first reply rather than draining to a timeout.
    asgiref's ApplicationCommunicator cancels the application task whenever
    receive_output times out, so a helper that always drains to empty kills
    the very connection it is meant to be testing - which looked exactly like
    an application bug.
    """
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout

    while loop.time() < deadline:
        try:
            frame = await communicator.receive_from(timeout=deadline - loop.time())
        except asyncio.TimeoutError:
            return False
        if isinstance(frame, str) and json.loads(frame).get('type') == 'response_chunk':
            return True
    return False


async def still_alive(communicator):
    """A turn still completes, so the socket survived whatever came before."""
    await communicator.send_json_to({'type': 'final_transcript', 'text': 'Hello there'})
    return await wait_for_reply(communicator)


# ------------------------------------------------------- malformed frames

@pytest.mark.parametrize('payload', [
    '{not json',
    '[]',
    '"just a string"',
    '123',
    'null',
    '{"type": null}',
    '{"no_type": true}',
    '{}',
])
async def test_odd_frames_do_not_drop_the_connection(payload):
    async with open_socket() as socket:
        await socket.send_to(text_data=payload)
        assert await still_alive(socket), f'connection died on {payload!r}'


async def test_unknown_message_type_is_ignored():
    async with open_socket() as socket:
        await socket.send_to(text_data=json.dumps({'type': 'not_a_real_type', 'text': 'x'}))
        assert await still_alive(socket)


async def test_oversized_frame_is_refused_without_closing():
    async with open_socket() as socket:
        await socket.send_to(text_data=json.dumps({'type': 'text_input', 'text': 'x' * 50000}))
        assert await still_alive(socket)


async def test_text_input_with_a_non_string_payload():
    async with open_socket() as socket:
        for value in (None, 123, {'nested': True}, ['a', 'b']):
            await socket.send_to(
                text_data=json.dumps({'type': 'final_transcript', 'text': value})
            )
        assert await still_alive(socket)


async def test_empty_and_whitespace_turns_are_ignored():
    async with open_socket() as socket:
        for text in ('', '   ', '\n\t'):
            await socket.send_to(
                text_data=json.dumps({'type': 'final_transcript', 'text': text})
            )
        assert await still_alive(socket)


# ----------------------------------------------------------- audio frames

async def test_audio_before_any_control_frame():
    async with open_socket() as socket:
        await socket.send_to(bytes_data=b'\x00\x01' * 800)
        assert await still_alive(socket)


async def test_a_flood_of_audio_does_not_stall_the_socket():
    """The queue is bounded, so chunks are dropped rather than blocking."""
    async with open_socket() as socket:
        for _ in range(400):
            await socket.send_to(bytes_data=b'\x00\x01' * 800)
        assert await still_alive(socket)


# --------------------------------------------------------------- ordering

async def test_stop_before_start():
    async with open_socket() as socket:
        await socket.send_to(text_data=json.dumps({'type': 'stop_recording'}))
        assert await still_alive(socket)


async def test_repeated_interrupts():
    async with open_socket() as socket:
        for _ in range(25):
            await socket.send_to(text_data=json.dumps({'type': 'interrupt'}))
        assert await still_alive(socket)


async def test_interrupt_between_two_turns():
    async with open_socket() as socket:
        await socket.send_to(
            text_data=json.dumps({'type': 'final_transcript', 'text': 'Hello there'})
        )
        await socket.send_to(text_data=json.dumps({'type': 'interrupt'}))
        assert await still_alive(socket)


async def test_two_turns_sent_back_to_back():
    async with open_socket() as socket:
        await socket.send_to(
            text_data=json.dumps({'type': 'final_transcript', 'text': 'Hello there'})
        )
        await socket.send_to(
            text_data=json.dumps({'type': 'final_transcript', 'text': 'Good morning'})
        )
        assert await still_alive(socket)


# ------------------------------------------------------------- disconnect

async def test_disconnecting_mid_turn_is_clean():
    """Cancelling the workers must not raise into the server."""
    async with open_socket('midturn') as socket:
        await socket.send_json_to({'type': 'final_transcript', 'text': 'Hello there'})
        await asyncio.sleep(0.05)   # do not let the turn finish
    await asyncio.sleep(0.3)        # any late cancellation surfaces here
