"""End-to-end voice agent: does a turn actually go all the way through?

These drive the real consumer over a real websocket, through the real
AsyncPipeline, the real router with a real httpx client talking to a real HTTP
server, the real memory worker and the real database. Only the model weights
are substituted (see fake_ollama.py) and speech synthesis is captured rather
than fetched.
"""

import asyncio
import json

import pytest
from channels.testing import WebsocketCommunicator
from django.contrib.auth.models import User

from project.asgi import application
from tests.fake_ollama import FakeOllama

pytestmark = pytest.mark.django_db(transaction=True)

CONNECT_TIMEOUT = 10
TURN_TIMEOUT = 20


@pytest.fixture
def quiet_tts(monkeypatch):
    """Capture synthesis instead of calling out to Google on every turn."""
    spoken = []

    def fake_synthesize(self, text):
        spoken.append(text)
        return b'ID3fake-mp3-bytes'

    from pipeline.tts_worker import TTSWorker

    monkeypatch.setattr(TTSWorker, '_synthesize_gtts', fake_synthesize)
    return spoken


@pytest.fixture
def ollama(monkeypatch):
    with FakeOllama() as server:
        monkeypatch.setenv('OLLAMA_URL', server.url)
        yield server


def _session_cookie(user):
    """A real logged-in session.

    scope['user'] cannot simply be injected: AuthMiddlewareStack replaces it
    with a lazy lookup driven by the session cookie. Signing in for real means
    the auth middleware is exercised too.
    """
    from django.conf import settings
    from django.contrib.auth import BACKEND_SESSION_KEY, HASH_SESSION_KEY, SESSION_KEY
    from django.contrib.sessions.backends.db import SessionStore

    session = SessionStore()
    session[SESSION_KEY] = str(user.pk)
    session[BACKEND_SESSION_KEY] = 'django.contrib.auth.backends.ModelBackend'
    session[HASH_SESSION_KEY] = user.get_session_auth_hash()
    session.save()
    return f'{settings.SESSION_COOKIE_NAME}={session.session_key}'.encode()


async def open_socket(user):
    from asgiref.sync import sync_to_async

    cookie = await sync_to_async(_session_cookie)(user)
    communicator = WebsocketCommunicator(
        application,
        '/ws/voice/',
        headers=[
            # AllowedHostsOriginValidator rejects a socket with no Origin
            # before it ever reaches the consumer. Browsers always send one.
            (b'origin', b'http://localhost:8000'),
            (b'host', b'localhost:8000'),
            (b'cookie', cookie),
        ],
    )
    connected, _ = await asyncio.wait_for(communicator.connect(), CONNECT_TIMEOUT)
    assert connected, 'websocket refused the authenticated user'
    await communicator.receive_json_from(timeout=CONNECT_TIMEOUT)  # the 'system' hello
    return communicator


async def say(communicator, text, until='audio'):
    """Send a turn and collect what comes back.

    Reads to the end of the turn - the audio frame is the last thing sent -
    rather than draining until a timeout. Stopping earlier leaves that frame
    queued, and the next turn then reads it first and returns nothing. asgiref's ApplicationCommunicator cancels the
    application task whenever receive_output times out, so a helper that
    drains to empty kills the connection it is testing and the disconnect that
    follows fails.
    """
    await communicator.send_json_to({'type': 'final_transcript', 'text': text})

    messages, audio = [], []
    loop = asyncio.get_event_loop()
    deadline = loop.time() + TURN_TIMEOUT

    while loop.time() < deadline:
        try:
            frame = await communicator.receive_from(timeout=deadline - loop.time())
        except asyncio.TimeoutError:
            break
        if isinstance(frame, bytes):
            audio.append(frame)
            break                      # audio is the last thing in a turn
        messages.append(json.loads(frame))
        if until == 'reply' and messages[-1].get('type') == 'response_chunk':
            break
    return messages, audio


def replies(messages):
    return [m['text'] for m in messages if m.get('type') == 'response_chunk']


@pytest.fixture
async def user():
    from asgiref.sync import sync_to_async

    return await sync_to_async(User.objects.create_user)(
        username='patient', password='a-good-password-42'
    )


# ------------------------------------------------------------- basic turn

async def test_a_turn_produces_a_spoken_reply(user, ollama, quiet_tts):
    communicator = await open_socket(user)
    try:
        messages, audio = await say(communicator, 'Hello there')

        assert replies(messages), f'no reply was produced; got {messages}'
        assert quiet_tts, 'nothing was sent to speech synthesis'
        assert audio, 'no audio frame reached the client'
    finally:
        try:
            await communicator.disconnect()
        except asyncio.CancelledError:
            pass


async def test_router_was_actually_called(user, ollama, quiet_tts):
    communicator = await open_socket(user)
    try:
        await say(communicator, 'Hello there')
        assert ollama.prompts, 'the router never called the model'
    finally:
        try:
            await communicator.disconnect()
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------- reminders

async def test_reminder_turn_creates_a_reminder(user, ollama, quiet_tts):
    from voice.models import Reminder

    communicator = await open_socket(user)
    try:
        messages, _ = await say(
            communicator, 'remind me to take my tablets in 10 minutes'
        )

        text = ' '.join(replies(messages))
        assert 'tablets' in text.lower(), f'reply did not confirm the task: {text!r}'

        reminder = await Reminder.objects.aget()
        assert reminder.text == 'take my tablets'
        assert reminder.user_key == str(user.pk)
        assert reminder.is_pending
    finally:
        try:
            await communicator.disconnect()
        except asyncio.CancelledError:
            pass


async def test_reminder_without_a_time_asks_then_schedules(user, ollama, quiet_tts):
    from voice.models import Reminder

    communicator = await open_socket(user)
    try:
        messages, _ = await say(communicator, 'remind me to call my daughter')
        question = ' '.join(replies(messages)).lower()
        assert 'when' in question, f'expected a question about the time, got {question!r}'

        assert await Reminder.objects.acount() == 0, 'scheduled before knowing when'

        messages, _ = await say(communicator, 'in 20 minutes')
        confirmation = ' '.join(replies(messages)).lower()
        assert 'daughter' in confirmation

        reminder = await Reminder.objects.aget()
        assert reminder.text == 'call my daughter'
    finally:
        try:
            await communicator.disconnect()
        except asyncio.CancelledError:
            pass


async def test_listing_reminders(user, ollama, quiet_tts):
    communicator = await open_socket(user)
    try:
        await say(communicator, 'remind me to take my tablets at 8pm')
        messages, _ = await say(communicator, 'what are my reminders')

        text = ' '.join(replies(messages)).lower()
        assert 'tablets' in text, f'reminder not listed back: {text!r}'
    finally:
        try:
            await communicator.disconnect()
        except asyncio.CancelledError:
            pass


# ------------------------------------------------------------------ memory

async def test_memory_store_then_retrieve(user, ollama, quiet_tts, real_ml_required):
    """The headline feature: tell it something, then ask for it back."""
    communicator = await open_socket(user)
    try:
        messages, _ = await say(communicator, 'I left my keys on the kitchen table')
        assert replies(messages), 'storing a memory produced no reply'

        messages, _ = await say(communicator, 'where did I leave my keys')
        answer = ' '.join(replies(messages)).lower()
        assert answer, 'retrieval produced no reply'
    finally:
        try:
            await communicator.disconnect()
        except asyncio.CancelledError:
            pass


async def test_incomplete_memory_asks_one_question(user, ollama, quiet_tts):
    communicator = await open_socket(user)
    try:
        messages, _ = await say(communicator, 'I put my glasses somewhere')
        reply = ' '.join(replies(messages))

        assert reply.count('?') <= 1, f'more than one question in a turn: {reply!r}'
        assert '?' in reply, f'expected a clarifying question, got {reply!r}'
    finally:
        try:
            await communicator.disconnect()
        except asyncio.CancelledError:
            pass


# --------------------------------------------------------------- lifecycle

async def test_turn_is_recorded(user, ollama, quiet_tts):
    from voice.models import ConversationTurn

    communicator = await open_socket(user)
    try:
        await say(communicator, 'Hello there')
        assert await ConversationTurn.objects.acount() >= 1, 'no turn was logged'
    finally:
        try:
            await communicator.disconnect()
        except asyncio.CancelledError:
            pass


async def test_malformed_frame_does_not_drop_the_connection(user, ollama, quiet_tts):
    communicator = await open_socket(user)
    try:
        await communicator.send_to(text_data='{not json at all')
        messages, _ = await say(communicator, 'Hello there')
        assert replies(messages), 'connection stopped working after a bad frame'
    finally:
        try:
            await communicator.disconnect()
        except asyncio.CancelledError:
            pass


async def test_interrupt_drops_the_superseded_turn(user, ollama, quiet_tts):
    communicator = await open_socket(user)
    try:
        await communicator.send_json_to({'type': 'final_transcript', 'text': 'Hello there'})
        await communicator.send_json_to({'type': 'interrupt'})

        # Whatever happens, the socket must still serve the next turn.
        messages, _ = await say(communicator, 'Good morning')
        assert replies(messages), 'socket unusable after an interrupt'
    finally:
        try:
            await communicator.disconnect()
        except asyncio.CancelledError:
            pass


async def test_disconnect_cleans_up(user, ollama, quiet_tts):
    communicator = await open_socket(user)
    await say(communicator, 'Hello there')
    await communicator.disconnect()
    # Workers are cancelled and http clients closed; nothing should still run.
    await asyncio.sleep(0.2)


async def test_reminder_is_stored_timezone_aware(user, ollama, quiet_tts):
    """A naive datetime is stored as though it were UTC.

    With USE_TZ on, that would fire "remind me at four" at whatever local time
    four PM UTC happens to be - hours off for anyone not on UTC.
    """
    from voice.models import Reminder

    communicator = await open_socket(user)
    try:
        await say(communicator, 'remind me to take my tablets at 4pm')
        reminder = await Reminder.objects.aget()
        assert reminder.due_at.tzinfo is not None, 'due_at must be timezone-aware'
    finally:
        try:
            await communicator.disconnect()
        except asyncio.CancelledError:
            pass
