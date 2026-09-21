import json
import logging
import sys
import uuid
from pathlib import Path

from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncJsonWebsocketConsumer

ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_APP_DIR = ROOT_DIR / 'src' / 'synapse'
if str(SRC_APP_DIR) not in sys.path:
    sys.path.append(str(SRC_APP_DIR))

logger = logging.getLogger(__name__)

MAX_TEXT_FRAME = 4000


class VoiceConsumer(AsyncJsonWebsocketConsumer):
    """Websocket endpoint for the voice assistant."""

    async def connect(self):
        user = self.scope.get('user')
        if user is None or not user.is_authenticated:
            # The pipeline reads and writes a person's memories and reminders,
            # so an anonymous socket has no business opening one.
            logger.warning("Rejected unauthenticated voice connection")
            await self.close(code=4401)
            return

        self.user_key = str(user.pk)
        self.session_id = str(uuid.uuid4())
        await self.accept()

        from pipeline.pipeline import AsyncPipeline
        from pipeline.reminder_scheduler import ReminderScheduler

        self.pipeline = AsyncPipeline(self, user_key=self.user_key)
        await self._create_session()

        self.scheduler = ReminderScheduler.instance()
        self.scheduler.register(self.user_key, self)

        await self.send_json({
            'type': 'system',
            'message': 'Connected',
            'session_id': self.session_id,
        })

    async def disconnect(self, close_code):
        scheduler = getattr(self, 'scheduler', None)
        if scheduler is not None:
            scheduler.unregister(getattr(self, 'user_key', None), self)

        pipeline = getattr(self, 'pipeline', None)
        if pipeline is not None:
            await pipeline.cleanup()

    async def receive(self, text_data=None, bytes_data=None):
        pipeline = getattr(self, 'pipeline', None)
        if pipeline is None:
            return

        if bytes_data:
            try:
                pipeline.audio_queue.put_nowait(bytes_data)
            except Exception:
                # Dropping a chunk under back-pressure is better than
                # stalling the socket behind a full queue.
                logger.debug("Audio queue full; dropped a chunk")
            return

        if not text_data:
            return
        if len(text_data) > MAX_TEXT_FRAME:
            logger.warning("Ignoring oversized control frame")
            return

        try:
            data = json.loads(text_data)
        except json.JSONDecodeError:
            # A malformed frame used to raise straight out of receive() and
            # take the connection down with it.
            logger.debug("Ignoring malformed control frame")
            return

        if not isinstance(data, dict):
            return

        message_type = data.get('type')

        if message_type == 'start_recording':
            pipeline.stt_worker.reset()
            await pipeline.interrupt()
        elif message_type == 'stop_recording':
            await pipeline.stt_worker.force_finalize()
        elif message_type == 'interrupt':
            await pipeline.interrupt()
        elif message_type in ('command', 'final_transcript', 'text_input'):
            text = data.get('text')
            # A frame carrying a number, a list or an object for `text` used to
            # reach .strip() and take the connection down with an AttributeError.
            if isinstance(text, str):
                await pipeline.handle_text(text)
            else:
                logger.debug('Ignoring %s frame with non-string text', message_type)

    # ------------------------------------------------------------------
    # outbound
    # ------------------------------------------------------------------

    async def send_transcript(self, text, is_final=False):
        await self.send_json({'type': 'transcript', 'text': text, 'is_final': is_final})

    async def send_decision(self, decision):
        await self.send_json({
            'type': 'decision',
            'intent': decision.get('intent'),
            'needs_gpt': decision.get('needs_gpt'),
            'memory_action': decision.get('memory_action'),
            'confidence': decision.get('confidence'),
        })

    async def send_response_chunk(self, text):
        await self.send_json({'type': 'response_chunk', 'text': text})

    async def send_audio_chunk(self, audio_bytes):
        await self.send(bytes_data=audio_bytes)

    async def send_memory_update(self, action, entity, value):
        await self.send_json({
            'type': 'memory_update', 'action': action, 'entity': entity, 'value': value,
        })

    async def send_status(self, status):
        await self.send_json({'type': 'status', 'message': status})

    async def send_reminder(self, text, spoken_time=''):
        """Deliver a reminder that has come due."""
        await self.send_json({
            'type': 'reminder', 'text': text, 'spoken_time': spoken_time,
        })
        spoken = f"This is your reminder to {text}."
        await self.send_response_chunk(spoken)
        tts = getattr(getattr(self, 'pipeline', None), 'tts_worker', None)
        if tts is not None:
            await tts._speak(spoken, self.pipeline.generation)

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    @database_sync_to_async
    def _create_session(self):
        from voice.models import ConversationSession

        return ConversationSession.objects.create(
            session_id=self.session_id, user_key=self.user_key
        )

    @database_sync_to_async
    def save_turn(self, user_text, decision, spoken_response, latency_ms=0):
        from voice.models import ConversationSession, ConversationTurn

        session = ConversationSession.objects.filter(session_id=self.session_id).first()
        if session is None:
            return None
        return ConversationTurn.objects.create(
            session=session,
            user_text=user_text or '',
            qwen_intent=decision or {},
            spoken_response=spoken_response or '',
            memory_action=(decision or {}).get('intent', ''),
            memory_updated=bool((decision or {}).get('needs_memory_storage')),
            latency_ms=int(latency_ms or 0),
        )
