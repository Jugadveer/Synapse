import asyncio
import io
import logging
import os

from pipeline.safety import apply_safety_rules
from pipeline.worker import PipelineWorker

logger = logging.getLogger(__name__)


class TTSWorker(PipelineWorker):
    """Speaks the assistant's reply using gTTS."""

    name = 'tts'
    input_queue_name = 'response_queue'

    def __init__(self, pipeline):
        super().__init__(pipeline)
        self.tts_disabled = False

    async def handle(self, item):
        generation = item.get('generation')
        # The person started speaking again: their turn wins.
        if self.pipeline.is_stale(generation):
            return

        text = (item.get('response') or '').strip()
        if not text:
            return

        # Every reply leaves through here, so this is where the cognitive
        # safety rules belong.
        text = apply_safety_rules(text, item.get('decision'), item.get('user_text'))
        if not text:
            return

        if not item.get('response_chunk_sent'):
            await self.pipeline.consumer.send_response_chunk(text)

        latency = self.pipeline.mark_turn_latency()
        await self._persist_turn(item, text, latency)
        await self._speak(text, generation)

    async def _persist_turn(self, item, spoken, latency):
        """Record the exchange. ConversationTurn had a model and a migration
        but nothing ever wrote a row, so no conversation was ever logged."""
        save_turn = getattr(self.pipeline.consumer, 'save_turn', None)
        if save_turn is None:
            return
        try:
            await save_turn(
                item.get('user_text', ''), item.get('decision', {}), spoken, latency
            )
        except Exception as e:
            logger.warning(f"Could not record conversation turn: {e}")

    async def _speak(self, text, generation):
        if self.tts_disabled:
            return

        audio_bytes = await asyncio.to_thread(self._synthesize_gtts, text)
        if not audio_bytes:
            return

        # Synthesis takes a moment; check again before playing over the user.
        if self.pipeline.is_stale(generation):
            logger.info("Dropping synthesised audio: turn was interrupted")
            return

        await self.pipeline.consumer.send_audio_chunk(audio_bytes)

    def _synthesize_gtts(self, text):
        try:
            from gtts import gTTS
        except ImportError as e:
            self.tts_disabled = True
            logger.error(f"gTTS is not installed; speech output disabled: {e}")
            return b''

        lang = os.getenv('TTS_GTTS_LANG', 'en').strip() or 'en'
        try:
            buffer = io.BytesIO()
            gTTS(text=text, lang=lang, slow=False).write_to_fp(buffer)
            return buffer.getvalue()
        except Exception as e:
            # Network hiccup or an unsupported language; the text reply has
            # already been sent, so stay quiet rather than failing the turn.
            logger.error(f"gTTS synthesis failed: {e}")
            return b''
