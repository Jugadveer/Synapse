import asyncio
import io
import logging
import os

logger = logging.getLogger(__name__)


class TTSWorker:
    """Text-to-speech worker using gTTS."""
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.tts_disabled = False
    
    async def run(self):
        """Consume text and stream audio."""
        while True:
            try:
                item = await self.pipeline.response_queue.get()
                text = item.get('response', '')
                if not text:
                    continue

                # Only send the text chunk here if it was not already sent by the reasoning worker.
                if not item.get('response_chunk_sent'):
                    await self.pipeline.consumer.send_response_chunk(text)
                
                # Generate streaming audio
                await self._stream_tts(text)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"TTS error: {e}")
    
    async def _stream_tts(self, text):
        """Stream TTS audio."""
        if self.tts_disabled:
            return
        try:
            audio_bytes = await asyncio.to_thread(self._synthesize_gtts, text)
            if not audio_bytes:
                return
            
            # Send audio bytes to client
            await self.pipeline.consumer.send_audio_chunk(audio_bytes)
        
        except Exception as e:
            logger.error(f"TTS generation error: {e}")

    def _synthesize_gtts(self, text):
        """Run blocking gTTS synthesis in a worker thread."""
        try:
            from gtts import gTTS
        except Exception as import_error:
            self.tts_disabled = True
            logger.error(f"gTTS is not available; install gTTS package. Error: {import_error}")
            return b''

        lang = os.getenv('TTS_GTTS_LANG', 'en').strip() or 'en'
        try:
            tts = gTTS(text=text, lang=lang, slow=False)
            buffer = io.BytesIO()
            tts.write_to_fp(buffer)
            return buffer.getvalue()
        except Exception as synth_error:
            logger.error(f"gTTS synthesis failed: {synth_error}")
            return b''
