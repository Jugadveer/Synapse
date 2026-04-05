import asyncio
import logging
import os
import time

import numpy as np

logger = logging.getLogger(__name__)


def detect_language_fast(text):
    """Detect language from text using Unicode ranges."""
    if not text:
        return "en"
    hindi_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    total_chars = len([c for c in text if c.isalpha()])
    if total_chars > 0 and hindi_chars / total_chars > 0.3:
        return "hi"
    return "en"


class STTWorker:
    """Local STT worker using faster-whisper with pause/manual finalize."""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.model_name = os.getenv('FASTER_WHISPER_MODEL', 'base').strip()
        self.device = os.getenv('FASTER_WHISPER_DEVICE', 'auto').strip().lower()
        self.compute_type = os.getenv('FASTER_WHISPER_COMPUTE_TYPE', 'int8').strip().lower()
        
        # Audio config (EXPECTED INPUT FORMAT)
        self.sample_rate = 16000
        self.bytes_per_sample = 2  # int16 PCM
        
        # State
        self.partial_text = ""
        self.last_sent_text = ""
        self.locked_language = None
        self.last_voice_ts = 0.0
        self.audio_buffer = bytearray()

        self.force_finalize_event = asyncio.Event()
        self.model = None

    def reset(self):
        logger.info("[STT] Resetting state")
        self.partial_text = ""
        self.last_sent_text = ""
        self.last_voice_ts = 0.0

    async def run(self):
        """Main async worker loop."""
        try:
            self.model = await asyncio.to_thread(self._load_model)
            logger.info(
                f"[STT] faster-whisper ready model={self.model_name} device={self.device} compute_type={self.compute_type}"
            )
            await self._run_batch_mode()
                    
        except Exception as e:
            logger.error(f"[STT] Worker failed: {e}")
        logger.info("[STT] Worker shutdown")

    def _load_model(self):
        from faster_whisper import WhisperModel

        return WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
        )

    async def _run_batch_mode(self):
        """Buffered local STT with pause detection."""
        min_voice_bytes = int(self.sample_rate * self.bytes_per_sample * 0.6)
        max_pause_s = 1.0

        while not self.pipeline.shutdown_event.is_set():
            if self.force_finalize_event.is_set():
                self.force_finalize_event.clear()
                # Allow short utterances on explicit stop.
                min_force_bytes = int(self.sample_rate * self.bytes_per_sample * 0.25)
                if len(self.audio_buffer) >= min_force_bytes:
                    await self._transcribe_buffer_batch()
                self.reset()
                self.audio_buffer = bytearray()
                continue

            try:
                chunk = await asyncio.wait_for(self.pipeline.audio_queue.get(), timeout=0.2)
            except asyncio.TimeoutError:
                # Finalize if user paused after speaking.
                if self.audio_buffer and self.last_voice_ts and (time.time() - self.last_voice_ts) > max_pause_s:
                    if len(self.audio_buffer) >= min_voice_bytes:
                        await self._transcribe_buffer_batch()
                    self.reset()
                    self.audio_buffer = bytearray()
                continue

            if not chunk:
                continue

            self.audio_buffer.extend(chunk)

            try:
                audio_data = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
                rms = float(np.sqrt(np.mean(audio_data ** 2))) if audio_data.size else 0.0
                if rms >= 500:
                    self.last_voice_ts = time.time()
            except Exception:
                pass

            # Auto-finalize on pause even if silence chunks are still being streamed.
            if self.audio_buffer and self.last_voice_ts and len(self.audio_buffer) >= min_voice_bytes:
                if (time.time() - self.last_voice_ts) > max_pause_s:
                    await self._transcribe_buffer_batch()
                    self.reset()
                    self.audio_buffer = bytearray()
                    continue

            # Guard against infinite growth if silence detection fails.
            max_buffer = int(self.sample_rate * self.bytes_per_sample * 20)
            if len(self.audio_buffer) > max_buffer and len(self.audio_buffer) >= min_voice_bytes:
                await self._transcribe_buffer_batch()
                self.reset()
                self.audio_buffer = bytearray()

    async def _transcribe_buffer_batch(self):
        """Transcribe buffered PCM audio with faster-whisper."""
        try:
            if not self.model or not self.audio_buffer:
                return

            pcm = np.frombuffer(bytes(self.audio_buffer), dtype=np.int16).astype(np.float32)
            if pcm.size == 0:
                return

            audio = pcm / 32768.0
            segments, info = await asyncio.to_thread(
                self.model.transcribe,
                audio,
                language=self.locked_language,
                vad_filter=True,
                beam_size=1,
                temperature=0.0,
                condition_on_previous_text=False,
            )

            text = " ".join((segment.text or '').strip() for segment in segments).strip()
            if not text:
                return

            if self.locked_language is None:
                detected_language = getattr(info, 'language', None)
                self.locked_language = detected_language or detect_language_fast(text)
                logger.info(f"[STT] Language locked to: {self.locked_language}")

            self.partial_text = text
            self.last_sent_text = text
            logger.info(f"[STT] Output (final): {text}")
            await self._finalize_text()

        except Exception as e:
            logger.error(f"[STT] faster-whisper transcription failed: {e}")

    async def _finalize_text(self):
        """Send final transcript to pipeline."""
        final_text = self.partial_text.strip()
        if final_text:
            logger.info(f"[STT] Final: {final_text}")

            await self.pipeline.consumer.send_transcript(
                final_text,
                is_final=True
            )

            await self.pipeline.text_queue.put({
                'type': 'final',
                'text': final_text
            })

    async def force_finalize(self):
        """Request immediate finalize of current batch audio buffer."""
        self.force_finalize_event.set()
