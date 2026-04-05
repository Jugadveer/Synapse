import json
import sys
from pathlib import Path

from channels.generic.websocket import AsyncJsonWebsocketConsumer


ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_APP_DIR = ROOT_DIR / 'src' / 'synapse'
if str(SRC_APP_DIR) not in sys.path:
    sys.path.append(str(SRC_APP_DIR))


class VoiceConsumer(AsyncJsonWebsocketConsumer):
    """FinalEclipse websocket endpoint bridged to src/synapse voice pipeline."""

    async def connect(self):
        await self.accept()
        from pipeline.pipeline import AsyncPipeline

        self.pipeline = AsyncPipeline(self)
        await self.send_json({'type': 'system', 'message': 'Connected'})

    async def disconnect(self, close_code):
        if hasattr(self, 'pipeline'):
            await self.pipeline.cleanup()

    async def receive(self, text_data=None, bytes_data=None):
        if bytes_data and hasattr(self, 'pipeline'):
            await self.pipeline.audio_queue.put(bytes_data)
            return

        if not text_data:
            return

        try:
            data = json.loads(text_data)
        except json.JSONDecodeError:
            return

        msg_type = data.get('type')
        if msg_type == 'start_recording':
            if hasattr(self, 'pipeline'):
                self.pipeline.stt_worker.reset()
            return

        if msg_type == 'stop_recording':
            if hasattr(self, 'pipeline'):
                await self.pipeline.stt_worker.force_finalize()
            return

        if msg_type == 'interrupt':
            if hasattr(self, 'pipeline'):
                await self.pipeline.interrupt()
            return

        if msg_type in ('command', 'final_transcript', 'text_input'):
            text = (data.get('text') or '').strip()
            if text and hasattr(self, 'pipeline'):
                await self.pipeline.handle_text(text)

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
        await self.send_json({'type': 'memory_update', 'action': action, 'entity': entity, 'value': value})

    async def send_status(self, status):
        await self.send_json({'type': 'status', 'message': status})