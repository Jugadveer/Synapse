import json
import asyncio
import time
import uuid
from channels.generic.websocket import AsyncJsonWebsocketConsumer
from channels.db import database_sync_to_async
from asgiref.sync import sync_to_async

from pipeline.pipeline import AsyncPipeline
from voice.models import ConversationSession, ConversationTurn


class VoiceConsumer(AsyncJsonWebsocketConsumer):
    """WebSocket consumer for real-time voice streaming and response."""
    
    async def connect(self):
        self.session_id = str(uuid.uuid4())
        await self.accept()

        # Accept immediately so the client does not reconnect while heavy models warm up.
        self.pipeline = AsyncPipeline(self)
        
        # Create session in DB
        await self._create_session()
        
        await self.send_json({
            'type': 'system',
            'message': 'Connected',
            'session_id': self.session_id
        })
    
    async def disconnect(self, close_code):
        if hasattr(self, 'pipeline'):
            await self.pipeline.cleanup()
    
    async def receive(self, text_data=None, bytes_data=None):
        """Receive audio chunks or text commands from client."""
        if bytes_data:
            # Audio chunk
            await self.pipeline.audio_queue.put(bytes_data)
        elif text_data:
            data = json.loads(text_data)

            if data.get('type') == 'start_recording':
                self.pipeline.stt_worker.reset()
            elif data.get('type') == 'interrupt':
                # Cancel ongoing TTS and start fresh
                await self.pipeline.interrupt()
            elif data.get('type') in ('command', 'final_transcript', 'text_input'):
                # Treat typed input the same as a final whisper transcript
                await self.pipeline.handle_text(data.get('text'))
    
    async def send_transcript(self, text, is_final=False):
        """Send transcript update to frontend."""
        await self.send_json({
            'type': 'transcript',
            'text': text,
            'is_final': is_final
        })
    
    async def send_decision(self, decision):
        """Send Qwen decision for transparency."""
        await self.send_json({
            'type': 'decision',
            'intent': decision.get('intent'),
            'needs_gpt': decision.get('needs_gpt'),
            'memory_action': decision.get('memory_action'),
            'confidence': decision.get('confidence')
        })
    
    async def send_response_chunk(self, text):
        """Send response text chunk."""
        await self.send_json({
            'type': 'response_chunk',
            'text': text
        })
    
    async def send_audio_chunk(self, audio_bytes):
        """Send audio chunk to client."""
        await self.send_bytes(audio_bytes)
    
    async def send_memory_update(self, action, entity, value):
        """Send memory update event."""
        await self.send_json({
            'type': 'memory_update',
            'action': action,
            'entity': entity,
            'value': value
        })
    
    async def send_status(self, status):
        """Send status message."""
        await self.send_json({
            'type': 'status',
            'message': status
        })
    
    @database_sync_to_async
    def _create_session(self):
        return ConversationSession.objects.create(session_id=self.session_id)
    
    @database_sync_to_async
    def _save_turn(self, user_text, qwen_decision, gpt_response, spoken_response, latency_ms):
        session = ConversationSession.objects.get(session_id=self.session_id)
        return ConversationTurn.objects.create(
            session=session,
            user_text=user_text,
            qwen_intent=qwen_decision,
            gpt_response=gpt_response,
            spoken_response=spoken_response,
            latency_ms=latency_ms
        )
