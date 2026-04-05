import asyncio
import time
import json
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class AsyncPipeline:
    """
    Orchestrates the entire voice processing pipeline:
    Audio -> Whisper -> Qwen -> (opt) GPT -> TTS -> Audio Output
    
    Uses concurrent asyncio queues for non-blocking execution.
    """
    
    def __init__(self, consumer):
        self.consumer = consumer
        
        # Queues for inter-worker communication
        self.audio_queue = asyncio.Queue(maxsize=100)
        self.text_queue = asyncio.Queue(maxsize=50)
        self.intent_queue = asyncio.Queue(maxsize=50)
        self.clarification_queue = asyncio.Queue(maxsize=50)  # 🔥 Confidence/slot gating
        self.gpt_input_queue = asyncio.Queue(maxsize=50)  # 🔥 After clarification check
        self.response_queue = asyncio.Queue(maxsize=50)
        self.tts_queue = asyncio.Queue(maxsize=100)
        
        # Control signals
        self.interrupt_event = asyncio.Event()
        self.shutdown_event = asyncio.Event()
        
        # Metrics
        self.start_time = time.time()
        self.turn_start_time = None

        # Pending clarification state for memory follow-ups.
        self.pending_memory_clarification = None

        # 🧠 Conversation state for dementia-safe cognition
        self.conversation_state = {
            'last_intent': None,
            'pending_slots': {},  # Missing required fields
            'user_profile': {},  # User preferences, patterns
            'context_window': [],  # Last 5 turns for context
            'last_confirmation': None,  # Last thing confirmed
        }
        
        # Import workers
        from pipeline.stt_worker import STTWorker
        from pipeline.qwen_router import QwenRouter
        from pipeline.clarification_worker import ClarificationWorker
        from pipeline.gpt_worker import GPTWorker
        from pipeline.memory_worker import MemoryWorker
        from pipeline.tts_worker import TTSWorker
        
        # Initialize workers
        self.stt_worker = STTWorker(self)
        self.qwen_router = QwenRouter(self)
        self.clarification_worker = ClarificationWorker(self)  # 🔥 New gate
        self.gpt_worker = GPTWorker(self)
        self.memory_worker = MemoryWorker(self)
        self.tts_worker = TTSWorker(self)
        
        # Task tracking
        self.workers = []
        self._start_workers()
    
    def _start_workers(self):
        """Start all worker coroutines."""
        self.workers = [
            asyncio.create_task(self.stt_worker.run()),
            asyncio.create_task(self.qwen_router.run()),
            asyncio.create_task(self.clarification_worker.run()),  # 🔥 Gate between intent and execution
            asyncio.create_task(self.gpt_worker.run()),
            asyncio.create_task(self.memory_worker.run()),
            asyncio.create_task(self.tts_worker.run()),
        ]
    
    async def handle_text(self, text):
        """Handle direct text input (e.g., via UI)."""
        self.turn_start_time = time.time()
        await self.text_queue.put({
            'type': 'final',
            'text': text,
        })
    
    async def interrupt(self):
        """Interrupt ongoing processing and TTS."""
        self.interrupt_event.set()
        await asyncio.sleep(0.01)
        self.interrupt_event.clear()
    
    async def cleanup(self):
        """Graceful shutdown of pipeline."""
        self.shutdown_event.set()
        await asyncio.sleep(0.1)
        
        # Cancel all worker tasks
        for worker in self.workers:
            if not worker.done():
                worker.cancel()
        
        await asyncio.gather(*self.workers, return_exceptions=True)
    
    def mark_turn_latency(self):
        """Calculate and log turn latency."""
        if self.turn_start_time:
            latency = (time.time() - self.turn_start_time) * 1000
            logger.info(f"Turn latency: {latency:.0f}ms")
            return latency
        return 0

    def update_conversation_context(self, user_text, decision):
        """Update conversation state with new turn."""
        # Add to context window
        self.conversation_state['context_window'].append({
            'user_text': user_text,
            'intent': decision.get('intent'),
            'timestamp': time.time(),
        })
        # Keep last 5 turns
        if len(self.conversation_state['context_window']) > 5:
            self.conversation_state['context_window'].pop(0)
        
        # Update last intent
        self.conversation_state['last_intent'] = decision.get('intent')
        
        # Track pending slots from decision
        if decision.get('missing_slots'):
            self.conversation_state['pending_slots'] = decision.get('missing_slots', {})
