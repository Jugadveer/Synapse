import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class AsyncPipeline:
    """
    Orchestrates the voice turn:
    audio -> STT -> router -> (clarify | reason) -> TTS -> audio out.

    Workers are independent coroutines joined by queues, so a slow reasoning
    call never blocks transcription of the next utterance.
    """

    def __init__(self, consumer, user_key=None):
        self.consumer = consumer
        # Scopes memory so one person's recollections are not served to another.
        self.user_key = user_key

        self.audio_queue = asyncio.Queue(maxsize=100)
        self.text_queue = asyncio.Queue(maxsize=50)
        self.intent_queue = asyncio.Queue(maxsize=50)
        self.gpt_input_queue = asyncio.Queue(maxsize=50)
        self.response_queue = asyncio.Queue(maxsize=50)

        self.shutdown_event = asyncio.Event()

        # Barge-in. Interrupting bumps the generation; work tagged with an
        # older generation is dropped instead of being spoken over the user.
        # The previous implementation set an Event and cleared it 10ms later,
        # which no worker ever read.
        self.generation = 0

        self.start_time = time.time()
        self.turn_start_time = None

        self.pending_memory_clarification = None

        self.conversation_state = {
            'last_intent': None,
            'pending_slots': {},
            'user_profile': {},
            'context_window': [],
            'last_confirmation': None,
        }

        from pipeline.clarification_worker import ClarificationWorker
        from pipeline.gpt_worker import GPTWorker
        from pipeline.memory_worker import MemoryWorker
        from pipeline.qwen_router import QwenRouter
        from pipeline.stt_worker import STTWorker
        from pipeline.tts_worker import TTSWorker

        self.stt_worker = STTWorker(self)
        self.qwen_router = QwenRouter(self)
        self.clarification_worker = ClarificationWorker(self)
        self.gpt_worker = GPTWorker(self)
        self.memory_worker = MemoryWorker(self)
        self.tts_worker = TTSWorker(self)

        self._worker_objects = [
            self.stt_worker,
            self.qwen_router,
            self.clarification_worker,
            self.gpt_worker,
            self.memory_worker,
            self.tts_worker,
        ]
        self.workers = []
        self._start_workers()

    def _start_workers(self):
        self.workers = [asyncio.create_task(w.run()) for w in self._worker_objects]

    # ------------------------------------------------------------------
    # turn lifecycle
    # ------------------------------------------------------------------

    async def handle_text(self, text):
        """Handle typed input, treated the same as a final transcript."""
        if not isinstance(text, str):
            return
        text = text.strip()
        if not text:
            return
        self.turn_start_time = time.time()
        await self.text_queue.put({'type': 'final', 'text': text, 'generation': self.generation})

    def is_stale(self, generation):
        """True when work from `generation` has been superseded by a barge-in."""
        return generation is not None and generation != self.generation

    async def interrupt(self):
        """Stop speaking and discard in-flight work for the current turn."""
        self.generation += 1
        drained = sum(
            self._drain(q) for q in (self.text_queue, self.intent_queue,
                                     self.gpt_input_queue, self.response_queue)
        )
        logger.info(f"Interrupted; generation={self.generation}, dropped {drained} queued items")

    @staticmethod
    def _drain(queue):
        dropped = 0
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                return dropped
            dropped += 1

    def mark_turn_latency(self):
        if not self.turn_start_time:
            return 0
        latency = (time.time() - self.turn_start_time) * 1000
        logger.info(f"Turn latency: {latency:.0f}ms")
        return latency

    def update_conversation_context(self, user_text, decision):
        self.conversation_state['context_window'].append({
            'user_text': user_text,
            'intent': decision.get('intent'),
            'timestamp': time.time(),
        })
        if len(self.conversation_state['context_window']) > 5:
            self.conversation_state['context_window'].pop(0)

        self.conversation_state['last_intent'] = decision.get('intent')
        if decision.get('missing_slots'):
            self.conversation_state['pending_slots'] = decision.get('missing_slots', {})

    # ------------------------------------------------------------------
    # shutdown
    # ------------------------------------------------------------------

    async def cleanup(self):
        """Stop workers and release the resources they hold."""
        self.shutdown_event.set()

        for task in self.workers:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)

        # Each worker owning an HTTP client used to leak it on every
        # disconnect; nothing closed them.
        await asyncio.gather(
            *(w.aclose() for w in self._worker_objects),
            return_exceptions=True,
        )
        logger.info("Pipeline cleaned up")
