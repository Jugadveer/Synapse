"""Intent router.

Runs the fine-tuned Qwen 2.5 3B model through Ollama to decide what a turn is
asking for, and handles the deterministic cases (reminders) locally so they
behave the same way every time.
"""

import json
import logging
import os
import re
from datetime import datetime

import httpx

from pipeline.reminder_parser import looks_like_reminder, parse_reminder
from pipeline.reminder_scheduler import create_reminder, pending_reminders
from pipeline.worker import PipelineWorker

logger = logging.getLogger(__name__)

LIST_REMINDERS = re.compile(
    r'\b(what|which|any|list|tell me).{0,25}\breminders?\b|\bmy reminders?\b'
)


class QwenRouter(PipelineWorker):
    """Classifies each turn and dispatches it."""

    name = 'router'
    input_queue_name = 'text_queue'

    def __init__(self, pipeline):
        super().__init__(pipeline)
        self.ollama_url = os.getenv('OLLAMA_URL', 'http://127.0.0.1:11434').rstrip('/')
        self.ollama_model = os.getenv('OLLAMA_QWEN_MODEL', 'qwen2.5:3b-instruct')
        self.http = httpx.AsyncClient(timeout=90.0)

    async def aclose(self):
        await self.http.aclose()

    # ------------------------------------------------------------------
    # dispatch
    # ------------------------------------------------------------------

    async def handle(self, item):
        if item.get('type') != 'final':
            return

        generation = item.get('generation')
        if self.pipeline.is_stale(generation):
            return

        user_text = (item.get('text') or '').strip()
        if not user_text:
            return

        pending = self.pipeline.pending_memory_clarification
        if pending and pending.get('kind') == 'reminder':
            if await self._resume_reminder(user_text, pending, generation):
                return

        if LIST_REMINDERS.search(user_text.lower()):
            await self._list_reminders(generation)
            return

        if looks_like_reminder(user_text):
            await self._handle_reminder(user_text, generation)
            return

        if pending:
            if await self._resume_memory(user_text, pending, generation):
                return

        memory_decision = await self._analyze_memory_turn(user_text)
        if memory_decision and memory_decision.get('intent') in (
            'memory_store', 'memory_retrieve', 'memory_clarify'
        ):
            await self._handle_memory(user_text, memory_decision, generation)
            return

        decision = self._normalize(await self._classify(user_text))
        await self._dispatch_general(user_text, decision, generation)

    async def on_error(self, item, exc):
        await self._respond(
            item.get('text', ''),
            {'intent': 'unclear'},
            "Sorry, I didn't follow that. Could you say it again?",
            item.get('generation'),
        )

    # ------------------------------------------------------------------
    # reminders
    # ------------------------------------------------------------------

    async def _handle_reminder(self, user_text, generation, carried=None):
        request = parse_reminder(user_text, now=datetime.now())
        if request is None:
            return

        task = request.task or (carried or {}).get('task', '')
        due_at = request.due_at or (carried or {}).get('due_at')
        spoken = request.spoken_time or (carried or {}).get('spoken_time', '')

        if due_at is None:
            await self._ask_for_reminder_detail(
                user_text, 'time', task, None, '',
                "When would you like me to remind you?", generation,
            )
            return

        if not task:
            await self._ask_for_reminder_detail(
                user_text, 'task', '', due_at, spoken,
                "What should I remind you about?", generation,
            )
            return

        self.pipeline.pending_memory_clarification = None
        await create_reminder(
            user_key=self.pipeline.user_key,
            text=task,
            due_at=due_at,
            spoken_time=spoken,
            session_id=getattr(self.pipeline.consumer, 'session_id', None),
        )
        logger.info(f"Reminder scheduled for {due_at:%Y-%m-%d %H:%M}: {task}")

        decision = {
            'intent': 'command', 'is_fast_response': True, 'needs_reasoning': False,
            'needs_memory_storage': False, 'confidence': 0.99,
        }
        await self._respond(
            user_text, decision, f"I'll remind you to {task} at {spoken}.", generation
        )

    async def _ask_for_reminder_detail(self, user_text, missing, task, due_at, spoken, question, generation):
        self.pipeline.pending_memory_clarification = {
            'kind': 'reminder',
            'missing': missing,
            'task': task,
            'due_at': due_at,
            'spoken_time': spoken,
            'original_text': user_text,
        }
        decision = {
            'intent': 'command', 'is_fast_response': True, 'needs_clarification': True,
            'needs_reasoning': False, 'confidence': 0.9,
        }
        await self._respond(user_text, decision, question, generation)

    async def _resume_reminder(self, user_text, pending, generation):
        """Second turn of a reminder: fold the answer into what we already have."""
        missing = pending.get('missing')

        if missing == 'time':
            # Re-parse with the trigger restored so bare answers still parse.
            probe = user_text if looks_like_reminder(user_text) else f"remind me {user_text}"
            request = parse_reminder(probe, now=datetime.now())
            if request and request.due_at:
                await self._handle_reminder(
                    f"remind me to {pending['task']} {user_text}"
                    if pending.get('task') else probe,
                    generation,
                    carried=pending,
                )
                return True
            await self._respond(
                user_text, {'intent': 'command', 'needs_clarification': True},
                "Sorry, when should I remind you? For example, in ten minutes, or at four o'clock.",
                generation,
            )
            return True

        if missing == 'task':
            task = user_text.strip(' .!?')
            if task:
                self.pipeline.pending_memory_clarification = None
                await create_reminder(
                    user_key=self.pipeline.user_key,
                    text=task,
                    due_at=pending['due_at'],
                    spoken_time=pending.get('spoken_time', ''),
                    session_id=getattr(self.pipeline.consumer, 'session_id', None),
                )
                await self._respond(
                    user_text,
                    {'intent': 'command', 'is_fast_response': True, 'confidence': 0.99},
                    f"I'll remind you to {task} at {pending.get('spoken_time', 'that time')}.",
                    generation,
                )
                return True

        return False

    async def _list_reminders(self, generation):
        reminders = await pending_reminders(self.pipeline.user_key)
        if not reminders:
            text = "You don't have any reminders set at the moment."
        else:
            first = reminders[0]
            text = f"You asked me to remind you to {first.text} at {first.spoken_time or 'later'}."
            if len(reminders) > 1:
                text += f" There are {len(reminders)} in total."
        await self._respond('', {'intent': 'memory_retrieve', 'is_fast_response': True}, text, generation)

    # ------------------------------------------------------------------
    # memory
    # ------------------------------------------------------------------

    async def _handle_memory(self, user_text, raw_decision, generation):
        decision = self._normalize(raw_decision)
        intent = decision.get('intent')

        if intent == 'memory_store' and self._needs_clarification(decision):
            question = await self._clarification_question(user_text, decision)
            self.pipeline.pending_memory_clarification = {
                'kind': 'memory',
                'original_text': user_text,
                'entity': decision.get('memory_entity') or self._entity_for(user_text, decision),
                'entity_type': decision.get('memory_entity_type', 'fact'),
                'memory_content': decision.get('memory_content') or user_text,
            }
            decision = {**decision, 'needs_clarification': True, 'needs_memory_storage': False}
            await self._respond(user_text, decision, question, generation)
            return

        if intent == 'memory_store':
            self.pipeline.pending_memory_clarification = None
            await self._store_and_confirm(
                user_text,
                decision,
                entity=decision.get('memory_entity') or self._entity_for(user_text, decision),
                entity_type=decision.get('memory_entity_type', 'fact'),
                value=decision.get('memory_value') or decision.get('memory_content') or user_text,
                generation=generation,
            )
            return

        if intent == 'memory_retrieve':
            context = await self.pipeline.memory_worker.retrieve_context(
                decision.get('memory_query') or user_text
            )
            text = await self._compose_retrieve_response(user_text, decision, context)
            await self._respond(user_text, decision, text, generation)
            return

        await self._dispatch_general(user_text, decision, generation)

    async def _resume_memory(self, user_text, pending, generation):
        raw = await self._analyze_memory_turn(user_text, pending)
        if not raw or raw.get('intent') not in ('memory_store', 'memory_clarify'):
            return False

        decision = self._normalize(raw)
        if decision.get('needs_clarification'):
            question = (decision.get('clarification_question')
                        or decision.get('fast_response')
                        or 'Could you tell me a little more?')
            await self._respond(user_text, decision, question, generation)
            return True

        self.pipeline.pending_memory_clarification = None
        await self._store_and_confirm(
            user_text,
            decision,
            entity=(decision.get('memory_entity') or pending.get('entity')
                    or self._entity_for(user_text, decision)),
            entity_type=decision.get('memory_entity_type', pending.get('entity_type', 'fact')),
            value=(decision.get('memory_value') or decision.get('memory_content')
                   or pending.get('memory_content') or user_text),
            generation=generation,
        )
        return True

    async def _store_and_confirm(self, user_text, decision, entity, entity_type, value, generation):
        stored = await self.pipeline.memory_worker.store_memory(
            entity=entity, entity_type=entity_type, value=value
        )
        if not stored:
            await self._respond(
                user_text, decision,
                "I had trouble saving that. Could you tell me once more?", generation,
            )
            return

        text = await self._compose_store_response(user_text, value)
        await self._respond(user_text, {**decision, 'needs_memory_storage': True}, text, generation)

    # ------------------------------------------------------------------
    # general routing
    # ------------------------------------------------------------------

    async def _dispatch_general(self, user_text, decision, generation):
        context = ''
        if decision.get('needs_memory_retrieval'):
            context = await self.pipeline.memory_worker.retrieve_context(
                decision.get('memory_query') or user_text
            )

        if decision.get('needs_reasoning'):
            await self.pipeline.consumer.send_decision(decision)
            await self.pipeline.intent_queue.put({
                'user_text': user_text,
                'decision': decision,
                'memory_context': context,
                'generation': generation,
            })
            return

        await self._respond(
            user_text, decision, self._fallback_text(decision, context), generation
        )

    def _fallback_text(self, decision, context):
        if decision.get('needs_clarification'):
            return decision.get('fast_response') or 'Could you tell me a bit more?'
        intent = decision.get('intent')
        if intent == 'memory_retrieve':
            if context:
                return context.splitlines()[0]
            return "I don't have that written down yet."
        if intent == 'unclear':
            return 'Could you tell me a bit more?'
        return decision.get('fast_response') or 'Okay.'

    async def _respond(self, user_text, decision, text, generation):
        await self.pipeline.consumer.send_decision(decision)
        await self.pipeline.response_queue.put({
            'user_text': user_text,
            'decision': decision,
            'response': text,
            'generation': generation,
        })

    # ------------------------------------------------------------------
    # decision shaping
    # ------------------------------------------------------------------

    def _normalize(self, decision):
        decision = decision or {}
        intent = decision.get('intent', 'unclear')
        is_fast = decision.get('is_fast_response', decision.get('is_fast', False))
        needs_memory = bool(decision.get('needs_memory', False))
        needs_reasoning = bool(decision.get('needs_reasoning', False))

        needs_store = decision.get('needs_memory_storage', intent == 'memory_store')
        needs_retrieve = decision.get('needs_memory_retrieval', intent == 'memory_retrieve')

        # Keep confidence absent rather than inventing one. A default of 0.7
        # sat below the clarification threshold, so every turn was diverted
        # into a clarifying question and the reasoning layer was unreachable.
        confidence = decision.get('confidence')
        try:
            confidence = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence = None

        normalized = {
            **decision,
            'intent': intent,
            'is_fast_response': bool(is_fast),
            'needs_memory': needs_memory,
            'needs_reasoning': needs_reasoning,
            'needs_gpt': needs_reasoning,
            'needs_memory_storage': bool(needs_store),
            'needs_memory_retrieval': bool(needs_retrieve),
            'confidence': confidence,
        }
        if normalized['is_fast_response'] and not normalized.get('fast_response'):
            normalized['fast_response'] = 'Okay.'
        return normalized

    def _needs_clarification(self, decision):
        if decision.get('intent') != 'memory_store':
            return False
        completeness = decision.get('information_completeness') or {}
        if completeness.get('should_ask'):
            return True
        if not completeness.get('is_complete', True):
            return True
        if completeness.get('missing_fields'):
            return True
        confidence = decision.get('confidence')
        return confidence is not None and confidence < 0.85

    def _entity_for(self, user_text, decision):
        text = (user_text or '').lower()
        entity_map = {
            'keys': ['key', 'keys'],
            'wallet': ['wallet'],
            'glasses': ['glasses', 'spectacles'],
            'phone': ['phone', 'mobile'],
            'remote': ['remote'],
            'documents': ['document', 'documents', 'papers'],
            'medicine': ['medicine', 'medication', 'pill', 'pills', 'pillbox', 'tablets'],
            'book': ['book', 'books'],
            'hearing aid': ['hearing aid'],
        }
        for entity, terms in entity_map.items():
            if any(re.search(rf'\b{re.escape(t)}\b', text) for t in terms):
                return entity
        return 'memory' if decision.get('intent') == 'memory_store' else 'user'

    # ------------------------------------------------------------------
    # model calls
    # ------------------------------------------------------------------

    async def _classify(self, user_text):
        prompt = f"""You are a strict decision layer for a dementia-safe voice assistant.
Output JSON only (no markdown).

Schema:
{{
  "intent": "command|memory_store|memory_retrieve|unclear|casual|question",
  "is_fast": true,
  "needs_memory": false,
  "needs_reasoning": false,
  "fast_response": "short direct reply if fast",
  "memory_query": "",
  "memory_content": "",
  "confidence": 0.0
}}

User: "{user_text}"
"""
        result = await self._llm_json(prompt)
        if result:
            return result
        return {
            'intent': 'unclear',
            'is_fast': True,
            'needs_memory': False,
            'needs_reasoning': False,
            'fast_response': 'Could you say that again?',
            'confidence': 0.2,
        }

    async def _analyze_memory_turn(self, user_text, pending=None):
        pending_text = (pending or {}).get('original_text', '')

        prompt = f"""You are a memory-intent analyst for a voice assistant.
Output JSON only.

Return this schema exactly:
{{
  "intent": "memory_store|memory_retrieve|memory_clarify|other",
  "is_fast": true,
  "needs_memory": false,
  "needs_reasoning": false,
  "needs_memory_storage": false,
  "needs_memory_retrieval": false,
  "needs_clarification": false,
  "information_completeness": {{
    "is_complete": true,
    "missing_fields": [],
    "should_ask": false
  }},
  "memory_entity": "",
  "memory_entity_type": "fact",
  "memory_value": "",
  "memory_query": "",
  "clarification_question": "",
  "confidence": 0.0
}}

Rules:
- If the person is stating something they want remembered, intent=memory_store.
- If they are asking where, what or when something was, intent=memory_retrieve.
- Treat a memory as structured: object, type, location, time.
- If an important field is missing or ambiguous, set is_complete=false and
  should_ask=true and write one short clarifying question.
- Preserve the person's own wording in memory_value.
- Never invent details and never assume a missing one.
- Ask at most one short, natural question.
- Use the pending original turn as context when one is given.

Pending original turn: "{pending_text}"
User message: "{user_text}"
"""
        return await self._llm_json(prompt)

    async def _clarification_question(self, user_text, decision):
        model_question = decision.get('clarification_question')
        if model_question:
            return model_question.strip()

        prompt = f"""Ask one short, warm, natural clarifying question so this memory can be
stored accurately. Do not say the input was vague. Ask only one question.

User message: {user_text}"""
        generated = await self._llm_text(prompt)
        return (generated or '').strip() or 'Could you give me one more detail so I remember it properly?'

    async def _compose_store_response(self, user_text, memory_value):
        memory_value = (memory_value or '').strip()
        if not memory_value:
            return 'Got it. I will remember that.'

        prompt = f"""Write one short, warm acknowledgement that a memory has been saved.
Do not start with "Okay". Do not say "I found". One sentence only.

They said: {user_text}
Saved: {memory_value}"""
        generated = await self._llm_text(prompt)
        return (generated or '').strip() or f"Got it. I'll remember that {memory_value}."

    async def _compose_retrieve_response(self, user_text, decision, memory_context):
        memory_context = (memory_context or '').strip()
        if not memory_context:
            return decision.get('fast_response') or "I don't have that written down yet."

        prompt = f"""Answer the person's question in one short sentence using the context.
Do not say "I found". Do not start with "Okay".

Question: {user_text}
Context: {memory_context}"""
        generated = await self._llm_text(prompt)
        if generated:
            return generated.strip()

        first = memory_context.splitlines()[0].strip()
        if first.lower().startswith('i '):
            return 'You ' + first[2:]
        if first.lower().startswith('my '):
            return 'Your ' + first[3:]
        return f"You told me {first}."

    async def _llm_json(self, prompt, default=None):
        try:
            resp = await self.http.post(
                f"{self.ollama_url}/api/generate",
                json={
                    'model': self.ollama_model,
                    'prompt': prompt,
                    'stream': False,
                    'format': 'json',
                    'options': {'temperature': 0.1},
                },
            )
            resp.raise_for_status()
            body = resp.json().get('response', '')
            match = re.search(r'\{.*\}', body, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except (httpx.HTTPError, json.JSONDecodeError, ValueError) as e:
            logger.error(f"Router JSON call failed: {e}")
        return default

    async def _llm_text(self, prompt, default=None):
        try:
            resp = await self.http.post(
                f"{self.ollama_url}/api/generate",
                json={
                    'model': self.ollama_model,
                    'prompt': prompt,
                    'stream': False,
                    'options': {'temperature': 0.2},
                },
            )
            resp.raise_for_status()
            return resp.json().get('response', '').strip() or default
        except (httpx.HTTPError, ValueError) as e:
            logger.error(f"Router text call failed: {e}")
            return default
