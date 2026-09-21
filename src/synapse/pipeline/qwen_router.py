"""Intent router.

Runs the fine-tuned Qwen 2.5 3B model through Ollama to decide what a turn is
asking for, and handles the deterministic cases (reminders) locally so they
behave the same way every time.
"""

import json
import logging
import os
import re
import httpx
from django.utils import timezone

from pipeline.phrasing import (
    acknowledge_memory, answer_from_memory, confirm_reminder, describe_reminder,
)
from pipeline.prompts import classify_prompt, memory_analyst_prompt
from pipeline.reminder_parser import looks_like_reminder, parse_reminder
from pipeline.reminder_scheduler import create_reminder, pending_reminders
from pipeline.worker import PipelineWorker

logger = logging.getLogger(__name__)

LIST_REMINDERS = re.compile(
    r'\b(what|which|any|list|tell me).{0,25}\breminders?\b|\bmy reminders?\b'
)

#: Words that carry no memory on their own. A turn made only of these is small
#: talk, whatever the router says it is.
SMALL_TALK = frozenset({
    'hello', 'hi', 'hey', 'good', 'morning', 'afternoon', 'evening', 'night',
    'thanks', 'thank', 'you', 'ok', 'okay', 'yes', 'no', 'yeah', 'nope',
    'bye', 'goodbye', 'how', 'are', 'is', 'it', 'going', 'please', 'sorry',
    'right', 'sure', 'fine', 'well', 'today', 'and', 'the', 'a', 'i', 'am',
    'im', 'm', 'very', 'much', 'lovely', 'nice', 'to', 'see', 'hear', 'from',
    'there',
})


#: A question, however it is phrased. Checked first, because "where did I put
#: my glasses" contains the same verb as "I put my glasses on the shelf".
#: Word boundaries are spelled with \s rather than \b so the pattern
#: survives being edited by tools that mangle backslash escapes.
QUESTION = re.compile(
    r"^\s*(?:where|what|when|who|whom|which|how|why|do|does|did|can|could"
    r"|would|will|have|has|is|are|am|shall|should)(?:\s|$)"
    r"|\?\s*$"
)

#: Declarative statements complete enough to store as they stand.
#: The "I left/put/kept" form needs a place as well as an object: "I put my
#: glasses somewhere" is a memory with a hole in it, and belongs on the
#: clarification path that asks one short question rather than here.
MEMORY_STATEMENT = re.compile(
    r"(?:^|\s)(?:"
    r"i\s+(?:left|put|kept|placed|stored|hid|moved|parked)\s+.*?"
    r"\s(?:in|on|at|under|behind|beside|by|near|inside)\s+\w"
    r"|please\s+remember|remember\s+that"
    r"|don\'?t\s+let\s+me\s+forget"
    r"|my\s+[a-z]+\s+(?:is|are|was|were)\s+"
    r"(?:in|on|at|under|behind|beside|called|named)\s)"
)


def looks_like_memory_statement(text):
    """Recognise a memory worth storing without asking the model.

    Reminders are handled deterministically because that makes them behave the
    same way every time; memory statements need the same treatment. A 1.5b
    router classifies "I left my keys on the kitchen table" as a retrieval, so
    nothing gets stored and the person is told "Okay." The rule below settles
    the unambiguous cases and leaves the rest to the model.
    """
    lowered = (text or '').strip().lower()
    if not lowered or QUESTION.search(lowered):
        return False
    return bool(MEMORY_STATEMENT.search(lowered)) and is_storable(lowered)


def is_storable(text):
    """Whether a turn contains anything worth remembering.

    A small router classifies almost anything as memory_store - the 0.5b model
    answers that for greetings, at confidence 1.0 - and the slot checks do not
    catch it, because a greeting is missing both an object and a location
    rather than exactly one of the two. Without this, "Hello there" ends up in
    the person's memories.
    """
    tokens = re.findall(r"[a-z']+", (text or '').lower())
    if len(tokens) < 3:
        return False
    return not all(token in SMALL_TALK for token in tokens)


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

        if looks_like_memory_statement(user_text):
            self.pipeline.pending_memory_clarification = None
            decision = self._normalize({
                'intent': 'memory_store', 'is_fast': True, 'needs_memory': True,
                'needs_memory_storage': True, 'confidence': 0.95,
            })
            await self._store_and_confirm(
                user_text, decision,
                entity=self._entity_for(user_text, decision),
                entity_type='location',
                value=user_text,
                generation=generation,
            )
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
        request = parse_reminder(user_text, now=timezone.localtime())
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
            user_text, decision, confirm_reminder(task, spoken), generation
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
            request = parse_reminder(probe, now=timezone.localtime())
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
                    confirm_reminder(task, pending.get('spoken_time', '')),
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
            text = describe_reminder(first.text, first.spoken_time)
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
        if not is_storable(value):
            logger.info(f'Refusing to store small talk as a memory: {value!r}')
            await self._respond(
                user_text, {**decision, 'intent': 'casual', 'needs_memory_storage': False},
                'Hello. It is good to hear from you.', generation,
            )
            return

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
        result = await self._llm_json(classify_prompt(user_text))
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
        return await self._llm_json(memory_analyst_prompt(user_text, pending_text))

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
        """Confirm a stored memory.

        This used to ask the router model for the wording. At 1.5b it answered
        "Thank you for remembering to save that" - the assistant thanking the
        person for doing the assistant's own job. A confirmation is a short,
        predictable sentence, so it is built rather than generated.
        """
        return acknowledge_memory(memory_value or user_text)

    async def _compose_retrieve_response(self, user_text, decision, memory_context):
        memory_context = (memory_context or '').strip()
        if not memory_context:
            return "I don't have that written down yet."

        prompt = f"""Answer the person's question in one short sentence using the context.
Do not say "I found". Do not start with "Okay".

Question: {user_text}
Context: {memory_context}"""
        generated = await self._llm_text(prompt)
        if generated:
            return generated.strip()

        # The old fallback swapped only a leading "I" or "My", so a memory
        # came back as "You left my keys on the table".
        return answer_from_memory(memory_context.splitlines()[0])

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
