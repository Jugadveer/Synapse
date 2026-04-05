import asyncio
from datetime import datetime, timedelta
import json
import logging
import os
import re

import httpx

logger = logging.getLogger(__name__)


class QwenRouter:
    """
    Qwen 2.5 3B router via Ollama.
    Uses OUMI fine-tuned model (served by Ollama) for:
    - intent classification
    - fast-path response
    - memory/reasoning routing
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.ollama_url = os.getenv('OLLAMA_URL', 'http://127.0.0.1:11434')
        self.ollama_model = os.getenv('OLLAMA_QWEN_MODEL', 'qwen2.5:3b-instruct')
        self.http = httpx.AsyncClient(timeout=90.0)
    
    async def run(self):
        """Main worker loop."""
        try:
            while not self.pipeline.shutdown_event.is_set():
                try:
                    text_data = await asyncio.wait_for(
                        self.pipeline.text_queue.get(),
                        timeout=2.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                if text_data['type'] == 'final':
                    user_text = text_data['text']

                    pending = getattr(self.pipeline, 'pending_memory_clarification', None)
                    if pending:
                        pending_decision = await self._analyze_memory_turn(user_text, pending)
                        if pending_decision and pending_decision.get('intent') in ('memory_store', 'memory_clarify'):
                            decision = self._normalize_decision(pending_decision)

                            if decision.get('needs_clarification'):
                                self.pipeline.pending_memory_clarification = {
                                    'original_text': pending.get('original_text', user_text),
                                    'entity': decision.get('memory_entity') or pending.get('entity') or self._extract_memory_entity(user_text, decision),
                                    'entity_type': decision.get('memory_entity_type', pending.get('entity_type', 'fact')),
                                    'memory_content': pending.get('memory_content') or pending.get('original_text', user_text),
                                    'clarification_prompt': decision.get('clarification_question') or decision.get('fast_response') or 'Could you clarify?',
                                }

                                await self.pipeline.consumer.send_decision(decision)
                                await self.pipeline.response_queue.put({
                                    'user_text': user_text,
                                    'decision': decision,
                                    'response': decision.get('clarification_question') or decision.get('fast_response') or 'Could you clarify what you mean?',
                                })
                                continue

                            self.pipeline.pending_memory_clarification = None
                            await self.pipeline.consumer.send_decision(decision)

                            await self.pipeline.memory_worker.store_memory(
                                entity=decision.get('memory_entity') or pending.get('entity') or self._extract_memory_entity(user_text, decision),
                                entity_type=decision.get('memory_entity_type', pending.get('entity_type', 'fact')),
                                value=decision.get('memory_value') or decision.get('memory_content') or pending.get('memory_content') or user_text,
                            )

                            store_response = await self._compose_memory_store_response(
                                user_text,
                                decision.get('memory_value') or decision.get('memory_content') or pending.get('memory_content') or user_text,
                            )
                            decision['fast_response'] = store_response
                            await self.pipeline.response_queue.put({
                                'user_text': user_text,
                                'decision': decision,
                                'response': store_response,
                            })
                            continue

                    memory_decision = await self._analyze_memory_turn(user_text)
                    if memory_decision and memory_decision.get('intent') in ('memory_store', 'memory_retrieve', 'memory_clarify'):
                        decision = self._normalize_decision(memory_decision)

                        if decision.get('intent') == 'memory_store' and self._needs_memory_clarification(decision):
                            clarification = await self._generate_memory_clarification_question(user_text, decision)
                            self.pipeline.pending_memory_clarification = {
                                'original_text': user_text,
                                'entity': decision.get('memory_entity') or self._extract_memory_entity(user_text, decision),
                                'entity_type': decision.get('memory_entity_type', 'fact'),
                                'memory_content': decision.get('memory_content') or user_text,
                                'clarification_prompt': clarification,
                            }
                            decision = {
                                **decision,
                                'needs_clarification': True,
                                'fast_response': clarification,
                                'clarification_question': clarification,
                                'needs_memory_storage': False,
                            }

                            await self.pipeline.consumer.send_decision(decision)
                            await self.pipeline.response_queue.put({
                                'user_text': user_text,
                                'decision': decision,
                                'response': clarification,
                            })
                            continue

                        if decision.get('needs_clarification'):
                            self.pipeline.pending_memory_clarification = {
                                'original_text': user_text,
                                'entity': decision.get('memory_entity') or self._extract_memory_entity(user_text, decision),
                                'entity_type': decision.get('memory_entity_type', 'fact'),
                                'memory_content': decision.get('memory_content') or user_text,
                                'clarification_prompt': decision.get('clarification_question') or decision.get('fast_response') or 'Could you clarify?',
                            }

                            await self.pipeline.consumer.send_decision(decision)
                            await self.pipeline.response_queue.put({
                                'user_text': user_text,
                                'decision': decision,
                                'response': decision.get('clarification_question') or decision.get('fast_response') or 'Could you clarify what you mean?',
                            })
                            continue

                        if decision.get('intent') == 'memory_store':
                            self.pipeline.pending_memory_clarification = None
                            await self.pipeline.consumer.send_decision(decision)

                            if decision.get('needs_clarification'):
                                clarification = decision.get('clarification_question') or decision.get('fast_response') or 'Could you tell me a little more?'
                                self.pipeline.pending_memory_clarification = {
                                    'original_text': user_text,
                                    'entity': decision.get('memory_entity') or self._extract_memory_entity(user_text, decision),
                                    'entity_type': decision.get('memory_entity_type', 'fact'),
                                    'memory_content': decision.get('memory_content') or user_text,
                                    'clarification_prompt': clarification,
                                }
                                await self.pipeline.response_queue.put({
                                    'user_text': user_text,
                                    'decision': decision,
                                    'response': clarification,
                                })
                                continue

                            decision['needs_memory_storage'] = True

                            await self.pipeline.memory_worker.store_memory(
                                entity=decision.get('memory_entity') or self._extract_memory_entity(user_text, decision),
                                entity_type=decision.get('memory_entity_type', 'fact'),
                                value=decision.get('memory_content') or user_text,
                            )

                            store_response = await self._compose_memory_store_response(
                                user_text,
                                decision.get('memory_content') or user_text,
                            )
                            decision['fast_response'] = store_response
                            await self.pipeline.response_queue.put({
                                'user_text': user_text,
                                'decision': decision,
                                'response': store_response,
                            })
                            continue

                        if decision.get('intent') == 'memory_retrieve':
                            await self.pipeline.consumer.send_decision(decision)

                            memory_context = await self.pipeline.memory_worker.retrieve_context(
                                decision.get('memory_query') or user_text
                            )
                            response_text = await self._compose_memory_retrieve_response(user_text, decision, memory_context)
                            decision['fast_response'] = response_text
                            await self.pipeline.response_queue.put({
                                'user_text': user_text,
                                'decision': decision,
                                'response': response_text,
                            })
                            continue

                    local_decision = self._handle_local_command(user_text)
                    if local_decision is not None:
                        decision = self._normalize_decision(local_decision)
                    else:
                        decision = await self._classify(user_text)
                        decision = self._normalize_decision(decision)

                    if decision.get('intent') == 'memory_store' and self._needs_memory_clarification(decision):
                        clarification = await self._generate_memory_clarification_question(user_text, decision)
                        self.pipeline.pending_memory_clarification = {
                            'original_text': user_text,
                            'entity': self._extract_memory_entity(user_text, decision),
                            'entity_type': 'fact',
                            'memory_content': decision.get('memory_content') or user_text,
                            'clarification_prompt': clarification,
                        }
                        decision = {
                            **decision,
                            'intent': 'memory_store',
                            'is_fast_response': True,
                            'needs_memory_storage': False,
                            'needs_reasoning': False,
                            'needs_clarification': True,
                            'fast_response': clarification,
                            'clarification_question': clarification,
                            'confidence': min(float(decision.get('confidence', 0.7)), 0.6),
                        }
                    elif decision.get('intent') == 'memory_store':
                        self.pipeline.pending_memory_clarification = None

                    await self.pipeline.consumer.send_decision(decision)

                    if decision.get('needs_memory_storage'):
                        memory_entity = self._extract_memory_entity(user_text, decision)
                        memory_value = decision.get('memory_content') or user_text
                        if decision.get('intent') == 'memory_store':
                            memory_value = user_text
                        await self.pipeline.memory_worker.store_memory(
                            entity=memory_entity,
                            entity_type='fact',
                            value=memory_value,
                        )

                        if decision.get('intent') == 'memory_store' and not decision.get('needs_clarification'):
                            decision['fast_response'] = await self._compose_memory_store_response(user_text, memory_value)

                    memory_context = ''
                    if decision.get('needs_memory_retrieval'):
                        memory_context = await self.pipeline.memory_worker.retrieve_context(
                            decision.get('memory_query') or user_text
                        )

                    if decision.get('intent') == 'memory_retrieve' and memory_context:
                        decision['fast_response'] = await self._compose_memory_retrieve_response(user_text, decision, memory_context)

                    if decision.get('is_fast_response'):
                        if decision.get('intent') == 'memory_store' and not decision.get('needs_clarification'):
                            decision['fast_response'] = decision.get('fast_response') or 'Got it. I will remember that.'
                        if decision.get('intent') == 'memory_retrieve' and memory_context:
                            decision['fast_response'] = decision.get('fast_response') or await self._compose_memory_retrieve_response(user_text, decision, memory_context)
                        await self.pipeline.response_queue.put({
                            'user_text': user_text,
                            'decision': decision,
                            'response': decision.get('fast_response') or 'Got it.'
                        })
                    elif decision.get('needs_reasoning'):
                        await self.pipeline.intent_queue.put({
                            'user_text': user_text,
                            'decision': decision,
                            'memory_context': memory_context,
                        })
                    else:
                        fallback = self._compose_non_reasoning_response(decision, memory_context)
                        await self.pipeline.response_queue.put({
                            'user_text': user_text,
                            'decision': decision,
                            'response': fallback,
                        })
        except Exception as e:
            logger.error(f"Qwen Router error: {e}")

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

User: "{user_text}""".strip()

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
            response_text = resp.json().get('response', '')
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception as e:
            logger.error(f"Qwen classification error: {e}")

        return {
            'intent': 'unclear',
            'is_fast': True,
            'needs_memory': False,
            'needs_reasoning': False,
            'fast_response': 'Could you say that again?',
            'confidence': 0.2,
        }

    def _handle_local_command(self, user_text):
        """Handle high-confidence commands locally for predictable behavior."""
        text = (user_text or '').strip()
        lowered = text.lower()

        reminder_phrase = any(
            phrase in lowered
            for phrase in ('set a reminder', 'set reminder', 'remind me')
        )
        if not reminder_phrase:
            return None

        duration_match = re.search(
            r'\b(?:in\s+)?(\d+)\s*(second|seconds|minute|minutes|hour|hours|day|days)\b',
            lowered,
        )
        if duration_match:
            amount = int(duration_match.group(1))
            unit = duration_match.group(2)
            seconds_per_unit = {
                'second': 1,
                'seconds': 1,
                'minute': 60,
                'minutes': 60,
                'hour': 3600,
                'hours': 3600,
                'day': 86400,
                'days': 86400,
            }
            delta_seconds = amount * seconds_per_unit[unit]
            reminder_time = datetime.now() + timedelta(seconds=delta_seconds)
            human_time = reminder_time.strftime('%I:%M %p').lstrip('0')
            human_date = reminder_time.strftime('%b %d')
            normalized_unit = unit if amount != 1 else unit.rstrip('s')

            return {
                'intent': 'command',
                'is_fast': True,
                'needs_memory': True,
                'needs_reasoning': False,
                'needs_memory_storage': True,
                'memory_content': f"Reminder: in {amount} {normalized_unit} at {human_time} on {human_date}",
                'fast_response': f"Reminder set for {amount} {normalized_unit}. I will remind you at {human_time}.",
                'confidence': 0.99,
            }

        number_only_match = re.search(r'\bset (?:a )?reminder(?: for)?\s+(\d+)\b', lowered)
        if number_only_match:
            amount = int(number_only_match.group(1))
            return {
                'intent': 'command',
                'is_fast': True,
                'needs_memory': False,
                'needs_reasoning': False,
                'fast_response': (
                    f"I heard {amount}. Should that be minutes, hours, or days?"
                ),
                'confidence': 0.95,
            }

        return {
            'intent': 'command',
            'is_fast': True,
            'needs_memory': False,
            'needs_reasoning': False,
            'fast_response': 'Sure. Tell me when to remind you, for example: in 10 minutes.',
            'confidence': 0.9,
        }

    def _handle_local_memory_statement(self, user_text):
        """Detect declarative memory statements before model classification."""
        text = (user_text or '').strip()
        lowered = text.lower()

        if 'book' in lowered and ('page' in lowered or 'read' in lowered or 'reading' in lowered):
            return {
                'intent': 'memory_store',
                'is_fast': True,
                'needs_memory': True,
                'needs_reasoning': False,
                'needs_memory_storage': False,
                'needs_clarification': True,
                'memory_content': text,
                'fast_response': 'Which book do you mean?',
                'confidence': 0.6,
            }

        store_triggers = (
            'i left', 'i kept', 'i put', 'i placed', 'my', 'remember that', 'please remember'
        )
        if not any(trigger in lowered for trigger in store_triggers):
            return None

        object_terms = (
            'key', 'keys', 'wallet', 'glasses', 'phone', 'remote', 'papers', 'documents',
            'book', 'books', 'tablet', 'medication', 'medicine', 'pill', 'pillbox', 'card', 'cards'
        )
        location_terms = (
            'desk', 'table', 'counter', 'shelf', 'drawer', 'cabinet', 'nightstand',
            'kitchen', 'office', 'room', 'bed', 'sofa', 'chair'
        )

        has_object = any(term in lowered for term in object_terms)
        has_location = any(term in lowered for term in location_terms)

        if not (has_object and has_location):
            return None

        return {
            'intent': 'memory_store',
            'is_fast': True,
            'needs_memory': True,
            'needs_reasoning': False,
            'needs_memory_storage': True,
            'memory_content': text,
            'fast_response': 'Got it. I will remember that.',
            'confidence': 0.95,
        }

    def _handle_local_memory_retrieve(self, user_text):
        """Detect common retrieval questions and extract a search query generically."""
        text = (user_text or '').strip()
        lowered = text.lower()

        retrieval_triggers = (
            'where did i leave', 'where did i put', 'where is my', 'where are my',
            'do you remember where', 'can you find', 'what did i say about',
            'remind me where', 'i forgot where', 'where did we leave'
        )
        if not any(trigger in lowered for trigger in retrieval_triggers):
            return None

        query = self._extract_memory_query(text)
        if not query:
            return {
                'intent': 'memory_retrieve',
                'is_fast': True,
                'needs_memory': True,
                'needs_reasoning': False,
                'needs_memory_retrieval': True,
                'memory_query': text,
                'fast_response': 'Let me check my memory for that.',
                'confidence': 0.8,
            }

        return {
            'intent': 'memory_retrieve',
            'is_fast': True,
            'needs_memory': True,
            'needs_reasoning': False,
            'needs_memory_retrieval': True,
            'memory_query': query,
            'fast_response': 'Let me check my memory for that.',
            'confidence': 0.92,
        }

    def _extract_memory_query(self, user_text):
        """Extract a compact search query from a retrieval question."""
        text = (user_text or '').lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)

        stop_words = {
            'where', 'did', 'i', 'leave', 'put', 'is', 'are', 'my', 'the', 'a', 'an',
            'you', 'do', 'does', 'can', 'could', 'would', 'me', 'for', 'that', 'what',
            'said', 'about', 'remember', 'find', 'didn', 't', 'we'
        }
        tokens = [token for token in text.split() if token and token not in stop_words]

        if not tokens:
            return ""

        query = " ".join(tokens).strip()
        return query

    def _needs_memory_clarification(self, decision):
        """Return True when the model says the memory turn is incomplete."""
        if decision.get('intent') != 'memory_store':
            return False

        completeness = decision.get('information_completeness') or {}
        return (
            completeness.get('should_ask')
            or not completeness.get('is_complete', True)
            or len(completeness.get('missing_fields', [])) > 0
            or float(decision.get('confidence', 0.0)) < 0.85
        )

    async def _generate_memory_clarification_question(self, user_text, decision):
        """Generate a short clarifying question for a vague memory turn."""
        prompt = f"""Ask one short, natural, friendly clarifying question so this memory can be stored accurately.
Do not mention that the input is vague.
Do not ask more than one question.
Sound supportive and conversational.
Do not sound robotic or technical.
Preserve the topic of the user message.

User message: {user_text}
Candidate memory type: {decision.get('intent', 'memory_store')}"""

        generated = await self._llm_text(prompt, default=None)
        if generated:
            return generated.strip()

        return 'Could you give me one more detail so I can remember it accurately?'

    def _normalize_decision(self, decision):
        intent = decision.get('intent', 'unclear')
        is_fast = decision.get('is_fast_response', decision.get('is_fast', False))
        needs_memory = decision.get('needs_memory', False)
        needs_reasoning = decision.get('needs_reasoning', False)

        needs_memory_storage = decision.get('needs_memory_storage', intent == 'memory_store')
        needs_memory_retrieval = decision.get('needs_memory_retrieval', intent == 'memory_retrieve')

        if needs_memory:
            needs_memory_storage = needs_memory_storage or intent == 'memory_store'
            needs_memory_retrieval = needs_memory_retrieval or intent == 'memory_retrieve'

        normalized = {
            **decision,
            'intent': intent,
            'is_fast_response': bool(is_fast),
            'needs_reasoning': bool(needs_reasoning),
            'needs_gpt': bool(needs_reasoning),
            'needs_memory_retrieval': bool(needs_memory_retrieval),
            'needs_memory_storage': bool(needs_memory_storage),
            'confidence': float(decision.get('confidence', 0.7)),
        }

        if normalized['is_fast_response'] and not normalized.get('fast_response'):
            normalized['fast_response'] = 'Okay.'

        normalized['needs_memory'] = bool(needs_memory)
        return normalized

    def _compose_non_reasoning_response(self, decision, memory_context):
        intent = decision.get('intent')
        if decision.get('needs_clarification'):
            return decision.get('fast_response') or 'Could you clarify what exactly you mean?'
        if intent == 'memory_store':
            return decision.get('fast_response') or 'Got it. I will remember that.'
        if intent == 'memory_retrieve':
            if memory_context:
                first = memory_context.splitlines()[0]
                return first if first else 'I could not find that in memory yet.'
            return 'I could not find that in memory yet.'
        if intent == 'unclear':
            return 'Could you tell me a bit more detail?'
        return decision.get('fast_response') or 'Okay.'

    async def _compose_memory_store_response(self, user_text, memory_value):
        """Generate a natural acknowledgement for stored memory."""
        memory_value = (memory_value or '').strip()
        if not memory_value:
            return 'Got it. I will remember that.'

        prompt = f"""Write one short, natural acknowledgement for a memory assistant.
Do not use the word 'Okay' by itself.
Do not say 'I found'.
Do not repeat the memory verbatim unless needed.

User said: {user_text}
Memory to store: {memory_value}"""

        generated = await self._llm_text(prompt, default=None)
        if generated:
            return generated.strip()

        return f"Got it. I’ll remember that {memory_value}."

    async def _compose_memory_retrieve_response(self, user_text, decision, memory_context):
        """Generate a natural retrieval response from memory context."""
        memory_context = (memory_context or '').strip()
        if not memory_context:
            return decision.get('fast_response') or 'I could not find that in memory yet.'

        prompt = f"""Answer the user's memory question in one short sentence.
Do not say 'I found'.
Do not say 'Okay'.
Use the memory context naturally.

User question: {user_text}
Memory context: {memory_context}"""

        generated = await self._llm_text(prompt, default=None)
        if generated:
            return generated.strip()

        first_line = memory_context.splitlines()[0].strip()
        if first_line.lower().startswith('i '):
            return 'You ' + first_line[2:]
        if first_line.lower().startswith('my '):
            return 'Your ' + first_line[3:]
        return f"You mentioned {first_line}."

    async def _analyze_memory_turn(self, user_text, pending=None):
        """Use the model to decide whether this turn stores, retrieves, or needs clarification."""
        pending_text = ''
        if pending:
            pending_text = pending.get('original_text', '')

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
- If the user is stating something they want remembered, intent=memory_store.
- If the statement is vague or missing important details, set intent=memory_store, information_completeness.is_complete=false, information_completeness.should_ask=true, and ask one short clarifying question.
- If the user is asking where/what/when something was, intent=memory_retrieve.
- Preserve the user's wording in memory_value when storing.
-If the user is asking to be reminded about something check the context and see if any time is mentioned. 
- Do not invent details.
- Treat memory like structured information: object, type, location, time.
- If any of these are missing or ambiguous, mark the turn as incomplete.
- Never assume missing details.
- Always prefer asking a clarification question over storing incomplete memory.
- A statement like "I read till page 78" is incomplete because the book is missing.
- A statement like "I left my key on the table" is incomplete because key type and exact location may be unclear so you need to ask that
- Ask only one short, natural question.
- Prefer asking over assuming. If confidence is low or any important field is ambiguous, ask for clarification.
- Use the pending original turn as context if provided.

Pending original turn: "{pending_text}"
User message: "{user_text}""".strip()

        return await self._llm_json(prompt, default=None)

    async def _llm_json(self, prompt, default=None):
        """Call the router model and parse a JSON response."""
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
            response_text = resp.json().get('response', '')
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception as e:
            logger.error(f"LLM JSON parse error: {e}")

        return default

    async def _llm_text(self, prompt, default=None):
        """Call the router model and return plain text."""
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
            response_text = resp.json().get('response', '').strip()
            return response_text or default
        except Exception as e:
            logger.error(f"LLM text generation error: {e}")
            return default

    def _extract_memory_entity(self, user_text, decision):
        """Derive a better entity label than the generic 'user'."""
        text = (user_text or '').lower()

        entity_map = {
            'keys': ['key', 'keys'],
            'wallet': ['wallet'],
            'glasses': ['glasses'],
            'phone': ['phone'],
            'remote': ['remote'],
            'documents': ['document', 'documents', 'papers'],
            'medicine': ['medicine', 'medication', 'pill', 'pills', 'pillbox'],
            'book': ['book', 'books'],
        }

        for entity, terms in entity_map.items():
            if any(term in text for term in terms):
                return entity

        if decision.get('intent') == 'memory_store':
            return 'memory'

        return 'user'

    def _handle_pending_memory_clarification(self, user_text):
        """Legacy no-op kept for compatibility."""
        return None
