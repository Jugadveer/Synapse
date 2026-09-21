"""Prompts for the intent router.

Single source of truth, imported by both the runtime router and the fine-tune
dataset builder. They used to be separate: the model was trained on a short
three-field schema with the system prompt "You are an intent classifier",
while the router asked it at runtime for an eight-field schema with a very
different instruction. The model had never seen the fields the router parsed,
so `confidence` was never returned and the pipeline filled in a default that
sat below its own clarification threshold.
"""

SYSTEM_PROMPT = "You are a decision layer for a dementia-safe voice assistant. Output JSON only."

CLASSIFY_SCHEMA = {
    "intent": "command|memory_store|memory_retrieve|unclear|casual|question",
    "is_fast": True,
    "needs_memory": False,
    "needs_reasoning": False,
    "fast_response": "short direct reply if fast",
    "memory_query": "",
    "memory_content": "",
    "confidence": 0.0,
}

MEMORY_SCHEMA = {
    "intent": "memory_store|memory_retrieve|memory_clarify|other",
    "is_fast": True,
    "needs_memory": False,
    "needs_reasoning": False,
    "needs_memory_storage": False,
    "needs_memory_retrieval": False,
    "needs_clarification": False,
    "information_completeness": {
        "is_complete": True,
        "missing_fields": [],
        "should_ask": False,
    },
    "memory_entity": "",
    "memory_entity_type": "fact",
    "memory_value": "",
    "memory_query": "",
    "clarification_question": "",
    "confidence": 0.0,
}

CLASSIFY_TEMPLATE = """You are a strict decision layer for a dementia-safe voice assistant.
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

Rules:
- confidence is how sure you are, from 0.0 to 1.0. Always give one.
- Set needs_reasoning only when the turn needs open conversation.
- Keep fast_response to one short sentence.

User: "{user_text}"
"""

MEMORY_TEMPLATE = """You are a memory-intent analyst for a voice assistant.
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
- confidence is how sure you are, from 0.0 to 1.0. Always give one.
- Use the pending original turn as context when one is given.

Pending original turn: "{pending_text}"
User message: "{user_text}"
"""


def classify_prompt(user_text):
    return CLASSIFY_TEMPLATE.format(user_text=user_text)


def memory_analyst_prompt(user_text, pending_text=''):
    return MEMORY_TEMPLATE.format(user_text=user_text, pending_text=pending_text or '')
