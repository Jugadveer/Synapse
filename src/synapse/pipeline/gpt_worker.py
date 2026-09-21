import logging
import os

import httpx

from pipeline.worker import PipelineWorker

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a compassionate voice assistant for a person living with dementia.
- Keep responses SHORT (1-2 sentences max)
- Use simple, clear language and short sentences
- Be warm and reassuring; acknowledge feelings first
- Never correct, contradict, argue or rush the person
- Never invent details about their life; if you do not know, say so gently"""


class GPTWorker(PipelineWorker):
    """Reasoning layer backed by the Mistral chat API.

    Named GPTWorker for continuity with the rest of the pipeline.
    """

    name = 'reasoning'
    input_queue_name = 'gpt_input_queue'

    def __init__(self, pipeline):
        super().__init__(pipeline)
        self.api_key = os.getenv('MISTRAL_API_KEY', '').strip()
        self.model = os.getenv('MISTRAL_MODEL', 'mistral-small-latest')
        self.base_url = os.getenv('MISTRAL_BASE_URL', 'https://api.mistral.ai/v1').rstrip('/')
        self.http = httpx.AsyncClient(timeout=90.0)

        if not self.api_key:
            logger.warning("MISTRAL_API_KEY is not set; the reasoning layer will use fallbacks")

    async def aclose(self):
        await self.http.aclose()

    async def handle(self, item):
        generation = item.get('generation')
        if self.pipeline.is_stale(generation):
            return

        user_text = item['user_text']
        decision = item['decision']

        memory_context = item.get('memory_context', '')
        if not memory_context and decision.get('needs_memory_retrieval'):
            memory_context = await self._fetch_memory(decision.get('memory_query'))

        response = await self._generate_response(user_text, memory_context, decision)

        if self.pipeline.is_stale(generation):
            return

        await self.pipeline.consumer.send_response_chunk(response)
        await self.pipeline.response_queue.put({
            'user_text': user_text,
            'decision': decision,
            'response': response,
            'response_chunk_sent': True,
            'generation': generation,
        })

    async def on_error(self, item, exc):
        await self.pipeline.response_queue.put({
            'user_text': item.get('user_text', ''),
            'decision': item.get('decision', {}),
            'response': "Sorry, I lost my train of thought. Could you say that again?",
            'generation': item.get('generation'),
        })

    async def _fetch_memory(self, query):
        if not query:
            return ""
        return await self.pipeline.memory_worker.retrieve_context(query)

    async def _generate_response(self, user_text, memory_context, decision):
        if not self.api_key:
            return "I can still help with reminders and remembering things, but I need to be set up for longer conversations."

        user_prompt = (
            f"What I remember: {memory_context}\n\nThey said: {user_text}"
            if memory_context
            else f"They said: {user_text}"
        )

        payload = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': user_prompt},
            ],
            'max_tokens': 140,
            'temperature': 0.4,
        }
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }

        try:
            resp = await self.http.post(
                f"{self.base_url}/chat/completions", json=payload, headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
            return data['choices'][0]['message']['content'].strip()
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
            logger.error(f"Mistral request failed: {e}")
            return "I didn't quite catch that. Could you tell me again?"
