import asyncio
import logging
import os

import httpx

logger = logging.getLogger(__name__)


class GPTWorker:
    """
    Complex reasoning worker using Mistral Small API.
    Kept as GPTWorker class name to avoid pipeline import churn.
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.api_key = os.getenv('MISTRAL_API_KEY', '')
        self.model = os.getenv('MISTRAL_MODEL', 'mistral-small-latest')
        self.base_url = os.getenv('MISTRAL_BASE_URL', 'https://api.mistral.ai/v1')
        self.http = httpx.AsyncClient(timeout=90.0)
    
    async def run(self):
        """Main worker loop."""
        try:
            while not self.pipeline.shutdown_event.is_set():
                try:
                    intent_data = await asyncio.wait_for(
                        self.pipeline.gpt_input_queue.get(),
                        timeout=2.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                user_text = intent_data['user_text']
                decision = intent_data['decision']
                
                memory_context = intent_data.get('memory_context', '')
                if not memory_context and decision.get('needs_memory_retrieval'):
                    memory_context = await self._fetch_memory(decision.get('memory_query'))
                
                response = await self._generate_streaming_response(
                    user_text,
                    memory_context,
                    decision
                )
                
                await self.pipeline.response_queue.put({
                    'user_text': user_text,
                    'decision': decision,
                    'response': response,
                    'response_chunk_sent': True
                })
        
        except Exception as e:
            logger.error(f"Mistral Worker error: {e}")
    
    async def _fetch_memory(self, query):
        """Fetch relevant memory from FAISS."""
        if not query:
            return ""
        
        try:
            return await self.pipeline.memory_worker.retrieve_context(query)
        except Exception as e:
            logger.error(f"Memory fetch error: {e}")
            return ""
    
    async def _generate_streaming_response(self, user_text, memory_context, decision):
        """Generate reasoning response using Mistral Small API."""
        if not self.api_key:
            return "I need API setup for complex reasoning."
        
        system_prompt = """You are a compassionate voice assistant for dementia patients.
- Keep responses SHORT (1-2 sentences max)
- Use simple, clear language
- Be warm and reassuring
- Never confuse or rush the user"""
        
        user_prompt = (
            f"Memory context: {memory_context}\n\nUser: {user_text}"
            if memory_context
            else f"User: {user_text}"
        )
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            payload = {
                'model': self.model,
                'messages': messages,
                'max_tokens': 140,
                'temperature': 0.4,
            }
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
            }

            resp = await self.http.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            text = data['choices'][0]['message']['content'].strip()
            await self.pipeline.consumer.send_response_chunk(text)
            return text
        
        except Exception as e:
            logger.error(f"Mistral API error: {e}")
            return "I didn't quite understand that. Can you try again?"
