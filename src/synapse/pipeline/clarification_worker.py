import asyncio
import logging

logger = logging.getLogger(__name__)


class ClarificationWorker:
    """
    Cognitive safety gate between QwenRouter and GPTWorker.
    
    Purpose:
    - Check confidence and slot completeness
    - Ask clarifying questions for low-confidence intents
    - Maintain context for dementia-safe interaction
    - Apply cognitive safety rules (repeat confirmations, simpler language)
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline

    async def run(self):
        """Main worker loop."""
        try:
            while not self.pipeline.shutdown_event.is_set():
                try:
                    decision = await asyncio.wait_for(
                        self.pipeline.intent_queue.get(),
                        timeout=2.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                if not decision:
                    continue

                user_text = decision.get('user_text', '')
                intent = decision.get('intent', 'unclear')
                confidence = float(decision.get('confidence', 0.5))

                # 🔥 Confidence threshold: prefer asking over assuming
                if confidence < 0.8:
                    # Ask for clarification instead of executing
                    clarification = await self._generate_safety_question(decision, user_text)
                    decision['needs_clarification'] = True
                    decision['fast_response'] = clarification
                    
                    # Track pending state
                    self.pipeline.conversation_state['pending_slots'] = {
                        'original_intent': intent,
                        'confidence': confidence,
                        'user_text': user_text,
                    }

                    await self.pipeline.consumer.send_decision(decision)
                    await self.pipeline.response_queue.put({
                        'user_text': user_text,
                        'decision': decision,
                        'response': clarification,
                    })
                    continue

                # Check for required slots based on intent
                missing_slots = self._check_required_slots(intent, decision, user_text)
                if missing_slots:
                    # Ask for missing information
                    clarification = await self._generate_slot_question(intent, missing_slots, user_text)
                    decision['needs_clarification'] = True
                    decision['missing_slots'] = missing_slots
                    decision['fast_response'] = clarification

                    self.pipeline.conversation_state['pending_slots'] = missing_slots

                    await self.pipeline.consumer.send_decision(decision)
                    await self.pipeline.response_queue.put({
                        'user_text': user_text,
                        'decision': decision,
                        'response': clarification,
                    })
                    continue

                # High confidence + all slots → execute (send to GPT for reasoning)
                self.pipeline.update_conversation_context(user_text, decision)

                # Add cognitive safety context
                decision['safe_context'] = self._build_safe_context()
                
                # Route to GPT worker for reasoning
                await self.pipeline.gpt_input_queue.put({
                    'user_text': user_text,
                    'decision': decision,
                    'memory_context': decision.get('memory_context', ''),
                })

        except Exception as e:
            logger.error(f"Clarification Worker error: {e}")

    def _check_required_slots(self, intent, decision, user_text):
        """Check for required slots based on intent type."""
        missing = {}

        # Reminder slots
        if intent == 'command' and 'reminder' in user_text.lower():
            memory_content = decision.get('memory_content', '').lower()
            if 'reminder' not in memory_content or not any(term in user_text.lower() for term in ('in', 'at', 'reminder', 'minutes', 'hour', 'hours')):
                missing['reminder_time'] = 'When should I remind you?'
            if 'reminder' in memory_content and not any(task_word in memory_content for task_word in ('medicine', 'call', 'task', 'appointment', 'meeting', 'take')):
                missing['reminder_task'] = 'What should I remind you about?'

        # Memory store slots
        if intent == 'memory_store':
            text_lower = user_text.lower()
            has_object = any(term in text_lower for term in ('key', 'wallet', 'glasses', 'phone', 'book', 'medicine', 'remote', 'documents'))
            has_location = any(term in text_lower for term in ('table', 'desk', 'shelf', 'drawer', 'kitchen', 'room', 'bed', 'office', 'counter', 'sofa'))
            
            if has_object and not has_location:
                missing['location'] = 'Where did you leave it?'
            if has_location and not has_object:
                missing['object'] = 'What item are you talking about?'

        return missing

    def _build_safe_context(self):
        """Build context for safe GPT reasoning."""
        state = self.pipeline.conversation_state
        return {
            'last_intent': state.get('last_intent'),
            'context_window': state.get('context_window', []),
            'user_profile': state.get('user_profile', {}),
        }

    def _wrap_with_safety(self, response, decision, user_text):
        """Wrap response with cognitive safety rules for dementia users."""
        intent = decision.get('intent')
        confidence = decision.get('confidence', 0.7)

        # Rule 1: Always confirm important actions
        if intent in ('memory_store', 'command'):
            if 'reminder' in user_text.lower():
                # Repeat the reminder back
                return f"{response} Can you tell me what it's for so I remember it correctly?"
            if 'key' in user_text.lower() or 'wallet' in user_text.lower():
                # Confirm where it is
                return f"{response} Just to be sure, where did you leave it?"

        # Rule 2: Use simpler sentences for low confidence
        if confidence < 0.7:
            # Split long sentences
            if len(response) > 80:
                sentences = response.split('. ')
                response = sentences[0] + '.'

        # Rule 3: Natural, supportive tone
        if intent == 'memory_retrieve':
            if 'could not find' in response.lower():
                return "I haven't found that yet. Can you remind me what you're looking for?"

        return response
