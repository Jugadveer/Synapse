import logging
import re

from pipeline.worker import PipelineWorker

logger = logging.getLogger(__name__)

#: Ask rather than assume below this, but only when the router actually
#: reported a confidence. A missing confidence is not evidence of doubt - it
#: used to be filled in with a 0.7 default that sat under the threshold, so
#: every single turn was diverted into a clarifying question and the reasoning
#: layer was unreachable.
CONFIDENCE_THRESHOLD = 0.8

OBJECT_TERMS = (
    'key', 'keys', 'wallet', 'glasses', 'phone', 'remote', 'papers', 'documents',
    'book', 'books', 'tablet', 'medication', 'medicine', 'pill', 'pills',
    'pillbox', 'card', 'cards', 'hearing aid', 'stick', 'cane',
)
LOCATION_TERMS = (
    'desk', 'table', 'counter', 'shelf', 'drawer', 'cabinet', 'nightstand',
    'kitchen', 'office', 'room', 'bed', 'bedroom', 'sofa', 'chair', 'bag',
    'pocket', 'car', 'hall', 'bathroom',
)


def mentions_any(text, terms):
    """Whole-word membership test.

    A plain substring test matched 'in' inside 'remind', so the reminder-time
    slot check could never fire.
    """
    lowered = (text or '').lower()
    return any(re.search(rf'\b{re.escape(term)}\b', lowered) for term in terms)


class ClarificationWorker(PipelineWorker):
    """
    Safety gate between the router and the reasoning layer.

    Prefers one short clarifying question over acting on a guess, which is the
    behaviour the dementia-dialogue literature calls incremental clarification.
    """

    name = 'clarification'
    input_queue_name = 'intent_queue'

    async def handle(self, item):
        generation = item.get('generation')
        if self.pipeline.is_stale(generation):
            return

        user_text = item.get('user_text', '')
        decision = item.get('decision', {})
        intent = decision.get('intent', 'unclear')
        confidence = decision.get('confidence')

        # Only treat confidence as a signal when the router reported one.
        explicitly_unsure = confidence is not None and float(confidence) < CONFIDENCE_THRESHOLD
        missing_slots = self._check_required_slots(intent, decision, user_text)

        if explicitly_unsure or missing_slots:
            reason = 'low confidence' if explicitly_unsure else f'missing {sorted(missing_slots)}'
            logger.info(f"[clarification] asking instead of acting ({reason})")

            question = (
                self._slot_question(missing_slots)
                if missing_slots
                else decision.get('clarification_question') or 'Could you tell me a little more?'
            )

            decision = {**decision, 'needs_clarification': True, 'fast_response': question}
            if missing_slots:
                decision['missing_slots'] = missing_slots

            self.pipeline.conversation_state['pending_slots'] = missing_slots or {
                'original_intent': intent,
                'confidence': confidence,
                'user_text': user_text,
            }

            await self.pipeline.consumer.send_decision(decision)
            await self.pipeline.response_queue.put({
                'user_text': user_text,
                'decision': decision,
                'response': question,
                'generation': generation,
            })
            return

        self.pipeline.update_conversation_context(user_text, decision)
        decision = {**decision, 'safe_context': self._build_safe_context()}

        await self.pipeline.gpt_input_queue.put({
            'user_text': user_text,
            'decision': decision,
            'memory_context': item.get('memory_context', ''),
            'generation': generation,
        })

    def _check_required_slots(self, intent, decision, user_text):
        """Required fields that the turn did not supply."""
        missing = {}

        if intent == 'memory_store':
            has_object = mentions_any(user_text, OBJECT_TERMS)
            has_location = mentions_any(user_text, LOCATION_TERMS)
            if has_object and not has_location:
                missing['location'] = 'Where did you leave it?'
            elif has_location and not has_object:
                missing['object'] = 'What was it you put there?'

        return missing

    @staticmethod
    def _slot_question(missing_slots):
        """One question only, even when several slots are missing."""
        return next(iter(missing_slots.values()))

    def _build_safe_context(self):
        state = self.pipeline.conversation_state
        return {
            'last_intent': state.get('last_intent'),
            'context_window': state.get('context_window', []),
            'user_profile': state.get('user_profile', {}),
        }
