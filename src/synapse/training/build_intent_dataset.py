"""Build the fine-tuning dataset for the intent router.

The previous generator emitted three fields - intent, is_fast, needs_memory -
under the system prompt "You are an intent classifier". At runtime the router
asked for an eight-field schema (and a twelve-field one for memory turns)
using completely different instruction text. The model had never seen
`confidence`, `fast_response` or `information_completeness`, so it never
produced them and the pipeline substituted defaults that broke its own gating.

Every example here is built from the same prompt templates the router sends,
imported from pipeline.prompts, so train and serve cannot drift apart again.
"""

import argparse
import json
import random
import sys
from pathlib import Path

SRC_APP_DIR = Path(__file__).resolve().parents[1]
if str(SRC_APP_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_APP_DIR))

from pipeline.prompts import (  # noqa: E402
    SYSTEM_PROMPT,
    classify_prompt,
    memory_analyst_prompt,
)

OBJECTS = [
    'keys', 'wallet', 'glasses', 'phone', 'hearing aid', 'walking stick',
    'pills', 'medicine', 'reading book', 'bank card', 'photo album', 'remote',
]
PLACES = [
    'kitchen table', 'bedside drawer', 'coat pocket', 'blue bowl by the door',
    'bathroom shelf', 'living room sofa', 'top of the fridge', 'handbag',
]
PEOPLE = ['Priya', 'my daughter', 'my son', 'Nurse Sarah', 'my neighbour Tom', 'Dr Khan']
TASKS = [
    'take my tablets', 'call my daughter', 'go to the doctor', 'feed the cat',
    'put the bins out', 'drink some water', 'switch off the cooker',
]
TIMES = ['in ten minutes', 'in two hours', 'at 4pm', 'at 9:30 am', 'tomorrow at 11am']


def chat_record(prompt, answer):
    """One OUMI chat row. The user turn is the exact runtime prompt."""
    return {
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': json.dumps(answer, separators=(',', ':'))},
        ]
    }


def classify_answer(intent, *, is_fast=True, needs_memory=False, needs_reasoning=False,
                    fast_response='', memory_query='', memory_content='', confidence=0.9):
    return {
        'intent': intent,
        'is_fast': is_fast,
        'needs_memory': needs_memory,
        'needs_reasoning': needs_reasoning,
        'fast_response': fast_response,
        'memory_query': memory_query,
        'memory_content': memory_content,
        'confidence': round(confidence, 2),
    }


def memory_answer(intent, *, needs_storage=False, needs_retrieval=False, needs_clarification=False,
                  is_complete=True, missing_fields=(), entity='', entity_type='fact',
                  value='', query='', question='', confidence=0.9):
    return {
        'intent': intent,
        'is_fast': True,
        'needs_memory': needs_storage or needs_retrieval,
        'needs_reasoning': False,
        'needs_memory_storage': needs_storage,
        'needs_memory_retrieval': needs_retrieval,
        'needs_clarification': needs_clarification,
        'information_completeness': {
            'is_complete': is_complete,
            'missing_fields': list(missing_fields),
            'should_ask': needs_clarification,
        },
        'memory_entity': entity,
        'memory_entity_type': entity_type,
        'memory_value': value,
        'memory_query': query,
        'clarification_question': question,
        'confidence': round(confidence, 2),
    }


def entity_for(obj):
    return obj.split()[-1] if obj else 'memory'


# ---------------------------------------------------------------- classify

def classify_examples(rng, count):
    rows = []
    while len(rows) < count:
        kind = rng.choice(['command', 'memory_store', 'memory_retrieve', 'casual', 'question', 'unclear'])

        if kind == 'command':
            text = rng.choice([
                f"Remind me to {rng.choice(TASKS)} {rng.choice(TIMES)}",
                f"Set a reminder for {rng.choice(TASKS)} {rng.choice(TIMES)}",
                "Turn the volume up", "Stop talking please",
            ])
            answer = classify_answer('command', fast_response='Of course.', confidence=rng.uniform(0.9, 0.99))

        elif kind == 'memory_store':
            obj, place = rng.choice(OBJECTS), rng.choice(PLACES)
            text = rng.choice([
                f"I put my {obj} on the {place}",
                f"I left my {obj} in the {place}",
                f"Remember that my {obj} are on the {place}",
            ])
            answer = classify_answer(
                'memory_store', needs_memory=True, memory_content=text,
                fast_response="I'll remember that.", confidence=rng.uniform(0.88, 0.98),
            )

        elif kind == 'memory_retrieve':
            obj = rng.choice(OBJECTS)
            text = rng.choice([
                f"Where did I put my {obj}?",
                f"Do you remember where my {obj} are?",
                f"I can't find my {obj}",
            ])
            answer = classify_answer(
                'memory_retrieve', needs_memory=True, memory_query=obj,
                fast_response='Let me check.', confidence=rng.uniform(0.85, 0.97),
            )

        elif kind == 'casual':
            text = rng.choice([
                'Hello there', 'Good morning', 'Thank you, that is kind',
                'How are you today?', "I'm feeling a bit tired",
            ])
            answer = classify_answer(
                'casual', fast_response='Hello. It is good to hear from you.',
                confidence=rng.uniform(0.85, 0.96),
            )

        elif kind == 'question':
            person = rng.choice(PEOPLE)
            text = rng.choice([
                f"Tell me about {person}", 'What should I do this afternoon?',
                'Can you explain what is happening tomorrow?',
            ])
            answer = classify_answer(
                'question', is_fast=False, needs_reasoning=True,
                confidence=rng.uniform(0.8, 0.93),
            )

        else:
            text = rng.choice([
                'The thing with the... you know', 'Um', 'I was going to say something',
                'It is the one from before', 'That thing',
            ])
            answer = classify_answer(
                'unclear', fast_response='Could you tell me a bit more?',
                confidence=rng.uniform(0.2, 0.45),
            )

        rows.append(chat_record(classify_prompt(text), answer))
    return rows


# ------------------------------------------------------------ memory turns

def memory_examples(rng, count):
    """Memory-analyst turns, including the incomplete ones that must ask."""
    rows = []
    while len(rows) < count:
        kind = rng.choice([
            'store_complete', 'store_no_place', 'store_no_object',
            'retrieve', 'vague', 'other', 'resumed',
        ])

        if kind == 'store_complete':
            obj, place = rng.choice(OBJECTS), rng.choice(PLACES)
            text = f"I put my {obj} on the {place}"
            answer = memory_answer(
                'memory_store', needs_storage=True, entity=entity_for(obj),
                entity_type='location', value=text, confidence=rng.uniform(0.9, 0.99),
            )
            prompt = memory_analyst_prompt(text)

        elif kind == 'store_no_place':
            obj = rng.choice(OBJECTS)
            text = f"I put my {obj} somewhere safe"
            answer = memory_answer(
                'memory_store', needs_clarification=True, is_complete=False,
                missing_fields=['location'], entity=entity_for(obj),
                entity_type='location', value=text,
                question='Where did you put them?', confidence=rng.uniform(0.55, 0.75),
            )
            prompt = memory_analyst_prompt(text)

        elif kind == 'store_no_object':
            place = rng.choice(PLACES)
            text = f"I left it on the {place}"
            answer = memory_answer(
                'memory_store', needs_clarification=True, is_complete=False,
                missing_fields=['object'], entity_type='location', value=text,
                question='What was it you left there?', confidence=rng.uniform(0.5, 0.7),
            )
            prompt = memory_analyst_prompt(text)

        elif kind == 'retrieve':
            obj = rng.choice(OBJECTS)
            text = rng.choice([f"Where are my {obj}?", f"Where did I leave my {obj}?"])
            answer = memory_answer(
                'memory_retrieve', needs_retrieval=True, entity=entity_for(obj),
                query=obj, confidence=rng.uniform(0.88, 0.98),
            )
            prompt = memory_analyst_prompt(text)

        elif kind == 'vague':
            text = rng.choice([
                'I read up to page 78', 'I finished that chapter', 'I spoke to them earlier',
            ])
            answer = memory_answer(
                'memory_store', needs_clarification=True, is_complete=False,
                missing_fields=['subject'], entity='memory', value=text,
                question='Which book do you mean?' if 'page' in text or 'chapter' in text
                else 'Who did you speak to?',
                confidence=rng.uniform(0.45, 0.65),
            )
            prompt = memory_analyst_prompt(text)

        elif kind == 'resumed':
            obj, place = rng.choice(OBJECTS), rng.choice(PLACES)
            original = f"I put my {obj} somewhere safe"
            text = f"on the {place}"
            answer = memory_answer(
                'memory_store', needs_storage=True, entity=entity_for(obj),
                entity_type='location', value=f"I put my {obj} on the {place}",
                confidence=rng.uniform(0.88, 0.97),
            )
            prompt = memory_analyst_prompt(text, original)

        else:
            text = rng.choice(['What is the weather like?', 'Turn the light on', 'Hello'])
            answer = memory_answer('other', confidence=rng.uniform(0.85, 0.96))
            prompt = memory_analyst_prompt(text)

        rows.append(chat_record(prompt, answer))
    return rows


def build(samples, seed=7, memory_ratio=0.5):
    rng = random.Random(seed)
    memory_count = int(samples * memory_ratio)
    rows = classify_examples(rng, samples - memory_count) + memory_examples(rng, memory_count)
    rng.shuffle(rows)
    return rows


def write_jsonl(path, rows):
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out_dir', default='./data')
    parser.add_argument('--samples', type=int, default=1200)
    parser.add_argument('--eval_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = build(args.samples, seed=args.seed)
    split = int(len(rows) * (1.0 - args.eval_ratio))

    write_jsonl(out_dir / 'dataset_train.jsonl', rows[:split])
    write_jsonl(out_dir / 'dataset_eval.jsonl', rows[split:])

    print(f"Wrote {split} train and {len(rows) - split} eval rows to {out_dir}")


if __name__ == '__main__':
    main()
