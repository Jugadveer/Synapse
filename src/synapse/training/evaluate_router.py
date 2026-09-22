"""Compare the fine-tuned router against the base model.

Loss says the model fits the training distribution. What matters is whether it
returns the right intent, and whether it returns every field the pipeline
reads - the base model omits `confidence` often enough that the clarification
gate cannot use it.

Run from this directory:

    python evaluate_router.py
"""

import argparse
import json
import os
import re
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TRANSFORMERS_NO_TF', '1')

import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
BASE = 'Qwen/Qwen2.5-1.5B-Instruct'
ADAPTER = HERE / 'artifacts' / 'router-lora'

#: Turns a person would actually say, with the intent the pipeline needs.
PRACTICAL = [
    ('Hello there', 'casual'),
    ('Good morning, how are you?', 'casual'),
    ('Where did I put my glasses?', 'memory_retrieve'),
    ('Where are my keys?', 'memory_retrieve'),
    ('I left my keys on the table', 'memory_store'),
    ('My daughter is called Priya', 'memory_store'),
    ('Tell me about my appointment', 'question'),
    ('Um, the thing with the... you know', 'unclear'),
]

CLASSIFY_FIELDS = {
    'intent', 'is_fast', 'needs_memory', 'needs_reasoning',
    'fast_response', 'memory_query', 'memory_content', 'confidence',
}


def load(with_adapter):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map='auto' if torch.cuda.is_available() else None,
    )
    if with_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(ADAPTER))
    return tokenizer, model.eval()


@torch.inference_mode()
def ask(tokenizer, model, prompt):
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors='pt').to(model.device)
    output = model.generate(
        **inputs, max_new_tokens=220, do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    reply = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    match = re.search(r'\{.*\}', reply, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def score(tokenizer, model, label):
    from pipeline.prompts import classify_prompt

    print(f'\n=== {label} ===')

    correct = valid_json = has_confidence = exact_schema = 0
    for text, expected in PRACTICAL:
        answer = ask(tokenizer, model, classify_prompt(text))
        if answer is None:
            print(f'  MISS {text[:34]:34} -> unparseable')
            continue
        valid_json += 1
        got = answer.get('intent')
        confidence = answer.get('confidence')
        has_confidence += confidence is not None
        exact_schema += set(answer) == CLASSIFY_FIELDS
        hit = got == expected
        correct += hit
        print(f'  {"ok " if hit else "MISS"} {text[:34]:34} want {expected:16} '
              f'got {str(got):16} conf={confidence}')

    total = len(PRACTICAL)
    print(f'  intent correct   {correct}/{total}')
    print(f'  parseable json   {valid_json}/{total}')
    print(f'  confidence given {has_confidence}/{total}')
    print(f'  exact schema     {exact_schema}/{total}')
    return correct, valid_json, has_confidence, exact_schema


def score_eval_set(tokenizer, model, label, limit):
    """Held-out rows from the generated dataset."""
    path = HERE / 'data' / 'dataset_eval.jsonl'
    rows = [json.loads(line) for line in open(path, encoding='utf-8')][:limit]

    correct = parsed = 0
    for row in rows:
        prompt = row['messages'][1]['content']
        expected = json.loads(row['messages'][2]['content'])['intent']
        answer = ask(tokenizer, model, prompt)
        if answer is None:
            continue
        parsed += 1
        correct += answer.get('intent') == expected

    print(f'  held-out set     {correct}/{len(rows)} intents, {parsed}/{len(rows)} parseable')
    return correct, len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval_rows', type=int, default=40)
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(HERE.parent))
    global SYSTEM_PROMPT
    from pipeline.prompts import SYSTEM_PROMPT as prompt

    SYSTEM_PROMPT = prompt

    results = {}
    for label, with_adapter in (('base model', False), ('fine-tuned', True)):
        tokenizer, model = load(with_adapter)
        practical = score(tokenizer, model, label)
        held_out = score_eval_set(tokenizer, model, label, args.eval_rows)
        results[label] = (practical, held_out)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print('\n' + '=' * 62)
    (bc, bj, bconf, bs), (bh, bn) = results['base model']
    (fc, fj, fconf, fs), (fh, fn) = results['fine-tuned']
    print(f'{"":18}{"base":>10}{"fine-tuned":>14}')
    print(f'{"intent (spoken)":18}{bc:>7}/{len(PRACTICAL)}{fc:>11}/{len(PRACTICAL)}')
    print(f'{"intent (held-out)":18}{bh:>7}/{bn}{fh:>11}/{fn}')
    print(f'{"confidence given":18}{bconf:>7}/{len(PRACTICAL)}{fconf:>11}/{len(PRACTICAL)}')
    print(f'{"exact schema":18}{bs:>7}/{len(PRACTICAL)}{fs:>11}/{len(PRACTICAL)}')


if __name__ == '__main__':
    main()
