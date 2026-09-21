# Intent router fine-tune (Qwen 2.5 3B + LoRA via OUMI)

The router decides what each spoken turn is asking for. It is called twice per
turn at most: once to classify, and once to analyse memory intent.

## Why the dataset is generated from the runtime prompts

Both prompt templates live in [`pipeline/prompts.py`](../pipeline/prompts.py)
and are imported by `build_intent_dataset.py`. Training rows therefore contain
the exact text the router sends at serve time.

This was previously not the case. The model was trained on

```json
{"intent": "memory_store", "is_fast": false, "needs_memory": true}
```

under the system prompt *"You are an intent classifier"*, while the router
asked for an eight-field schema with different instructions. `confidence` was
never in the training data, so the model never returned it, so the pipeline
filled in a default of `0.7` — which sat below its own `0.8` clarification
threshold. Every reasoning turn was diverted into a clarifying question and the
Mistral layer was unreachable in practice.

`tests/test_training_dataset.py` fails if the two schemas drift apart again.

## Output schemas

Classification (8 fields):

```json
{"intent": "command|memory_store|memory_retrieve|unclear|casual|question",
 "is_fast": true, "needs_memory": false, "needs_reasoning": false,
 "fast_response": "", "memory_query": "", "memory_content": "", "confidence": 0.0}
```

Memory analysis (14 fields) adds `needs_memory_storage`,
`needs_memory_retrieval`, `needs_clarification`, `information_completeness`
(`is_complete`, `missing_fields`, `should_ask`), `memory_entity`,
`memory_entity_type`, `memory_value` and `clarification_question`.

## What the data teaches

Roughly half the rows are memory-analysis turns, and a large share of those are
deliberately **incomplete** — "I put my keys somewhere safe", "I read up to page
78" — labelled `should_ask: true` with one short clarifying question.

That bias is the point. The dementia-dialogue literature finds that assistants
fail these users by acting on a guess and by interrupting too early; the
recommended pattern is *incremental clarification* — one short question, then
proceed. Asking is trained as the preferred behaviour whenever a field is
missing or ambiguous.

## Build the dataset

```bash
python build_intent_dataset.py --out_dir ./data --samples 1200
```

Generation is seeded, so the same seed gives the same dataset.

## Train

```bash
oumi train ./oumi_lora_qwen25_3b.yaml
```

`r=16`, `alpha=32`, `learning_rate=2e-4`. The rate was `2e-5` — a full
fine-tune value that barely moves a LoRA adapter of this size.

## Merge and serve

```bash
oumi merge-lora --base Qwen/Qwen2.5-3B-Instruct \
  --lora ./artifacts/qwen25-3b-intent-lora \
  --output ./artifacts/qwen25-3b-intent-merged
```

Then register the merged model with Ollama and point `OLLAMA_QWEN_MODEL` at it.

PowerShell helpers: `./train.ps1`, `./merge_lora.ps1`.

## Improving it

The generated set is templated and gets the schema right, not the long tail of
real speech. The highest-value next step is replacing generated rows with real
transcripts from `ConversationTurn`, which now records every exchange
(`user_text` and the routed `qwen_intent`).
