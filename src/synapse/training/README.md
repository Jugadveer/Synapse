# Intent router fine-tune

The router decides what each spoken turn is asking for. It is called at most
twice per turn: once to classify, once to analyse memory intent.

## Results

Qwen 2.5 **1.5B**, QLoRA, 3 epochs. Train loss 0.052, eval loss 0.043.

| | base | fine-tuned |
| --- | --- | --- |
| intent, spoken cases | 4/8 | **8/8** |
| intent, held-out set | 28/40 | **40/40** |
| `confidence` returned | 5/8 | **8/8** |
| exact schema | 5/8 | **8/8** |

The `confidence` row matters most. The base model omits the field on three
turns in eight, and the clarification gate cannot work without it — a missing
confidence used to be filled in with a default that sat below the gate's own
threshold, so every reasoning turn was diverted into a question.

`qwen2.5:0.5b-instruct` was measured and rejected: it answers `memory_store`
to everything, including greetings, at confidence 1.0 (2/8).

## Why the dataset is generated from the runtime prompts

Both prompt templates live in [`pipeline/prompts.py`](../pipeline/prompts.py)
and are imported by `build_intent_dataset.py`, so a training row contains the
exact text the router sends at serve time.

This was not always so. The model was trained on

```json
{"intent": "memory_store", "is_fast": false, "needs_memory": true}
```

under the system prompt *"You are an intent classifier"*, while the router
asked for an eight-field schema with different instructions. `confidence` was
never in the training data, so the model never produced it.

`tests/test_training_dataset.py` fails if the two drift apart again.

## Output schemas

Classification, 8 fields:

```json
{"intent": "command|memory_store|memory_retrieve|unclear|casual|question",
 "is_fast": true, "needs_memory": false, "needs_reasoning": false,
 "fast_response": "", "memory_query": "", "memory_content": "", "confidence": 0.0}
```

Memory analysis, 14 fields, adds `needs_memory_storage`,
`needs_memory_retrieval`, `needs_clarification`, `information_completeness`
(`is_complete`, `missing_fields`, `should_ask`), `memory_entity`,
`memory_entity_type`, `memory_value` and `clarification_question`.

## What the data teaches

About half the rows are memory-analysis turns, and a large share of those are
deliberately **incomplete** — "I put my keys somewhere safe", "I read up to
page 78" — labelled `should_ask: true` with one short clarifying question.

That bias is the point. The dementia-dialogue literature finds that assistants
fail these users by acting on a guess and by interrupting too early; the
recommended pattern is *incremental clarification*. Asking is trained as the
preferred behaviour whenever a field is missing or ambiguous.

## Running it

```bash
pip install -r ../../../requirements-ml.txt -r requirements-training.txt
```

Training on a GPU needs a CUDA build of torch — the default wheel is CPU-only,
and a 1.5B LoRA would take many hours on CPU:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Then:

```bash
python build_intent_dataset.py --out_dir ./data --samples 1200
python train_router.py                 # ~45 min on an RTX 4050, ~5 GB VRAM
python evaluate_router.py              # base vs fine-tuned
```

Generation is seeded, so the same seed gives the same dataset. Loss is computed
on the assistant turn only; including the prompt would spend most of the
gradient teaching the model to reproduce a fixed instruction block it is always
given anyway.

## Serving it

Ollama dropped LoRA support ("LoRA adapters are no longer supported" as of
0.34), so the adapter has to be folded into the weights first:

```bash
python merge_router.py                 # writes artifacts/router-merged
python serve_router.py --port 11500
# then in .env:  OLLAMA_URL=http://127.0.0.1:11500
```

`serve_router.py` implements the one endpoint the router calls
(`/api/generate`), so nothing in the application changes.

### Latency

2.0s for a classification and 4.0s for the longer memory-analyst prompt, at
~21 tokens a second in fp16. The first call after startup is slower while
kernels warm up.

Three things were wrong when this first measured 26 seconds a call:

- **VRAM.** Ollama still held its own model, leaving under a gigabyte spare on
  a 6 GB card. The allocator thrashes there and throughput drops roughly
  tenfold, with wild variance — 14 to 56 seconds for the same work. `ollama
  stop` first; the server now reports free VRAM at startup and warns below
  1.2 GB.
- **A stopping criterion that was not needed.** It decoded the whole generated
  tail on every token to find the closing brace. The model emits
  end-of-sequence after 50 to 110 tokens on its own, so this only added cost.
- **`device_map`.** Passing it measured 21.7 tokens a second against 17.0 for
  loading and then calling `.to('cuda')`, for the same memory.

4-bit NF4 was tried and rejected: 14.9 tokens a second against fp16's 21.7.
It saves 2 GB but dequantisation costs more than the memory is worth at this
size and batch of one.

## Improving it

The generated set gets the schema right, not the long tail of real speech. The
highest-value next step is replacing generated rows with real transcripts from
`ConversationTurn`, which records every exchange (`user_text` and the routed
`qwen_intent`).
