# Synapse

**A voice-first companion for people living with dementia.** Talk to it, and it
remembers things for you — where you put your keys, when to take your tablets,
what your daughter is called — and tells you back when you ask.

Built around one idea: an assistant for someone with memory difficulty has to
be *predictable*. So the things that matter most — reminders, remembering,
knowing when to ask instead of guess — are settled by rules that behave the
same way every time, and the language model handles the rest.

```
you : I left my keys on the kitchen table
bot : I'll remember that you left your keys on the kitchen table.

you : where did I leave my keys
bot : You left your keys on the kitchen table.

you : remind me to take my tablets in ten minutes
bot : I'll remind you to take your tablets at 7:46 PM.
```

---

## What it does

| | |
| --- | --- |
| 🎙️ **Voice companion** | Speech in, spoken replies. Semantic memory you can ask questions of. |
| ⏰ **Reminders** | Parsed from ordinary speech, stored, and delivered when due — even if you closed the app. |
| 🧩 **Cognitive games** | Four memory and attention exercises, no backend needed. |
| 🩺 **Screening** | Indicators from a voice recording or an MRI slice. Read [the limits](#the-audio-indicator) before trusting either. |

## How it works

Each stage is an independent coroutine joined by queues, so a slow reasoning
call never blocks transcription of the next thing you say.

```
  microphone
      │
      ▼
  ┌─────────┐   ┌────────┐   ┌───────────────┐   ┌───────────┐
  │   STT   │──►│ router │──►│ clarification │──►│ reasoning │
  │ Whisper │   │ Qwen   │   │     gate      │   │  Mistral  │
  └─────────┘   └────┬───┘   └───────┬───────┘   └─────┬─────┘
                     │               │                 │
                     │    ┌──────────┴─────────────────┘
                     │    │
                     ▼    ▼
                ┌──────────────┐   ┌────────┐   ┌─────┐
                │ safety rules │──►│  TTS   │──►│ you │
                └──────────────┘   └────────┘   └─────┘
                     ▲
        ┌────────────┴────────────┐
        │  memory (FAISS)         │
        │  reminders (scheduler)  │
        └─────────────────────────┘
```

**Reminders and clear memory statements never reach the model.** A 1.5B router
classifies *"I left my keys on the kitchen table"* as a **retrieval** — so
nothing gets stored and you are told "Okay." Those cases are decided in Python;
ambiguous ones go to the model.

**The clarification gate asks one short question rather than acting on a
guess.** That is the *incremental clarification* pattern from the
[dementia-dialogue literature](https://www.frontiersin.org/journals/dementia/articles/10.3389/frdem.2024.1343052/full),
which finds that assistants fail these users by guessing and by interrupting.

**Every reply passes the safety rules** before it is spoken: one question per
turn, short sentences, no bald failure messages.

**Barge-in works.** Interrupting bumps a generation counter and work from the
superseded turn is dropped rather than spoken over you.

### Choices made for this audience

- **`STT_MAX_PAUSE_SECONDS` defaults to 2.5, not 1.0.** People with dementia
  pause longer mid-sentence, and cutting them off is a documented failure mode.
- **Anything quoted back is moved into the second person.** Echoed verbatim,
  "remind me to take my tablets" becomes the assistant talking about its own
  tablets.
- **Confirmations are built, not generated.** Asked for the wording, a 1.5B
  model produced *"Thank you for remembering to save that"* — the assistant
  thanking you for doing its job.
- **A memory with a hole in it is asked about.** *"I put my glasses
  somewhere"* is not stored; it cannot answer the question it exists to answer.

---

## Quickstart

```bash
git clone https://github.com/Jugadveer/Synapse.git
cd Synapse
python -m venv .venv && .venv/Scripts/activate      # source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                                 # then fill in SECRET_KEY
python manage.py migrate
python manage.py runserver
```

That gets you the dashboard, reminders, conversation and the games.

For speech, semantic memory and the screening models, add the inference wheels
(several GB) and a small local model for the router:

```bash
pip install -r requirements-ml.txt
ollama pull qwen2.5:1.5b-instruct
```

Everything degrades cleanly: without the wheels the app still runs, and
anything needing a model says so rather than failing.

### Configuration

One `.env` at the repo root; see [`.env.example`](.env.example).

| Variable | Notes |
| --- | --- |
| `SECRET_KEY` | 50+ characters. The app refuses to start in production without one. |
| `DEBUG` | `False` in production — that switch enables HTTPS redirect, HSTS, secure cookies and `X-Frame-Options`. |
| `ALLOWED_HOSTS` | Comma-separated. Also derives `CSRF_TRUSTED_ORIGINS`. |
| `TIME_ZONE` | **Set this.** Reminders are spoken in local terms, so `Asia/Kolkata` and `UTC` are not interchangeable. |
| `OLLAMA_URL`, `OLLAMA_QWEN_MODEL` | Intent router. `qwen2.5:1.5b-instruct` needs ~1.2 GB and runs on a 4 GB GPU. |
| `MISTRAL_API_KEY` | Reasoning layer, for open conversation. |
| `STT_MAX_PAUSE_SECONDS` | Silence before a turn is finalised. Default `2.5`. |

> `qwen2.5:0.5b-instruct` was measured and rejected. It answers `memory_store`
> to everything — including "Hello there" — at confidence 1.0, scoring 2/8 on
> intent classification. 1.5B is the smallest that actually classifies.

---

## The models

### Intent router

Qwen 2.5 1.5B, fine-tuned with QLoRA on the dataset in
[`src/synapse/training/`](src/synapse/training/). The dataset is generated from
the *same prompt templates the router sends at runtime*, so train and serve
cannot drift apart — a test fails if they do.

| | base | fine-tuned |
| --- | --- | --- |
| intent, spoken cases | 4/8 | **8/8** |
| intent, held-out set | 28/40 | **40/40** |
| `confidence` returned | 5/8 | **8/8** |
| exact schema | 5/8 | **8/8** |

The `confidence` row is the important one: the base model omits the field on
three turns in eight, and the clarification gate cannot work without it.

> **Not yet wired in.** Ollama 0.34 dropped LoRA support, so
> [`merge_router.py`](src/synapse/training/merge_router.py) merges the weights
> and [`serve_router.py`](src/synapse/training/serve_router.py) serves them over
> the same `/api/generate` contract. It answers 8/8 correctly but takes ~26s a
> call against Ollama's 0.5–1.8s, so the default stays on the base model.
> Unresolved.

### The audio indicator

It reports **three** outcomes — no indication, inconclusive, some indication —
because separation is not good enough for a binary verdict to be honest.
Forced to choose, the model either misses cases or flags most healthy people.

Measured on **held-out speakers**: 36 people never used for training, model
selection, or choosing the cuts.

| band | share of recordings | dementia rate within |
| --- | --- | --- |
| no indication | 15.5% | 9.1% |
| **inconclusive** | **66.2%** | 36.2% |
| some indication | 18.3% | 69.2% |

Base rate is 38%. It commits on **34%** of recordings and is **79.2% correct**
when it does; forced to answer every time it was 59% correct. Speaker-level
ROC-AUC is 0.766 ± 0.015.

Two thirds of the time the honest answer is "I cannot tell", and it says so.

> **This is not a diagnostic tool.** Even "some indication" only doubles the
> prior. Treat it as a prompt to talk to a doctor, nothing more.

```bash
python FinalEclipse/project/synapse/app/data/evaluate_audio_model.py
```

<details>
<summary><b>How it got here, and what was tried</b></summary>

The original model scored **7.7% sensitivity** — it answered "No Dementia" to
almost everyone, telling 24 of 26 people who had dementia they were clear.
Three things were wrong:

1. **The split leaked.** `train_test_split` stratified on the label alone put
   68% of validation speakers into training too, so the evaluation partly
   measured whether the model recognised a familiar voice.
2. **The threshold was never chosen.** Left at 0.5 with unbalanced classes,
   the model collapsed onto the majority answer.
3. **The features were a generation behind.**

Averaged over 5 seeds × 5 folds, because a single split moves by several points:

| approach | speaker ROC-AUC |
| --- | --- |
| MFCC mean-pooled (original) | 0.655 |
| + deltas, std, pause structure | no change |
| Whisper transcripts → linguistic features | **chance (0.49)** |
| WavLM-base-plus + MFCC | 0.738 |
| wav2vec2-**large** + MFCC | 0.749–0.753 |
| **wav2vec2-base layer 7 + MFCC** | **0.766** |

Four negative results worth keeping:

- **Richer hand-crafted features did nothing.** Mean-pooling was never the
  bottleneck.
- **A bigger pretrained model did not help.** wav2vec2-large scores *below*
  base at every layer, for 2.5× the cost.
- **Linguistic features scored at chance** — the opposite of the published
  result. ADReSS uses the Cookie Theft picture description, so every
  participant says something comparable; this corpus is scraped interviews on
  unrelated topics, where transcript statistics measure subject matter.
- **Recording conditions are not a confound.** Bandwidth, rolloff, noise floor
  and duration predict the label at AUC 0.50, class means within 0.2%.

One subtle failure, recorded because it cost an afternoon: the threshold was
first chosen from raw out-of-fold scores while the shipped model is
isotonic-calibrated. Those are different distributions, and held-out
specificity collapsed from 61% to 29%. Thresholds are now chosen by scoring
out-of-fold *through the calibrated estimator*.

</details>

### The MRI classifier

A four-class Keras CNN over 128×128 slices. **It has never been evaluated** —
there is no held-out split in the repository — and the interface says so.

---

## Why this corpus caps out

The audio model is limited by its data, not its architecture. 352 scraped
interview clips from 178 people, labelled by whether someone was *later
reported* to have dementia — the clip may predate any symptoms by years, and
there is no common elicitation task.

Published 90%+ figures come from clinically collected corpora.
[`docs/DATA_ACCESS.md`](docs/DATA_ACCESS.md) has the two worth applying for and
a draft request to send. Every model-side avenue in the table above has been
tried and measured.

---

## Tests

```bash
python -m pytest
```

313 tests covering the memory store, reminder parsing, timezone handling, the
websocket protocol, worker resilience, risk mapping, the scan endpoints, auth
flows, train/serve schema parity, the real inference paths, and the voice agent
end to end against a live model.

The suite substitutes the multi-gigabyte inference wheels with deterministic
stand-ins when they are absent, and uses the real ones when they are present —
so it runs anywhere, and means something where it matters.

## Deployment

Two things with different hosting needs. See
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

| | Where |
| --- | --- |
| `index.html` — the games | **Vercel** (`vercel.json` is set up) |
| The assistant | Anything running a persistent process: Render, Railway, Fly.io. `Dockerfile` included. |

The assistant **cannot** run on serverless: it holds websockets open, runs a
background scheduler that must outlive any request, and keeps state on disk.
`manage.py check --deploy` passes clean.

## Known limitations

- The audio indicator declines to answer two thirds of the time. That is the
  honest behaviour, not a bug.
- The MRI model is unvalidated.
- The fine-tuned router is measured but not deployed (latency, above).
- SQLite by default. Move off it before more than a handful of people use it.
- The FAISS store is a local directory; put it on a persistent volume.

## Layout

```
manage.py                  Django entry point
FinalEclipse/project/      The Django project — auth, dashboard, scans, websocket
src/synapse/
  pipeline/                Voice workers: STT, router, clarification, reasoning, TTS,
                           safety rules, phrasing, reminder parser and scheduler
  models_wrapper/          FAISS-backed semantic memory
  voice/                   Django app: sessions, memories, reminders, turns
  training/                Router fine-tune: dataset, QLoRA, merge, serve, evaluate
tests/                     pytest suite
docs/                      Deployment and data access
index.html                 Cognitive games, served at /game/
```

## Acknowledgements

Design informed by work on
[spoken dialogue for cognitive assistants](https://journals.sagepub.com/doi/full/10.1177/1460458215593329)
and the [ADReSS challenge](https://arxiv.org/pdf/2004.06833) on dementia
recognition from spontaneous speech.

Intended as a research and coursework project. Not a medical device, and not a
substitute for clinical assessment.
