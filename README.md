# Synapse

A voice-first assistant and cognitive-health dashboard for people living with
dementia. It does three things:

- **Voice companion** — speech in, an intent router, semantic memory ("where
  did I put my keys?"), and spoken replies.
- **Reminders** — scheduled, stored, and delivered when they come due.
- **Screening** — dementia risk indicators from an audio recording or an MRI slice.

## Layout

```
manage.py                     Django entry point (the only one)
FinalEclipse/project/         The Django project
  project/settings.py           settings, reads .env
  synapse/                      auth, dashboard, scan endpoints, websocket consumer
src/synapse/                  Library imported by the project
  pipeline/                     voice workers: STT, router, clarification, reasoning, TTS
  models_wrapper/               FAISS-backed semantic memory
  voice/                        Django app: sessions, memories, reminders, turns
  training/                     LoRA fine-tune for the intent router
tests/                        pytest suite
index.html                    cognitive games, served at /game/
```

`src/synapse` is a plain library, not a second Django project. It used to
contain a full duplicate — settings, urls, asgi, wsgi and a second `manage.py`
— which was never importable because of how the path was built, and had quietly
drifted from the copy that actually ran.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```

For speech, semantic memory and the screening models, also install the
inference wheels (large):

```bash
pip install -r requirements-ml.txt
```

Without them the app still runs — reminders, conversation and the dashboard
work, and the features that need a model say so instead of failing.

Then create `.env` from the template and fill in your keys:

```bash
cp .env.example .env
python manage.py migrate
python manage.py runserver
```

## Configuration

All settings come from a single `.env` at the repo root. Nothing is read from a
second file; two were loaded before, and because `load_dotenv` does not
override an existing variable, empty placeholders in the first silently
shadowed the real keys in the second.

| Variable | Purpose |
| --- | --- |
| `SECRET_KEY` | Django secret. Required when `DEBUG=False`. |
| `DEBUG` | `True` for development. |
| `ALLOWED_HOSTS` | Comma-separated; defaults to localhost. |
| `OLLAMA_URL`, `OLLAMA_QWEN_MODEL` | Intent router. |
| `MISTRAL_API_KEY`, `MISTRAL_MODEL` | Reasoning layer. |
| `TTS_GTTS_LANG` | Speech output language. |
| `STT_MAX_PAUSE_SECONDS` | Silence before a turn is finalised (default `2.5`). |

## The voice pipeline

Each worker is an independent coroutine joined by queues:

```
audio ─► STT ─► router ─┬─► clarification ─► reasoning ─┐
                        └─────────── fast reply ────────┴─► safety ─► TTS ─► audio
```

- The **router** runs a fine-tuned Qwen 2.5 3B through Ollama. Reminders are
  handled deterministically in Python so they behave identically every time.
- The **clarification gate** asks one short question rather than acting on a
  guess — the incremental-clarification pattern from the dementia-dialogue
  literature.
- **Safety rules** run on every reply before it is spoken: one question per
  turn, short sentences.
- **Barge-in** bumps a generation counter; work from a superseded turn is
  dropped rather than spoken over the person.

`STT_MAX_PAUSE_SECONDS` defaults to 2.5 rather than 1.0 because people living
with dementia pause longer mid-sentence, and cutting them off is a documented
failure mode for voice assistants.

## Screening models

Two separate models, two separate label vocabularies:

| | Input | Labels |
| --- | --- | --- |
| Audio | 60 acoustic features → scikit-learn | `Dementia`, `No Dementia` |
| MRI | 128×128 slice → Keras CNN | `No Impairment`, `Very Mild`, `Mild`, `Moderate` |

`synapse/utils.py` maps each vocabulary to a risk level separately, and returns
`None` for a label it does not recognise rather than guessing.

> **On the audio model.** It uses mean-pooled acoustic features (MFCC, chroma,
> spectral contrast, ZCR). In the published ADReSS/DementiaBank benchmarks that
> family of approach sits around 62–63% accuracy, against roughly 77% for
> linguistic features from transcripts and 90%+ for multimodal models. Averaging
> over the whole clip also discards the temporal and linguistic signal that
> carries most of the discriminative power. Treat its output as a rough
> indicator, not a diagnosis. The pipeline already transcribes speech with
> Whisper, so adding a linguistic feature path is the obvious next step.

## Tests

```bash
python -m pytest
```

98 tests covering the memory store, reminder parsing, risk mapping, worker
resilience, the scan endpoints and train/serve schema parity.

## Fine-tuning the router

See [`src/synapse/training/README.md`](src/synapse/training/README.md). The
dataset is generated from the same prompt templates the router sends at
runtime, so the two cannot drift apart.
