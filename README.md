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
| Audio | wav2vec2 embeddings + acoustic summary → calibrated logistic regression | `Dementia`, `Inconclusive`, `No Dementia` |
| MRI | 128×128 slice → Keras CNN | `No Impairment`, `Very Mild`, `Mild`, `Moderate` |

`synapse/utils.py` maps each vocabulary to a risk level separately, and returns
`None` for a label it does not recognise rather than guessing.

> ### What the audio indicator is worth
>
> It reports **three** outcomes, not two: no indication, inconclusive, or some
> indication. Separation is not good enough for a binary verdict to be honest —
> forced to choose, the model either misses cases or flags most healthy people.
>
> On **held-out speakers** — 36 people never used for training, model
> selection, or choosing the cuts:
>
> | band | share of recordings | dementia rate within |
> | --- | --- | --- |
> | no indication | 15.5% | 9.1% |
> | **inconclusive** | **66.2%** | 36.2% |
> | some indication | 18.3% | 69.2% |
>
> Base rate is 38%. It commits on **34%** of recordings and is **79.2%
> correct** when it does. Forced to answer every time it was 59% correct.
> Speaker-level ROC-AUC is 0.766 ± 0.015.
>
> Two thirds of the time the honest answer is "I cannot tell", and it now says
> that instead of guessing.
>
> **It is not a diagnostic tool.** Even "some indication" only doubles the
> prior. Treat it as a prompt to talk to a doctor, nothing more.
>
> ```bash
> python FinalEclipse/project/synapse/app/data/evaluate_audio_model.py
> ```
>
> #### How it got here
>
> The original model scored **7.7% sensitivity** — it answered "No Dementia"
> to almost everyone, telling 24 of 26 people who had dementia that they were
> clear. Three things were wrong:
>
> 1. **The split leaked.** `train_test_split` stratified on the label alone
>    put 68% of validation speakers into training too, so the evaluation
>    partly measured whether the model recognised a familiar voice.
>    `prepare_data.py` now splits by speaker, and asserts no overlap.
> 2. **The threshold was never chosen.** Left at 0.5, with unbalanced classes,
>    the model collapsed onto the majority answer. It is now selected for a
>    sensitivity target and recorded in the model card.
> 3. **The features were the previous generation.** Replaced (see below).
>
> #### What was tried
>
> Averaged over 5 seeds × 5 folds, because a single split moves by several
> points — an earlier run had me convinced layer 8 beat layer 7 on what turned
> out to be fold noise.
>
> | approach | speaker ROC-AUC |
> | --- | --- |
> | MFCC mean-pooled (original) | 0.655 |
> | + deltas, std, pause structure | no change |
> | Whisper transcripts → linguistic features | **chance (0.49)** |
> | WavLM-base-plus + MFCC | 0.738 |
> | wav2vec2-**large** + MFCC (4 layers tried) | 0.749–0.753 |
> | multi-layer combinations | 0.731–0.752 |
> | embeddings alone, no MFCC | 0.734 |
> | **wav2vec2-base layer 7 + MFCC (shipped)** | **0.766 ± 0.015** |
>
> Four negative results worth keeping, so nobody repeats them:
>
> **Richer hand-crafted features did nothing.** Adding deltas, per-coefficient
> standard deviations and pause statistics moved AUC by less than noise. The
> problem was never that mean-pooling threw away the signal.
>
> **A bigger pretrained model did not help.** wav2vec2-large scores *below*
> base at every layer tried, for 2.5× the inference cost.
>
> **Linguistic features scored at chance** — every individual feature within
> 0.08 of 0.5. This is the opposite of the published result, where transcript
> features beat acoustic ones comfortably, and the reason is the corpus.
> ADReSS uses the Cookie Theft picture description, so every participant says
> something comparable and lexical diversity means something. These are
> scraped interviews on unrelated topics, where transcript statistics measure
> subject matter and interview style instead.
>
> **Recording conditions are not a confound.** Worth stating because it was
> the obvious worry: spectral bandwidth, rolloff, noise floor and clip
> duration predict the label at AUC 0.50, and the class means differ by under
> 0.2%. The model is not keying on microphone era.
>
> **Window averaging was tried and dropped.** Averaging four windows of one
> recording scored no better on held-out speakers (auc 0.736 against 0.732)
> for four times the inference cost — the per-speaker gain came from averaging
> across *different recordings* of a person, which one upload does not give.
>
> One subtle failure is worth recording: the threshold was first chosen from
> raw out-of-fold scores while the shipped model is isotonic-calibrated. Those
> are different distributions, and held-out specificity collapsed from 61% to
> 29%. `train_model.py` now scores out-of-fold *through the calibrated
> estimator*, so the threshold lives on the scale the deployed model produces.
>
> #### Why this is not at published state-of-the-art
>
> Reported 90%+ figures come from ADReSS/DementiaBank: balanced, clinically
> collected, one standardised elicitation task, verified diagnoses. This
> corpus is 352 interview clips from 178 people, scraped, with labels inferred
> from whether someone was later reported to have dementia — the clip may well
> predate any symptoms, and recording conditions differ systematically between
> the two classes. Closing the remaining gap needs that data, not a better
> classifier. Applying for DementiaBank access is the highest-value next step.
>
> The MRI model has no held-out split in the repository (`dataset/` is ignored
> and absent), so it has **not** been evaluated and the interface says so.

## Tests

```bash
python -m pytest
```

131 tests covering the memory store, reminder parsing, risk mapping, worker
resilience, the scan endpoints, train/serve schema parity, the real inference
paths and the voice agent end to end.

## Fine-tuning the router

See [`src/synapse/training/README.md`](src/synapse/training/README.md). The
dataset is generated from the same prompt templates the router sends at
runtime, so the two cannot drift apart.
