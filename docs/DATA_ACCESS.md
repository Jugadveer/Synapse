# Getting data that can support a real model

The audio indicator is limited by the corpus, not the classifier. Everything
reasonable on the model side has been tried and measured (see the README); the
ceiling is a speaker-level ROC-AUC around 0.77. Published work reaches 90%+ on
clinically collected corpora.

## Why this corpus caps out

`resources/` holds 352 interview clips from 178 people, scraped from public
recordings, labelled by whether the person was later reported to have dementia.
Three problems follow from that, none of which a better model fixes:

1. **The label is not a diagnosis at the time of recording.** A clip may predate
   any symptoms by years. Some of the "dementia" recordings almost certainly
   show no impairment at all.
2. **There is no common task.** People are talking about different things in
   different settings. That is why transcript-derived linguistic features score
   at chance here while they beat acoustic features on standardised corpora —
   with no shared elicitation task, lexical statistics measure subject matter.
3. **It is small.** 178 speakers, and 61 of them contribute a single clip.

Checked and ruled out: recording conditions are *not* a confound. Spectral
bandwidth, rolloff, noise floor and duration predict the label at AUC 0.50,
and the class means are within 0.2% of each other.

## Two datasets worth applying for

Both require a request from you personally, with an institutional affiliation.
Neither can be downloaded by a third party, and neither charges a fee.

### PROCESS-2 — the easier one, start here

<https://huggingface.co/datasets/CognoSpeak/PROCESS-2>

400 participants: 200 healthy controls, 150 mild cognitive impairment, 50
dementia. Roughly 21 hours of audio with manually verified transcripts and
predefined train/test partitions. Three tasks per participant: semantic
fluency, phonemic fluency, and Cookie Theft picture description.

Access is gated on Hugging Face — you click *Request access* on the dataset
page, describe your affiliation and intended use, and agree to the data use
agreement. Redistribution and commercial use are prohibited.

This is the better first target: the request is lighter than a full DUA, and
the standardised tasks are exactly what makes linguistic features work.

### DementiaBank Pitt Corpus — the benchmark

<https://talkbank.org/dementia/access/English/Pitt.html>

The corpus behind most published numbers, and the source of the ADReSS
challenge subsets. Access goes to academic researchers with a verified
university or research-lab position, by email agreeing to the TalkBank Ground
Rules. No fee. No further IRB approval is needed — review already happened at
the contributing institution.

If you use it, citation is mandatory: at least one corpus reference plus
acknowledgement of NIA grants AG03705 and AG05133.

## Draft request

Fill in the bracketed parts and send from your university address. The
institutional email matters more than the wording.

> Subject: Request for access to the DementiaBank Pitt Corpus
>
> Dear TalkBank team,
>
> I am [name], [role — e.g. an undergraduate student] at [institution],
> working under the supervision of [supervisor, department].
>
> I would like to request access to the DementiaBank Pitt Corpus for a project
> on detecting cognitive decline from spontaneous speech. The work is
> non-commercial and for research and coursework only. I intend to train and
> evaluate speech-based classifiers using speaker-disjoint splits, and to
> report aggregate performance metrics only — no audio, transcripts or
> participant-level data would be redistributed or published.
>
> I have read the TalkBank Ground Rules and agree to abide by them, including
> restricting access to myself and named collaborators, using the data solely
> for bona fide research, and citing the corpus along with the supporting NIA
> grants AG03705 and AG05133 in any resulting work.
>
> [If applicable: My supervisor, copied here, can confirm my affiliation.]
>
> Thank you for considering this request.
>
> [name, institutional email, department, institution]

## What to do once you have it

The pipeline in `synapse/app/data/` transfers directly. Point
`prepare_data.py` at the new corpus and keep the speaker grouping — splits that
share speakers are what made the original evaluation meaningless.

Expect the biggest gain from the thing that failed here: with a standardised
elicitation task, transcript features become informative, and the pipeline
already runs Whisper. Fuse those with the wav2vec2 embeddings that
`extract_features.py` produces.

Re-run `evaluate_audio_model.py` on the new holdout before changing any of the
wording in the interface. The numbers shown to people come from the model card,
so they update themselves — but the framing in the README does not.
