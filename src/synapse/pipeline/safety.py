"""Cognitive-safety rules applied to every spoken reply.

These rules existed as ClarificationWorker._wrap_with_safety but had no
callers, so none of them ever ran. They now sit on the single path every
response takes on its way to the person.

The shape of the rules follows the dementia-communication literature: short
simple sentences, and never more than one question in a turn, because a
stacked question is hard to hold in working memory.
"""

import re

MAX_SPOKEN_CHARS = 220
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')


def _sentences(text):
    return [s for s in _SENTENCE_SPLIT.split(text.strip()) if s]


def keep_single_question(text):
    """Keep at most one question, so the person has one thing to answer."""
    sentences = _sentences(text)
    questions = [i for i, s in enumerate(sentences) if s.rstrip().endswith('?')]
    if len(questions) <= 1:
        return text
    # Keep everything up to and including the first question.
    return ' '.join(sentences[:questions[0] + 1])


def shorten(text, limit=MAX_SPOKEN_CHARS):
    """Trim a long reply to whole sentences within the limit."""
    text = text.strip()
    if len(text) <= limit:
        return text

    kept, total = [], 0
    for sentence in _sentences(text):
        if kept and total + len(sentence) + 1 > limit:
            break
        kept.append(sentence)
        total += len(sentence) + 1
    return ' '.join(kept) if kept else text[:limit].rstrip()


def soften_not_found(text):
    """Avoid a bald failure message, which reads as a reproach."""
    if 'could not find' in text.lower() or 'couldn\'t find' in text.lower():
        return "I don't have that written down yet. Can you tell me again?"
    return text


def apply_safety_rules(text, decision=None, user_text=None):
    """Run every rule over a reply that is about to be spoken."""
    if not text:
        return text

    result = soften_not_found(text.strip())
    result = keep_single_question(result)
    result = shorten(result)
    return result.strip()
