"""Cognitive-safety rules, in detail.

Every reply passes through these on its way to being spoken, so they see
whatever the reasoning model produces: no punctuation, walls of text, several
questions at once, other alphabets. Getting them wrong degrades every turn.
"""

import pytest

from pipeline.safety import (
    MAX_SPOKEN_CHARS, apply_safety_rules, keep_single_question, shorten,
    soften_not_found,
)


# --------------------------------------------------------- one question

def test_a_statement_followed_by_one_question_is_kept():
    text = 'Your keys are on the table. Would you like me to remind you later?'
    assert keep_single_question(text) == text


def test_everything_after_the_first_question_is_dropped():
    text = 'Where did you leave it? Was it the kitchen? Or upstairs?'
    assert keep_single_question(text) == 'Where did you leave it?'


def test_a_statement_after_a_question_is_also_dropped():
    """Anything past the question is another thing to hold in mind."""
    text = 'Which room was it? I can note it down for you.'
    assert keep_single_question(text) == 'Which room was it?'


def test_text_with_no_question_is_untouched():
    text = 'I have made a note of that. It is saved now.'
    assert keep_single_question(text) == text


def test_a_question_mark_inside_a_sentence():
    text = 'You asked "where is it?" and I looked.'
    assert keep_single_question(text)


# ---------------------------------------------------------- shortening

def test_short_text_is_returned_unchanged():
    text = 'Your keys are on the kitchen table.'
    assert shorten(text) == text


def test_long_text_is_cut_at_a_sentence_boundary():
    text = ' '.join(f'Sentence number {i} here.' for i in range(40))
    result = shorten(text, limit=120)

    assert len(result) <= 120
    assert result.endswith('.')


def test_one_very_long_sentence_is_still_bounded():
    """Nothing to cut on, so it is cut anyway rather than spoken whole."""
    text = 'word ' * 200
    result = shorten(text, limit=100)
    assert len(result) <= 100


def test_shortening_keeps_at_least_one_sentence():
    text = 'This first sentence is already longer than the limit allows. And more.'
    assert shorten(text, limit=20)


@pytest.mark.parametrize('text', ['', '   ', '.', '...', '!!!', '?'])
def test_shortening_degenerate_input(text):
    assert isinstance(shorten(text), str)


# ------------------------------------------------------------ softening

def test_a_bald_failure_is_rephrased():
    result = soften_not_found('I could not find that in memory yet.')
    assert 'could not find' not in result.lower()


def test_the_contracted_form_is_caught_too():
    result = soften_not_found("I couldn't find that anywhere.")
    assert "couldn't find" not in result.lower()


def test_other_text_is_left_alone():
    text = 'Your glasses are in the drawer.'
    assert soften_not_found(text) == text


# ----------------------------------------------------------- end to end

def test_every_rule_applies_together():
    text = (
        'I could not find that in memory yet. Where did you leave it? '
        'Was it the kitchen? ' + 'Extra padding. ' * 40
    )
    result = apply_safety_rules(text)

    assert len(result) <= MAX_SPOKEN_CHARS
    assert result.count('?') <= 1
    assert 'could not find' not in result.lower()


def test_a_reply_with_no_punctuation_survives():
    text = 'your keys are on the kitchen table where you left them this morning'
    result = apply_safety_rules(text)
    assert result.strip()
    assert len(result) <= MAX_SPOKEN_CHARS


@pytest.mark.parametrize('text', [
    '',
    '   ',
    '\n\n',
    '?',
    '...',
    'A',
    'x' * 5000,
])
def test_degenerate_replies_do_not_raise(text):
    result = apply_safety_rules(text)
    assert isinstance(result, str)
    assert len(result) <= MAX_SPOKEN_CHARS


def test_none_passes_through():
    assert apply_safety_rules(None) is None


def test_other_alphabets_are_not_mangled():
    """Hindi is a supported input language, so replies come back in it."""
    text = 'आपकी चाबी मेज़ पर है।'
    result = apply_safety_rules(text)
    assert 'चाबी' in result


def test_rules_are_idempotent():
    """Applying them twice must not keep eroding the reply."""
    text = 'I could not find that. Where did you leave it? Was it upstairs?'
    once = apply_safety_rules(text)
    assert apply_safety_rules(once) == once


def test_a_reply_at_exactly_the_limit_is_untouched():
    text = 'a' * MAX_SPOKEN_CHARS
    assert apply_safety_rules(text) == text
