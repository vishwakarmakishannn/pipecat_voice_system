"""Deterministic spoken-digit normalization with token alignment.

This is data normalization, not intent routing.  The grammar is deliberately
small, versioned, and reusable by any typed field contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


NORMALIZATION_VERSION = "spoken-digits-en-hi-v1"

# Locale data belongs here rather than in application routing code.  Ambiguous
# homophones such as "to" and "for" are intentionally excluded.
_DIGITS = {
    "zero": "0",
    "oh": "0",
    "o": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "shunya": "0",
    "ek": "1",
    "do": "2",
    "teen": "3",
    "char": "4",
    "chaar": "4",
    "paanch": "5",
    "panch": "5",
    "chhe": "6",
    "che": "6",
    "saat": "7",
    "aath": "8",
    "nau": "9",
}
_REPETITIONS = {
    "double": 2,
    "triple": 3,
    "dabal": 2,
    "tripal": 3,
}
_TOKEN = re.compile(r"[A-Za-z]+|\d+")


@dataclass(frozen=True)
class DigitSequence:
    value: str
    start: int
    end: int
    source_tokens: tuple[str, ...]


def extract_digit_sequences(text: str) -> list[DigitSequence]:
    """Extract unambiguous contiguous digit grammar spans in source order."""
    matches = list(_TOKEN.finditer(text or ""))
    output: list[DigitSequence] = []
    digits: list[str] = []
    source_tokens: list[str] = []
    start: int | None = None
    end: int | None = None
    repeat = 1

    def flush() -> None:
        nonlocal digits, source_tokens, start, end, repeat
        if digits and start is not None and end is not None:
            output.append(
                DigitSequence("".join(digits), start, end, tuple(source_tokens))
            )
        digits = []
        source_tokens = []
        start = None
        end = None
        repeat = 1

    for match in matches:
        token = match.group(0).casefold()
        if token in _REPETITIONS:
            if repeat != 1:
                flush()
            repeat = _REPETITIONS[token]
            if start is None:
                start = match.start()
            source_tokens.append(token)
            end = match.end()
            continue
        token_digits = token if token.isdigit() else _DIGITS.get(token)
        if token_digits is None:
            flush()
            continue
        if start is None:
            start = match.start()
        source_tokens.append(token)
        digits.append(token_digits * repeat)
        repeat = 1
        end = match.end()
    # A dangling repetition word is not evidence for extra digits.
    flush()
    return output


def unique_sequence_for_length(text: str, digit_count: int) -> DigitSequence | None:
    candidates = [
        sequence
        for sequence in extract_digit_sequences(text)
        if len(sequence.value) == digit_count
    ]
    distinct = {candidate.value: candidate for candidate in candidates}
    return next(iter(distinct.values())) if len(distinct) == 1 else None

