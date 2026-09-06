"""Word-boundary matching shared by the label, overlay, and ranking code.

Drug names are matched as whole words so `dolo` never hits `dolonex` and
`iron` never hits `environment`.
"""
from __future__ import annotations

import re
from functools import lru_cache


@lru_cache(maxsize=2048)
def term_pattern(term: str) -> re.Pattern[str]:
    """Alphanumeric-bounded match. Safe for terms ending in punctuation or digits."""
    return re.compile(rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])", re.I)


def contains_term(text: str, term: str) -> bool:
    if not text or not term:
        return False
    return bool(term_pattern(term).search(text.lower()))


def contains_any_term(text: str, terms) -> bool:
    return any(contains_term(text, t) for t in terms)


def has_word(text: str, words) -> bool:
    """`\\b`-bounded match, for the curated token lists in overlay/alternatives."""
    low = (text or "").lower()
    return any(re.search(rf"\b{re.escape(w.lower())}\b", low) for w in words)
