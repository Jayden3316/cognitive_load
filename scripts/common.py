"""Shared token rules for the CrisperWhisper -> MFA -> merge steps.

Both `03_prepare_mfa_corpus.py` (writes MFA input text) and
`05_merge_alignments.py` (maps MFA word timings back onto CrisperWhisper
tokens) must agree on how a CrisperWhisper token becomes MFA words, so that
logic lives here and nowhere else.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

try:
    from num2words import num2words
except ImportError:  # optional; digits are then left for MFA's G2P
    num2words = None

# Filled pauses. CrisperWhisper 2.0 emits these as bracketed tags ([UH], [UM]),
# but bare spellings are accepted too in case a model/version writes them plainly.
FILLERS = {
    "uh", "um", "uhm", "umm", "uhh", "er", "erm", "ah", "eh",
    "hmm", "hm", "mm", "mhm", "mmm",
}
# What MFA should "hear" for each filler (spellings present in the MFA English dictionaries).
FILLER_TO_MFA = {"uhm": "um", "umm": "um", "uhh": "uh", "erm": "um", "hm": "hmm", "mmm": "mm"}

_BRACKETED = re.compile(r"^[\[<{(](.+?)[\]>})]$")
_EDGE_PUNCT = re.compile(r"^[^\w'\-‐‑–—]+|[^\w'\-‐‑–—]+$")
_KEY_STRIP = re.compile(r"[^a-z0-9']")
_PARTIAL = re.compile(r"\w[-‐‑–—]$")  # "th-", "wh—"
_DIGITS = re.compile(r"\d+(?:\.\d+)?")


@dataclass
class Token:
    """One CrisperWhisper word plus the flags the analysis cares about."""

    idx: int
    word: str        # as emitted by CrisperWhisper
    start: float
    end: float
    kind: str        # "word" | "filler" | "event" | "partial"
    mfa_words: list[str]  # what this token contributes to MFA's transcript (may be empty)


def classify(raw: str) -> tuple[str, str]:
    """Return (kind, core text) for a raw CrisperWhisper token."""
    text = raw.strip()
    m = _BRACKETED.match(text)
    if m:
        inner = m.group(1).strip().lower()
        return ("filler", inner) if inner in FILLERS else ("event", inner)
    core = _EDGE_PUNCT.sub("", text)
    if not any(c.isalnum() for c in core):
        return "word", ""  # punctuation-only token, dropped by the caller
    if core.lower() in FILLERS:
        return "filler", core.lower()
    if _PARTIAL.search(core):
        return "partial", core
    return "word", core


def to_mfa_words(kind: str, core: str) -> list[str]:
    """Words MFA should align for this token.

    * fillers  -> plain spelling ("um") so they get real phones, not `spn`
    * events   -> kept bracketed ("[laughter]"); MFA maps bracketed words to the
                  spoken-noise phone, so the noise still occupies time
    * partials -> trailing hyphen dropped ("th"); MFA's G2P guesses a pronunciation
    * digits   -> spelled out when num2words is available
    """
    if kind == "filler":
        return [FILLER_TO_MFA.get(core, core)]
    if kind == "event":
        return [f"[{core.replace(' ', '_')}]"]
    if kind == "partial":
        core = core.rstrip("-‐‑–—")
    out: list[str] = []
    for piece in core.split():
        if num2words is not None and _DIGITS.fullmatch(piece):
            out.extend(num2words(float(piece) if "." in piece else int(piece)).replace("-", " ").split())
        elif piece:
            out.append(piece)
    return out


def match_key(word: str) -> str:
    """Loose comparison key between our expected MFA words and MFA's output labels."""
    return _KEY_STRIP.sub("", word.lower())


def load_tokens(cw_json: Path) -> tuple[dict, list[Token]]:
    """Read a CrisperWhisper result JSON and return (raw dict, classified tokens)."""
    data = json.loads(cw_json.read_text(encoding="utf-8"))
    tokens = []
    for i, w in enumerate(data.get("words") or []):
        kind, core = classify(w["word"])
        if not core:
            continue  # pure punctuation
        start = float(w["start"]) if w.get("start") is not None else float("nan")
        end = float(w["end"]) if w.get("end") is not None else start
        tokens.append(Token(i, w["word"].strip(), start, end, kind, to_mfa_words(kind, core)))
    return data, tokens


def read_manifest(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return {row["id"]: row for row in csv.DictReader(f)}


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
