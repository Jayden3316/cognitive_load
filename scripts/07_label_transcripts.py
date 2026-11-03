"""Step 7: labelled verbatim transcripts, cross-checked by two more models.

Base: CrisperWhisper's verbatim tokens with MFA timings (data/aligned/tokens/<id>.csv) and pauses
(data/aligned/pauses/<id>.csv). Two independent second opinions (step 6):
    data/second_opinion/qwen3_omni.json        Qwen3-Omni, tagged verbatim (laughter, swearing, ...)
    data/second_opinion/whisper_large_v3.json  Whisper large-v3, plain text

Labels in the transcript:
    [HESITATION: uh] / [HESITATION: um]   filled pause, the sound actually made (CrisperWhisper)
    [PAUSE 0.8s]                          silent pause of at least --min-pause seconds (MFA timings)
    [LAUGHTER]                            from CrisperWhisper or Qwen3-Omni
    [SWEAR: word]                         profanity in any language, written out
    [NON-ENGLISH: words]                  words not in English
    [COUGH] [BREATH] [SIGH] ...           other non-speech sounds heard by Qwen3-Omni
    [CUT OFF]                             the recording starts or ends in the middle of speech
    th-                                   a cut-off (partial) word

Words: CrisperWhisper's, except where Whisper large-v3 and Qwen3-Omni both heard the same different
words (2 of 3 agree); every such change is listed in review_notes. Repetitions and partial words
that only CrisperWhisper kept are left in: the other two models tidy those away.

Writes data/transcripts/labelled/<id>.txt and data/transcripts/verbatim_transcripts.csv.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf

TAG = re.compile(r"\[([A-Za-z][A-Za-z \-]*?)(?::\s*([^\]]*))?\]")
EVENT_TAGS = {"LAUGHTER", "COUGH", "BREATH", "SIGH", "CLICK", "BACKGROUND SPEECH", "SNIFF", "GASP"}


FILLERS = {"uh", "um", "er", "erm", "ah", "hmm", "mm", "eh", "uhm"}
CONTRACTIONS = [("n't", " not"), ("'re", " are"), ("'s", " is"), ("'ll", " will"), ("'ve", " have"),
                ("'m", " am"), ("'d", " would")]
BRITISH = [(r"our\b", "or"), (r"ise\b", "ize"), (r"ised\b", "ized"), (r"ising\b", "izing"), (r"yse\b", "yze"),
           (r"tre\b", "ter"), (r"ll(ed|ing)\b", r"l\1")]


def key(w: str) -> str:
    return re.sub(r"[^a-z0-9]", "", w.lower())


def cosmetic(a: str, b: str) -> bool:
    """Same words, different spelling (colour/color) or contraction (they're/they are)."""
    def norm(s: str) -> str:
        s = s.lower().replace("’", "'")
        for c, full in CONTRACTIONS:
            s = s.replace(c, full)
        s = re.sub(r"[^a-z' ]", "", s)
        for pat, rep in BRITISH:
            s = re.sub(pat, rep, s)
        return re.sub(r"[^a-z]", "", s)
    return norm(a) == norm(b)


def parse_tagged(text: str) -> list[tuple[str, str]]:
    """Qwen3-Omni output -> [(kind, value)]: ("word", w) or (TAG, value)."""
    items, pos = [], 0
    for m in TAG.finditer(text):
        items += [("word", w) for w in text[pos:m.start()].split() if key(w)]
        items.append((m.group(1).strip().upper(), (m.group(2) or "").strip()))
        pos = m.end()
    items += [("word", w) for w in text[pos:].split() if key(w)]
    return items


def speech_at_edge(wav: Path, first: float, last: float, dur: float) -> tuple[bool, bool]:
    """Does speech run into the start / end of the file? Loud audio within 0.15 s of the edge and
    a word touching it."""
    x, sr = sf.read(str(wav), dtype="float32")
    n = int(sr * 0.15)
    db = lambda a: 20 * np.log10(np.sqrt(np.mean(a ** 2)) + 1e-9)  # noqa: E731
    speech = np.percentile([db(x[i:i + n]) for i in range(0, len(x) - n, n)], 75)
    head = max(db(x[i:i + n // 3]) for i in range(0, n, n // 3))
    tail = max(db(x[len(x) - n + i:len(x) - n + i + n // 3]) for i in range(0, n - n // 3 + 1, n // 3))
    return (first < 0.3 and head > speech - 12), (dur - last < 0.3 and tail > speech - 12)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--aligned", type=Path, default=Path("data/aligned"))
    ap.add_argument("--second-opinion", type=Path, default=Path("data/second_opinion"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/transcripts/labelled"))
    ap.add_argument("--csv", type=Path, default=Path("data/transcripts/verbatim_transcripts.csv"))
    ap.add_argument("--min-pause", type=float, default=0.5, help="shortest silent pause to label (s)")
    args = ap.parse_args()

    qwen = json.loads((args.second_opinion / "qwen3_omni.json").read_text())
    whisper = json.loads((args.second_opinion / "whisper_large_v3.json").read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in csv.DictReader(args.manifest.open()):
        rid = m["id"]
        toks = list(csv.DictReader((args.aligned / "tokens" / f"{rid}.csv").open()))
        pauses = list(csv.DictReader((args.aligned / "pauses" / f"{rid}.csv").open()))
        notes: list[str] = []
        if not toks:
            text = "[NO SPEECH: background noise only]"
            (args.out_dir / f"{rid}.txt").write_text(text + "\n")
            rows.append({**{k: m[k] for k in ("id", "participant", "session", "prompt", "condition", "duration_s")},
                         "transcript": text, "n_words": 0, "n_hesitations": 0, "n_pauses": 0, "n_laughter": 0,
                         "n_swears": 0, "n_partial_words": 0, "review_notes": "recording has no speech; re-record"})
            continue

        # Base stream: one slot per CrisperWhisper token; labels go before/after slots.
        words = [t["word"] for t in toks]
        kinds = [t["kind"] for t in toks]
        before: dict[int, list[str]] = {}
        after: dict[int, list[str]] = {}

        # 2-of-3 word corrections: Whisper and Qwen agree on the same replacement / insertion.
        cw_idx = [i for i, k in enumerate(kinds) if k == "word"]
        cw_keys = [key(words[i]) for i in cw_idx]
        q_items = parse_tagged(qwen.get(rid, ""))
        q_words = [v for k, v in q_items if k == "word"]
        w_words = [w for w in whisper[rid]["text"].split() if key(w)]

        def edits(other: list[str]) -> dict[tuple[int, int], list[str]]:
            sm = difflib.SequenceMatcher(None, cw_keys, [key(w) for w in other], autojunk=False)
            return {(i1, i2): other[j1:j2] for op, i1, i2, j1, j2 in sm.get_opcodes()
                    if op in ("replace", "insert")}

        eq, ew = edits(q_words), edits(w_words)
        for span, alt_q in eq.items():
            alt = ew.get(span)  # Whisper's spelling and casing
            if alt is None or [key(w) for w in alt] != [key(w) for w in alt_q]:
                continue
            (i1, i2), cw_text = span, " ".join(words[cw_idx[i]] for i in range(*span))
            if i2 > i1 and (not alt or len(alt) < i2 - i1):
                continue  # the others dropped words CrisperWhisper heard: keep them (repetitions etc.)
            if i2 > i1 and cosmetic(cw_text, " ".join(alt)):
                continue  # spelling / contraction only: keep CrisperWhisper's verbatim form
            new = " ".join(f"[HESITATION: {key(w)}]" if key(w) in FILLERS else w for w in alt)
            if i2 > i1:
                words[cw_idx[i1]] = new
                for i in range(i1 + 1, i2):
                    words[cw_idx[i]] = ""
                notes.append(f"'{cw_text}' -> '{new}' (Whisper and Qwen agree)")
            else:
                anchor = cw_idx[i1] if i1 < len(cw_idx) else len(words) - 1
                # Fillers are not in the word sequence compared above: skip one CrisperWhisper already has.
                lo = cw_idx[i1 - 1] + 1 if i1 > 0 else 0
                hi = cw_idx[i1] if i1 < len(cw_idx) else len(words)
                if all(key(w) in FILLERS for w in alt) and any(kinds[i] == "filler" for i in range(lo, hi)):
                    continue
                (before if i1 < len(cw_idx) else after).setdefault(anchor, []).append(new)
                notes.append(f"added '{new}' (Whisper and Qwen heard it; CrisperWhisper did not)")

        # Qwen3-Omni tags, placed after the CrisperWhisper word aligned to the word before the tag.
        sm = difflib.SequenceMatcher(None, [key(w) for w in q_words], cw_keys, autojunk=False)
        q2cw = {}
        for blk in sm.get_matching_blocks():
            for n in range(blk.size):
                q2cw[blk.a + n] = cw_idx[blk.b + n]
        qi, swears, cut_tags = -1, 0, 0
        for kind, val in q_items:
            if kind == "word":
                qi += 1
                continue
            prev = max((q2cw[i] for i in range(qi + 1) if i in q2cw), default=-1)
            nearby = range(max(0, prev - 2), min(len(words), prev + 4))
            if kind == "SWEAR":
                hit = next((i for i in nearby if key(words[i]) == key(val)), None)
                if hit is not None:
                    words[hit] = f"[SWEAR: {words[hit]}]"
                else:
                    (after if prev >= 0 else before).setdefault(max(prev, 0), []).append(f"[SWEAR: {val}]")
                    notes.append(f"swear word '{val}' heard only by Qwen3-Omni")
                swears += 1
            elif kind == "NON-ENGLISH":
                (after if prev >= 0 else before).setdefault(max(prev, 0), []).append(f"[NON-ENGLISH: {val}]")
                notes.append(f"non-English '{val}' (Qwen3-Omni)")
            elif kind in EVENT_TAGS:
                if kind == "LAUGHTER" and any(kinds[i] == "event" and "laugh" in words[i].lower() for i in nearby):
                    continue
                label = f"[{kind}: {val}]" if val else f"[{kind}]"
                (after if prev >= 0 else before).setdefault(max(prev, 0), []).append(label)
            elif kind == "CUT OFF":
                cut_tags += 1

        # Cut-off start / end, measured on the audio.
        starts = [float(t["start"]) for t in toks]
        ends = [float(t["end"]) for t in toks]
        cut_start, cut_end = speech_at_edge(Path(m["wav"]), min(starts), max(ends), float(m["duration_s"]))
        # Qwen3-Omni also marks a sentence broken off by the end of the recording.
        q_tags = [k for k, _ in q_items]
        cut_start |= bool(q_tags) and q_tags[0] == "CUT OFF"
        cut_end |= bool(q_tags) and q_tags[-1] == "CUT OFF"

        out: list[str] = ["[CUT OFF]"] if cut_start else []
        pause_at = {round(float(p["start"]), 2): float(p["duration"]) for p in pauses
                    if float(p["duration"]) >= args.min_pause}
        n_partial = 0
        for i, (w, k) in enumerate(zip(words, kinds)):
            out += before.get(i, [])
            if k == "filler":
                out.append(f"[HESITATION: {key(w) or w}]")
            elif k == "event":
                out.append("[LAUGHTER]" if "laugh" in w.lower() else w.upper())
            elif w:
                out.append(w)
                n_partial += k == "partial"
            out += after.get(i, [])
            gap = pause_at.get(round(ends[i], 2))
            if gap is not None:
                out.append(f"[PAUSE {gap:.1f}s]")
        if cut_end:
            out.append("[CUT OFF]")
        text = " ".join(out)
        n_laugh = text.count("[LAUGHTER]")
        (args.out_dir / f"{rid}.txt").write_text(text + "\n")
        rows.append({**{k: m[k] for k in ("id", "participant", "session", "prompt", "condition", "duration_s")},
                     "transcript": text,
                     "n_words": sum(1 for w, k in zip(words, kinds) if k == "word" and w),
                     "n_hesitations": text.count("[HESITATION:"), "n_pauses": len(pause_at), "n_laughter": n_laugh,
                     "n_swears": swears, "n_partial_words": n_partial, "review_notes": "; ".join(notes)})

    with args.csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} transcripts -> {args.out_dir}, {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
