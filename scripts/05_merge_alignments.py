"""Step 5: merge MFA word timings back onto the CrisperWhisper verbatim tokens.

CrisperWhisper decides *what* was said (fillers, repetitions, partial words,
events); MFA refines *when*. MFA's word labels are matched to the expected
words (same normalisation as step 3, via common.py) by sequence alignment, so
an occasional OOV / <unk> / skipped chunk only affects those tokens: anything
MFA could not time keeps its CrisperWhisper timestamps, and the `source`
column records which aligner each timestamp came from.

Outputs in --out-dir:
    tokens/<id>.csv          one row per token: word, kind, start, end, source, is_repetition
    pauses/<id>.csv          silent gaps >= --min-pause between speech tokens
    textgrids/<id>.TextGrid  tiers words / disfluencies / pauses, for checking in Praat
    all_tokens.csv, all_pauses.csv   the above for every recording, with manifest columns
    summary.csv              per-recording counts, rates and MFA coverage

Usage:
    python scripts/05_merge_alignments.py --transcripts data/transcripts/crisperwhisper \
        --mfa-dir data/mfa_aligned --out-dir data/aligned
"""

from __future__ import annotations

import argparse
import math
import sys
from difflib import SequenceMatcher
from pathlib import Path

from praatio import textgrid

from common import Token, load_tokens, match_key, read_manifest, write_csv

SPEECH_KINDS = {"word", "filler", "partial"}
META_FIELDS = ["participant", "session", "prompt", "condition"]
TOKEN_FIELDS = ["id", "idx", "word", "kind", "start", "end", "duration", "source",
                "is_repetition", "cw_start", "cw_end"]
PAUSE_FIELDS = ["id", "start", "end", "duration", "prev_word", "next_word", "events_inside"]
SUMMARY_FIELDS = ["id", *META_FIELDS, "duration_s", "speech_start", "speech_end",
                  "n_words", "n_fillers", "n_partials", "n_repetitions", "n_events",
                  "n_pauses", "total_pause_s", "mean_pause_s", "wpm", "fillers_per_min",
                  "has_mfa", "mfa_coverage"]


def read_mfa_words(path: Path) -> list[tuple[float, float, str]]:
    """All non-empty intervals from every '... words' tier, in time order."""
    tg = textgrid.openTextgrid(str(path), includeEmptyIntervals=False)
    entries = []
    for name in tg.tierNames:
        if name.lower().endswith("words"):
            entries += [(e.start, e.end, e.label) for e in tg.getTier(name).entries
                        if e.label.strip() and e.label.strip() != "<eps>"]
    return sorted(entries)


def apply_mfa(tokens: list[Token], mfa_words: list[tuple[float, float, str]]) -> list[str]:
    """Overwrite token times with MFA's where the word sequences line up; return per-token source."""
    expected = [(ti, w) for ti, t in enumerate(tokens) for w in t.mfa_words]
    sm = SequenceMatcher(None, [match_key(w) for _, w in expected],
                         [match_key(lbl) for _, _, lbl in mfa_words], autojunk=False)
    hits: dict[int, list[tuple[float, float]]] = {}
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        # 'replace' blocks of equal length are label mismatches at the same position
        # (e.g. "[laughter]" exported as "[bracketed]", or a G2P'd partial word).
        if op == "equal" or (op == "replace" and i2 - i1 == j2 - j1):
            for k in range(i2 - i1):
                s, e, _ = mfa_words[j1 + k]
                hits.setdefault(expected[i1 + k][0], []).append((s, e))
    sources = []
    for ti, t in enumerate(tokens):
        if ti in hits:
            t.start = min(s for s, _ in hits[ti])
            t.end = max(e for _, e in hits[ti])
            sources.append("mfa")
        else:
            sources.append("crisperwhisper")
    return sources


def repetition_flags(tokens: list[Token]) -> list[bool]:
    """True when a word repeats the previous word, ignoring fillers/events in between ("I um I")."""
    flags, prev = [], None
    for t in tokens:
        if t.kind in ("filler", "event"):
            flags.append(False)
            continue
        key = match_key(" ".join(t.mfa_words))
        flags.append(t.kind == "word" and key != "" and key == prev)
        prev = key
    return flags


def find_pauses(tokens: list[Token], min_pause: float) -> list[dict]:
    speech = [t for t in tokens if t.kind in SPEECH_KINDS and not math.isnan(t.start)]
    events = [t for t in tokens if t.kind == "event"]
    pauses = []
    for a, b in zip(speech, speech[1:]):
        gap = b.start - a.end
        if gap >= min_pause:
            inside = [e.word for e in events if e.start < b.start and e.end > a.end]
            pauses.append({"start": round(a.end, 3), "end": round(b.start, 3), "duration": round(gap, 3),
                           "prev_word": a.word, "next_word": b.word, "events_inside": " ".join(inside)})
    return pauses


def tidy(entries: list[tuple[float, float, str]], duration: float) -> list[tuple[float, float, str]]:
    """Make intervals valid for a TextGrid: inside [0, duration], ordered, non-overlapping, non-empty."""
    out, last = [], 0.0
    for s, e, label in sorted(entries):
        s, e = max(s, last, 0.0), min(e, duration)
        if e - s >= 0.001:
            out.append((round(s, 3), round(e, 3), label))
            last = e
    return out


def write_textgrid(path: Path, tokens: list[Token], reps: list[bool], pauses: list[dict], duration: float) -> None:
    words = [(t.start, t.end, t.word) for t in tokens if not math.isnan(t.start)]
    disfl = [(t.start, t.end, "repetition" if r else t.kind) for t, r in zip(tokens, reps)
             if (r or t.kind in ("filler", "partial", "event")) and not math.isnan(t.start)]
    sil = [(p["start"], p["end"], f"{p['duration']:.2f}") for p in pauses]
    tg = textgrid.Textgrid()
    for name, entries in (("words", words), ("disfluencies", disfl), ("pauses", sil)):
        tg.addTier(textgrid.IntervalTier(name, tidy(entries, duration), 0.0, duration))
    path.parent.mkdir(parents=True, exist_ok=True)
    tg.save(str(path), format="long_textgrid", includeBlankSpaces=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transcripts", type=Path, default=Path("data/transcripts/crisperwhisper"))
    ap.add_argument("--mfa-dir", type=Path, default=Path("data/mfa_aligned"))
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/aligned"))
    ap.add_argument("--min-pause", type=float, default=0.25, help="shortest silent gap counted as a pause (s)")
    args = ap.parse_args()

    manifest = read_manifest(args.manifest)
    mfa_files = {p.stem: p for p in args.mfa_dir.rglob("*.TextGrid")} if args.mfa_dir.exists() else {}
    jsons = sorted(args.transcripts.glob("*.json"))
    if not jsons:
        sys.exit(f"no CrisperWhisper JSON in {args.transcripts}")

    all_tokens, all_pauses, summary = [], [], []
    for js in jsons:
        rec_id = js.stem
        meta = {k: manifest.get(rec_id, {}).get(k, "") for k in META_FIELDS}
        data, tokens = load_tokens(js)
        duration = float(manifest.get(rec_id, {}).get("duration_s") or data.get("duration") or 0.0)
        cw_times = [(t.start, t.end) for t in tokens]

        if rec_id in mfa_files:
            sources = apply_mfa(tokens, read_mfa_words(mfa_files[rec_id]))
        else:
            print(f"WARNING {rec_id}: no MFA TextGrid, keeping CrisperWhisper timings", file=sys.stderr)
            sources = ["crisperwhisper"] * len(tokens)
        reps = repetition_flags(tokens)
        pauses = find_pauses(tokens, args.min_pause)

        rows = [{"id": rec_id, "idx": t.idx, "word": t.word, "kind": t.kind,
                 "start": round(t.start, 3), "end": round(t.end, 3), "duration": round(t.end - t.start, 3),
                 "source": src, "is_repetition": int(rep), "cw_start": cs, "cw_end": ce}
                for t, src, rep, (cs, ce) in zip(tokens, sources, reps, cw_times)]
        pause_rows = [{"id": rec_id, **p} for p in pauses]
        write_csv(args.out_dir / "tokens" / f"{rec_id}.csv", rows, TOKEN_FIELDS)
        write_csv(args.out_dir / "pauses" / f"{rec_id}.csv", pause_rows, PAUSE_FIELDS)
        write_textgrid(args.out_dir / "textgrids" / f"{rec_id}.TextGrid", tokens, reps, pauses, duration)
        all_tokens += [{**r, **meta} for r in rows]
        all_pauses += [{**r, **meta} for r in pause_rows]

        speech = [t for t in tokens if t.kind in SPEECH_KINDS]
        span = (speech[-1].end - speech[0].start) if speech else 0.0
        n_words = sum(t.kind == "word" for t in tokens)
        n_fillers = sum(t.kind == "filler" for t in tokens)
        total_pause = sum(p["duration"] for p in pauses)
        summary.append({
            "id": rec_id, **meta, "duration_s": round(duration, 3),
            "speech_start": round(speech[0].start, 3) if speech else "",
            "speech_end": round(speech[-1].end, 3) if speech else "",
            "n_words": n_words, "n_fillers": n_fillers,
            "n_partials": sum(t.kind == "partial" for t in tokens),
            "n_repetitions": sum(reps),
            "n_events": sum(t.kind == "event" for t in tokens),
            "n_pauses": len(pauses), "total_pause_s": round(total_pause, 3),
            "mean_pause_s": round(total_pause / len(pauses), 3) if pauses else "",
            "wpm": round(n_words / (span / 60), 1) if span > 0 else "",
            "fillers_per_min": round(n_fillers / (span / 60), 2) if span > 0 else "",
            "has_mfa": int(rec_id in mfa_files),
            "mfa_coverage": round(sources.count("mfa") / len(sources), 3) if sources else "",
        })
        s = summary[-1]
        print(f"{rec_id:<40} words={n_words:<4} fillers={n_fillers:<3} pauses={len(pauses):<3} "
              f"mfa_coverage={s['mfa_coverage']}")

    write_csv(args.out_dir / "all_tokens.csv", all_tokens, TOKEN_FIELDS + META_FIELDS)
    write_csv(args.out_dir / "all_pauses.csv", all_pauses, PAUSE_FIELDS + META_FIELDS)
    write_csv(args.out_dir / "summary.csv", summary, SUMMARY_FIELDS)
    low = [s["id"] for s in summary if s["has_mfa"] and s["mfa_coverage"] != "" and s["mfa_coverage"] < 0.9]
    if low:
        print(f"\nCheck these in Praat, MFA timed < 90% of tokens: {', '.join(low)}", file=sys.stderr)
    print(f"\n{len(summary)} recordings -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
