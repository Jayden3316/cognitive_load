"""Step 3: turn CrisperWhisper output into an MFA corpus.

Layout produced (one folder per speaker so MFA can do per-speaker adaptation):

    data/mfa_corpus/<participant>/<id>.wav
    data/mfa_corpus/<participant>/<id>.TextGrid

Each TextGrid has one interval tier named after the speaker, with one interval per
utterance chunk. The recordings are ~1 min of continuous speech, while MFA works
best on segments under ~30 s, so the CrisperWhisper word timings are used to cut
the transcript at silent gaps: every gap >= --hard-gap starts a new chunk, and any
chunk still longer than --max-len is split recursively at its largest internal gap.
Chunk edges are padded by --pad seconds (never past the middle of the gap), so a
small CrisperWhisper timing error cannot clip a word out of its chunk.

Token handling (see common.py): fillers become plain words ("um"), non-speech
events stay bracketed ("[laughter]" -> spoken-noise), partial words lose their
trailing hyphen and are given pronunciations by MFA's G2P.

Usage:
    python scripts/03_prepare_mfa_corpus.py --transcripts data/transcripts/crisperwhisper --out-dir data/mfa_corpus
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import wave
from pathlib import Path

from praatio import textgrid

from common import Token, load_tokens, read_manifest


def fill_missing_times(tokens: list[Token]) -> list[Token]:
    """Give untimed tokens the previous token's end so they stay in reading order."""
    last = 0.0
    for t in tokens:
        if math.isnan(t.start):
            t.start = t.end = last
        if math.isnan(t.end) or t.end < t.start:
            t.end = t.start
        last = t.end
    return tokens


def split_long(group: list[Token], max_len: float) -> list[list[Token]]:
    """Recursively cut at the largest gap, considering only cuts that leave both sides
    at least max_len/3 long, so near-equal gaps do not peel off one-word chunks."""
    start, end = group[0].start, group[-1].end
    if len(group) < 2 or end - start <= max_len:
        return [group]
    cuts = range(1, len(group))
    balanced = [i for i in cuts if group[i - 1].end - start >= max_len / 3 and end - group[i].start >= max_len / 3]
    mid = (start + end) / 2
    cut = max(balanced or cuts, key=lambda i: (round(group[i].start - group[i - 1].end, 2),
                                               -abs(group[i].start - mid)))
    return split_long(group[:cut], max_len) + split_long(group[cut:], max_len)


def chunk(tokens: list[Token], hard_gap: float, max_len: float) -> list[list[Token]]:
    groups: list[list[Token]] = [[tokens[0]]]
    for prev, cur in zip(tokens, tokens[1:]):
        if cur.start - prev.end >= hard_gap:
            groups.append([cur])
        else:
            groups[-1].append(cur)
    return [g for group in groups for g in split_long(group, max_len)]


def intervals_for(groups: list[list[Token]], duration: float, pad: float) -> list[tuple[float, float, str]]:
    out = []
    for i, g in enumerate(groups):
        start = max(0.0, g[0].start - pad)
        end = min(duration, g[-1].end + pad)
        if i > 0:
            start = max(start, (groups[i - 1][-1].end + g[0].start) / 2)
        if i + 1 < len(groups):
            end = min(end, (g[-1].end + groups[i + 1][0].start) / 2)
        text = " ".join(w for t in g for w in t.mfa_words)
        if end - start < 0.1:  # MFA skips intervals under 100 ms
            print(f"  note: very short chunk {start:.2f}-{end:.2f} '{text}'", file=sys.stderr)
        out.append((round(start, 3), round(end, 3), text))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transcripts", type=Path, default=Path("data/transcripts/crisperwhisper"))
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/mfa_corpus"))
    ap.add_argument("--hard-gap", type=float, default=1.0, help="always split at silences at least this long (s)")
    ap.add_argument("--max-len", type=float, default=15.0, help="max chunk length before forced split (s)")
    ap.add_argument("--pad", type=float, default=0.15, help="padding around each chunk (s)")
    args = ap.parse_args()

    manifest = read_manifest(args.manifest)
    jsons = sorted(args.transcripts.glob("*.json"))
    if not jsons:
        sys.exit(f"no CrisperWhisper JSON in {args.transcripts}")

    n_chunks, longest = 0, 0.0
    for js in jsons:
        rec_id = js.stem
        if rec_id not in manifest:
            print(f"skip {rec_id}: not in {args.manifest}", file=sys.stderr)
            continue
        row = manifest[rec_id]
        _, tokens = load_tokens(js)
        tokens = fill_missing_times([t for t in tokens if t.mfa_words])
        if not tokens:
            print(f"skip {rec_id}: empty transcript", file=sys.stderr)
            continue

        speaker_dir = args.out_dir / row["participant"]
        speaker_dir.mkdir(parents=True, exist_ok=True)
        wav_dst = speaker_dir / f"{rec_id}.wav"
        shutil.copyfile(row["wav"], wav_dst)
        with wave.open(str(wav_dst)) as w:
            duration = w.getnframes() / w.getframerate()

        entries = intervals_for(chunk(tokens, args.hard_gap, args.max_len), duration, args.pad)
        tg = textgrid.Textgrid()
        tg.addTier(textgrid.IntervalTier(row["participant"], entries, 0.0, duration))
        tg.save(str(speaker_dir / f"{rec_id}.TextGrid"), format="long_textgrid", includeBlankSpaces=True)

        n_chunks += len(entries)
        longest = max(longest, max(e - s for s, e, _ in entries))
        print(f"{rec_id:<40} {len(tokens):>4} tokens -> {len(entries):>2} chunks")

    print(f"\n{n_chunks} chunks (longest {longest:.1f}s) -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
