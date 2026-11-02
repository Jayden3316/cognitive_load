"""Step 1: convert raw recordings to 16 kHz mono 16-bit WAV with clean, unique IDs.

Recording metadata is parsed from the folder/file names, e.g.

    audio/LCC_Data/Abhijith/Day 1/banana with rules abhijith.mp3
        -> id = abhijith_day1_banana_rules
    audio/LCC_Data/Ranesh/Without Rules/Tigers without Rules.m4a
        -> id = ranesh_na_tiger_norules     (no "Day N" in the path)

and written to a manifest CSV that every later step joins on.

Usage:
    python scripts/01_prepare_audio.py --audio-root audio/LCC_Data --out-dir data/wav
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import wave
from pathlib import Path

from common import write_csv

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".mp4", ".webm", ".opus"}
DEFAULT_PROMPTS = ["banana", "table", "tiger", "tree", "water"]
MANIFEST_FIELDS = ["id", "participant", "session", "prompt", "condition", "duration_s", "wav", "source"]


def parse_metadata(rel: Path, prompts: list[str]) -> dict:
    """Infer participant/session/prompt/condition from a path relative to the audio root."""
    participant = re.sub(r"\W+", "", rel.parts[0]).lower()
    lowered = [p.lower() for p in rel.parts]
    joined = " / ".join(lowered)

    session = "na"
    for part in lowered[1:]:
        if m := re.search(r"day\s*(\d+)", part):
            session = f"day{m.group(1)}"

    # The file name wins; the folder ("With Rules/", "Without Rules/") is the fallback.
    name = lowered[-1]
    if re.search(r"without\s*rules", name) or (not re.search(r"with\s*rules", name) and "without rules" in joined):
        condition = "norules"
    elif re.search(r"with\s*rules", joined):
        condition = "rules"
    else:
        condition = "unknown"

    prompt = "unknown"
    if m := re.search(r"\b(" + "|".join(map(re.escape, prompts)) + r")s?\b", name):
        prompt = m.group(1)

    return {
        "id": f"{participant}_{session}_{prompt}_{condition}",
        "participant": participant,
        "session": session,
        "prompt": prompt,
        "condition": condition,
    }


def wav_duration(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def convert(ffmpeg: str, src: Path, dst: Path, sample_rate: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.wav")
    cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
           "-vn", "-ac", "1", "-ar", str(sample_rate), "-sample_fmt", "s16", str(tmp)]
    subprocess.run(cmd, check=True)
    tmp.replace(dst)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio-root", type=Path, default=Path("audio/LCC_Data"),
                    help="folder whose immediate subfolders are participants")
    ap.add_argument("--out-dir", type=Path, default=Path("data/wav"))
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS, help="known prompt words (singular)")
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if shutil.which(args.ffmpeg) is None:
        sys.exit(f"ffmpeg not found ({args.ffmpeg!r}); install it or pass --ffmpeg")

    # Never pick up our own outputs if they live under the audio root.
    generated = {args.out_dir.resolve(), args.manifest.resolve().parent}
    sources = sorted(p for p in args.audio_root.rglob("*")
                     if p.is_file() and p.suffix.lower() in AUDIO_EXTS
                     and not generated & set(p.resolve().parents))
    if not sources:
        sys.exit(f"no audio files under {args.audio_root}")

    rows, seen, problems = [], {}, []
    for src in sources:
        meta = parse_metadata(src.relative_to(args.audio_root), args.prompts)
        if meta["id"] in seen:
            problems.append(f"duplicate id {meta['id']}: {seen[meta['id']]} and {src}")
            continue
        if "unknown" in (meta["prompt"], meta["condition"]):
            problems.append(f"could not parse prompt/condition: {src}")
        seen[meta["id"]] = src

        dst = args.out_dir / f"{meta['id']}.wav"
        if args.overwrite or not dst.exists():
            convert(args.ffmpeg, src, dst, args.sample_rate)
        rows.append({**meta, "duration_s": f"{wav_duration(dst):.3f}",
                     "wav": dst.as_posix(), "source": src.as_posix()})
        print(f"{meta['id']:<40} {rows[-1]['duration_s']:>8}s  <- {src}")

    participants = {r["participant"] for r in rows}
    if len(participants) == 1 and len(rows) > len(args.prompts) * 2:
        problems.append(f"every file was assigned participant {participants.pop()!r}; --audio-root should be "
                        "the folder whose subfolders are the participants (e.g. audio/LCC_Data)")

    write_csv(args.manifest, rows, MANIFEST_FIELDS)
    print(f"\n{len(rows)} recordings -> {args.out_dir}, manifest -> {args.manifest}")
    for p in problems:
        print("WARNING:", p, file=sys.stderr)
    return 1 if any(p.startswith("duplicate") for p in problems) else 0


if __name__ == "__main__":
    sys.exit(main())
