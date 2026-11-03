"""Step 2: verbatim transcription with word timestamps using CrisperWhisper 2.0.

For every WAV in --wav-dir this writes

    <out-dir>/<id>.json   full result: text, word list [{word, start, end}], run metadata
    <out-dir>/<id>.txt    the verbatim transcript as plain text

Already-transcribed files are skipped (safe to re-run after a Colab disconnect);
use --overwrite to redo them.

Usage (Colab GPU):
    python scripts/02_transcribe_crisperwhisper.py --wav-dir data/wav --out-dir data/transcripts/crisperwhisper

Docs: https://github.com/nyrahealth/CrisperWhisper/blob/main/DOCS.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def words_to_list(words, offset: float = 0.0) -> list[dict]:
    return [{"word": w.word, "start": round(float(w.start) + offset, 3), "end": round(float(w.end) + offset, 3)}
            for w in (words or [])]


def pause_cuts(x, sr: int, max_len: float = 30.0, window: float = 0.2) -> list[float]:
    """Cut points (s) so every piece is at most max_len: the quietest `window`-long stretch in the
    middle third of each over-long piece. Each piece then fits one 30 s Whisper window, so nothing
    is stitched across windows (the stitch is where the 30-39 s JAM clips looped)."""
    import numpy as np

    hop = int(sr * 0.01)
    energy = np.array([np.mean(x[i:i + hop] ** 2) for i in range(0, len(x) - hop + 1, hop)])
    smooth = np.convolve(energy, np.ones(int(window / 0.01)), mode="same")

    def cut(lo: int, hi: int) -> list[int]:  # in 10 ms frames
        if (hi - lo) * 0.01 <= max_len:
            return []
        a, b = lo + (hi - lo) // 3, lo + 2 * (hi - lo) // 3
        c = a + int(np.argmin(smooth[a:b]))
        return cut(lo, c) + [c] + cut(c, hi)

    return [c * 0.01 for c in cut(0, len(smooth))]


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav-dir", type=Path, default=Path("data/wav"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/transcripts/crisperwhisper"))
    ap.add_argument("--model", default="large",
                    help="size shorthand (large|medium|small|turbo), HF id, or local path")
    ap.add_argument("--backend", default="auto", choices=["auto", "ct2", "transformers"])
    ap.add_argument("--compute-type", default="float16", help="e.g. float16, int8_float16 (ct2)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--ct2-cache", type=Path, default=None,
                    help="where the converted CTranslate2 model is cached; put it on Drive to skip reconversion")
    ap.add_argument("--language", default="en")
    ap.add_argument("--mode", default="verbatim", choices=["verbatim", "intended"])
    ap.add_argument("--dual", action="store_true",
                    help="also store the clean 'intended' transcript from the same pass (ct2 backend only)")
    ap.add_argument("--no-hallucination-mitigation", action="store_true")
    ap.add_argument("--longform-strategy", default="continuation",
                    choices=["continuation", "chunked_lcs", "token_lcs"],
                    help="how audio over 30 s is stitched; the LCS strategies decode each window without "
                         "the previous text as context, which avoids repetition loops at the boundary")
    ap.add_argument("--chunk-duration", type=float, default=30.0,
                    help="long-form window in seconds (<= 30); a shorter window moves the cut away from "
                         "the end of the 30 s encoder window, where the 30-39 s JAM clips looped")
    ap.add_argument("--stride", type=float, default=26.0, help="seconds between long-form windows")
    ap.add_argument("--split-at-pause", action="store_true",
                    help="cut audio over 30 s at its quietest point near the middle and transcribe each "
                         "piece in one window (no long-form stitching); word times are shifted back")
    ap.add_argument("--glob", default="*.wav", help="which files in --wav-dir to process")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N files (smoke test)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    wavs = sorted(args.wav_dir.glob(args.glob))[: args.limit]
    if not wavs:
        sys.exit(f"no files matching {args.glob} in {args.wav_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    todo = [w for w in wavs if args.overwrite or not (args.out_dir / f"{w.stem}.json").exists()]
    print(f"{len(wavs)} files, {len(wavs) - len(todo)} already done, {len(todo)} to transcribe")
    if not todo:
        return 0

    from crisperwhisper import CrisperWhisperModel  # heavy import, only when there is work

    t0 = time.time()
    model = CrisperWhisperModel(args.model, backend=args.backend, compute_type=args.compute_type,
                                device=args.device, cache_dir=args.ct2_cache)
    print(f"loaded {args.model} on backend={model.backend} in {time.time() - t0:.0f}s")
    if args.dual and model.backend != "ct2":
        sys.exit("--dual needs the ct2 backend (transcribe_dual is not available on transformers)")

    common = dict(language=args.language, word_timestamps=True,
                  hallucination_mitigation=not args.no_hallucination_mitigation,
                  longform_strategy=args.longform_strategy,
                  chunk_duration=args.chunk_duration, stride=args.stride)
    run_info = {
        "model": args.model,
        "backend": model.backend,
        "compute_type": args.compute_type,
        "crisperwhisper_version": package_version("crisperwhisper"),
        "language": args.language,
        "hallucination_mitigation": common["hallucination_mitigation"],
        "longform_strategy": args.longform_strategy,
        "chunk_duration": args.chunk_duration,
        "stride": args.stride,
        "split_at_pause": args.split_at_pause,
    }

    def run(audio, **kw):
        if args.dual:
            return model.transcribe_dual(audio, modes=("verbatim", "intended"), **kw, **common)
        return model.transcribe(audio, mode=args.mode, **kw, **common), None

    failures = []
    for n, wav in enumerate(todo, 1):
        try:
            if args.split_at_pause:
                import soundfile as sf

                x, sr = sf.read(str(wav), dtype="float32")
                bounds = [0.0, *pause_cuts(x, sr), len(x) / sr]
                texts, intended_texts, words, segments, proc = [], [], [], [], 0.0
                for s, e in zip(bounds, bounds[1:]):
                    result, intended = run(x[int(s * sr):int(e * sr)], sr=sr)
                    texts.append(result.text.strip())
                    intended_texts.append(intended.text.strip() if intended else "")
                    words += words_to_list(result.words, offset=s)
                    segments.append([round(s, 3), round(e, 3)])
                    proc += result.processing_time
                text, duration = " ".join(t for t in texts if t), len(x) / sr
                intended_text = " ".join(t for t in intended_texts if t) if args.dual else None
            else:
                result, intended = run(str(wav))
                text, words, segments = result.text, words_to_list(result.words), None
                duration, proc = result.duration, result.processing_time
                intended_text = intended.text if intended is not None else None

            record = {
                "id": wav.stem,
                "audio": wav.as_posix(),
                **run_info,
                "mode": result.mode,
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "duration": duration,
                "processing_time": proc,
                "text": text,
                "words": words,
            }
            if segments:
                record["segments"] = segments
            if intended_text is not None:
                record["intended_text"] = intended_text
            write_atomic(args.out_dir / f"{wav.stem}.json", json.dumps(record, ensure_ascii=False, indent=2))
            write_atomic(args.out_dir / f"{wav.stem}.txt", text.strip() + "\n")
            print(f"[{n}/{len(todo)}] {wav.stem}: {len(record['words'])} words, "
                  f"{duration:.1f}s audio in {proc:.1f}s")
        except Exception:
            failures.append(wav.stem)
            print(f"[{n}/{len(todo)}] FAILED {wav.stem}\n{traceback.format_exc()}", file=sys.stderr)

    if failures:
        print(f"\n{len(failures)} failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"\ndone in {time.time() - t0:.0f}s -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
