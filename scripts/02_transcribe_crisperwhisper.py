"""Step 2: verbatim transcription with word timestamps using CrisperWhisper 2.0.

For every WAV in --wav-dir this writes

    <out-dir>/<id>.json   full result: text, word list [{word, start, end}], run metadata
    <out-dir>/<id>.txt    the verbatim transcript as plain text

Already-transcribed files are skipped (safe to re-run after a Colab disconnect);
use --overwrite to redo them.

Parallelism: CrisperWhisper has no cross-file batching. transcribe() decodes one
file at a time (hallucination repair, attention capture for word timings and
the longform continuation are all per-sample steps), and the engine's
generate_batch is a Python loop. To use spare VRAM, --workers N starts N processes,
each with its own copy of the model (~4 GB in fp16), each taking every Nth file.

Usage (Colab GPU):
    python scripts/02_transcribe_crisperwhisper.py --wav-dir data/wav --out-dir data/transcripts/crisperwhisper
    python scripts/02_transcribe_crisperwhisper.py ... --workers 3      # 16 GB GPU

Docs: https://github.com/nyrahealth/CrisperWhisper/blob/main/DOCS.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
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


def words_to_list(words) -> list[dict]:
    return [{"word": w.word, "start": round(float(w.start), 3), "end": round(float(w.end), 3)}
            for w in (words or [])]


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def prepare_ct2_model(args) -> None:
    """Convert the checkpoint to CTranslate2 once, before workers start, so they
    don't all race to convert into the same cache directory (conversion is CPU-only)."""
    from crisperwhisper.model import _backend_available, resolve_model_id

    if args.backend == "transformers" or (args.backend == "auto" and not _backend_available("ct2")):
        return
    from crisperwhisper.converter import ensure_ct2_model

    ensure_ct2_model(resolve_model_id(args.model), quantization=args.compute_type, cache_dir=args.ct2_cache)


def launch_workers(args, n_todo: int) -> int:
    """Re-run this script as N single-worker processes, one shard each."""
    n = min(args.workers, n_todo)
    print(f"preparing model, then starting {n} workers")
    prepare_ct2_model(args)
    procs = [subprocess.Popen([sys.executable, __file__, *sys.argv[1:], "--workers", "1", "--shard", f"{k}/{n}"])
             for k in range(n)]
    codes = [p.wait() for p in procs]
    failed = [k for k, c in enumerate(codes) if c != 0]
    if failed:
        print(f"workers {failed} reported failures; re-run to retry the missing files", file=sys.stderr)
    return 1 if failed else 0


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
    ap.add_argument("--glob", default="*.wav", help="which files in --wav-dir to process")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N files (smoke test)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel model copies on the GPU (each ~4 GB fp16 for large); 2-3 fit in 16 GB")
    ap.add_argument("--shard", default="0/1", help=argparse.SUPPRESS)  # set by --workers: "k/N"
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # stdout is a pipe under Colab's `!`; show progress live

    wavs = sorted(args.wav_dir.glob(args.glob))[: args.limit]
    if not wavs:
        sys.exit(f"no files matching {args.glob} in {args.wav_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    k, n_shards = map(int, args.shard.split("/"))
    tag = f"[w{k}] " if n_shards > 1 else ""
    # Shard the full sorted list (not the to-do list) so every worker sees the same split.
    todo = [w for w in wavs[k::n_shards] if args.overwrite or not (args.out_dir / f"{w.stem}.json").exists()]
    if n_shards == 1:
        print(f"{len(wavs)} files, {len(wavs) - len(todo)} already done, {len(todo)} to transcribe")
    if not todo:
        return 0
    if args.workers > 1:
        return launch_workers(args, len(todo))

    from crisperwhisper import CrisperWhisperModel  # heavy import, only when there is work

    t0 = time.time()
    model = CrisperWhisperModel(args.model, backend=args.backend, compute_type=args.compute_type,
                                device=args.device, cache_dir=args.ct2_cache)
    print(f"{tag}loaded {args.model} on backend={model.backend} in {time.time() - t0:.0f}s, {len(todo)} files")
    if args.dual and model.backend != "ct2":
        sys.exit("--dual needs the ct2 backend (transcribe_dual is not available on transformers)")

    common = dict(language=args.language, word_timestamps=True,
                  hallucination_mitigation=not args.no_hallucination_mitigation)
    run_info = {
        "model": args.model,
        "backend": model.backend,
        "compute_type": args.compute_type,
        "crisperwhisper_version": package_version("crisperwhisper"),
        "language": args.language,
        "hallucination_mitigation": common["hallucination_mitigation"],
    }

    failures = []
    for n, wav in enumerate(todo, 1):
        try:
            intended = None
            if args.dual:
                result, intended = model.transcribe_dual(str(wav), modes=("verbatim", "intended"), **common)
            else:
                result = model.transcribe(str(wav), mode=args.mode, **common)

            record = {
                "id": wav.stem,
                "audio": wav.as_posix(),
                **run_info,
                "mode": result.mode,
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "duration": result.duration,
                "processing_time": result.processing_time,
                "text": result.text,
                "words": words_to_list(result.words),
            }
            if intended is not None:
                record["intended_text"] = intended.text
            write_atomic(args.out_dir / f"{wav.stem}.json", json.dumps(record, ensure_ascii=False, indent=2))
            write_atomic(args.out_dir / f"{wav.stem}.txt", result.text.strip() + "\n")
            print(f"{tag}[{n}/{len(todo)}] {wav.stem}: {len(record['words'])} words, "
                  f"{result.duration:.1f}s audio in {result.processing_time:.1f}s")
        except Exception:
            failures.append(wav.stem)
            print(f"{tag}[{n}/{len(todo)}] FAILED {wav.stem}\n{traceback.format_exc()}", file=sys.stderr)

    if failures:
        print(f"\n{tag}{len(failures)} failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"\n{tag}done in {time.time() - t0:.0f}s -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
