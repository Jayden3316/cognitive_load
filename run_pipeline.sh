#!/usr/bin/env bash
# Run the whole pipeline (or chosen steps) on a data folder, e.g. on Google Drive.
#
#   bash run_pipeline.sh DATA_DIR [STEP ...]
#
# DATA_DIR mirrors the local project layout:
#   DATA_DIR/audio/LCC_Data/<participant>/...    recordings (input)
#   DATA_DIR/data/...                            everything the pipeline writes
#   DATA_DIR/cache/ct2/                          converted CrisperWhisper model (reused across sessions)
#
# Steps: 1 audio -> WAV, 2 CrisperWhisper, 3 MFA corpus, 4 MFA, 5 merge (default: all).
# Every step skips work already done, so after a Colab disconnect just run it again.
#
# Environment overrides:
#   AUDIO_DIR    input recordings              [DATA_DIR/audio/LCC_Data]
#   OUT_DIR      output root                   [DATA_DIR/data]
#   CW_WORKERS   parallel CrisperWhisper copies on the GPU (~4 GB each)  [1]
#   CW_ARGS      extra args for step 2, e.g. "--limit 1" or "--dual"
#   MIN_PAUSE    pause threshold in seconds    [0.25]
#   EXTRA        "ct2" (Linux + NVIDIA) or "torch" (CPU / Windows / macOS)  [ct2]
#   plus the MFA_* variables documented in scripts/04_run_mfa.sh
set -euo pipefail

if [[ $# -lt 1 ]]; then
    sed -n '2,20p' "$0"
    exit 1
fi
DATA_DIR=$(realpath "$1"); shift
STEPS=${*:-1 2 3 4 5}
AUDIO_DIR=${AUDIO_DIR:-$DATA_DIR/audio/LCC_Data}
OUT_DIR=${OUT_DIR:-$DATA_DIR/data}
CW_WORKERS=${CW_WORKERS:-1}
CW_ARGS=${CW_ARGS:-}
MIN_PAUSE=${MIN_PAUSE:-0.25}
EXTRA=${EXTRA:-ct2}

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$REPO"

has_step() { [[ " $STEPS " == *" $1 "* ]]; }

# Python env for steps 1, 2, 3, 5 (uv venv in the repo, on local disk). MFA has its own env (step 4).
if has_step 1 || has_step 2 || has_step 3 || has_step 5; then
    command -v uv >/dev/null || pip install -q uv
    uv sync -q --extra "$EXTRA"
fi
PY="${UV_PROJECT_ENVIRONMENT:-$REPO/.venv}/bin/python"

echo "== data: $DATA_DIR   audio: $AUDIO_DIR   output: $OUT_DIR   steps: $STEPS"

if has_step 1; then
    echo "== [1/5] audio -> 16 kHz WAV"
    "$PY" scripts/01_prepare_audio.py --audio-root "$AUDIO_DIR" \
        --out-dir "$OUT_DIR/wav" --manifest "$OUT_DIR/manifest.csv"
fi

if has_step 2; then
    echo "== [2/5] CrisperWhisper"
    # shellcheck disable=SC2086
    "$PY" scripts/02_transcribe_crisperwhisper.py --wav-dir "$OUT_DIR/wav" \
        --out-dir "$OUT_DIR/transcripts/crisperwhisper" --ct2-cache "$DATA_DIR/cache/ct2" \
        --workers "$CW_WORKERS" $CW_ARGS
fi

if has_step 3; then
    echo "== [3/5] MFA corpus"
    "$PY" scripts/03_prepare_mfa_corpus.py --transcripts "$OUT_DIR/transcripts/crisperwhisper" \
        --manifest "$OUT_DIR/manifest.csv" --out-dir "$OUT_DIR/mfa_corpus"
fi

if has_step 4; then
    echo "== [4/5] Montreal Forced Aligner"
    bash scripts/04_run_mfa.sh "$OUT_DIR/mfa_corpus" "$OUT_DIR/mfa_aligned"
fi

if has_step 5; then
    echo "== [5/5] merge"
    "$PY" scripts/05_merge_alignments.py --transcripts "$OUT_DIR/transcripts/crisperwhisper" \
        --mfa-dir "$OUT_DIR/mfa_aligned" --manifest "$OUT_DIR/manifest.csv" \
        --out-dir "$OUT_DIR/aligned" --min-pause "$MIN_PAUSE"
fi

echo "== done: results in $OUT_DIR/aligned"
