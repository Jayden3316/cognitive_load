#!/usr/bin/env bash
# Step 4: forced alignment with Montreal Forced Aligner v3.
#
# MFA depends on Kaldi/OpenFst binaries that are only distributed through
# conda-forge, so it cannot live in the uv venv. Instead of a full conda install
# this uses micromamba (a single static binary) to build an isolated MFA prefix;
# nothing is added to PATH and Colab's own Python is untouched.
#
# Usage:  bash scripts/04_run_mfa.sh [CORPUS_DIR] [OUTPUT_DIR]
#
# Environment overrides (defaults in brackets):
#   MFA_ENV        prefix for the MFA environment      [$HOME/.mfa_env]
#   MAMBA_BIN      micromamba executable                [$HOME/.local/bin/micromamba]
#   MFA_SPEC       conda version spec                   [montreal-forced-aligner>=3.4,<4]
#   MFA_MODEL      Hugging Face MFA model               [english_mfa  (= MontrealCorpusTools/english_mfa)]
#   MFA_DIALECT    dictionary + G2P inside the model    [english_india_mfa]
#   NUM_JOBS       parallel jobs                        [nproc]
#   MFA_ROOT_DIR   MFA's model cache / temp directory   [$HOME/Documents/MFA]
#   MFA_EXTRA      extra flags for `mfa align_hf`, e.g. "--fine_tune"
set -euo pipefail

CORPUS=${1:-data/mfa_corpus}
OUTPUT=${2:-data/mfa_aligned}
MFA_ENV=${MFA_ENV:-$HOME/.mfa_env}
MAMBA_BIN=${MAMBA_BIN:-$HOME/.local/bin/micromamba}
MFA_SPEC=${MFA_SPEC:-"montreal-forced-aligner>=3.4,<4"}
MFA_MODEL=${MFA_MODEL:-english_mfa}
MFA_DIALECT=${MFA_DIALECT:-english_india_mfa}
NUM_JOBS=${NUM_JOBS:-$(nproc)}
MFA_EXTRA=${MFA_EXTRA:-}
export MFA_ROOT_DIR=${MFA_ROOT_DIR:-$HOME/Documents/MFA}
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$HOME/.micromamba}  # micromamba's package cache

if [[ ! -x "$MAMBA_BIN" ]]; then
    echo ">> installing micromamba -> $MAMBA_BIN"
    mkdir -p "$(dirname "$MAMBA_BIN")"
    tmp=$(mktemp -d)
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$tmp" bin/micromamba
    mv "$tmp/bin/micromamba" "$MAMBA_BIN"
    rm -rf "$tmp"
fi

if [[ ! -x "$MFA_ENV/bin/mfa" ]]; then
    echo ">> creating MFA environment at $MFA_ENV ($MFA_SPEC); takes a few minutes"
    "$MAMBA_BIN" create -y -q -p "$MFA_ENV" -c conda-forge "$MFA_SPEC"
fi

mfa() { "$MAMBA_BIN" run -p "$MFA_ENV" mfa "$@"; }

echo ">> MFA $(mfa version)"
echo ">> aligning $CORPUS with $MFA_MODEL (dialect $MFA_DIALECT), $NUM_JOBS jobs -> $OUTPUT"
mkdir -p "$OUTPUT"
# --use_g2p: words missing from the dictionary (partial words, names, Indian
# English vocabulary) get pronunciations from the model's G2P instead of <unk>.
# shellcheck disable=SC2086
mfa align_hf "$CORPUS" "$MFA_MODEL" "$OUTPUT" \
    --dialect "$MFA_DIALECT" \
    --use_g2p \
    --output_format long_textgrid \
    --include_original_text \
    --clean \
    --num_jobs "$NUM_JOBS" \
    $MFA_EXTRA

echo ">> done; TextGrids in $OUTPUT"
