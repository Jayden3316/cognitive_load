# Cognitive load of prescriptive rules: transcription and alignment pipeline

This repo turns the JAM (Just A Minute) recordings into **verbatim, time-aligned transcripts** for the
HS5652 project. It keeps filled pauses, repetitions, partial words and silent pauses, because the
project's metrics (WPM, rule-violation timestamps, violation rate per 5 s bin, ideas per minute)
all depend on them.

```
audio/*.mp3|m4a ──(1) ffmpeg──▶ 16 kHz WAV + manifest
                 ──(2) CrisperWhisper 2.0, verbatim ──▶ words + fillers + word timestamps (~41 ms)
                 ──(3) chunk at pauses, normalise tokens ──▶ MFA corpus (TextGrid per recording)
                 ──(4) Montreal Forced Aligner v3, Indian English dictionary + G2P ──▶ refined word timings
                 ──(5) merge ──▶ tokens.csv, pauses.csv, Praat TextGrids, summary.csv
```

CrisperWhisper decides **what** was said. MFA refines **when** it was said. The merge step keeps
CrisperWhisper's tokens, swaps in MFA's timestamps wherever the two agree, and records which tool
each timestamp came from.

## Layout

```
pyproject.toml                       uv project (CrisperWhisper + praatio + num2words)
notebooks/colab_pipeline.ipynb       runs all five steps on Colab
scripts/
  common.py                          token rules shared by steps 3 and 5 (fillers, events, partials)
  01_prepare_audio.py                convert to WAV, parse participant/day/prompt/condition from file names
  02_transcribe_crisperwhisper.py    CrisperWhisper 2.0 verbatim transcription with word timestamps
  03_prepare_mfa_corpus.py           CrisperWhisper JSON → MFA corpus (speaker dirs + TextGrids)
  04_run_mfa.sh                      installs MFA v3 (micromamba) and runs `mfa align_hf`
  05_merge_alignments.py             MFA timings → verbatim tokens; pauses; summary
audio/                               raw recordings (git-ignored)
data/                                all generated output (git-ignored)
```

Run every script **from the repo root**, because paths in `data/manifest.csv` are relative to it.

## Running on Colab (recommended)

1. Copy this folder, including `audio/`, to Google Drive, e.g. `MyDrive/cognitive_load`.
2. Open `notebooks/colab_pipeline.ipynb` in Colab and pick a GPU runtime. A T4 is enough: the 2B model is about 4 GB in fp16.
3. Set `REPO_DIR` in the first cell and run all cells. Every step skips files it has already done,
   so after a disconnect you can rerun from the top.

The notebook does the following:

```bash
pip install uv
UV_PROJECT_ENVIRONMENT=/content/.venv uv sync --extra ct2     # CrisperWhisper on the CTranslate2 GPU backend
PY=/content/.venv/bin/python
$PY scripts/01_prepare_audio.py
$PY scripts/02_transcribe_crisperwhisper.py --ct2-cache /content/drive/MyDrive/cognitive_load_cache/ct2
$PY scripts/03_prepare_mfa_corpus.py
MFA_ENV=/content/mfa_env bash scripts/04_run_mfa.sh
$PY scripts/05_merge_alignments.py
```

The venv is called directly (`$PY`) instead of through `uv run`. A plain `uv run` re-syncs to the
default dependencies and would remove the `ct2` extra.

**Model licence:** the CrisperWhisper 2.0 weights use the Nyra Health *Non-Commercial Research
Licence*. If Hugging Face asks you to accept terms, accept them on
[the model page](https://huggingface.co/nyralabs/CrisperWhisper2.0_large) and add an `HF_TOKEN` Colab secret.

## Running locally

Steps 1, 2, 3 and 5 work on any OS through uv:

```bash
uv sync --extra torch          # pure-PyTorch backend (Windows/macOS/CPU); use --extra ct2 on Linux + NVIDIA
uv run --extra torch python scripts/02_transcribe_crisperwhisper.py --backend transformers --limit 1
```

Step 1 needs `ffmpeg` on PATH. Step 4 needs Linux or macOS. On Windows, use WSL or the official Docker image
`mmcauliffe/montreal-forced-aligner`.

## Step details and design decisions

### 1. `01_prepare_audio.py`
- Converts `mp3`/`m4a` to 16 kHz mono 16-bit PCM WAV. MFA resamples to 16 kHz anyway, and Kaldi needs 16-bit audio.
- Builds IDs of the form `participant_session_prompt_condition`, e.g. `abhijith_day1_banana_rules` or `ranesh_na_tiger_norules`.
  - The condition comes from "with rules" / "without rules" in the file name, or from the folder name if the file name has neither.
  - The prompt is matched against `--prompts` (banana, table, tiger, tree, water); plurals are allowed.
  - The session is `dayN` when the path contains "Day N", otherwise `na`. Ranesh's folders have no day.
- Writes `data/manifest.csv`. The script warns when it can't parse a file, and stops if two files map to the same ID.

### 2. `02_transcribe_crisperwhisper.py`
- Calls `CrisperWhisperModel(...).transcribe(wav, mode="verbatim", word_timestamps=True, language="en")`.
  Hallucination mitigation is on, and long-form audio (over 30 s) uses the default `continuation` strategy.
- Writes `data/transcripts/crisperwhisper/<id>.json` with `text`, `words: [{word, start, end}]`, and the
  model, backend and package version. It also writes a plain `.txt` copy.
- Options: `--model` (`large`, `medium`, `small`, `turbo`, or an HF id/path), `--backend`, `--compute-type`,
  `--dual` (also store the clean *intended* transcript, ct2 only), `--limit N`, `--overwrite`.
- Files are written atomically and existing outputs are skipped, so a Colab disconnect never leaves half-written JSON.

### 3. `03_prepare_mfa_corpus.py`
MFA aligns best on short segments (under about 30 s; shorter is faster and more robust), and each recording is
30–40 s of continuous speech. The CrisperWhisper timings are used to split each transcript into chunks:
- Every silence of at least `--hard-gap` (default 1.0 s) starts a new chunk.
- A chunk still longer than `--max-len` (15 s) is split at its largest internal gap, recursively. Only cuts that
  leave at least a third of `--max-len` on each side are considered, so a slightly longer gap at the edge can't
  peel off a one-word chunk.
- Each chunk is padded by `--pad` (0.15 s), but never past the middle of the neighbouring gap. A small
  CrisperWhisper timing error can't push a word outside its chunk.

Tokens are mapped to MFA words in `common.py`:

| CrisperWhisper token | kind | sent to MFA as | why |
|---|---|---|---|
| `[UM]`, `[UH]`, `um`, `uhm` | filler | `um`, `uh` | in the dictionary, so it gets real phones and a real duration |
| `[laughter]`, `[breath]`, … | event | `[laughter]` | MFA maps bracketed words to spoken noise (`spn`), so the noise still takes up time |
| `th-`, `wh—` | partial | `th` | gets a pronunciation from the Indian English G2P |
| `25` | word | `twenty five` | digits have no pronunciation |
| anything else | word | as is (MFA strips punctuation) | |

Each speaker gets a folder named after the participant. This lets MFA's per-speaker (fMLLR) adaptation pool
all ten recordings by the same person, which helps with accent.

### 4. `04_run_mfa.sh`
- **Why conda appears at all:** MFA depends on Kaldi, OpenFst and kalpy binaries, and those are only published on
  conda-forge. Even MFA's documented "pip" route starts with `conda create … kaldi pynini`. To avoid a full
  conda installation, the script downloads **micromamba** (a single static binary). It uses micromamba to build an
  isolated prefix (`$MFA_ENV`) containing `montreal-forced-aligner>=3.4,<4`, then calls `micromamba run -p $MFA_ENV mfa …`.
  Nothing is activated, and Colab's Python and the uv venv are untouched.
- Runs `mfa align_hf CORPUS english_mfa OUT --dialect english_india_mfa --use_g2p`. This is the MFA ≥ 3.4 interface:
  the Hugging Face model `MontrealCorpusTools/english_mfa` bundles the acoustic model, the dictionaries
  (US/UK/India/Nigeria) and the G2P models. `--use_g2p` generates pronunciations for out-of-vocabulary words
  instead of mapping them to `<unk>`.
- You can override these with environment variables: `MFA_DIALECT=english_us_mfa`, `NUM_JOBS`, `MFA_SPEC`,
  `MFA_EXTRA="--fine_tune"` (an extra boundary fine-tuning pass; slower but can tighten boundaries).
- The spec is pinned to `<4`. MFA 4.0 is scheduled to turn `align_hf` into the default `mfa align` and rename
  the old command to `mfa align_legacy`.

### 5. `05_merge_alignments.py`
- Reads every `*words` tier from MFA's TextGrids. It matches MFA's word labels to the expected word sequence with
  `difflib.SequenceMatcher`, on lower-cased alphanumeric keys. Matched tokens take MFA's times. Unmatched tokens
  keep CrisperWhisper's times. Unmatched tokens include words in a chunk MFA failed to align, and length-mismatched edits.
- A **pause** is a silent gap of at least `--min-pause` (default 0.25 s) between consecutive speech tokens (words,
  fillers or partials). Non-speech events inside a gap are listed in `events_inside`, not treated as speech.
- A **repetition** is a word identical to the previous word, ignoring fillers in between. "I um I" counts.

Outputs in `data/aligned/`:

| file | contents |
|---|---|
| `tokens/<id>.csv` | `idx, word, kind (word/filler/partial/event), start, end, duration, source (mfa/crisperwhisper), is_repetition, cw_start, cw_end` |
| `pauses/<id>.csv` | `start, end, duration, prev_word, next_word, events_inside` |
| `textgrids/<id>.TextGrid` | tiers `words`, `disfluencies`, `pauses`; open with the WAV in Praat to check alignments |
| `all_tokens.csv`, `all_pauses.csv` | all recordings stacked, with participant/session/prompt/condition columns |
| `summary.csv` | per recording: `n_words, n_fillers, n_partials, n_repetitions, n_events, n_pauses, total_pause_s, mean_pause_s, wpm, fillers_per_min, mfa_coverage` |

WPM counts lexical words only (no fillers, partials or events) over the speaking span, from the first to the last
speech token. This excludes lead-in and trailing silence.

## Quality checks

- **`mfa_coverage`** in `summary.csv` is the share of tokens timed by MFA. The merge step lists every recording
  below 0.9. Open those `textgrids/<id>.TextGrid` files in Praat, together with `data/wav/<id>.wav`.
- Fillers are what CrisperWhisper is designed for, but spot-check a few recordings by ear anyway. This matters
  most for short `ah`/`eh` interjections, and for repetitions the model may normalise.
- If a recording starts with the moderator announcing the topic or rules, that speech is transcribed too. Trim it
  before step 2, or drop those tokens before analysis.

## Notes on the current data

- 50 recordings: Abhijith, Akshara, Ranesh, Sanjith and Venkat, 10 each (5 prompts × with/without rules). `audio/LCC_Data/Sharma/` is empty.
- Recordings are 29–39 s long (Abhijith's are about 30.5–31.7 s; Akshara's and Venkat's run longer).
- Abhijith's files are 48 kHz stereo mp3; the others are m4a. ffmpeg reports `Header missing` once per Abhijith
  file. Each of those files has one corrupt mp3 frame, so about 26 ms of audio is dropped. This is harmless, and
  the conversion still succeeds.

## References

- CrisperWhisper 2.0: [model card](https://huggingface.co/nyralabs/CrisperWhisper2.0_large), [DOCS.md](https://github.com/nyrahealth/CrisperWhisper/blob/main/DOCS.md)
- Montreal Forced Aligner: [installation](https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html), [corpus structure](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/corpus_structure.html), [`align` / `align_hf`](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/workflows/alignment.html), [HF model versions](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/models/model_versions.html)
