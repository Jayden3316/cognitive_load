# Run notes (bodhanai cluster, 2026-11-02)

How the verbatim transcripts in `data/` were produced on this machine, and what was changed or found
along the way. The pipeline itself is described in `README.md`.

## Data

- Source: the project's Google Drive folder (shared by link, so the link stays out of this public
  repo; on the cluster it is in `data/DRIVE_SOURCE.txt`), downloaded with
  `uvx --from gdown==5.2.0 gdown --folder <url>` into `audio/LCC_Data/` (50 files, 24 MB).
  Every folder holds at most 10 files, so gdown's 50-files-per-folder limit did not cut anything.
  The Day2-4 folders of Akshara and Venkat, and all of Sharma's folders, are empty on Drive.
- 50 recordings: Abhijith, Akshara, Ranesh, Sanjith, Venkat, 10 each (5 prompts x with/without rules).

## Commands

The models (CrisperWhisper 2.0, Whisper large-v3, the AST tagger, MFA's english_mfa, Qwen3-Omni) live
in `/projects/data/speechteam/venkata/.cache/hf_models`, not in `~/.cache` (the shared `/home` disk is
full). Export this before any step that loads a model:

```bash
export HF_HUB_CACHE=/projects/data/speechteam/venkata/.cache/hf_models
```

```bash
uv sync --extra ct2                                            # .venv (Python 3.14, crisperwhisper 2.0.3, ctranslate2 4.7.1)
.venv/bin/python scripts/01_prepare_audio.py
.venv/bin/python scripts/02_transcribe_crisperwhisper.py --ct2-cache data/cache/ct2 --split-at-pause --dual   # on 1 H100
.venv/bin/python scripts/03_prepare_mfa_corpus.py
MFA_ENV=/projects/data/speechteam/venkata/envs/mfa MAMBA_ROOT_PREFIX=/projects/data/speechteam/venkata/.micromamba \
  MFA_ROOT_DIR=$PWD/data/cache/mfa NUM_JOBS=16 bash scripts/04_run_mfa.sh    # MFA 3.4.3
.venv/bin/python scripts/05_merge_alignments.py
```

## Changes to `scripts/02_transcribe_crisperwhisper.py`

New options; the defaults keep the original behaviour.

- `--split-at-pause` (used for the final run). Audio over 30 s is cut at the quietest 0.2 s stretch in
  the middle third, and each piece is transcribed in a single 30 s window; word times are shifted back.
  The JSON records the cut as `segments`.
- `--longform-strategy`, `--chunk-duration`, `--stride` expose CrisperWhisper's long-form settings.

Why: the clips are 29-39 s, just over the 30 s Whisper window. With the default long-form
`continuation` stitching, 3 of 50 transcripts looped at the window boundary (e.g. one phrase repeated
155 times in `ranesh_na_water_norules`), and the word list had duplicated zero-duration words at 29.98 s.
Changing the strategy (`chunked_lcs`, `token_lcs`) or the window (20 s, 24 s) only moved the loops to
other files. Splitting at a pause removes long-form stitching altogether. In the final run, 47 clips
were cut once (between 10.6 s and 21.9 s), no cut falls inside a word, and every transcript passes the
checks: text equals the word list, no zero-duration words, and no phrase repeated except one real
restart (`sanjith_day1_table_rules`, "you can find water along with" at 17.1 s and 20.4 s, also present
in the unsplit run). The superseded unsplit run is kept in `data/transcripts/superseded/`.

## Findings

- **Ranesh's five "with rules" recordings contain no speech.** Each is about 30 s of flat background
  noise (median -48 dB, 3-4 dB spread) after a short burst in the first second; his "without rules"
  files are speech at about -23 dB with 11-21 dB spread. Raising the gain did not recover any words.
  They are transcribed as empty and are not aligned. They need to be re-recorded or re-shared.
- MFA 3.4.3 crashes in TextGrid export (`KeyError` on the speaker) when a file has an utterance but no
  aligned phones; an earlier pass with a spurious 9-word transcript of a noise-only file triggered it.
  With the final transcripts no such file exists, and all 45 recordings with speech align.

## Labelled transcripts (steps 6-7, added here)

```bash
.venv/bin/python scripts/06a_second_opinion_whisper_ast.py                  # Whisper large-v3 + AST tagger, 1 GPU
HF_HUB_CACHE=/projects/data/speechteam/venkata/.cache/hf_models \
  /projects/data/speechteam/venkata/envs/qwen_omni/bin/python scripts/06b_second_opinion_qwen_omni.py   # 1 GPU
.venv/bin/python scripts/07_label_transcripts.py
```

Labels: `[HESITATION: uh|um]` (filled pause), `[PAUSE 0.8s]` (silent pause >= 0.5 s, MFA timings),
`[LAUGHTER]`, `[SWEAR: word]`, `[NON-ENGLISH: ...]`, `[COUGH]`/`[BREATH]`/..., `[CUT OFF]` (recording
starts or ends mid-speech), `th-` (partial word). Words are CrisperWhisper's unless Whisper large-v3
and Qwen3-Omni both heard the same different word(s); spelling-only and contraction-only differences
are ignored. Each change is listed in `review_notes` (9 recordings).

What the three models found: 125 hesitations, 204 pauses >= 0.5 s, 24 partial words, 1 laughter
(Sanjith, table, without rules, 6.9 s), 0 swear words, 0 non-English words. Qwen3-Omni was told to
write swearing in any language uncensored and to tag laughter; it found none, and the AST tagger's
laughter score never exceeds 0.05. If people swore or laughed in these recordings, the models are
missing it: those spots need to be found by ear and added.

Abhijith's recordings begin mid-word (about 0.1 s of digital silence, then speech at full level), and
most end mid-sentence at about 30.5 s; CrisperWhisper heard the clipped "(B)ananas" and "(T)igers" as
"Panas" and "Pigeons". The transcripts now read `[CUT OFF] Bananas ...` / `[CUT OFF] Tigers ...`.

## Results

- `data/transcripts/verbatim_transcripts.csv`: one row per recording: id, participant, session, prompt,
  condition, duration, the labelled `transcript`, counts (words, hesitations, pauses, laughter, swears,
  partial words) and `review_notes`. Also one `.txt` per recording in `data/transcripts/labelled/`.
- `data/second_opinion/`: raw outputs of Whisper large-v3, Qwen3-Omni and the AST tagger.
- `data/transcripts/crisperwhisper/<id>.json|.txt`: per recording, with word timestamps.
- `data/aligned/`: tokens, pauses, TextGrids and `summary.csv`. 45 recordings with speech: 3,376 words,
  125 fillers, 24 partial words, 16 repetitions, 444 pauses; MFA timed at least 99.1% of the tokens in
  every recording.
- Logs: `data/logs/`.
