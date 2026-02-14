# Bot or Not Challenge - Multi-Tier Bot Detector (EN + FR)

A conservative, multi-tier scoring pipeline to detect bot accounts in social media datasets (Bot or Not Challenge format).  
It supports English (EN) and French (FR) datasets by using the dataset's `"lang"` field and switching language-specific signals.

## What This Repository Contains

- `detector.py`: Main detector and CLI entrypoint.
- `dataset.posts&users.<id>.json`: Input datasets with users + posts.
- `dataset.bots.<id>.txt`: Ground-truth bot IDs (for evaluation on practice sets).
- `detections*.txt`: Output files produced by runs.

## High-Level Flow

The pipeline runs in this order:

1. Load a `dataset.posts&users.*.json` file.
2. Read dataset metadata (`id`, `lang`, totals).
3. Group posts by `author_id`.
4. Score each user with `score_user(...)` across Tier 1, Tier 2, and Tier 3 signals.
5. Flag users whose score is at or above threshold (default `3`).
6. Write detected IDs to a text file (one ID per line).
7. Optionally evaluate against `dataset.bots.*.txt` using challenge scoring.

## Dataset Types (Both Are Supported)

### 1) Posts + Users dataset (`dataset.posts&users.*.json`)

Used for detection. Expected fields include:

- `lang`: Language code (`"en"` or `"fr"`).
- `users`: User objects (used to map IDs to usernames).
- `posts`: Post objects with at least:
  - `author_id`
  - `created_at` (ISO timestamp)
  - `text`

### 2) Bot labels dataset (`dataset.bots.*.txt`)

Used for evaluation only. One bot user ID per line.

## EN/FR Language Handling

Language is read from `data["lang"]` in `detect_bots(...)`, then passed into:

- `score_user(..., dataset_lang=lang, ...)`
- Language-aware tier functions:
  - `tier1a_meta_text(..., lang=...)`
  - `tier2d_just_frequency(..., lang=...)`
  - `tier3c_fun_fact(..., lang=...)`
  - `tier3d_repetitive_opener(..., lang=...)`

Behavior differences:

- EN uses EN phrase/pattern lists.
- FR adds FR phrase/pattern lists and FR regex patterns (for example, `viens/vient de` style usage in Tier 2d).

## Scoring System

Default threshold: `DETECTION_THRESHOLD = 3`

### Tier 1 (near-certain, strongest)

- `T1a_meta_text`: Leaked LLM/meta prompt text.
- `T1b_encoding`: Control-character/garbled text artifacts.

### Tier 2 (strong)

- `T2a_same_second`: Same-second posting bursts.
- `T2b_interval_cv`: Low CV of posting intervals (regular timing).
- `T2c_template`: Zero URL + zero hashtag pattern at high volume.
- `T2d_just_freq`: High "just" (EN) or `viens/vient de` (FR) pattern rate.
- `T2e_zero_engage`: Zero URL + zero mention engagement pattern.

### Tier 3 (supporting)

- `T3a_hashtags`: Elevated hashtag density.
- `T3b_low_url`: Very low URL rate (gated by Tier 2 evidence).
- `T3c_fun_fact`: Repeated "fun fact"/FR equivalent pattern.
- `T3d_rep_opener`: Repetitive opener phrases (known + dynamic detection).
- `T3e_length_uniform`: Uniform post length.
- `T3f_spam_exempt`: Strong negative score to exempt spam-like non-target accounts.

## Conservative Guardrails

The detector includes anti-false-positive rules:

- Tier-3-only safety: if no Tier 1 and no Tier 2 signal fired, force score below threshold.
- Tier 2b corroboration: marginal CV cases are downscored if mention activity is too low.
- Same-second corroboration: same-second bursts alone need stronger corroboration.
- Tier 3 gating: some weak Tier 3 signals count only if Tier 2 evidence exists.

## CLI Usage

Run all known datasets:

```bash
python detector.py all
```

Run a specific alias:

```bash
python detector.py 30
python detector.py 31
python detector.py 32
python detector.py 33
```

Run a custom dataset path:

```bash
python detector.py path/to/dataset.posts&users.custom.json
```

Useful options:

```bash
python detector.py 30 -t 3 -o detections.txt -v
python detector.py 31 -b data/dataset.bots.31.txt
```

Flags:

- `-t, --threshold`: Detection threshold (default `3`).
- `-o, --output`: Output detections file (default `detections.txt`).
- `-b, --bots`: Ground-truth bot file for evaluation (optional).
- `-v, --verbose`: Per-user scoring logs.

## Output Files

- Detection output format: one user ID per line.
- If running multiple datasets (`all`), output is auto-suffixed per dataset ID:
  - `detections.30.txt`
  - `detections.31.txt`
  - `detections.32.txt`
  - `detections.33.txt`

## Evaluation Metrics

When a bots label file is available, evaluation prints:

- TP, FP, FN
- Precision and Recall
- Challenge score: `+4*TP - 2*FP - 1*FN`
- Combined summary when running multiple datasets

## Core Functions (Code Map)

- Dataset/file helpers:
  - `_resolve_dataset_alias(...)`
  - `_default_bots_for_dataset(...)`
  - `_output_path_for_dataset(...)`
- Feature extraction:
  - `parse_timestamps(...)`
  - `compute_intervals(...)`
  - `compute_cv(...)`
- User scoring:
  - `score_user(...)`
- Dataset processing:
  - `detect_bots(...)`
- Evaluation/output:
  - `evaluate(...)`
  - `write_detections(...)`
- Entrypoint:
  - `main()`

## Notes

- The detector is intentionally conservative: it prefers avoiding false positives.
- EN/FR support is in one unified pipeline driven by dataset `lang`, not separate scripts.
