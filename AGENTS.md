# AGENTS.md

This file provides guidance to Qoder (qoder.com) when working with code in this repository.

## Commands

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

# Run the full pipeline (collect → analyze → predict → output)
python main.py

# Run specific modes
python main.py --status     # Show config and data file status
python main.py --analyze    # Run signal analysis only

# Run all tests
python -m pytest tests/ -v

# Run a single test file or test
python -m pytest tests/test_models.py -v
python -m pytest tests/test_models.py::TestResetEvent::test_json_roundtrip -v
```

## Architecture

The project is a linear prediction pipeline with four stages. Each stage is a Python package at the repo root, orchestrated by `main.py`:

```
collectors → analyzer → model (predictor) → output
```

**Data flow**: `collectors` load raw signals from JSON files in `data/` into pydantic model instances (`Tweet`, `ResetEvent`). `analyzer` extracts `AnalysisFeatures` (tweet counts, time since last reset, avg reset interval). `model/predictor` consumes features + raw data to produce a `PredictionResult`. `output` serializes results to JSON + text files in `output/`.

### Key design decisions

- **No fake prediction logic**: `PlaceholderPredictor.predict()` raises `NotImplementedError`. `LLMPredictor` and `StatisticalPredictor` are stub classes with the same behavior, reserved for future phases. Never add hardcoded/random prediction results — implement real logic or leave the stub.

- **Predictor extension pattern**: All predictors inherit `BasePredictor` (ABC) and implement `predict(tweets, reset_events, horizons) -> PredictionResult`. Override the `model_version` property to identify your predictor. See `model/predictor.py`.

- **Config singleton**: `config.py` creates a module-level `config = Config()` instance. `Config` reads from environment variables via `python-dotenv` at import time. All modules import `from config import config` — there is no config injection or factory.

- **sys.path manipulation**: `main.py` inserts `PROJECT_ROOT` into `sys.path` so that imports like `from model.data_models import ...` work when running `python main.py` directly. Tests rely on the same root-level package imports (pytest's rootdir is the repo root).

- **Pydantic v2 models**: Data models in `model/data_models.py` use pydantic v2. Do not use `json_encoders` in `model_config` (deprecated). Pydantic v2 serializes `datetime` to ISO format by default. Use `model_dump(mode="json")` for JSON-safe dicts and `model_dump_json()` for strings.

- **UTC everywhere**: All `datetime` fields are UTC. The analyzer's `_to_aware()` helper treats naive datetimes as UTC. Pass `timezone.utc` when constructing datetimes.

- **Collector persistence**: `TweetCollector` and `ResetHistoryCollector` each take a `Path` and handle their own JSON load/save. `ResetHistoryCollector.add_event()` is the only write API — it appends and rewrites the entire file.

### Data files

- `data/tweets.json` — array of Tweet objects (`timestamp`, `author`, `text`, `url`)
- `data/reset_history.json` — array of ResetEvent objects (`reset_time`, `source`, `confidence`, `notes`)
- `output/` — generated prediction files (gitignored except `.gitkeep` and `__init__.py`)

### Environment variables

See `.env.example`. Key variables: `TWITTER_BEARER_TOKEN`, `OPENAI_API_KEY`, `PREDICTION_HORIZONS` (default `5,24,48`), `CONFIDENCE_THRESHOLD` (default `0.5`), `DATA_DIR`, `OUTPUT_DIR`.

## Conventions

- Docstrings and comments are written in Chinese.
- The user expects automatic git commit and push after completing work in this repository.
- No frontend code — this is Phase 1 (prediction system only). Phase 5 will add a website.
