# AGENTS.md

This file provides guidance to Qoder (qoder.com) when working with code in this repository.

## Commands

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

# One-command prediction (collect → analyze → predict → output/prediction.json)
python predict.py

# Run data collection + LLM signal analysis (collect → save tweets.json → analyze signals)
python -m collectors

# Run the full prediction pipeline with verbose output
python main.py

# Run specific modes
python main.py --status     # Show config and data file status
python main.py --analyze    # Run signal analysis only (no prediction)
python main.py --predict    # Run prediction only (uses collected data)

# Run all tests
python -m pytest tests/ -v

# Run a single test file or test
python -m pytest tests/test_survival_model.py -v
python -m pytest tests/test_llm_signal.py -v
```

GitHub Actions workflow at `.github/workflows/predict.yml` runs `python predict.py` every 10 minutes via cron and auto-commits `output/prediction.json`. Also supports `workflow_dispatch` for manual triggers.

## Architecture

The project has two pipelines sharing the same data models:

**Collection + Signal pipeline** (`python -m collectors`):
```
RSS feeds / mock data → collectors → Tweet[] → save tweets.json
                                              → LLMAnalyzer → SignalScores[]
```

**Prediction pipeline** (`python predict.py` or `python main.py`):
```
collectors → analyzer (statistical) → LLMAnalyzer → build_features → ResetPredictor → output/prediction.json
```

`predict.py` is the unified entry point used by GitHub Actions. `main.py` provides the same pipeline with verbose CLI output and additional modes (`--status`, `--analyze`, `--predict`).

### Data flow

`collectors` gather raw signals from RSS feeds and mock data into `Tweet` objects. Each Tweet carries a `source` field identifying its origin (`tibo_rss`, `openai_rss`, `community_mock`, etc.). The `analyzer` layer has two components: `SignalAnalyzer` extracts statistical features (counts, time intervals), while `LLMAnalyzer` (Gemini or Mock) converts text into `SignalScores` — structured 0-1 floats for `reset_signal`, `limit_discussion`, `release_signal`, `community_pressure`. The LLM does **not** predict reset; it only extracts features.

`model/survival_model.py` contains `ResetPredictor`, a Discrete-Time Survival Model that takes `PredictionFeatures` (time ratio + LLM signals combined via `build_features()`) and outputs `PredictionExplanation` with 5h/24h/48h probabilities and human-readable reasons. The model uses a logistic hazard rate: `h = sigmoid(α + β_time × time_ratio + β_tibo × s_tibo + β_community × s_community + β_release × s_release)`, then `P(within T) = 1 - (1-h)^T`. Default parameters: α=-4.0, β_time=1.5, β_tibo=1.0, β_community=0.8, β_release=0.5. Parameters and horizons are customizable in the constructor.

### Key design decisions

- **No fake prediction logic**: `PlaceholderPredictor.predict()` raises `NotImplementedError`. The real prediction is in `ResetPredictor` (survival_model.py) which uses a principled logistic hazard model — not hardcoded/random results.

- **Unified Collector interface**: All collectors inherit `BaseCollector` and return `list[Tweet]`. Downstream modules never depend on specific data sources. `BaseRSSCollector` provides shared RSS parsing/dedup logic; `TiboRSSCollector`, `OpenAIRSSCollector`, and `CommunityCollector` are thin subclasses.

- **RSS URLs are configurable, not hardcoded**: Feed URLs come from `config.rss_feeds` (env vars `TIBO_RSS_URLS`, `OPENAI_RSS_URLS`, `COMMUNITY_RSS_URLS`). Empty by default — `python -m collectors` works via mock data from `data/sample_tweets.json`.

- **LLM analyzer dual mode**: `GeminiAnalyzer` calls the Gemini API (lazy import of `google-generativeai`). `MockLLMAnalyzer` uses keyword matching with no API key needed. Both return `list[SignalScores]` with identical structure. `__main__.py` auto-selects based on `config.has_gemini_credentials`.

- **Config singleton**: `config.py` creates a module-level `config = Config()` instance. All modules import `from config import config`.

- **Pydantic v2 models**: Do not use `json_encoders` (deprecated). Use `model_dump(mode="json")` for JSON-safe dicts. `Tweet.source` defaults to `"unknown"`. `SignalScores.to_features()` returns a `dict[str, float]` for survival model consumption.

- **UTC everywhere**: All `datetime` fields are UTC. The analyzer's `_to_aware()` helper treats naive datetimes as UTC.

- **Circular import avoidance**: `collectors/__init__.py` defines `BaseCollector` first, then imports RSS collector subclasses at the bottom of the file. `rss_base.py` imports `BaseCollector` from `collectors`.

### Data files

- `data/tweets.json` — collected Tweet objects (`timestamp`, `author`, `text`, `source`, `url`)
- `data/sample_tweets.json` — mock data for testing (6 cases: reset discussions, product updates, irrelevant)
- `data/reset_history.json` — historical ResetEvent objects
- `output/signal_analysis.json` — LLM signal analysis results (gitignored)
- `output/prediction_latest.json` — latest prediction output from `main.py --predict` (gitignored)
- `output/prediction.json` — final prediction output from `predict.py`, **tracked by git** (auto-committed by GitHub Actions). Format: `{updated_at, prediction: {within_5h, within_24h, within_48h}, confidence, signals, reasons}`

### Environment variables

See `.env.example`. Key variables: `GEMINI_API_KEY`, `GEMINI_MODEL`, `TIBO_RSS_URLS` / `OPENAI_RSS_URLS` / `COMMUNITY_RSS_URLS` (comma-separated), `PREDICTION_HORIZONS` (default `5,24,48`), `CONFIDENCE_THRESHOLD`, `DATA_DIR`, `OUTPUT_DIR`.

## Conventions

- Docstrings and comments are written in Chinese.
- The user expects automatic git commit and push after completing work in this repository.
- No frontend code — prediction system only. A website is planned for a later phase.
- X/Twitter API is intentionally not used as a primary data source. RSS feeds and public web feeds are preferred.
- The survival model (`ResetPredictor`) is intentionally interpretable (logistic hazard, not neural network) because historical reset data is sparse. Each factor's contribution can be traced in `reasons`.
- `build_features()` maps `SignalScores` to `PredictionFeatures`: `tibo_signal = 0.6 × reset_signal + 0.4 × limit_discussion`, `community_signal = community_pressure`, `release_signal = release_signal`. Values are averaged across all SignalScores.
