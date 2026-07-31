# AGENTS.md

This file provides guidance to Qoder (qoder.com) when working with code in this repository.

## Commands

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

# Run data collection + LLM signal analysis (collect → save tweets.json → analyze signals)
python -m collectors

# Run the full prediction pipeline (collect → analyze → predict → output)
python main.py

# Run specific modes
python main.py --status     # Show config and data file status
python main.py --analyze    # Run signal analysis only

# Run all tests
python -m pytest tests/ -v

# Run a single test file or test
python -m pytest tests/test_llm_signal.py -v
python -m pytest tests/test_rss_collectors.py::TestBaseRSSCollector::test_entry_to_tweet -v
```

## Architecture

The project has two pipelines sharing the same data models:

**Collection + Signal pipeline** (`python -m collectors`):
```
RSS feeds / mock data → collectors → Tweet[] → save tweets.json
                                              → LLMAnalyzer → SignalScores[]
```

**Prediction pipeline** (`python main.py`):
```
collectors → analyzer (statistical) → model (predictor) → output
```

### Data flow

`collectors` gather raw signals from RSS feeds and mock data into `Tweet` objects. Each Tweet carries a `source` field identifying its origin (`tibo_rss`, `openai_rss`, `community_mock`, etc.). The `analyzer` layer has two components: `SignalAnalyzer` extracts statistical features (counts, time intervals), while `LLMAnalyzer` (Gemini or Mock) converts text into `SignalScores` — structured 0-1 floats for `reset_signal`, `limit_discussion`, `release_signal`, `community_pressure`. The LLM does **not** predict reset; it only extracts features for the future `model/survival_model.py`.

### Key design decisions

- **No fake prediction logic**: `PlaceholderPredictor.predict()` raises `NotImplementedError`. `LLMPredictor` and `StatisticalPredictor` are stub classes with the same behavior. Never add hardcoded/random prediction results.

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

### Environment variables

See `.env.example`. Key variables: `GEMINI_API_KEY`, `GEMINI_MODEL`, `TIBO_RSS_URLS` / `OPENAI_RSS_URLS` / `COMMUNITY_RSS_URLS` (comma-separated), `PREDICTION_HORIZONS` (default `5,24,48`), `CONFIDENCE_THRESHOLD`, `DATA_DIR`, `OUTPUT_DIR`.

## Conventions

- Docstrings and comments are written in Chinese.
- The user expects automatic git commit and push after completing work in this repository.
- No frontend code — prediction system only. A website is planned for a later phase.
- X/Twitter API is intentionally not used as a primary data source. RSS feeds and public web feeds are preferred.
