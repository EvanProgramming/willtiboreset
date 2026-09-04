# WillTiboReset

> Will Tibo reset Codex/ChatGPT usage tomorrow?

Predicts the probability that Tibo/OpenAI will reset ChatGPT/Codex usage quotas within the next **5 hours**, **24 hours**, and **48 hours** using public internet signals.

---

## Project Goals

ChatGPT/Codex usage quotas are reset on a regular or irregular schedule, making it hard for users to anticipate the next reset. WillTiboReset solves this by:

1. **Collecting public signals**: Gathering information related to quota resets from public sources such as RSS feeds and community discussions.
2. **LLM signal extraction**: Using an LLM (Gemini API or keyword matching) to convert natural-language text into structured signal scores.
3. **Survival model prediction**: Computing reset probabilities for each time window based on a Discrete-Time Survival Model.
4. **Automated execution**: Running automatically every hour via GitHub Actions to keep predictions up to date.

No large neural network is used; the model is interpretable and suitable for scenarios with limited historical data.

---

## Data Flow

```
                    ┌──────────────────────────────────────────┐
                    │            Data Collection Layer         │
                    │  collectors/                             │
                    │                                          │
                    │  TiboRSSCollector   → Tibo-related RSS   │
                    │  OpenAIRSSCollector → OpenAI official RSS│
                    │  CommunityCollector → Community RSS + mock│
                    │                                          │
                    │  Output: list[Tweet] (uniform, w/ source)│
                    └────────────────────┬─────────────────────┘
                                         │
                                         ▼
                    ┌──────────────────────────────────────────┐
                    │            Signal Analysis Layer         │
                    │  analyzer/                               │
                    │                                          │
                    │  SignalAnalyzer  → statistical features  │
                    │    (hours_since_reset, avg_interval)     │
                    │                                          │
                    │  LLMAnalyzer     → structured scores     │
                    │    (Gemini API or Mock keyword matching) │
                    │    reset_signal, limit_discussion,       │
                    │    release_signal, community_pressure    │
                    └────────────────────┬─────────────────────┘
                                         │
                                         ▼
                    ┌──────────────────────────────────────────┐
                    │           Prediction Engine              │
                    │  model/survival_model.py                 │
                    │                                          │
                    │  build_features()  → PredictionFeatures  │
                    │  ResetPredictor()  → probability + rationale│
                    └────────────────────┬─────────────────────┘
                                         │
                                         ▼
                              output/prediction.json
```

## Model Pipeline

### 1. Feature Construction

Statistical features and LLM signals are combined into `PredictionFeatures`:

| Input Feature | Source | Description |
|---------|------|------|
| `hours_since_last_reset` | `SignalAnalyzer` | Hours since the last reset |
| `average_reset_interval` | `SignalAnalyzer` | Historical average reset interval |
| `tibo_signal` | `LLMAnalyzer` | `0.6 × reset_signal + 0.4 × limit_discussion` |
| `community_signal` | `LLMAnalyzer` | `community_pressure` |
| `release_signal` | `LLMAnalyzer` | `release_signal` |

### 2. Survival Model (Discrete-Time Survival Model)

Uses a logistic hazard-rate model without a neural network:

**Step 1** — Compute the time ratio:
```
time_ratio = hours_since_last_reset / average_reset_interval
```
A ratio greater than 1 means the average interval has been exceeded, so the likelihood of a reset rises.

**Step 2** — Compute the hourly hazard rate:
```
h = sigmoid(α + β_time × time_ratio + β_tibo × s_tibo + β_community × s_community + β_release × s_release)
```

Default parameters:

| Parameter | Default | Meaning |
|------|--------|------|
| α | -4.0 | Intercept (controls baseline hazard) |
| β_time | 1.5 | Time-ratio weight |
| β_tibo | 1.0 | Tibo reset/limit signal weight |
| β_community | 0.8 | Community-pressure signal weight |
| β_release | 0.5 | Product-release signal weight |

**Step 3** — Compute the window probability:
```
P(reset within T hours) = 1 - (1 - h)^T
```

### 3. Example Output

```json
{
  "updated_at": "2025-07-31T12:00:00+00:00",
  "prediction": {
    "within_5h": 0.42,
    "within_24h": 0.76,
    "within_48h": 0.91
  },
  "confidence": "medium",
  "signals": {
    "tweet_count": 6,
    "hours_since_last_reset": null,
    "average_reset_interval": null,
    "time_ratio": null,
    "hazard_rate": 0.036,
    "tibo_signal": 0.485,
    "community_signal": 0.167,
    "release_signal": 0.200,
    "llm_scores": { ... }
  },
  "reasons": [
    "No historical reset records; prediction uses the default baseline hazard.",
    "A small amount of reset/limit-related discussion was detected (signal strength 0.49)."
  ]
}
```

---

## How to Run

### Local Run

```bash
# 1. Install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. One-command prediction (collect → analyze → predict → output)
python predict.py

# Output file: output/prediction.json
```

### Other Commands

```bash
# Data collection + LLM signal analysis (no prediction)
python -m collectors

# Full prediction pipeline (verbose output)
python main.py

# Run prediction only (using already-collected data)
python main.py --predict

# Show project status only
python main.py --status

# Run signal analysis only
python main.py --analyze

# Run tests
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

### GitHub Actions (Automated)

The project is configured with `.github/workflows/predict.yml`:

- **Scheduled runs**: `python predict.py` executes automatically every hour.
- **Manual trigger**: Trigger manually from the GitHub repository Actions page.
- **Runtime state**: The workflow restores the last deployed JSON state, runs the normal prediction and build steps, and republishes the updated state through the Pages artifact. Generated runtime data is not committed to the source repository.

To enable Gemini API analysis, add `GEMINI_API_KEY` in the repository Settings → Secrets.

---

## API Configuration

All configuration is managed through environment variables or an `.env` file. Copy `.env.example` to `.env` and fill in the values as needed:

```bash
cp .env.example .env
```

### Environment Variables

| Variable | Required | Default | Description |
|--------|------|--------|------|
| `GEMINI_API_KEY` | No | Empty | Gemini API key; enables LLM semantic analysis. Leave empty to use Mock keyword matching. |
| `GEMINI_MODEL` | No | `gemini-2.0-flash` | Gemini model name |
| `TIBO_RSS_URLS` | No | Empty | Comma-separated Tibo RSS URLs |
| `OPENAI_RSS_URLS` | No | Empty | Comma-separated OpenAI RSS URLs |
| `COMMUNITY_RSS_URLS` | No | Empty | Comma-separated community RSS URLs |
| `PREDICTION_HORIZONS` | No | `5,24,48` | Prediction time windows in hours |
| `DATA_DIR` | No | `data` | Data storage directory |
| `OUTPUT_DIR` | No | `output` | Output directory |

> **Runs without any API key**: If RSS feeds are not configured, mock data from `data/sample_tweets.json` is used; if the Gemini API key is not configured, the Mock keyword-matching analyzer is used.

### GitHub Actions Secrets

Add the following in the repository Settings → Secrets and variables → Actions:

| Secret Name | Description |
|-------------|------|
| `GEMINI_API_KEY` | Gemini API key (optional) |
| `TIBO_RSS_URLS` | Tibo RSS URLs (optional) |
| `OPENAI_RSS_URLS` | OpenAI RSS URLs (optional) |
| `COMMUNITY_RSS_URLS` | Community RSS URLs (optional) |

---

## Project Structure

```
willtiboreset/
├── predict.py              # One-command prediction entry (collect → analyze → predict → output)
├── main.py                 # CLI entry point (verbose output)
├── config.py               # Configuration system (.env / environment variables)
├── requirements.txt        # Runtime dependencies
├── .env.example            # Environment variable example
│
├── .github/workflows/
│   └── predict.yml         # GitHub Actions (auto-predict every 10 minutes)
│
├── model/                  # Data models and predictors
│   ├── data_models.py      # ResetEvent, Tweet, SignalScores, PredictionFeatures
│   ├── predictor.py        # Predictor framework (BasePredictor)
│   └── survival_model.py   # ResetPredictor — survival-model prediction engine
│
├── collectors/             # Data collectors
│   ├── rss_base.py         # RSS base collector (parsing / deduplication)
│   ├── tibo_rss.py         # Tibo RSS collector
│   ├── openai_rss.py       # OpenAI RSS collector
│   ├── community.py        # Community signal collector (RSS + mock)
│   └── __main__.py         # Entry point: python -m collectors
│
├── analyzer/               # Signal analyzers
│   ├── __init__.py         # SignalAnalyzer (statistical features)
│   └── llm_signal.py       # LLMAnalyzer (Gemini + Mock)
│
├── output/                 # Outputs
│   └── prediction.json     # Final prediction result (auto-generated / updated)
│
├── data/                   # Data storage
│   ├── reset_history.json  # Historical reset events
│   ├── tweets.json         # Collected tweets
│   └── sample_tweets.json  # Mock data used when RSS is unavailable
│
└── tests/                  # Tests (128)
    ├── test_models.py
    ├── test_config.py
    ├── test_analyzer.py
    ├── test_collectors.py
    ├── test_rss_collectors.py
    ├── test_llm_signal.py
    └── test_survival_model.py
```

---

## Technology Choices

| Component | Choice | Reason |
|------|------|------|
| Data models | Pydantic v2 | Type safety, JSON serialization, validation |
| RSS parsing | feedparser | Mature and stable RSS/Atom parsing library |
| LLM signal analysis | Gemini API (optional) | Generous free tier, low latency |
| Prediction model | Logistic Hazard Model | Interpretable, low data requirements, no training |
| Configuration | python-dotenv | Environment variable management with `.env` support |
| CI/CD | GitHub Actions | Free and natively integrated with the Git repository |
| Data sources | RSS feeds | No dependency on the X/Twitter API; simple configuration |

---

## Roadmap

| Phase | Content | Status |
|-------|------|------|
| Phase 1 | Project structure, data models, full collect/analyze/predict/output pipeline | ✅ Complete |
| Phase 2 | RSS data collection layer + LLM signal analysis layer | ✅ Complete |
| Phase 3 | Core prediction engine — Discrete-Time Survival Model | ✅ Complete |
| Phase 4 | StatisticalPredictor — statistical model prediction | Planned |
| Phase 5 | Frontend display (website) | Planned |

## License

MIT
