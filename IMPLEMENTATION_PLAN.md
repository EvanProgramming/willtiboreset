# WillTiboReset Phase 1 Implementation Plan

> This document captures the implementation path for Phases A-D to prevent context loss.
> Goal: Transform the current skeleton with mock fallbacks into a long-running real AI forecasting system.

---

## Final Project Goal

WillTiboReset is a continuously running prediction system:

- Predicts the probability of a ChatGPT/Codex usage-quota reset within the next 5h / 24h / 48h.
- Data-source priority:
  1. Tibo X updates (highest priority)
  2. OpenAI official information
  3. Reddit / community discussions
- The model adapts as more historical data is collected, becoming more accurate over time.

---

## Phase A: Enforce Real Dependencies, Remove Mock Fallbacks

### Goal
Ensure the system no longer defaults to keyword matching or mock data. When a core dependency is missing, fail or warn explicitly.

### Change List

1. **[predict.py] `create_analyzer()`**
   - Remove the automatic `MockLLMAnalyzer` fallback.
   - Raise `RuntimeError` with a clear message when `GEMINI_API_KEY` is missing.

2. **[collectors/community.py] `CommunityCollector.collect()`**
   - Remove unconditional loading of `data/sample_tweets.json`.
   - Load mock data only when the environment variable `USE_MOCK_DATA=true` is set or in test mode.
   - Return an empty list with a warning when `COMMUNITY_RSS_URLS` is empty and mock data is not enabled.

3. **[predict.py] Startup checks**
   - Before collection, check `config.has_gemini_credentials`.
   - Check `config.has_rss_feeds`; warn if empty (empty RSS is allowed but must be explicit).
   - Promote empty `TIBO_RSS_URLS` to an error (core data source must be configured).

4. **Test updates**
   - Remove default test paths that rely on `MockLLMAnalyzer`.
   - Provide mock HTTP / client stubs for Gemini-related tests.

### Acceptance Criteria

- Running `python predict.py` without `GEMINI_API_KEY` fails immediately.
- Running `python predict.py` without `TIBO_RSS_URLS` fails immediately.
- With proper configuration: RSS → Gemini → Model → prediction.json.

---

## Phase B: Data Quality — authority_score and Interval Statistics

### Goal
Ensure every data point and historical event carries credibility / authority information, and enrich statistical features.

### Change List

1. **[model/data_models.py] `Tweet` model**
   - Add field `authority_score: float = Field(default=1.0, ge=0.0, le=1.0)`.

2. **[collectors/rss_base.py] `_entry_to_tweet()`**
   - Subclasses can pass a default `authority_score`.
   - `TiboRSSCollector` authority_score = 1.0
   - `OpenAIRSSCollector` authority_score = 0.9
   - `CommunityCollector` authority_score = 0.5

3. **[analyzer/__init__.py] `AnalysisFeatures`**
   - Add fields:
     - `median_reset_interval_hours: Optional[float] = None`
     - `std_reset_interval_hours: Optional[float] = None`
     - `min_reset_interval_hours: Optional[float] = None`
     - `max_reset_interval_hours: Optional[float] = None`
     - `interval_confidence: float = 0.0` (confidence based on sample size)

4. **[analyzer/__init__.py] `SignalAnalyzer.analyze()`**
   - When historical events >= 2, compute:
     - average / median / std / min / max interval
     - interval_confidence = min(1.0, n / 10) or a formula based on standard error
   - Keep existing output fields for downstream compatibility.

5. **Prediction output**
   - Add interval statistics to the `signals` section of `prediction.json`.

### Acceptance Criteria

- Every `Tweet` contains `authority_score`.
- `SignalAnalyzer` output includes median / std interval.
- Tests cover authority_score per source and interval statistics.

---

## Phase C: Adaptive Bayesian Survival Model and model_state.json

### Goal
Make model parameters and priors no longer hard-coded; compute and update them dynamically from historical data.

### Change List

1. **Add [model/model_state.py] `ModelState` data model**

   ```python
   class ModelState(BaseModel):
       average_interval_hours: float
       median_interval_hours: Optional[float]
       std_interval_hours: Optional[float]
       sample_count: int
       interval_confidence: float
       prior_weight: float
       params: dict[str, float]
       updated_at: datetime
   ```

2. **Add [update_model.py]**
   - Read `data/reset_history.json`.
   - Compute interval statistics.
   - Compute `prior_weight = max(0.0, 1.0 - sample_count / 20)` from sample_count.
   - Fine-tune `beta_time` and other parameters based on data stability.
   - Save `data/model_state.json`.

3. **[model/survival_model.py] `ResetPredictor`**
   - Add constructor argument `model_state_path: Optional[Path] = None`.
   - Load `model_state.json`:
     - If it exists: use its `average_interval_hours`, `params`, and `prior_weight`.
     - If it does not exist: fall back to current hard-coded defaults, but explicitly mark `prior_applied=True`.
   - `build_features()` accepts an `interval_statistics` dict and computes the posterior interval.

4. **[config.py]**
   - Add `model_state_path` property:
     ```python
     @property
     def model_state_path(self) -> Path:
         return self.data_dir / "model_state.json"
     ```

5. **[predict.py]**
   - Call or prompt for `update_model.py` before running.
   - Read `model_state.json` and pass it to `ResetPredictor`.

6. **Test updates**
   - In `test_survival_model.py`, add:
     - Model-state loading test.
     - Prior-weight decay test with sample_count.
     - End-to-end test for update_model.py.

### Acceptance Criteria

- `python update_model.py` successfully generates `data/model_state.json`.
- Predictions use dynamic parameters when model_state is present.
- Reasonable fallback remains when model_state is absent.
- Model output includes `interval_confidence`.

---

## Phase D: GitHub Actions Automation for update_model.py

### Goal
Have GitHub Actions update the model state daily / weekly automatically, enabling continuous self-improvement.

### Change List

1. **Add / modify [.github/workflows/update_model.yml]**

   ```yaml
   name: Update Model State
   on:
     workflow_dispatch:
     schedule:
       - cron: '0 0 * * *'  # Daily at UTC 00:00
   jobs:
     update:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-python@v5
           with:
             python-version: '3.12'
         - run: pip install -r requirements.txt
         - env:
             GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
           run: python update_model.py
         - run: |
             git config user.name "github-actions[bot]"
             git config user.email "github-actions[bot]@users.noreply.github.com"
             git add data/model_state.json
             git diff --staged --quiet || (
               git commit -m "chore: auto-update model state"
               git push
             )
   ```

2. **Modify [.github/workflows/predict.yml]**
   - Optionally add `python update_model.py` before the `Run prediction` step, depending on whether each prediction should refresh the state.
   - Or keep predict read-only for model_state and let the daily workflow handle updates.

3. **Add optional environment variables**
   - `UPDATE_MODEL_SCHEDULE`: controls update frequency.
   - `MIN_SAMPLES_FOR_UPDATE`: minimum sample count required to update.

### Acceptance Criteria

- GitHub Actions can run `update_model.py` successfully.
- `data/model_state.json` is committed automatically.
- `predict.yml` uses the latest model_state when running.

---

## Final Data Flow

```
RSS Feed (Tibo / OpenAI / Community)
        ↓
RSS Collector
        ↓
Tweet {timestamp, source, author, text, url, authority_score}
        ↓
Gemini API (required)
        ↓
SignalScores {reset_signal, limit_signal, release_signal, community_pressure, confidence, reasons}
        ↓
SignalAnalyzer → interval statistics + tweet features
        ↓
ResetPredictor (loads model_state.json)
        ↓
PredictionExplanation {5h, 24h, 48h, hazard_rate, time_ratio, reasons}
        ↓
output/prediction.json
```

---

## Known Issues Before Implementation

- `MockLLMAnalyzer` was the default fallback.
- `CommunityCollector` always loaded `data/sample_tweets.json`.
- `Tweet` had no `authority_score`.
- `SignalAnalyzer` only output the average interval.
- Model parameters `alpha/beta_*` were hard-coded.
- No `update_model.py` / `model_state.json`.
- GitHub Actions only ran predict, not model updates.

---

## Notes

- No frontend development.
- No neural-network training.
- All changes remain interpretable (Bayesian / survival-inspired).
- Comments in English, code in English.
