# WillTiboReset Phase 1 Acceptance Report

**Date:** 2026-08-01
**Tester:** AI Engineering Acceptance Test
**Scope:** Verify the complete pipeline `Real RSS Data → Gemini LLM → Adaptive Bayesian Survival Model → 5h/24h/48h Prediction → output/prediction.json`

---

## Overall Status

**FAIL (Configuration Blocked)**

Phase 1 does **not** PASS under the acceptance criterion that the full real-data pipeline must execute end-to-end. The local runtime environment lacks all required secrets (`GEMINI_API_KEY`, `TIBO_RSS_URLS`, etc.), so `python predict.py` fails at startup validation before any real RSS fetch or Gemini call can occur.

The failure is **not a code defect** — the code correctly rejects missing credentials instead of falling back to mock data. Once the secrets are configured in `.env` or GitHub Actions, the pipeline is structurally ready to run.

---

## Score

| Dimension        | Score | Notes                                                                 |
|------------------|-------|-----------------------------------------------------------------------|
| Architecture     | 9/10  | Clean separation: collectors → analyzer → LLM → survival model → output. Minor inconsistency: `collectors/__main__.py` still falls back to `MockLLMAnalyzer`. |
| Data Pipeline    | 6/10  | RSS collectors, deduplication, and persistence are implemented, but could not be exercised with real feeds locally. |
| LLM Integration  | 5/10  | `GeminiAnalyzer` is wired and required; no real API call was performed due to missing key. `predict.py` correctly errors instead of mocking. |
| Prediction Model | 9/10  | Adaptive Bayesian Survival Model behaves correctly across scenarios; all unit tests pass; monotonicity holds. |
| Automation       | 8/10  | GitHub Actions workflows are valid, use the correct secrets, and can run unattended. Cannot verify live run without secrets. |

**Total: 37/50**

---

## 1. Environment & Configuration Acceptance

### Checklist

| Config            | Expected            | Actual                          | Status |
|-------------------|---------------------|---------------------------------|--------|
| Gemini API        | Required, no mock fallback | `GEMINI_API_KEY` not set locally; code raises `RuntimeError` when missing | **FAIL** |
| Tibo RSS          | Required            | `TIBO_RSS_URLS` not set locally; code raises `RuntimeError` when missing | **FAIL** |
| OpenAI RSS        | Optional            | `OPENAI_RSS_URLS` not set locally | **FAIL** |
| Community RSS     | Optional            | `COMMUNITY_RSS_URLS` not set locally | **FAIL** |

### Evidence

Local environment check:

```text
GEMINI_API_KEY: NOT SET
GEMINI_MODEL: NOT SET
TIBO_RSS_URLS: NOT SET
OPENAI_RSS_URLS: NOT SET
COMMUNITY_RSS_URLS: NOT SET
USE_MOCK_DATA: NOT SET
```

`.env` file does not exist; only `.env.example` is present.

### Fallback Audit

- `predict.py` [`validate_configuration()`](file:///Users/evangong/Documents/Programming/AI/willtiboreset/predict.py#L104-L113) explicitly raises `RuntimeError` if `GEMINI_API_KEY` or `TIBO_RSS_URLS` are missing. **No silent mock fallback.**
- `collectors/__main__.py` [`_create_analyzer()`](file:///Users/evangong/Documents/Programming/AI/willtiboreset/collectors/__main__.py#L57-L68) still falls back to `MockLLMAnalyzer` when `GEMINI_API_KEY` is missing. This is inconsistent with Phase A requirements and should be aligned with `predict.py`.

---

## 2. Data Collection Acceptance

### Execution

```bash
python -m collectors
```

### Output

```text
TiboRSS:       0 条
OpenAI RSS:    0 条
Community:     0 条
去重后总计:    0 条
已保存到: data/tweets.json
⚠ 无数据可分析
```

### Assessment

- **Real data / Mock data:** No data was collected because no RSS URLs are configured.
- **Mock usage:** `CommunityCollector` only loads `data/sample_tweets.json` when `USE_MOCK_DATA=true`. In this run it was not enabled, so no mock data was injected.
- **Implementation status:** Multiple RSS URLs, feed parsing, URL/text deduplication, and persistence are implemented in [`collectors/rss_base.py`](file:///Users/evangong/Documents/Programming/AI/willtiboreset/collectors/rss_base.py) and [`collectors/community.py`](file:///Users/evangong/Documents/Programming/AI/willtiboreset/collectors/community.py).

### Verdict

**FAIL** for real-data validation (blocked by missing RSS URLs); **PASS** for implementation correctness and no silent mock fallback.

---

## 3. Gemini LLM Acceptance

### Execution

Not executed because the prerequisite RSS collection produced zero items and `GEMINI_API_KEY` is absent.

### Code Verification

- [`analyzer/llm_signal.py`](file:///Users/evangong/Documents/Programming/AI/willtiboreset/analyzer/llm_signal.py) implements `GeminiAnalyzer.analyze_tweets()` and `analyze_batch()`, returning `SignalScores` with fields:
  - `reset_signal`
  - `limit_discussion`
  - `release_signal`
  - `community_pressure`
  - `confidence`
  - `reason`
- `predict.py` will not proceed without a real `GEMINI_API_KEY`.

### Verdict

**FAIL** for real API validation; **PASS** for no keyword-matching fallback in the production entrypoint.

---

## 4. Adaptive Bayesian Survival Model Acceptance

### Implementation Check

The model in [`model/survival_model.py`](file:///Users/evangong/Documents/Programming/AI/willtiboreset/model/survival_model.py) includes:

- `reset_history` consumption via `SignalAnalyzer` and `update_model.py`
- Prior interval `DEFAULT_RESET_INTERVAL_HOURS = 48.0`
- Bayesian shrinkage in `_compute_posterior_interval()`
- `time_ratio = hours_since_last_reset / average_reset_interval`
- Logistic hazard rate: `h = sigmoid(α + β_time·time_ratio + β_tibo·s_tibo + β_community·s_community + β_release·s_release)`
- Window probability: `P = 1 - (1 - h)^T`

### Scenario Validation

| Scenario | hours_since | time_ratio | 5h prob | 24h prob | 48h prob |
|----------|-------------|------------|---------|----------|----------|
| Just reset | 1 | 0.02 | 0.0894 | 0.3619 | 0.5929 |
| Near average interval | 45 | 0.94 | 0.3026 | 0.8227 | 0.9686 |
| Beyond average interval | 80 | 1.67 | 0.6347 | 0.9920 | 0.9999 |

### Checks

- **Monotonicity:** `within_5h <= within_24h <= within_48h` holds for all scenarios.
- **Trend:** probabilities strictly increase from Scenario 1 → 2 → 3 as expected.

### Verdict

**PASS**

---

## 5. End-to-End Test

### Execution

```bash
python predict.py
```

### Output

```text
RuntimeError: GEMINI_API_KEY 未配置。请在 .env 文件或 GitHub Actions Secrets 中设置。
```

### Assessment

The pipeline intentionally fails fast at configuration validation. No `output/prediction.json` was generated because real credentials are absent.

### Verdict

**FAIL** — real E2E execution blocked by missing configuration.

---

## 6. Adaptive Capability Check

### Mechanism

- [`update_model.py`](file:///Users/evangong/Documents/Programming/AI/willtiboreset/update_model.py) reads `data/reset_history.json`, computes interval statistics, and writes `data/model_state.json`.
- [`model/model_state.py`](file:///Users/evangong/Documents/Programming/AI/willtiboreset/model/model_state.py) defines `ModelState` and `ModelStateManager`.
- `ResetPredictor` loads `model_state.json` and uses adaptive `params` and `average_interval_hours`.
- Prior weight decays with sample count: `prior_weight = max(0.0, 1.0 - sample_count / 20)`.

### Execution

```bash
python update_model.py
```

### Output

```text
样本 interval 数: 10
后验平均间隔: 58.96h
中位间隔: 42.92
标准差: 83.5
间隔置信度: 60%
先验权重: 50.00%
参数: {'alpha': -4.0, 'beta_time': 1.62, 'beta_tibo': 1.0, 'beta_community': 0.8, 'beta_release': 0.5}
```

### Verdict

**PASS**

---

## 7. GitHub Actions Acceptance

### Workflows

- [`.github/workflows/predict.yml`](file:///Users/evangong/Documents/Programming/AI/willtiboreset/.github/workflows/predict.yml): runs every 10 minutes; calls `update_model.py` then `predict.py`; uses `secrets.GEMINI_API_KEY`, `secrets.TIBO_RSS_URLS`, etc.; commits `output/prediction.json` and `data/model_state.json`.
- [`.github/workflows/update_model.yml`](file:///Users/evangong/Documents/Programming/AI/willtiboreset/.github/workflows/update_model.yml): runs daily at UTC 00:00; calls `update_model.py`; commits `data/model_state.json`.

### Validation

- YAML syntax was reviewed; workflow structure is correct.
- Secret names match the expected configuration keys.
- `permissions: contents: write` is set for automated commits.
- No `actionlint` available locally for deeper static analysis.

### Verdict

**PASS** (structural). Live run can only be confirmed after secrets are configured in the GitHub repository.

---

## Completed

- [x] `predict.py` enforces real `GEMINI_API_KEY` and `TIBO_RSS_URLS`.
- [x] RSS collectors support multiple feeds, deduplication, and structured `Tweet` output.
- [x] `Tweet` includes `authority_score` (Tibo=1.0, OpenAI=0.9, Community=0.5).
- [x] `SignalAnalyzer` outputs interval statistics (median, std, min, max, confidence).
- [x] Adaptive Bayesian Survival Model with `time_ratio`, hazard rate, and window probabilities.
- [x] `update_model.py` + `model_state.json` adaptive update mechanism.
- [x] GitHub Actions workflows for prediction and model-state refresh.
- [x] 142 unit tests pass.

---

## Problems Found

| # | Problem | Severity | Location |
|---|---------|----------|----------|
| 1 | **Local secrets missing** — real E2E cannot run. | Blocker | Environment |
| 2 | **`collectors/__main__.py` still falls back to `MockLLMAnalyzer`** when `GEMINI_API_KEY` is missing. This contradicts the Phase A "no silent mock fallback" rule enforced in `predict.py`. | Medium | [`collectors/__main__.py`](file:///Users/evangong/Documents/Programming/AI/willtiboreset/collectors/__main__.py#L57-L68) |
| 3 | **No live validation** of RSS fetch, Gemini API call, or GitHub Actions run. | Medium | Deployment |

---

## Before Phase 2

Before starting the React website (Phase 2), the following must be resolved:

1. **Configure all required secrets** in GitHub Actions:
   - `GEMINI_API_KEY`
   - `TIBO_RSS_URLS`
   - `OPENAI_RSS_URLS` (optional but recommended)
   - `COMMUNITY_RSS_URLS` (optional)
2. **Run `python predict.py` in GitHub Actions** and confirm `output/prediction.json` is generated with real data.
3. **Fix `collectors/__main__.py`** to reject missing `GEMINI_API_KEY` instead of falling back to `MockLLMAnalyzer`, matching `predict.py` behavior.
4. **Add at least one real Tibo RSS URL** and verify RSS collection returns non-zero tweets.
5. **Confirm `update_model.yml` runs successfully** and commits updated `data/model_state.json`.
6. (Recommended) Add integration tests that exercise RSS parsing and model prediction with real or recorded data.

---

## Conclusion

WillTiboReset Phase 1 is **architecturally complete and individually component-tested**, but it has **not yet been proven end-to-end with real data** because the local environment lacks the required secrets. The system correctly refuses to run without real credentials, which is the intended behavior. Once secrets are provided, the expected Phase 1 PASS criteria should be re-evaluated by running `python predict.py` and inspecting `output/prediction.json`.
