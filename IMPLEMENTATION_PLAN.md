# WillTiboReset Phase 1 实施计划

> 本文档用于固化 Phase A-D 的实施路径，防止上下文丢失。
> 目标：把当前带有 mock fallback 的骨架改造为长期运行的真实 AI forecasting system。

---

## 项目最终目标

WillTiboReset 是一个持续运行的预测系统：

- 预测未来 5h / 24h / 48h 内 ChatGPT/Codex 使用额度 reset 概率
- 数据来源优先级：
  1. Tibo X 动态（最高优先级）
  2. OpenAI 官方信息
  3. Reddit / 社区讨论
- 模型随历史数据增加而自适应，越跑越准

---

## Phase A：强制真实依赖，移除 mock fallback

### 目标
让系统不再默认使用 keyword matching 或 mock 数据。所有核心依赖缺失时明确报错或 warning。

### 改动清单

1. **[predict.py] `create_analyzer()`**
   - 删除 `MockLLMAnalyzer` 自动 fallback
   - `GEMINI_API_KEY` 缺失时抛出 `RuntimeError` 并给出清晰提示

2. **[collectors/community.py] `CommunityCollector.collect()`**
   - 删除无条件加载 `data/sample_tweets.json`
   - mock 数据仅在显式环境变量 `USE_MOCK_DATA=true` 或测试模式下加载
   - 配置 `COMMUNITY_RSS_URLS` 为空且未启用 mock 时，返回空列表并 warning

3. **[predict.py] 启动检查**
   - 收集前检查 `config.has_gemini_credentials`
   - 检查 `config.has_rss_feeds`，若为空则 warning（允许空 RSS 但需显式提示）
   - `TIBO_RSS_URLS` 为空时提升为 error（核心数据源必须配置）

4. **测试更新**
   - 删除依赖 `MockLLMAnalyzer` 的默认测试路径
   - 为 Gemini 相关测试提供 mock HTTP/client stub

### 验收标准

- 未配置 `GEMINI_API_KEY` 时运行 `python predict.py` 直接报错
- 未配置 `TIBO_RSS_URLS` 时运行 `python predict.py` 报错
- 配置后流程：RSS → Gemini → Model → prediction.json

---

## Phase B：数据质量 — authority_score 与 interval statistics

### 目标
让每条数据和每个历史事件都携带可信度/权威性信息，并让统计特征更丰富。

### 改动清单

1. **[model/data_models.py] `Tweet` 模型**
   - 新增字段 `authority_score: float = Field(default=1.0, ge=0.0, le=1.0)`

2. **[collectors/rss_base.py] `_entry_to_tweet()`**
   - 子类可传入默认 authority_score
   - `TiboRSSCollector` authority_score = 1.0
   - `OpenAIRSSCollector` authority_score = 0.9
   - `CommunityCollector` authority_score = 0.5

3. **[analyzer/__init__.py] `AnalysisFeatures`**
   - 新增字段：
     - `median_reset_interval_hours: Optional[float] = None`
     - `std_reset_interval_hours: Optional[float] = None`
     - `min_reset_interval_hours: Optional[float] = None`
     - `max_reset_interval_hours: Optional[float] = None`
     - `interval_confidence: float = 0.0`（基于样本量的置信度）

4. **[analyzer/__init__.py] `SignalAnalyzer.analyze()`**
   - 在历史事件 ≥ 2 时计算：
     - average / median / std / min / max interval
     - interval_confidence = min(1.0, n / 10) 或基于标准误差的公式
   - 输出时保留现有字段以兼容下游

5. **预测输出**
   - `prediction.json` 的 `signals` 区域增加 interval statistics

### 验收标准

- 每条 `Tweet` 包含 `authority_score`
- `SignalAnalyzer` 输出包含 median / std interval
- 测试覆盖不同 source 的 authority_score 和 interval statistics

---

## Phase C：Adaptive Bayesian Survival Model 与 model_state.json

### 目标
让模型参数和先验不再硬编码，而是根据历史数据动态计算和更新。

### 改动清单

1. **新增 [model/model_state.py] `ModelState` 数据模型**

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

2. **新增 [update_model.py]**
   - 读取 `data/reset_history.json`
   - 计算 interval statistics
   - 根据 sample_count 计算 `prior_weight = max(0.0, 1.0 - sample_count / 20)`
   - 根据数据稳定性微调 `beta_time` 等参数
   - 保存 `data/model_state.json`

3. **[model/survival_model.py] `ResetPredictor`**
   - 构造函数增加 `model_state_path: Optional[Path] = None`
   - 加载 `model_state.json`：
     - 若存在：使用其中的 `average_interval_hours`、`params`、`prior_weight`
     - 若不存在：退化为当前硬编码默认值，但明确标记 `prior_applied=True`
   - `build_features()` 接受 `interval_statistics` 字典，计算 posterior interval

4. **[config.py]**
   - 新增 `model_state_path` 属性：
     ```python
     @property
     def model_state_path(self) -> Path:
         return self.data_dir / "model_state.json"
     ```

5. **[predict.py]**
   - 运行前调用或提示需要 `update_model.py`
   - 读取 `model_state.json` 并传入 `ResetPredictor`

6. **测试更新**
   - `test_survival_model.py` 增加：
     - model_state 加载测试
     - prior weight 随 sample_count 衰减测试
     - update_model.py 端到端测试

### 验收标准

- `python update_model.py` 成功生成 `data/model_state.json`
- 有 model_state 时预测使用动态参数
- 无 model_state 时仍有合理 fallback
- 模型输出包含 `interval_confidence`

---

## Phase D：GitHub Actions 自动化 update_model.py

### 目标
让 GitHub Actions 每日/每周自动更新模型状态，实现持续自我改进。

### 改动清单

1. **新增/修改 [.github/workflows/update_model.yml]**

   ```yaml
   name: Update Model State
   on:
     workflow_dispatch:
     schedule:
       - cron: '0 0 * * *'  # 每天 UTC 00:00
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

2. **修改 [.github/workflows/predict.yml]**
   - 在 `Run prediction` 步骤前增加 `python update_model.py`（可选，取决于是否希望每次预测前刷新）
   - 或者保持 predict 只读取 model_state，由每日 workflow 负责更新

3. **新增环境变量（可选）**
   - `UPDATE_MODEL_SCHEDULE`：控制更新频率
   - `MIN_SAMPLES_FOR_UPDATE`：最少样本数才更新

### 验收标准

- GitHub Actions 能成功运行 `update_model.py`
- `data/model_state.json` 被自动提交
- `predict.yml` 运行时使用最新 model_state

---

## 数据流最终形态

```
RSS Feed (Tibo / OpenAI / Community)
        ↓
RSS Collector
        ↓
Tweet {timestamp, source, author, text, url, authority_score}
        ↓
Gemini API (必须)
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

## 当前已知问题（实施前状态）

- `MockLLMAnalyzer` 是默认 fallback
- `CommunityCollector` 始终加载 `data/sample_tweets.json`
- `Tweet` 无 `authority_score`
- `SignalAnalyzer` 只输出 average interval
- 模型参数 `alpha/beta_*` 硬编码
- 无 `update_model.py` / `model_state.json`
- GitHub Actions 只运行 predict，不更新模型

---

## 备注

- 不开发前端
- 不训练神经网络
- 所有改动保持可解释性（Bayesian / survival-inspired）
- 中文注释、英文代码
