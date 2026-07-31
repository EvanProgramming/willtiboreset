# WillTiboReset

> Will Tibo reset Codex/ChatGPT usage tomorrow?

通过公开互联网信号预测 Tibo/OpenAI 是否会在未来 **5 小时**、**24 小时**、**48 小时** 内重置 ChatGPT/Codex 使用额度。

## 当前状态：Phase 3

- **Phase 1** ✅ 项目结构、数据模型、收集/分析/输出管道框架
- **Phase 2** ✅ RSS 数据采集层 + LLM 信号分析层
- **Phase 3** ✅ 核心预测引擎（Discrete-Time Survival Model）
  - 可解释的 logistic hazard rate 模型
  - 输入：时间特征 + LLM 信号 → 输出：5h/24h/48h reset 概率
  - 模型解释：返回驱动概率的关键原因
  - 可自定义参数和预测窗口

---

## 项目结构

```
willtiboreset/
├── main.py                 # 入口文件，CLI 命令
├── config.py               # 配置系统（.env / 环境变量 / RSS_CONFIG）
├── requirements.txt        # 运行依赖
├── requirements-dev.txt    # 开发依赖（pytest）
├── .env.example            # 环境变量示例
│
├── model/                  # 数据模型与预测器
│   ├── __init__.py         # 统一导出
│   ├── data_models.py      # ResetEvent, Tweet, SignalScores, PredictionFeatures, PredictionExplanation
│   ├── predictor.py        # 预测器框架（BasePredictor + 占位实现）
│   └── survival_model.py   # ResetPredictor — 核心生存模型预测引擎
│
├── collectors/             # 数据收集器
│   ├── __init__.py         # BaseCollector, TweetCollector, ResetHistoryCollector
│   ├── rss_base.py         # RSS 基础收集器（解析/去重）
│   ├── tibo_rss.py         # Tibo RSS 收集器
│   ├── openai_rss.py       # OpenAI RSS 收集器
│   ├── community.py        # 社区信号收集器（RSS + mock）
│   └── __main__.py         # 入口: python -m collectors
│
├── analyzer/               # 信号分析器
│   ├── __init__.py         # SignalAnalyzer, AnalysisFeatures
│   └── llm_signal.py       # LLM 信号分析器（Gemini + Mock）
│
├── output/                 # 输出格式化
│   └── __init__.py         # OutputFormatter（JSON + 文本）
│
├── data/                   # 数据存储
│   ├── reset_history.json  # 历史重置事件
│   ├── tweets.json         # 收集的推文（运行后生成）
│   └── sample_tweets.json  # 测试/mock 数据
│
└── tests/                  # 测试
    ├── test_models.py
    ├── test_config.py
    ├── test_analyzer.py
    ├── test_collectors.py
    ├── test_rss_collectors.py
    ├── test_llm_signal.py
    └── test_survival_model.py
```

## 架构概览

### 数据采集 & LLM 信号分析（`python -m collectors`）

```
RSS Feeds / Mock 数据
        │
        ▼
  ┌─────────────┐
  │  collectors  │  TiboRSS / OpenAIRSS / Community
  └──────┬──────┘  → list[Tweet] (统一结构)
         │
    保存 tweets.json
         │
         ▼
  ┌─────────────┐
  │ LLMAnalyzer  │  Gemini / Mock
  └──────┬──────┘  → list[SignalScores]
         │
         ▼
  SignalScores: reset_signal, limit_discussion,
                release_signal, community_pressure
```

### 预测管道（`python main.py`）

```
  ┌─────────────┐
  │  collectors  │  收集原始信号 → Tweet / ResetEvent
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │   analyzer   │  统计特征 → AnalysisFeatures
  │ (statistical)│  (hours_since_reset, avg_interval)
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │ LLMAnalyzer  │  信号分析 → SignalScores
  │ (Gemini/Mock)│  (reset_signal, limit_discussion, ...)
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │ build_features│ 合并 → PredictionFeatures
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │ResetPredictor│  生存模型 → PredictionExplanation
  │(survival)    │  (5h/24h/48h 概率 + 解释)
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │   output     │  格式化输出 → JSON
  └─────────────┘
```

### 生存模型原理

模型使用 **Discrete-Time Survival Model**（离散时间生存模型），基于 logistic hazard rate：

1. **基线 hazard**：由时间比率驱动
   - `time_ratio = hours_since_last_reset / average_reset_interval`
   - 比率 > 1（超过平均间隔）→ hazard 显著上升

2. **信号调整**：LLM 信号通过 logistic 线性组合调整 hazard
   - `tibo_signal` → reset/limit 讨论，直接推高 hazard
   - `community_signal` → 社区压力，间接推高 hazard
   - `release_signal` → 产品发布信号，轻微推高 hazard

3. **每小时 hazard rate**：
   `h = sigmoid(α + β_time × time_ratio + β_tibo × s_tibo + β_community × s_community + β_release × s_release)`

4. **窗口概率**：`P(reset within T hours) = 1 - (1 - h)^T`

默认参数：α=-4.0, β_time=1.5, β_tibo=1.0, β_community=0.8, β_release=0.5

## 数据模型

### ResetEvent
历史重置事件记录。

| 字段 | 类型 | 说明 |
|------|------|------|
| `reset_time` | `datetime` | 重置发生时间（UTC） |
| `source` | `SignalSource` | 信息来源（twitter / manual / openai_status 等） |
| `confidence` | `float` | 可信度 0.0–1.0 |
| `notes` | `str` | 补充说明 |

### Tweet
统一信号数据单元（所有 Collector 输出）。

| 字段 | 类型 | 说明 |
|------|------|------|
| `timestamp` | `datetime` | 发布时间（UTC） |
| `author` | `str` | 作者用户名或站点名 |
| `text` | `str` | 正文内容（标题 + 摘要） |
| `source` | `str` | 数据来源标识（tibo_rss / openai_rss / community_mock 等） |
| `url` | `str?` | 原始链接（可选） |

### SignalScores
LLM 信号分析输出（供 survival_model.py 使用）。

| 字段 | 类型 | 说明 |
|------|------|------|
| `reset_signal` | `float` | 讨论额度重置的信号强度 0.0–1.0 |
| `limit_discussion` | `float` | 讨论使用限制/额度耗尽的信号强度 0.0–1.0 |
| `release_signal` | `float` | 暗示即将发布更新或变更的信号强度 0.0–1.0 |
| `community_pressure` | `float` | 社区对重置的压力或期待程度 0.0–1.0 |
| `confidence` | `float` | LLM 对以上评分的整体置信度 0.0–1.0 |
| `reason` | `list[str]` | 评分依据列表 |

### PredictionFeatures
生存模型预测输入特征（由统计特征 + LLM 信号合并）。

| 字段 | 类型 | 说明 |
|------|------|------|
| `hours_since_last_reset` | `float?` | 距上次 reset 的小时数，None = 无历史 |
| `average_reset_interval` | `float?` | 平均 reset 间隔（小时），None = 无历史 |
| `tibo_signal` | `float` | Tibo/Reset 相关信号强度 0.0–1.0 |
| `community_signal` | `float` | 社区压力信号强度 0.0–1.0 |
| `release_signal` | `float` | 产品发布信号强度 0.0–1.0 |

### PredictionExplanation
生存模型预测输出（含可解释说明）。

| 字段 | 类型 | 说明 |
|------|------|------|
| `probability` | `dict[str, float]` | 各窗口概率，如 `{"5h": 0.42, "24h": 0.76, "48h": 0.91}` |
| `reasons` | `list[str]` | 驱动概率的关键原因（人类可读） |
| `hazard_rate` | `float` | 当前每小时 hazard rate |
| `time_ratio` | `float?` | hours_since_last_reset / average_reset_interval |

### PredictionResult
预测结果（Phase 1 兼容格式），包含多个时间窗口。

| 字段 | 类型 | 说明 |
|------|------|------|
| `timestamp` | `datetime` | 预测生成时间 |
| `predictions` | `list[HorizonPrediction]` | 各时间窗口预测 |
| `signals_used` | `list[str]` | 使用的信号描述 |
| `model_version` | `str` | 模型版本标识 |
| `notes` | `str` | 额外说明 |

## 快速开始

### 1. 安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # 测试用
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 填入 API keys（Phase 1 可留空）
```

### 3. 运行

```bash
# 数据采集 + LLM 信号分析（收集 → 保存 tweets.json → 信号分析）
python -m collectors

# 完整预测管道（收集 → 统计分析 → LLM 信号 → 生存模型预测）
python main.py

# 仅运行预测（使用已收集的数据）
python main.py --predict

# 仅查看项目状态
python main.py --status

# 仅运行信号分析（不含预测）
python main.py --analyze
```

### 4. 测试

```bash
python -m pytest tests/ -v
```

## 预测引擎使用

### 基本用法

```python
from model.survival_model import ResetPredictor
from model.data_models import PredictionFeatures

predictor = ResetPredictor()
features = PredictionFeatures(
    hours_since_last_reset=20.0,
    average_reset_interval=24.0,
    tibo_signal=0.8,
    community_signal=0.3,
    release_signal=0.1,
)
result = predictor.predict(features)

print(result.probability)  # {"5h": 0.33, "24h": 0.86, "48h": 0.98}
print(result.reasons)      # ["距上次 reset 20.0 小时...", "Tibo/社区讨论 reset..."]
print(result.hazard_rate)  # 0.079
```

### 从分析结果构建特征

```python
from model.survival_model import build_features

# analysis_features 来自 SignalAnalyzer
# signal_scores 来自 LLMAnalyzer
features = build_features(
    hours_since_last_reset=analysis_features.hours_since_last_reset,
    average_reset_interval=analysis_features.avg_reset_interval_hours,
    signal_scores=signal_scores,
)
result = predictor.predict(features)
```

### 自定义模型参数

```python
predictor = ResetPredictor(
    params={"alpha": -3.0, "beta_time": 2.0},  # 覆盖默认参数
    horizons=[3, 12, 72],                        # 自定义预测窗口
    default_interval=48.0,                        # 无历史时的默认间隔
)
```

## 后续 Roadmap

| Phase | 内容 |
|-------|------|
| **Phase 1** ✅ | 项目结构、数据模型、收集/分析/输出管道框架 |
| **Phase 2** ✅ | RSS 数据采集层 + LLM 信号分析层（Gemini/Mock） |
| **Phase 3** ✅ | 核心预测引擎 — Discrete-Time Survival Model |
| Phase 4 | 实现 `StatisticalPredictor` — 统计模型预测 |
| Phase 5 | 前端展示（网站） |

## License

MIT
