# WillTiboReset

> Will Tibo reset Codex/ChatGPT usage tomorrow?

通过公开互联网信号预测 Tibo/OpenAI 是否会在未来 **5 小时**、**24 小时**、**48 小时** 内重置 ChatGPT/Codex 使用额度。

## 当前状态：Phase 2

- **Phase 1** ✅ 项目结构、数据模型、收集/分析/输出管道框架
- **Phase 2** ✅ RSS 数据采集层 + LLM 信号分析层
  - RSS 收集器（TiboRSSCollector、OpenAIRSSCollector、CommunityCollector）
  - LLM 信号分析器（GeminiAnalyzer + MockLLMAnalyzer）
  - 统一 Tweet 数据结构（含 source 字段）
  - SignalScores 结构化特征输出
  - 预测逻辑接口已定义但留空（不包含任何假逻辑）

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
│   ├── data_models.py      # ResetEvent, Tweet, SignalScores, PredictionResult
│   └── predictor.py        # 预测器框架（BasePredictor + 占位实现）
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
    └── test_llm_signal.py
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
  │   analyzer   │  提取统计特征 → AnalysisFeatures
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │    model     │  预测 → PredictionResult
  │ (predictor)  │  (接口已定义，逻辑待实现)
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │   output     │  格式化输出 → JSON + 文本
  └─────────────┘
```

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

### PredictionResult
预测结果，包含多个时间窗口。

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

# 完整预测管道（收集 → 分析 → 预测 → 输出）
python main.py

# 仅查看项目状态
python main.py --status

# 仅运行信号分析
python main.py --analyze
```

### 4. 测试

```bash
python -m pytest tests/ -v
```

## 预测器扩展指南

预测逻辑通过 `BasePredictor` 抽象基类定义，后续 Phase 实现新预测器只需继承并实现 `predict` 方法：

```python
from model.predictor import BasePredictor
from model.data_models import PredictionResult, ResetEvent, Tweet

class MyPredictor(BasePredictor):
    @property
    def model_version(self) -> str:
        return "my-model-1.0"

    def predict(
        self,
        tweets: list[Tweet],
        reset_events: list[ResetEvent],
        horizons: list[int],
    ) -> PredictionResult:
        # 实现预测逻辑
        ...
```

已有预留接口：
- **`LLMPredictor`** — 使用 OpenAI API 进行自然语言推理分析
- **`StatisticalPredictor`** — 基于历史重置事件的时间序列统计模型

## 后续 Roadmap

| Phase | 内容 |
|-------|------|
| **Phase 1** ✅ | 项目结构、数据模型、收集/分析/输出管道框架 |
| **Phase 2** ✅ | RSS 数据采集层 + LLM 信号分析层（Gemini/Mock） |
| Phase 3 | 实现 `model/survival_model.py` — 基于 SignalScores 的生存分析预测 |
| Phase 4 | 实现 `StatisticalPredictor` — 统计模型预测 |
| Phase 5 | 前端展示（网站） |

## License

MIT
