# WillTiboReset

> Will Tibo reset Codex/ChatGPT usage tomorrow?

通过公开互联网信号预测 Tibo/OpenAI 是否会在未来 **5 小时**、**24 小时**、**48 小时** 内重置 ChatGPT/Codex 使用额度。

## 当前状态：Phase 1

Phase 1 已实现完整的**数据收集 + 信号分析管道**框架，预测逻辑接口已定义但留空（不包含任何假逻辑），为后续接入 LLM 分析和统计预测模型做好准备。

---

## 项目结构

```
willtiboreset/
├── main.py                 # 入口文件，CLI 命令
├── config.py               # 配置系统（.env / 环境变量）
├── requirements.txt        # 运行依赖
├── requirements-dev.txt    # 开发依赖（pytest）
├── .env.example            # 环境变量示例
│
├── model/                  # 数据模型与预测器
│   ├── __init__.py         # 统一导出
│   ├── data_models.py      # ResetEvent, Tweet, PredictionResult
│   └── predictor.py        # 预测器框架（BasePredictor + 占位实现）
│
├── collectors/             # 数据收集器
│   └── __init__.py         # TweetCollector, ResetHistoryCollector
│
├── analyzer/               # 信号分析器
│   └── __init__.py         # SignalAnalyzer, AnalysisFeatures
│
├── output/                 # 输出格式化
│   └── __init__.py         # OutputFormatter（JSON + 文本）
│
├── data/                   # 数据存储
│   ├── reset_history.json  # 历史重置事件
│   └── tweets.json         # 收集的推文
│
└── tests/                  # 测试
    ├── test_models.py
    ├── test_config.py
    ├── test_analyzer.py
    └── test_collectors.py
```

## 架构概览

```
数据源 (Twitter/X, 社区报告, OpenAI Status)
        │
        ▼
  ┌─────────────┐
  │  collectors  │  收集原始信号 → Tweet / ResetEvent
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │   analyzer   │  提取特征 → AnalysisFeatures
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │    model     │  预测 → PredictionResult
  │ (predictor)  │  (Phase 1: 接口已定义，逻辑待实现)
  └──────┬──────┘
         │
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
Tibo/OpenAI 相关推文。

| 字段 | 类型 | 说明 |
|------|------|------|
| `timestamp` | `datetime` | 推文发布时间（UTC） |
| `author` | `str` | 作者用户名 |
| `text` | `str` | 推文正文 |
| `url` | `str?` | 推文链接（可选） |

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
# 完整管道（收集 → 分析 → 预测 → 输出）
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
| Phase 2 | 接入 Twitter/X API 实时推文收集 |
| Phase 3 | 实现 `LLMPredictor` — LLM 语义分析预测 |
| Phase 4 | 实现 `StatisticalPredictor` — 统计模型预测 |
| Phase 5 | 前端展示（网站） |

## License

MIT
