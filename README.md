# WillTiboReset

> Will Tibo reset Codex/ChatGPT usage tomorrow?

通过公开互联网信号预测 Tibo/OpenAI 是否会在未来 **5 小时**、**24 小时**、**48 小时** 内重置 ChatGPT/Codex 使用额度。

---

## 项目目标

ChatGPT/Codex 的使用额度会被定期或不定期重置，用户往往难以预判下一次重置时机。WillTiboReset 通过以下方式解决这一问题：

1. **采集公开信号**：从 RSS Feed、社区讨论等公开来源收集与额度重置相关的信息
2. **LLM 信号提取**：使用 LLM（Gemini API 或关键词匹配）将自然语言文本转换为结构化信号分数
3. **生存模型预测**：基于 Discrete-Time Survival Model 计算各时间窗口内的 reset 概率
4. **自动运行**：通过 GitHub Actions 每 10 分钟自动运行，持续更新预测结果

不使用大型神经网络，模型可解释，适合历史数据较少的场景。

---

## 数据流程

```
                    ┌─────────────────────────────────────────┐
                    │              数据采集层                   │
                    │  collectors/                             │
                    │                                          │
                    │  TiboRSSCollector   → Tibo 相关 RSS      │
                    │  OpenAIRSSCollector → OpenAI 官方 RSS     │
                    │  CommunityCollector  → 社区 RSS + mock    │
                    │                                          │
                    │  输出: list[Tweet] (统一结构，含 source)   │
                    └──────────────────┬──────────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │              信号分析层                   │
                    │  analyzer/                               │
                    │                                          │
                    │  SignalAnalyzer  → 统计特征               │
                    │    (hours_since_reset, avg_interval)     │
                    │                                          │
                    │  LLMAnalyzer     → 结构化信号分数          │
                    │    (Gemini API 或 Mock 关键词匹配)         │
                    │    reset_signal, limit_discussion,       │
                    │    release_signal, community_pressure    │
                    └──────────────────┬──────────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │              预测引擎                     │
                    │  model/survival_model.py                 │
                    │                                          │
                    │  build_features()  → PredictionFeatures  │
                    │  ResetPredictor()  → 概率 + 解释          │
                    └──────────────────┬──────────────────────┘
                                       │
                                       ▼
                              output/prediction.json
```

## 模型流程

### 1. 特征构建

将统计特征和 LLM 信号合并为 `PredictionFeatures`：

| 输入特征 | 来源 | 说明 |
|---------|------|------|
| `hours_since_last_reset` | `SignalAnalyzer` | 距上次 reset 的小时数 |
| `average_reset_interval` | `SignalAnalyzer` | 历史平均 reset 间隔 |
| `tibo_signal` | `LLMAnalyzer` | `0.6 × reset_signal + 0.4 × limit_discussion` |
| `community_signal` | `LLMAnalyzer` | `community_pressure` |
| `release_signal` | `LLMAnalyzer` | `release_signal` |

### 2. 生存模型 (Discrete-Time Survival Model)

使用 logistic hazard rate 模型，不使用神经网络：

**Step 1** — 计算时间比率：
```
time_ratio = hours_since_last_reset / average_reset_interval
```
比率 > 1 表示已超过平均间隔，reset 可能性上升。

**Step 2** — 计算每小时 hazard rate：
```
h = sigmoid(α + β_time × time_ratio + β_tibo × s_tibo + β_community × s_community + β_release × s_release)
```

默认参数：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| α | -4.0 | 截距（控制基线 hazard） |
| β_time | 1.5 | 时间比率权重 |
| β_tibo | 1.0 | Tibo reset/limit 信号权重 |
| β_community | 0.8 | 社区压力信号权重 |
| β_release | 0.5 | 产品发布信号权重 |

**Step 3** — 计算窗口概率：
```
P(reset within T hours) = 1 - (1 - h)^T
```

### 3. 输出示例

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
    "无历史 reset 记录，基于默认基线 hazard 预测",
    "检测到少量 reset/limit 相关讨论（信号强度 0.49）"
  ]
}
```

---

## 如何运行

### 本地运行

```bash
# 1. 安装依赖
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. 一键预测（采集 → 分析 → 预测 → 输出）
python predict.py

# 输出文件: output/prediction.json
```

### 其他命令

```bash
# 数据采集 + LLM 信号分析（不运行预测）
python -m collectors

# 完整预测管道（带详细输出）
python main.py

# 仅运行预测（使用已收集数据）
python main.py --predict

# 仅查看项目状态
python main.py --status

# 仅运行信号分析
python main.py --analyze

# 运行测试
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

### GitHub Actions（自动运行）

项目已配置 `.github/workflows/predict.yml`：

- **定时运行**：每 10 分钟自动执行 `python predict.py`
- **手动触发**：在 GitHub 仓库 Actions 页面手动触发
- **自动提交**：预测结果 `output/prediction.json` 自动 commit 到仓库

如需启用 Gemini API 分析，在仓库 Settings → Secrets 中添加 `GEMINI_API_KEY`。

---

## 如何配置 API

所有配置通过环境变量或 `.env` 文件管理。复制 `.env.example` 为 `.env` 并按需填写：

```bash
cp .env.example .env
```

### 环境变量

| 变量名 | 必填 | 默认值 | 说明 |
|--------|------|--------|------|
| `GEMINI_API_KEY` | 否 | 空 | Gemini API key，启用 LLM 语义分析。留空则使用 Mock 关键词匹配 |
| `GEMINI_MODEL` | 否 | `gemini-2.0-flash` | Gemini 模型名称 |
| `TIBO_RSS_URLS` | 否 | 空 | Tibo RSS 地址，逗号分隔 |
| `OPENAI_RSS_URLS` | 否 | 空 | OpenAI RSS 地址，逗号分隔 |
| `COMMUNITY_RSS_URLS` | 否 | 空 | 社区 RSS 地址，逗号分隔 |
| `PREDICTION_HORIZONS` | 否 | `5,24,48` | 预测时间窗口（小时） |
| `DATA_DIR` | 否 | `data` | 数据存储目录 |
| `OUTPUT_DIR` | 否 | `output` | 输出目录 |

> **无需任何 API key 即可运行**：未配置 RSS Feed 时使用 `data/sample_tweets.json` 中的 mock 数据，未配置 Gemini API key 时使用关键词匹配的 Mock 分析器。

### GitHub Actions Secrets 配置

在仓库 Settings → Secrets and variables → Actions 中添加：

| Secret 名称 | 说明 |
|-------------|------|
| `GEMINI_API_KEY` | Gemini API key（可选） |
| `TIBO_RSS_URLS` | Tibo RSS 地址（可选） |
| `OPENAI_RSS_URLS` | OpenAI RSS 地址（可选） |
| `COMMUNITY_RSS_URLS` | 社区 RSS 地址（可选） |

---

## 项目结构

```
willtiboreset/
├── predict.py              # 一键预测入口（采集 → 分析 → 预测 → 输出）
├── main.py                 # CLI 入口（带详细输出）
├── config.py               # 配置系统（.env / 环境变量）
├── requirements.txt        # 运行依赖
├── .env.example            # 环境变量示例
│
├── .github/workflows/
│   └── predict.yml         # GitHub Actions（每 10 分钟自动预测）
│
├── model/                  # 数据模型与预测器
│   ├── data_models.py      # ResetEvent, Tweet, SignalScores, PredictionFeatures
│   ├── predictor.py        # 预测器框架（BasePredictor）
│   └── survival_model.py   # ResetPredictor — 生存模型预测引擎
│
├── collectors/             # 数据收集器
│   ├── rss_base.py         # RSS 基础收集器（解析/去重）
│   ├── tibo_rss.py         # Tibo RSS 收集器
│   ├── openai_rss.py       # OpenAI RSS 收集器
│   ├── community.py        # 社区信号收集器（RSS + mock）
│   └── __main__.py         # 入口: python -m collectors
│
├── analyzer/               # 信号分析器
│   ├── __init__.py         # SignalAnalyzer（统计特征）
│   └── llm_signal.py       # LLMAnalyzer（Gemini + Mock）
│
├── output/                 # 输出
│   └── prediction.json     # 最终预测结果（自动生成/更新）
│
├── data/                   # 数据存储
│   ├── reset_history.json  # 历史重置事件
│   ├── tweets.json         # 收集的推文
│   └── sample_tweets.json  # mock 数据（无 RSS 时使用）
│
└── tests/                  # 测试（128 个）
    ├── test_models.py
    ├── test_config.py
    ├── test_analyzer.py
    ├── test_collectors.py
    ├── test_rss_collectors.py
    ├── test_llm_signal.py
    └── test_survival_model.py
```

---

## 技术选型

| 组件 | 选择 | 原因 |
|------|------|------|
| 数据模型 | Pydantic v2 | 类型安全、JSON 序列化、验证 |
| RSS 解析 | feedparser | 成熟稳定的 RSS/Atom 解析库 |
| LLM 信号分析 | Gemini API（可选） | 免费额度充足，延迟低 |
| 预测模型 | Logistic Hazard Model | 可解释、数据需求低、无需训练 |
| 配置 | python-dotenv | 环境变量管理，支持 .env 文件 |
| CI/CD | GitHub Actions | 免费、与 Git 仓库原生集成 |
| 数据源 | RSS Feed | 不依赖 X/Twitter API，配置简单 |

---

## Roadmap

| Phase | 内容 | 状态 |
|-------|------|------|
| Phase 1 | 项目结构、数据模型、收集/分析/预测/输出完整管道 | ✅ 完成 |
| Phase 2 | RSS 数据采集层 + LLM 信号分析层 | ✅ 完成 |
| Phase 3 | 核心预测引擎 — Discrete-Time Survival Model | ✅ 完成 |
| Phase 4 | StatisticalPredictor — 统计模型预测 | 计划中 |
| Phase 5 | 前端展示（网站） | 计划中 |

## License

MIT
