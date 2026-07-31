"""
WillTiboReset - 配置系统

从环境变量 / .env 文件加载配置，
提供全局配置单例 config。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

def _parse_csv_env(key: str) -> list[str]:
    """将逗号分隔的环境变量解析为列表"""
    raw = os.getenv(key, "")
    return [url.strip() for url in raw.split(",") if url.strip()]


# 加载 .env 文件（如果存在）
load_dotenv()


@dataclass
class Config:
    """全局配置"""

    # --- Twitter/X API ---
    twitter_bearer_token: str = field(
        default_factory=lambda: os.getenv("TWITTER_BEARER_TOKEN", "")
    )
    twitter_api_key: str = field(
        default_factory=lambda: os.getenv("TWITTER_API_KEY", "")
    )
    twitter_api_secret: str = field(
        default_factory=lambda: os.getenv("TWITTER_API_SECRET", "")
    )

    # --- OpenAI API ---
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    openai_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o")
    )

    # --- Gemini API (LLM 信号分析) ---
    gemini_api_key: str = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", "")
    )
    gemini_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    )

    # --- RSS Feed 配置 (RSS_CONFIG) ---
    # 不硬编码 URL，通过环境变量配置，逗号分隔
    rss_feeds: dict[str, list[str]] = field(
        default_factory=lambda: {
            "tibo": _parse_csv_env("TIBO_RSS_URLS"),
            "openai": _parse_csv_env("OPENAI_RSS_URLS"),
            "community": _parse_csv_env("COMMUNITY_RSS_URLS"),
        }
    )
    rss_request_timeout: int = field(
        default_factory=lambda: int(os.getenv("RSS_REQUEST_TIMEOUT", "30"))
    )

    # --- 预测模型配置 ---
    prediction_horizons: list[int] = field(
        default_factory=lambda: [
            int(h.strip())
            for h in os.getenv("PREDICTION_HORIZONS", "5,24,48").split(",")
            if h.strip()
        ]
    )
    confidence_threshold: float = field(
        default_factory=lambda: float(
            os.getenv("CONFIDENCE_THRESHOLD", "0.5")
        )
    )
    default_reset_interval_hours: float = field(
        default_factory=lambda: float(
            os.getenv("DEFAULT_RESET_INTERVAL_HOURS", "48")
        )
    )
    interval_prior_strength: float = field(
        default_factory=lambda: float(
            os.getenv("INTERVAL_PRIOR_STRENGTH", "1")
        )
    )

    # --- 数据路径 ---
    data_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("DATA_DIR", "data")
        )
    )
    output_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("OUTPUT_DIR", "output")
        )
    )

    @property
    def reset_history_path(self) -> Path:
        """历史重置事件 JSON 路径"""
        return self.data_dir / "reset_history.json"

    @property
    def tweets_path(self) -> Path:
        """推文 JSON 路径"""
        return self.data_dir / "tweets.json"

    @property
    def sample_tweets_path(self) -> Path:
        """样本推文 JSON 路径（用于测试/mock）"""
        return self.data_dir / "sample_tweets.json"

    @property
    def model_state_path(self) -> Path:
        """模型状态 JSON 路径"""
        return self.data_dir / "model_state.json"

    @property
    def has_gemini_credentials(self) -> bool:
        """是否已配置 Gemini 凭证"""
        return bool(self.gemini_api_key)

    @property
    def has_rss_feeds(self) -> bool:
        """是否已配置任何 RSS Feed URL"""
        return any(urls for urls in self.rss_feeds.values())

    @property
    def has_twitter_credentials(self) -> bool:
        """是否已配置 Twitter 凭证"""
        return bool(self.twitter_bearer_token or self.twitter_api_key)

    @property
    def has_openai_credentials(self) -> bool:
        """是否已配置 OpenAI 凭证"""
        return bool(self.openai_api_key)

    def ensure_dirs(self) -> None:
        """确保数据目录和输出目录存在"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


# 全局配置单例
config = Config()
