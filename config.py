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
