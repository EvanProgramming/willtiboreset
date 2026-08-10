"""
WillTiboReset - Configuration system

Loads configuration from environment variables / .env file,
provides the global config singleton.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

def _parse_csv_env(key: str) -> list[str]:
    """Parse a comma-separated environment variable into a list"""
    raw = os.getenv(key, "")
    return [url.strip() for url in raw.split(",") if url.strip()]


def _env_or_default(key: str, default: str) -> str:
    """Read environment variable; return default if unset or empty."""
    return os.getenv(key, default) or default


# Load .env file if it exists
load_dotenv()


@dataclass
class Config:
    """Global configuration"""

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
        default_factory=lambda: _env_or_default("OPENAI_MODEL", "gpt-4o")
    )

    # --- Gemini API (LLM signal analysis, deprecated) ---
    gemini_api_key: str = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", "")
    )
    gemini_model: str = field(
        default_factory=lambda: _env_or_default("GEMINI_MODEL", "gemini-2.0-flash")
    )

    # --- DeepSeek API (LLM signal analysis, currently preferred) ---
    deepseek_api_key: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", "")
    )
    deepseek_model: str = field(
        default_factory=lambda: _env_or_default("DEEPSEEK_MODEL", "deepseek-chat")
    )

    # --- RSS feed configuration (RSS_CONFIG) ---
    # URLs are not hardcoded; configured via environment variables, comma-separated
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

    # --- Prediction model configuration ---
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

    # --- Manual override: explicit future reset ---
    # Set to "true" when Tibo has announced a reset that RSS missed
    # (e.g., reply-only tweet). This pushes probability to near-max.
    explicit_future_reset: bool = field(
        default_factory=lambda: os.getenv("EXPLICIT_FUTURE_RESET", "").lower()
        in ("true", "1", "yes")
    )

    # --- Data paths ---
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
        """Path to the historical reset events JSON"""
        return self.data_dir / "reset_history.json"

    @property
    def tweets_path(self) -> Path:
        """Path to the tweets JSON"""
        return self.data_dir / "tweets.json"

    @property
    def sample_tweets_path(self) -> Path:
        """Path to the sample tweets JSON (used for testing / mock)"""
        return self.data_dir / "sample_tweets.json"

    @property
    def model_state_path(self) -> Path:
        """Path to the model state JSON"""
        return self.data_dir / "model_state.json"

    @property
    def prediction_history_path(self) -> Path:
        """Path to the prediction history JSON"""
        return self.data_dir / "prediction_history.json"

    @property
    def model_performance_path(self) -> Path:
        """Path to the model performance report JSON"""
        return self.output_dir / "model_performance.json"

    @property
    def has_gemini_credentials(self) -> bool:
        """Whether Gemini credentials are configured"""
        return bool(self.gemini_api_key)

    @property
    def has_deepseek_credentials(self) -> bool:
        """Whether DeepSeek credentials are configured"""
        return bool(self.deepseek_api_key)

    @property
    def has_rss_feeds(self) -> bool:
        """Whether any RSS feed URL is configured"""
        return any(urls for urls in self.rss_feeds.values())

    @property
    def has_openai_credentials(self) -> bool:
        """Whether OpenAI credentials are configured"""
        return bool(self.openai_api_key)

    def ensure_dirs(self) -> None:
        """Ensure data and output directories exist"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


# Global config singleton
config = Config()
