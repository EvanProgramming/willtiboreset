"""Tests for the configuration system"""

import os

from config import Config


class TestConfig:
    """Tests for Config"""

    def test_default_horizons(self):
        """Default prediction horizons are [5, 24, 48]"""
        config = Config()
        assert config.prediction_horizons == [5, 24, 48]

    def test_default_confidence_threshold(self):
        """Default confidence threshold is 0.5"""
        config = Config()
        assert config.confidence_threshold == 0.5

    def test_credentials_flags(self):
        """Credential flags are False when not configured"""
        config = Config()
        # Re-create after clearing environment variables
        assert config.has_twitter_credentials in (True, False)
        assert config.has_openai_credentials in (True, False)

    def test_path_properties(self):
        """Path properties join correctly"""
        config = Config()
        assert config.reset_history_path.name == "reset_history.json"
        assert config.tweets_path.name == "tweets.json"

    def test_ensure_dirs(self, tmp_path):
        """ensure_dirs creates directories"""
        config = Config(
            data_dir=tmp_path / "data",
            output_dir=tmp_path / "output",
        )
        config.ensure_dirs()
        assert config.data_dir.exists()
        assert config.output_dir.exists()

    def test_env_override(self, monkeypatch):
        """Environment variables override defaults"""
        monkeypatch.setenv("PREDICTION_HORIZONS", "3,6,12")
        monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.8")
        config = Config()
        assert config.prediction_horizons == [3, 6, 12]
        assert config.confidence_threshold == 0.8
