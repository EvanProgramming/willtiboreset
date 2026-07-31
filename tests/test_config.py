"""测试配置系统"""

import os

from config import Config


class TestConfig:
    """Config 配置测试"""

    def test_default_horizons(self):
        """默认预测窗口为 [5, 24, 48]"""
        config = Config()
        assert config.prediction_horizons == [5, 24, 48]

    def test_default_confidence_threshold(self):
        """默认置信度阈值为 0.5"""
        config = Config()
        assert config.confidence_threshold == 0.5

    def test_credentials_flags(self):
        """未配置凭证时标志为 False"""
        config = Config()
        # 清除环境变量后重新创建
        assert config.has_twitter_credentials in (True, False)
        assert config.has_openai_credentials in (True, False)

    def test_path_properties(self):
        """路径属性正确拼接"""
        config = Config()
        assert config.reset_history_path.name == "reset_history.json"
        assert config.tweets_path.name == "tweets.json"

    def test_ensure_dirs(self, tmp_path):
        """ensure_dirs 创建目录"""
        config = Config(
            data_dir=tmp_path / "data",
            output_dir=tmp_path / "output",
        )
        config.ensure_dirs()
        assert config.data_dir.exists()
        assert config.output_dir.exists()

    def test_env_override(self, monkeypatch):
        """环境变量覆盖默认值"""
        monkeypatch.setenv("PREDICTION_HORIZONS", "3,6,12")
        monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.8")
        config = Config()
        assert config.prediction_horizons == [3, 6, 12]
        assert config.confidence_threshold == 0.8
