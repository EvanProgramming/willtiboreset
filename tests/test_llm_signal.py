"""测试 LLM 信号分析器"""

from datetime import datetime, timezone

from analyzer.llm_signal import (
    GeminiAnalyzer,
    LLMAnalyzer,
    MockLLMAnalyzer,
    _dict_to_scores,
    _extract_json_array,
)
from model.data_models import SignalScores, Tweet


class TestMockLLMAnalyzer:
    """MockLLMAnalyzer 测试"""

    def test_analyze_reset_text(self):
        """重置相关文本得到高 reset_signal"""
        analyzer = MockLLMAnalyzer()
        scores = analyzer.analyze(["Codex 额度终于重置了！reset confirmed"])
        assert len(scores) == 1
        assert scores[0].reset_signal > 0.5
        assert scores[0].confidence > 0.5

    def test_analyze_limit_text(self):
        """限制相关文本得到高 limit_discussion"""
        analyzer = MockLLMAnalyzer()
        scores = analyzer.analyze(["I hit the rate limit, quota exhausted"])
        assert len(scores) == 1
        assert scores[0].limit_discussion > 0.5

    def test_analyze_release_text(self):
        """发布相关文本得到高 release_signal"""
        analyzer = MockLLMAnalyzer()
        scores = analyzer.analyze(["OpenAI announces new update, version release"])
        assert len(scores) == 1
        assert scores[0].release_signal > 0.5

    def test_analyze_irrelevant_text(self):
        """无关文本所有分数接近 0"""
        analyzer = MockLLMAnalyzer()
        scores = analyzer.analyze(["The weather in San Francisco is sunny today"])
        assert len(scores) == 1
        assert scores[0].reset_signal == 0.0
        assert scores[0].limit_discussion == 0.0
        assert scores[0].release_signal == 0.0
        assert scores[0].community_pressure == 0.0
        assert scores[0].confidence < 0.5

    def test_analyze_multiple_texts(self):
        """分析多条文本"""
        analyzer = MockLLMAnalyzer()
        texts = [
            "Codex 额度重置了",
            "OpenAI announces GPT-5",
            "The weather is nice today",
        ]
        scores = analyzer.analyze(texts)
        assert len(scores) == 3
        assert scores[0].reset_signal > 0
        assert scores[1].release_signal > 0
        assert scores[2].reset_signal == 0.0

    def test_analyze_empty_list(self):
        """空列表返回空列表"""
        analyzer = MockLLMAnalyzer()
        assert analyzer.analyze([]) == []

    def test_output_is_signal_scores(self):
        """输出类型为 SignalScores"""
        analyzer = MockLLMAnalyzer()
        scores = analyzer.analyze(["test text"])
        assert isinstance(scores[0], SignalScores)
        assert 0.0 <= scores[0].reset_signal <= 1.0
        assert 0.0 <= scores[0].confidence <= 1.0
        assert isinstance(scores[0].reason, list)

    def test_analyze_tweets(self):
        """直接分析 Tweet 对象"""
        analyzer = MockLLMAnalyzer()
        tweets = [
            Tweet(
                timestamp=datetime(2025, 7, 30, tzinfo=timezone.utc),
                author="user",
                text="Codex 额度重置了 reset",
                source="community_mock",
            ),
        ]
        scores = analyzer.analyze_tweets(tweets)
        assert len(scores) == 1
        assert scores[0].reset_signal > 0.5

    def test_analyze_batch(self):
        """批量聚合分析"""
        analyzer = MockLLMAnalyzer()
        texts = [
            "Codex 额度重置了 reset",
            "rate limit quota exhausted",
            "The weather is nice today",
        ]
        batch = analyzer.analyze_batch(texts)
        assert isinstance(batch, SignalScores)
        assert 0.0 <= batch.reset_signal <= 1.0
        assert isinstance(batch.reason, list)
        # 有 2 条相关，1 条无关，reset_signal 应该 > 0
        assert batch.reset_signal > 0

    def test_analyze_batch_empty(self):
        """空输入的批量分析"""
        analyzer = MockLLMAnalyzer()
        batch = analyzer.analyze_batch([])
        assert batch.reset_signal == 0.0
        assert batch.confidence == 0.0

    def test_to_features_compatibility(self):
        """to_features 输出兼容 survival_model.py"""
        analyzer = MockLLMAnalyzer()
        scores = analyzer.analyze(["Codex reset limit quota"])[0]
        features = scores.to_features()
        assert isinstance(features, dict)
        assert set(features.keys()) == {
            "reset_signal", "limit_discussion",
            "release_signal", "community_pressure", "confidence",
        }
        for v in features.values():
            assert isinstance(v, float)
            assert 0.0 <= v <= 1.0


class TestExtractJsonArray:
    """JSON 解析辅助函数测试"""

    def test_plain_json(self):
        text = '[{"reset_signal": 0.8, "confidence": 0.9}]'
        result = _extract_json_array(text)
        assert len(result) == 1
        assert result[0]["reset_signal"] == 0.8

    def test_markdown_code_block(self):
        text = '```json\n[{"reset_signal": 0.5}]\n```'
        result = _extract_json_array(text)
        assert len(result) == 1

    def test_with_surrounding_text(self):
        text = 'Here is the result:\n[{"reset_signal": 0.7}]\nDone.'
        result = _extract_json_array(text)
        assert len(result) == 1

    def test_invalid_json(self):
        assert _extract_json_array("not json at all") == []

    def test_empty_array(self):
        assert _extract_json_array("[]") == []


class TestDictToScores:
    """字典转 SignalScores 测试"""

    def test_valid_dict(self):
        d = {
            "reset_signal": 0.8,
            "limit_discussion": 0.6,
            "release_signal": 0.2,
            "community_pressure": 0.7,
            "confidence": 0.9,
            "reason": ["reason1", "reason2"],
        }
        scores = _dict_to_scores(d)
        assert scores.reset_signal == 0.8
        assert scores.reason == ["reason1", "reason2"]

    def test_clamp_values(self):
        """超出范围的值被截断到 [0, 1]"""
        d = {
            "reset_signal": 1.5,
            "limit_discussion": -0.5,
            "release_signal": 0.5,
            "community_pressure": 0.5,
            "confidence": 0.5,
        }
        scores = _dict_to_scores(d)
        assert scores.reset_signal == 1.0
        assert scores.limit_discussion == 0.0

    def test_missing_fields_default_zero(self):
        d = {"confidence": 0.5}
        scores = _dict_to_scores(d)
        assert scores.reset_signal == 0.0
        assert scores.confidence == 0.5

    def test_reason_as_string(self):
        d = {
            "reset_signal": 0.5,
            "limit_discussion": 0.5,
            "release_signal": 0.5,
            "community_pressure": 0.5,
            "confidence": 0.5,
            "reason": "single reason string",
        }
        scores = _dict_to_scores(d)
        assert scores.reason == ["single reason string"]


class TestGeminiAnalyzer:
    """GeminiAnalyzer 测试（不调用真实 API）"""

    def test_init(self):
        analyzer = GeminiAnalyzer(api_key="fake_key", model="gemini-2.0-flash")
        assert analyzer._api_key == "fake_key"
        assert analyzer._model == "gemini-2.0-flash"
        assert analyzer._client is None

    def test_is_llm_analyzer(self):
        analyzer = GeminiAnalyzer(api_key="fake_key")
        assert isinstance(analyzer, LLMAnalyzer)

    def test_analyze_empty(self):
        """空列表不触发 API 调用"""
        analyzer = GeminiAnalyzer(api_key="fake_key")
        assert analyzer.analyze([]) == []


class TestLLMAnalyzerInterface:
    """LLMAnalyzer 接口测试"""

    def test_mock_is_llm_analyzer(self):
        analyzer = MockLLMAnalyzer()
        assert isinstance(analyzer, LLMAnalyzer)

    def test_analyze_returns_list_of_signal_scores(self):
        analyzer = MockLLMAnalyzer()
        results = analyzer.analyze(["test1", "test2"])
        assert isinstance(results, list)
        assert all(isinstance(r, SignalScores) for r in results)
        assert len(results) == 2
