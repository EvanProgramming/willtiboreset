"""Tests for the LLM signal analyzer"""

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
    """Tests for MockLLMAnalyzer"""

    def test_analyze_reset_text(self):
        """Reset-related text yields high reset_intent / reset_confirmation"""
        analyzer = MockLLMAnalyzer()
        scores = analyzer.analyze(["Codex usage limit has been reset! reset confirmed"])
        assert len(scores) == 1
        assert scores[0].reset_intent > 0.5
        assert scores[0].reset_confirmation > 0.5
        assert scores[0].confidence > 0.5

    def test_analyze_limit_text(self):
        """Limit-related text yields high limit_complaint"""
        analyzer = MockLLMAnalyzer()
        scores = analyzer.analyze(["I hit the rate limit, quota exhausted"])
        assert len(scores) == 1
        assert scores[0].limit_complaint > 0.5

    def test_analyze_release_text(self):
        """Release-related text yields high official_change"""
        analyzer = MockLLMAnalyzer()
        scores = analyzer.analyze(["OpenAI announces new update, version release"])
        assert len(scores) == 1
        assert scores[0].official_change > 0.5

    def test_analyze_irrelevant_text(self):
        """Irrelevant text has all scores near 0"""
        analyzer = MockLLMAnalyzer()
        scores = analyzer.analyze(["The weather in San Francisco is sunny today"])
        assert len(scores) == 1
        assert scores[0].reset_intent == 0.0
        assert scores[0].limit_complaint == 0.0
        assert scores[0].official_change == 0.0
        assert scores[0].reset_confirmation == 0.0
        assert scores[0].confidence < 0.5

    def test_analyze_multiple_texts(self):
        """Analyze multiple texts"""
        analyzer = MockLLMAnalyzer()
        texts = [
            "Codex usage has been reset",
            "OpenAI announces GPT-5",
            "The weather is nice today",
        ]
        scores = analyzer.analyze(texts)
        assert len(scores) == 3
        assert scores[0].reset_intent > 0
        assert scores[1].official_change > 0
        assert scores[2].reset_intent == 0.0

    def test_analyze_empty_list(self):
        """Empty list returns empty list"""
        analyzer = MockLLMAnalyzer()
        assert analyzer.analyze([]) == []

    def test_output_is_signal_scores(self):
        """Output type is SignalScores"""
        analyzer = MockLLMAnalyzer()
        scores = analyzer.analyze(["test text"])
        assert isinstance(scores[0], SignalScores)
        assert 0.0 <= scores[0].reset_intent <= 1.0
        assert 0.0 <= scores[0].confidence <= 1.0
        assert isinstance(scores[0].reason, list)

    def test_analyze_tweets(self):
        """Analyze Tweet objects directly"""
        analyzer = MockLLMAnalyzer()
        tweets = [
            Tweet(
                timestamp=datetime(2025, 7, 30, tzinfo=timezone.utc),
                author="user",
                text="Codex usage has been reset",
                source="community_mock",
            ),
        ]
        scores = analyzer.analyze_tweets(tweets)
        assert len(scores) == 1
        assert scores[0].reset_intent > 0.5

    def test_analyze_batch(self):
        """Batch aggregation analysis"""
        analyzer = MockLLMAnalyzer()
        texts = [
            "Codex usage has been reset",
            "rate limit quota exhausted",
            "The weather is nice today",
        ]
        batch = analyzer.analyze_batch(texts)
        assert isinstance(batch, SignalScores)
        assert 0.0 <= batch.reset_intent <= 1.0
        assert isinstance(batch.reason, list)
        # 2 relevant and 1 irrelevant, so reset_intent should be > 0
        assert batch.reset_intent > 0

    def test_analyze_batch_empty(self):
        """Batch analysis with empty input"""
        analyzer = MockLLMAnalyzer()
        batch = analyzer.analyze_batch([])
        assert batch.reset_intent == 0.0
        assert batch.confidence == 0.0

    def test_to_features_compatibility(self):
        """to_features output is compatible with survival_model.py"""
        analyzer = MockLLMAnalyzer()
        scores = analyzer.analyze(["Codex reset limit quota"])[0]
        features = scores.to_features()
        assert isinstance(features, dict)
        assert set(features.keys()) == {
            "reset_intent", "limit_complaint",
            "official_change", "reset_confirmation", "confidence",
        }
        for v in features.values():
            assert isinstance(v, float)
            assert 0.0 <= v <= 1.0


class TestExtractJsonArray:
    """Tests for JSON extraction helper"""

    def test_plain_json(self):
        text = '[{"reset_intent": 0.8, "confidence": 0.9}]'
        result = _extract_json_array(text)
        assert len(result) == 1
        assert result[0]["reset_intent"] == 0.8

    def test_markdown_code_block(self):
        text = '```json\n[{"reset_intent": 0.5}]\n```'
        result = _extract_json_array(text)
        assert len(result) == 1

    def test_with_surrounding_text(self):
        text = 'Here is the result:\n[{"reset_intent": 0.7}]\nDone.'
        result = _extract_json_array(text)
        assert len(result) == 1

    def test_invalid_json(self):
        assert _extract_json_array("not json at all") == []

    def test_empty_array(self):
        assert _extract_json_array("[]") == []


class TestDictToScores:
    """Tests for dict-to-SignalScores conversion"""

    def test_valid_dict(self):
        d = {
            "reset_intent": 0.8,
            "limit_complaint": 0.6,
            "official_change": 0.2,
            "reset_confirmation": 0.7,
            "confidence": 0.9,
            "reason": ["reason1", "reason2"],
        }
        scores = _dict_to_scores(d)
        assert scores.reset_intent == 0.8
        assert scores.reason == ["reason1", "reason2"]

    def test_clamp_values(self):
        """Out-of-range values are clamped to [0, 1]"""
        d = {
            "reset_intent": 1.5,
            "limit_complaint": -0.5,
            "official_change": 0.5,
            "reset_confirmation": 0.5,
            "confidence": 0.5,
        }
        scores = _dict_to_scores(d)
        assert scores.reset_intent == 1.0
        assert scores.limit_complaint == 0.0

    def test_missing_fields_default_zero(self):
        d = {"confidence": 0.5}
        scores = _dict_to_scores(d)
        assert scores.reset_intent == 0.0
        assert scores.confidence == 0.5

    def test_reason_as_string(self):
        d = {
            "reset_intent": 0.5,
            "limit_complaint": 0.5,
            "official_change": 0.5,
            "reset_confirmation": 0.5,
            "confidence": 0.5,
            "reason": "single reason string",
        }
        scores = _dict_to_scores(d)
        assert scores.reason == ["single reason string"]

    def test_backward_compatible_old_field_names(self):
        """Legacy field names can still be parsed"""
        d = {
            "reset_signal": 0.8,
            "limit_discussion": 0.6,
            "release_signal": 0.2,
            "confidence": 0.9,
        }
        scores = _dict_to_scores(d)
        assert scores.reset_intent == 0.8
        assert scores.limit_complaint == 0.6
        assert scores.official_change == 0.2


class TestGeminiAnalyzer:
    """Tests for GeminiAnalyzer (no real API calls)"""

    def test_init(self):
        analyzer = GeminiAnalyzer(api_key="fake_key", model="gemini-2.0-flash")
        assert analyzer._api_key == "fake_key"
        assert analyzer._model == "gemini-2.0-flash"
        assert analyzer._client is None

    def test_is_llm_analyzer(self):
        analyzer = GeminiAnalyzer(api_key="fake_key")
        assert isinstance(analyzer, LLMAnalyzer)

    def test_analyze_empty(self):
        """Empty list does not trigger API call"""
        analyzer = GeminiAnalyzer(api_key="fake_key")
        assert analyzer.analyze([]) == []


class TestLLMAnalyzerInterface:
    """Tests for LLMAnalyzer interface"""

    def test_mock_is_llm_analyzer(self):
        analyzer = MockLLMAnalyzer()
        assert isinstance(analyzer, LLMAnalyzer)

    def test_analyze_returns_list_of_signal_scores(self):
        analyzer = MockLLMAnalyzer()
        results = analyzer.analyze(["test1", "test2"])
        assert isinstance(results, list)
        assert all(isinstance(r, SignalScores) for r in results)
        assert len(results) == 2
