"""
LLM 信号分析器。

将自然语言文本（推文、新闻）转换为结构化机器学习特征。
LLM 不负责直接预测 reset，只负责信号提取。
输出 SignalScores 将传递给 model/survival_model.py 作为预测输入。

通用接口：
    class LLMAnalyzer:
        analyze(texts: list[str]) -> list[SignalScores]

当前实现：
    - GeminiAnalyzer: 使用 Gemini API（优先）
    - MockLLMAnalyzer: 基于关键词匹配（无需 API key，用于测试）
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Optional

from model.data_models import SignalScores, Tweet


# ──────────────────────────────────────────────
# 通用接口
# ──────────────────────────────────────────────

class LLMAnalyzer(ABC):
    """
    LLM 信号分析器抽象基类。

    所有分析器（Gemini、OpenAI、Mock 等）继承此类。
    核心方法 analyze() 接收文本列表，返回每条文本的 SignalScores。
    """

    @abstractmethod
    def analyze(self, texts: list[str]) -> list[SignalScores]:
        """
        分析文本列表，返回结构化信号分数。

        Args:
            texts: 待分析的自然语言文本列表

        Returns:
            与输入等长的 SignalScores 列表，每个元素对应一条文本
        """
        ...

    def analyze_tweets(self, tweets: list[Tweet]) -> list[SignalScores]:
        """便捷方法：直接分析 Tweet 对象列表"""
        return self.analyze([t.text for t in tweets])

    def analyze_batch(self, texts: list[str]) -> SignalScores:
        """
        批量分析并返回聚合信号分数。

        对所有文本的 SignalScores 取平均值，
        适用于需要整体信号概览的场景。
        """
        scores = self.analyze(texts)
        if not scores:
            return SignalScores(
                reset_signal=0.0,
                limit_discussion=0.0,
                release_signal=0.0,
                community_pressure=0.0,
                confidence=0.0,
                reason=["无输入文本"],
            )
        n = len(scores)
        all_reasons: list[str] = []
        for s in scores:
            all_reasons.extend(s.reason[:2])
        return SignalScores(
            reset_signal=sum(s.reset_signal for s in scores) / n,
            limit_discussion=sum(s.limit_discussion for s in scores) / n,
            release_signal=sum(s.release_signal for s in scores) / n,
            community_pressure=sum(s.community_pressure for s in scores) / n,
            confidence=sum(s.confidence for s in scores) / n,
            reason=all_reasons[:5] if all_reasons else ["批量聚合"],
        )


# ──────────────────────────────────────────────
# Gemini 实现
# ──────────────────────────────────────────────

_SYSTEM_PROMPT = """你是一个信号分析助手，专门分析关于 ChatGPT/Codex 使用额度重置的社交媒体帖子和新闻。

对于每条文本，请评估以下维度（0.0 到 1.0 的浮点数）：

1. reset_signal: 文本是否讨论了使用额度/限制的重置（0=完全不相关，1=明确讨论重置发生）
2. limit_discussion: 文本是否讨论了使用限制/额度耗尽（0=完全不相关，1=明确讨论限制问题）
3. release_signal: 文本是否暗示即将发布更新或变更（0=完全不相关，1=明确暗示即将发布）
4. community_pressure: 文本是否反映了社区对重置的压力或期待（0=无压力，1=强烈压力）
5. confidence: 你对以上评分的整体置信度（0=非常不确定，1=非常确定）

同时提供 reason 列表，简述评分依据（每个理由不超过一句话）。

请严格以 JSON 格式返回结果。返回一个 JSON 数组，每个元素对应一条输入文本：
[
  {
    "reset_signal": 0.0,
    "limit_discussion": 0.0,
    "release_signal": 0.0,
    "community_pressure": 0.0,
    "confidence": 0.0,
    "reason": ["原因1", "原因2"]
  }
]

不要包含任何其他文字，只返回 JSON 数组。"""


def _extract_json_array(text: str) -> list[dict]:
    """从 LLM 响应中提取 JSON 数组"""
    # 尝试直接解析
    text = text.strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # 尝试从 markdown 代码块中提取
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(1).strip())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # 尝试找到第一个 [ 和最后一个 ]
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(text[start:end + 1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    return []


def _dict_to_scores(d: dict) -> SignalScores:
    """将字典转换为 SignalScores，处理类型和默认值"""
    def _get_float(key: str) -> float:
        val = d.get(key, 0.0)
        try:
            return max(0.0, min(1.0, float(val)))
        except (TypeError, ValueError):
            return 0.0

    def _get_reasons() -> list[str]:
        val = d.get("reason", [])
        if isinstance(val, list):
            return [str(r) for r in val]
        if isinstance(val, str):
            return [val]
        return []

    return SignalScores(
        reset_signal=_get_float("reset_signal"),
        limit_discussion=_get_float("limit_discussion"),
        release_signal=_get_float("release_signal"),
        community_pressure=_get_float("community_pressure"),
        confidence=_get_float("confidence"),
        reason=_get_reasons(),
    )


class GeminiAnalyzer(LLMAnalyzer):
    """
    Gemini API 信号分析器。

    使用 Google Gemini API 进行自然语言分析。
    需要配置 GEMINI_API_KEY 环境变量。
    google-generativeai 包为延迟导入，未安装时给出清晰错误。
    """

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self._api_key = api_key
        self._model = model
        self._client = None

    def _ensure_client(self):
        """延迟初始化 Gemini 客户端"""
        if self._client is not None:
            return
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai 未安装。"
                "请运行: pip install google-generativeai"
            )
        genai.configure(api_key=self._api_key)
        self._client = genai.GenerativeModel(
            self._model,
            system_instruction=_SYSTEM_PROMPT,
        )

    def analyze(self, texts: list[str]) -> list[SignalScores]:
        """调用 Gemini API 分析文本"""
        if not texts:
            return []

        self._ensure_client()

        # 构造用户输入
        lines = []
        for i, text in enumerate(texts, 1):
            lines.append(f"[{i}] {text}")
        user_input = "\n".join(lines)

        response = self._client.generate_content(user_input)
        response_text = response.text

        # 解析 JSON 响应
        raw_list = _extract_json_array(response_text)

        # 确保返回长度与输入一致
        results: list[SignalScores] = []
        for i, text in enumerate(texts):
            if i < len(raw_list):
                results.append(_dict_to_scores(raw_list[i]))
            else:
                results.append(SignalScores(
                    reset_signal=0.0,
                    limit_discussion=0.0,
                    release_signal=0.0,
                    community_pressure=0.0,
                    confidence=0.0,
                    reason=["LLM 响应不完整"],
                ))
        return results


# ──────────────────────────────────────────────
# Mock 实现（用于测试和无 API key 环境）
# ──────────────────────────────────────────────

# 关键词映射表
_RESET_KEYWORDS = [
    "reset", "重置", "额度重置", "usage reset", "limit reset",
    "quota reset", "resetted", "has been reset",
]
_LIMIT_KEYWORDS = [
    "limit", "限制", "额度", "quota", "用完", "exhausted",
    "rate limit", "usage limit", "capacity", "上限", "达到上限",
]
_RELEASE_KEYWORDS = [
    "release", "发布", "update", "更新", "announce", "公告",
    "launch", "推出", "new version", "rollout", "deploy",
]
_PRESSURE_KEYWORDS = [
    "please", "需要", "希望", "want", "when", "什么时候",
    "waiting", "等待", "急需", "anyone", "有人", "谁知道",
]


def _keyword_score(text_lower: str, keywords: list[str]) -> float:
    """基于关键词匹配计算分数"""
    matches = sum(1 for kw in keywords if kw in text_lower)
    if matches == 0:
        return 0.0
    if matches == 1:
        return 0.6
    if matches == 2:
        return 0.85
    return 1.0


class MockLLMAnalyzer(LLMAnalyzer):
    """
    Mock LLM 分析器。

    基于关键词匹配，无需 API key 或网络连接。
    用于测试、开发和无网络环境。
    评分逻辑简单但输出格式与 GeminiAnalyzer 完全一致。
    """

    def analyze(self, texts: list[str]) -> list[SignalScores]:
        results: list[SignalScores] = []
        for text in texts:
            text_lower = text.lower()

            reset = _keyword_score(text_lower, _RESET_KEYWORDS)
            limit = _keyword_score(text_lower, _LIMIT_KEYWORDS)
            release = _keyword_score(text_lower, _RELEASE_KEYWORDS)
            pressure = _keyword_score(text_lower, _PRESSURE_KEYWORDS)

            # 如果完全无匹配，置信度低
            has_any = any([reset, limit, release, pressure])
            confidence = 0.7 if has_any else 0.3

            reasons: list[str] = []
            if reset:
                reasons.append("检测到重置相关关键词")
            if limit:
                reasons.append("检测到限制/额度相关关键词")
            if release:
                reasons.append("检测到发布/更新相关关键词")
            if pressure:
                reasons.append("检测到社区压力/期待相关关键词")
            if not has_any:
                reasons.append("未检测到相关关键词，可能为无关内容")

            results.append(SignalScores(
                reset_signal=reset,
                limit_discussion=limit,
                release_signal=release,
                community_pressure=pressure,
                confidence=confidence,
                reason=reasons,
            ))
        return results


__all__ = [
    "LLMAnalyzer",
    "GeminiAnalyzer",
    "MockLLMAnalyzer",
]
