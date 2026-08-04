"""
LLM signal analyzer.

Converts natural language text (tweets, news) into structured machine-learning features.
The LLM does not directly predict reset; it only extracts signals.
Output SignalScores are passed to model/survival_model.py as prediction input.

Common interface:
    class LLMAnalyzer:
        analyze(texts: list[str]) -> list[SignalScores]

Current implementations:
    - GeminiAnalyzer: Uses Gemini API (preferred)
    - MockLLMAnalyzer: Keyword matching (no API key needed, for testing)
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Optional

from model.data_models import SignalScores, Tweet


# ──────────────────────────────────────────────
# Common interface
# ──────────────────────────────────────────────

class LLMAnalyzer(ABC):
    """
    Abstract base class for LLM signal analyzers.

    All analyzers (Gemini, OpenAI, Mock, etc.) inherit from this class.
    The core method analyze() receives a list of texts and returns SignalScores for each.
    """

    @abstractmethod
    def analyze(self, texts: list[str]) -> list[SignalScores]:
        """
        Analyze a list of texts and return structured signal scores.

        Args:
            texts: List of natural language texts to analyze

        Returns:
            A list of SignalScores with the same length as the input, one per text
        """
        ...

    def analyze_tweets(self, tweets: list[Tweet]) -> list[SignalScores]:
        """Convenience method: analyze a list of Tweet objects directly"""
        return self.analyze([t.text for t in tweets])

    def analyze_batch(self, texts: list[str]) -> SignalScores:
        """
        Batch analyze and return aggregated signal scores.

        Averages SignalScores across all texts,
        suitable for scenarios requiring an overall signal overview.
        """
        scores = self.analyze(texts)
        if not scores:
            return SignalScores(
                reset_intent=0.0,
                limit_complaint=0.0,
                official_change=0.0,
                reset_confirmation=0.0,
                confidence=0.0,
                reason=["No input texts"],
            )
        n = len(scores)
        all_reasons: list[str] = []
        for s in scores:
            all_reasons.extend(s.reason[:2])
        return SignalScores(
            reset_intent=sum(s.reset_intent for s in scores) / n,
            limit_complaint=sum(s.limit_complaint for s in scores) / n,
            official_change=sum(s.official_change for s in scores) / n,
            reset_confirmation=sum(s.reset_confirmation for s in scores) / n,
            confidence=sum(s.confidence for s in scores) / n,
            reason=all_reasons[:5] if all_reasons else ["Batch aggregation"],
        )


# ──────────────────────────────────────────────
# Gemini implementation
# ──────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a signal analysis assistant specialized in analyzing social media posts and news about ChatGPT/Codex usage quota resets.

For each text, evaluate the following dimensions as floating point numbers from 0.0 to 1.0. Use the FULL 0-1 scale; do not default to 0.0 or 1.0. Calibrate your scores so that small hints receive small positive values and only explicit statements receive high values.

1. reset_intent: Does the text discuss or imply an upcoming quota reset?
   - 0.0: no relevance
   - 0.1-0.3: vague hint, teaser, "there will be signs", "something is coming" (not about reset specifically)
   - 0.4-0.6: moderate signal, mentions reset in general terms or asks about timing
   - 0.7-0.9: strong signal, explicitly says a reset is coming soon
   - 1.0: guarantees an imminent reset, e.g. "we will reset usage limits in the next hour"

2. limit_complaint: Does the text reflect complaints about usage limits/quota exhaustion?
   - 0.0: none
   - 0.1-0.3: mild mention of limits/capacity
   - 0.4-0.6: explicit complaint about hitting limits
   - 0.7-1.0: widespread or urgent complaints about quotas being exhausted

3. official_change: Does the text come from an official source or imply a product/policy change?
   - 0.0: none
   - 0.1-0.3: minor product update mention
   - 0.4-0.6: official release or policy change discussed
   - 0.7-1.0: major launch/change explicitly announced by OpenAI/Tibo

4. reset_confirmation: Does the text explicitly confirm that a reset has occurred or will occur?
   - 0.0: no confirmation
   - 0.1-0.3: indirect or vague confirmation
   - 0.4-0.6: clear confirmation of a past or future reset
   - 0.7-1.0: explicit, unambiguous confirmation ("I have reset usage limits", "reset is live now")

5. confidence: Your overall confidence in the above scores (0=very uncertain, 1=very certain)

CRITICAL: distinguish past-tense confirmations from future-tense signals.
- If the text says a reset has ALREADY happened (e.g. "I have reset usage limits", "we have reset"), set reset_confirmation HIGH (0.8-1.0) but reset_intent LOW (0.0-0.2), because the reset already occurred and does NOT predict another one soon.
- If the text says a reset is coming / will happen soon (e.g. "we will reset", "reset coming", "lands in the next hour"), set reset_intent HIGH (0.7-1.0) and reset_confirmation HIGH (0.7-1.0).
- A vague teaser about future plans that does NOT mention resets (e.g. "there will be signs", "major breakthroughs") should score LOW but not necessarily zero: reset_intent 0.0-0.2, reset_confirmation 0.0, because it contains no actionable reset signal.

Examples:
- "I have reset usage limits for Codex and ChatGPT" -> reset_confirmation: 0.95, reset_intent: 0.05, official_change: 0.0, limit_complaint: 0.0
- "We will reset usage limits tonight" -> reset_intent: 0.9, reset_confirmation: 0.85, official_change: 0.0, limit_complaint: 0.0
- "There will be signs" -> reset_intent: 0.1, reset_confirmation: 0.0, official_change: 0.0, limit_complaint: 0.0
- "Users are hitting the rate limit again" -> limit_complaint: 0.7, reset_intent: 0.2, reset_confirmation: 0.0
- "New GPT model shipped today" -> official_change: 0.8, reset_intent: 0.0, reset_confirmation: 0.0

Scoring principles:
- A user simply complaining about limit does not mean a reset is imminent; limit_complaint should not directly push up reset probability.
- When Tibo or an official source clearly says "will reset soon / about to reset", reset_intent should be high.
- When an official release mentions new features but not reset, official_change can be high while reset_confirmation should remain low.
- Score 0.0 only when the text is completely irrelevant. Even weak hints should receive 0.1-0.2.

Also provide a reason list briefly explaining the scoring rationale (each reason no more than one sentence).

Return results strictly in JSON format. Return a JSON array where each element corresponds to one input text:
[
  {
    "reset_intent": 0.0,
    "limit_complaint": 0.0,
    "official_change": 0.0,
    "reset_confirmation": 0.0,
    "confidence": 0.0,
    "reason": ["reason1", "reason2"]
  }
]

Do not include any other text; return only the JSON array."""


def _extract_json_array(text: str) -> list[dict]:
    """Extract JSON array from LLM response"""
    # Try direct parsing first
    text = text.strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(1).strip())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Try finding the first [ and last ]
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
    """Convert dict to SignalScores, handling types, defaults, and legacy field names"""
    def _get_float(key: str, fallback_keys: Optional[list[str]] = None) -> float:
        keys = [key]
        if fallback_keys:
            keys.extend(fallback_keys)
        for k in keys:
            if k in d:
                val = d[k]
                try:
                    return max(0.0, min(1.0, float(val)))
                except (TypeError, ValueError):
                    return 0.0
        return 0.0

    def _get_reasons() -> list[str]:
        val = d.get("reason", [])
        if isinstance(val, list):
            return [str(r) for r in val]
        if isinstance(val, str):
            return [val]
        return []

    return SignalScores(
        reset_intent=_get_float("reset_intent", ["reset_signal"]),
        limit_complaint=_get_float("limit_complaint", ["limit_discussion"]),
        official_change=_get_float("official_change", ["release_signal"]),
        reset_confirmation=_get_float("reset_confirmation"),
        confidence=_get_float("confidence"),
        reason=_get_reasons(),
    )


class GeminiAnalyzer(LLMAnalyzer):
    """
    Gemini API signal analyzer.

    Uses the Google Gemini API for natural language analysis.
    Requires GEMINI_API_KEY environment variable.
    The google-generativeai package is lazily imported; a clear error is raised if not installed.
    """

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self._api_key = api_key
        self._model = model
        self._client = None

    def _ensure_client(self):
        """Lazily initialize the Gemini client"""
        if self._client is not None:
            return
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai is not installed. "
                "Please run: pip install google-generativeai"
            )
        genai.configure(api_key=self._api_key)
        self._client = genai.GenerativeModel(
            self._model,
            system_instruction=_SYSTEM_PROMPT,
        )

    def analyze(self, texts: list[str]) -> list[SignalScores]:
        """Call Gemini API to analyze texts"""
        if not texts:
            return []

        self._ensure_client()

        # Build user input
        lines = []
        for i, text in enumerate(texts, 1):
            lines.append(f"[{i}] {text}")
        user_input = "\n".join(lines)

        response = self._client.generate_content(user_input)
        response_text = response.text

        # Parse JSON response
        raw_list = _extract_json_array(response_text)

        # Ensure output length matches input
        results: list[SignalScores] = []
        for i, text in enumerate(texts):
            if i < len(raw_list):
                results.append(_dict_to_scores(raw_list[i]))
            else:
                results.append(SignalScores(
                    reset_intent=0.0,
                    limit_complaint=0.0,
                    official_change=0.0,
                    reset_confirmation=0.0,
                    confidence=0.0,
                    reason=["LLM response incomplete"],
                ))
        return results


# ──────────────────────────────────────────────
# DeepSeek implementation (OpenAI-compatible API)
# ──────────────────────────────────────────────

class DeepSeekAnalyzer(LLMAnalyzer):
    """
    DeepSeek API signal analyzer.

    Calls the DeepSeek model via an OpenAI-compatible interface.
    Requires DEEPSEEK_API_KEY environment variable.
    The openai package is lazily imported; a clear error is raised if not installed.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com/v1",
    ):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._client = None

    def _ensure_client(self):
        """Lazily initialize the DeepSeek client"""
        if self._client is not None:
            return
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai is not installed. Please run: pip install openai"
            )
        self._client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
        )

    def analyze(self, texts: list[str]) -> list[SignalScores]:
        """Call DeepSeek API to analyze texts"""
        if not texts:
            return []

        self._ensure_client()

        lines = []
        for i, text in enumerate(texts, 1):
            lines.append(f"[{i}] {text}")
        user_input = "\n".join(lines)

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
            temperature=0.1,
        )
        response_text = response.choices[0].message.content or ""

        raw_list = _extract_json_array(response_text)

        results: list[SignalScores] = []
        for i, text in enumerate(texts):
            if i < len(raw_list):
                results.append(_dict_to_scores(raw_list[i]))
            else:
                results.append(SignalScores(
                    reset_intent=0.0,
                    limit_complaint=0.0,
                    official_change=0.0,
                    reset_confirmation=0.0,
                    confidence=0.0,
                    reason=["LLM response incomplete"],
                ))
        return results


# ──────────────────────────────────────────────
# Mock implementation (for testing and environments without API keys)
# ──────────────────────────────────────────────

# Keyword mapping tables (English only)
_RESET_KEYWORDS = [
    "reset", "usage reset", "limit reset", "quota reset",
    "resetted", "has been reset",
]
_LIMIT_KEYWORDS = [
    "limit", "quota", "exhausted", "rate limit", "usage limit",
    "capacity", "capacity limit", "hit the cap",
]
_RELEASE_KEYWORDS = [
    "release", "update", "announce", "launch",
    "new version", "rollout", "deploy",
]
_PRESSURE_KEYWORDS = [
    "please", "want", "when", "waiting", "anyone",
    "urgent", "anyone knows", "does anyone",
]


def _keyword_score(text_lower: str, keywords: list[str]) -> float:
    """Compute score based on keyword matching"""
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
    Mock LLM analyzer.

    Based on keyword matching; no API key or network connection required.
    Used for testing, development, and offline environments.
    Scoring logic is simple but output format is fully consistent with GeminiAnalyzer.
    """

    def analyze(self, texts: list[str]) -> list[SignalScores]:
        results: list[SignalScores] = []
        for text in texts:
            text_lower = text.lower()

            reset = _keyword_score(text_lower, _RESET_KEYWORDS)
            limit = _keyword_score(text_lower, _LIMIT_KEYWORDS)
            release = _keyword_score(text_lower, _RELEASE_KEYWORDS)
            pressure = _keyword_score(text_lower, _PRESSURE_KEYWORDS)

            # Low confidence if there is no match at all
            has_any = any([reset, limit, release, pressure])
            confidence = 0.7 if has_any else 0.3

            reasons: list[str] = []
            if reset:
                reasons.append("Reset-related keywords detected")
            if limit:
                reasons.append("Limit/quota-related keywords detected")
            if release:
                reasons.append("Release/update-related keywords detected")
            if pressure:
                reasons.append("Community pressure/expectation keywords detected")
            if not has_any:
                reasons.append("No relevant keywords detected; possibly unrelated content")

            results.append(SignalScores(
                reset_intent=reset,
                limit_complaint=limit,
                official_change=release,
                reset_confirmation=reset if "reset" in text_lower else 0.0,
                confidence=confidence,
                reason=reasons,
            ))
        return results


__all__ = [
    "LLMAnalyzer",
    "GeminiAnalyzer",
    "DeepSeekAnalyzer",
    "MockLLMAnalyzer",
]
