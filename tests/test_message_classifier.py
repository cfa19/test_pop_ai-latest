"""
Tests for Message Worthiness Classifier

Tests cover:
    - Trivial message pre-filter (any language)
    - Heuristic classifier (English keywords)
    - Language detection (lingua-py)
    - Hybrid classifier flow
"""

from unittest.mock import MagicMock

import pytest

from src.utils.message_classifier import (
    MessageType,
    MessageWorthiness,
    classify_message_heuristic,
    classify_message_hybrid,
    classify_message_llm,
    detect_language,
    is_trivial_message,
)


# ===============================
# TRIVIAL MESSAGE PRE-FILTER
# ===============================
class TestIsTrivialMessage:
    """Tests for the trivial message pre-filter (any language, no API call)."""

    def test_english_greeting(self):
        assert is_trivial_message("hi") is True

    def test_english_thanks(self):
        assert is_trivial_message("thanks") is True

    def test_english_ok(self):
        assert is_trivial_message("ok") is True

    def test_spanish_greeting(self):
        assert is_trivial_message("hola") is True

    def test_spanish_thanks(self):
        assert is_trivial_message("gracias") is True

    def test_spanish_yes(self):
        assert is_trivial_message("sí") is True

    def test_french_greeting(self):
        assert is_trivial_message("bonjour") is True

    def test_french_thanks(self):
        assert is_trivial_message("merci") is True

    def test_long_message_not_trivial(self):
        assert is_trivial_message("I want to become a data scientist") is False

    def test_short_but_meaningful(self):
        # "I was fired" is 3 words but not in the trivial list
        assert is_trivial_message("I was fired") is False

    def test_four_words_not_trivial(self):
        # 4+ words are never trivial regardless of content
        assert is_trivial_message("ok ok ok ok") is False


# ===============================
# LANGUAGE DETECTION
# ===============================
class TestDetectLanguage:
    """Tests for lingua-py language detection."""

    def test_detect_english(self):
        result = detect_language("I want to become a senior developer")
        assert result == "en"

    def test_detect_spanish(self):
        result = detect_language("Quiero convertirme en desarrollador senior")
        assert result == "es"

    def test_detect_french(self):
        result = detect_language("Je voudrais devenir développeur senior")
        assert result == "fr"

    def test_long_english_text(self):
        text = (
            "I have been working as a software engineer for the past five "
            "years at a major tech company. My primary focus has been on "
            "backend development using Python and Go."
        )
        result = detect_language(text)
        assert result == "en"

    def test_long_spanish_text(self):
        text = (
            "He estado trabajando como ingeniero de software durante los "
            "últimos cinco años en una empresa de tecnología. Mi enfoque "
            "principal ha sido el desarrollo backend con Python y Go."
        )
        result = detect_language(text)
        assert result == "es"


# ===============================
# HEURISTIC CLASSIFIER
# ===============================
class TestClassifyMessageHeuristic:
    """Tests for the heuristic keyword-based classifier (English only)."""

    def test_high_value_career_goal(self):
        message = "I want to become a senior data scientist specializing in NLP within the next 2 years"
        result = classify_message_heuristic(message)
        assert result.worthiness == MessageWorthiness.HIGH
        assert result.score >= 70

    def test_high_value_skills(self):
        message = "I have 5 years of experience with Python, Django, and React. I've built several production APIs."
        result = classify_message_heuristic(message)
        assert result.worthiness == MessageWorthiness.HIGH
        assert result.score >= 70

    def test_high_value_achievement(self):
        message = "I led a team of 8 developers and delivered a platform that increased revenue by 30 percent"
        result = classify_message_heuristic(message)
        assert result.worthiness == MessageWorthiness.HIGH
        assert result.score >= 70

    def test_high_value_challenge(self):
        message = "I struggle with imposter syndrome when leading team meetings and presenting to executives"
        result = classify_message_heuristic(message)
        assert result.worthiness == MessageWorthiness.HIGH
        assert result.score >= 70

    def test_low_value_greeting(self):
        message = "Thanks!"
        result = classify_message_heuristic(message)
        assert result.worthiness == MessageWorthiness.LOW
        assert result.score < 40

    def test_low_value_ok(self):
        message = "Ok, got it"
        result = classify_message_heuristic(message)
        assert result.worthiness == MessageWorthiness.LOW
        assert result.score < 40

    def test_low_value_hello(self):
        message = "Hello!"
        result = classify_message_heuristic(message)
        assert result.worthiness == MessageWorthiness.LOW
        assert result.score < 40

    def test_medium_value_context(self):
        message = "I have been thinking about my career path and where I should go next"
        result = classify_message_heuristic(message)
        assert result.worthiness in [
            MessageWorthiness.MEDIUM,
            MessageWorthiness.HIGH,
        ]
        assert result.score >= 40

    def test_preference_detected(self):
        message = "I prefer remote work and I really enjoy building backend systems"
        result = classify_message_heuristic(message)
        assert result.score >= 60
        assert "preference" in result.reason.lower()

    def test_score_capped_at_100(self):
        message = (
            "My career goal is to become a CTO. I have experience with "
            "Python and I led a team that built and launched a product. "
            "I struggle with delegation but I prefer working on hard problems "
            "and I value autonomy and growth."
        )
        result = classify_message_heuristic(message)
        assert result.score <= 100


# ===============================
# LLM CLASSIFIER (MOCKED)
# ===============================
class TestClassifyMessageLlm:
    """Tests for the LLM classifier with mocked OpenAI calls."""

    @pytest.mark.asyncio
    async def test_llm_high_value(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            '{"worthiness": "high", "score": 88, '
            '"reason": "career transition goal", '
            '"message_type": "career_goal", '
            '"extracted_entities": {"skills": [], "goals": ["project manager"], "experiences": []}}'
        )
        mock_client.chat.completions.create.return_value = mock_response

        result = await classify_message_llm(
            "I want to transition into project management",
            mock_client,
        )

        assert result.worthiness == MessageWorthiness.HIGH
        assert result.score == 88
        assert result.message_type == MessageType.CAREER_GOAL

    @pytest.mark.asyncio
    async def test_llm_low_value(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            '{"worthiness": "low", "score": 5, '
            '"reason": "greeting", '
            '"message_type": "acknowledgment", '
            '"extracted_entities": {"skills": [], "goals": [], "experiences": []}}'
        )
        mock_client.chat.completions.create.return_value = mock_response

        result = await classify_message_llm("Hello!", mock_client)

        assert result.worthiness == MessageWorthiness.LOW
        assert result.score == 5

    @pytest.mark.asyncio
    async def test_llm_extracts_entities(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            '{"worthiness": "high", "score": 92, '
            '"reason": "mentions skills and experience", '
            '"message_type": "skill", '
            '"extracted_entities": {"skills": ["Python", "React"], "goals": [], "experiences": ["5 years"]}}'
        )
        mock_client.chat.completions.create.return_value = mock_response

        result = await classify_message_llm(
            "I have 5 years of experience with Python and React",
            mock_client,
        )

        assert result.extracted_entities is not None
        assert "Python" in result.extracted_entities["skills"]
        assert "React" in result.extracted_entities["skills"]


# ===============================
# HYBRID CLASSIFIER
# ===============================
class TestClassifyMessageHybrid:
    """Tests for the hybrid classifier flow."""

    @pytest.mark.asyncio
    async def test_trivial_message_skips_everything(self):
        mock_client = MagicMock()

        result = await classify_message_hybrid("ok", mock_client)

        assert result.worthiness == MessageWorthiness.LOW
        assert result.score == 10
        # No API calls should have been made
        mock_client.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_clear_high_english_skips_llm(self):
        mock_client = MagicMock()

        # Long message with multiple keyword categories -> score >=80, no LLM needed
        result = await classify_message_hybrid(
            "My career goal is to become a lead engineer. I have years of experience with "
            "Python and React, and I led a team that built and launched a production platform.",
            mock_client,
        )

        assert result.worthiness == MessageWorthiness.HIGH
        assert result.score >= 80
        # No LLM call needed - heuristic was clear enough
        mock_client.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_english_triggers_translation(self):
        mock_client = MagicMock()

        # Mock translation response
        mock_translate = MagicMock()
        mock_translate.choices = [MagicMock()]
        mock_translate.choices[0].message.content = "I have 10 years of experience as a Python developer and I want to become a tech lead"
        mock_client.chat.completions.create.return_value = mock_translate

        result = await classify_message_hybrid(
            "Tengo 10 años de experiencia como desarrollador Python y quiero convertirme en tech lead",
            mock_client,
        )

        # Translation should have been called
        mock_client.chat.completions.create.assert_called_once()
        assert result.original_language == "es"
        assert result.translated_text is not None

    @pytest.mark.asyncio
    async def test_uncertain_score_triggers_llm(self):
        mock_client = MagicMock()

        # Mock LLM classification response (called when heuristic is uncertain)
        mock_llm = MagicMock()
        mock_llm.choices = [MagicMock()]
        mock_llm.choices[0].message.content = (
            '{"worthiness": "high", "score": 82, '
            '"reason": "career reflection", '
            '"message_type": "career_goal", '
            '"extracted_entities": {"skills": [], "goals": ["career change"], "experiences": []}}'
        )
        mock_client.chat.completions.create.return_value = mock_llm

        # This message scores 60 (within ±10 of default threshold 60), triggering LLM
        result = await classify_message_hybrid(
            "I am interested in learning more about becoming a better professional",
            mock_client,
        )

        assert result.worthiness == MessageWorthiness.HIGH
        assert result.score == 82


# ===============================
# INTEGRATION TEST: CLASSIFIER ACCURACY
# ===============================
class TestClassifierAccuracy:
    """
    Test that the heuristic correctly classifies sample messages.
    These are the examples from the ticket.
    """

    HIGH_VALUE_MESSAGES = [
        "I want to transition into product management within the next year",
        "I have 3 years of experience as a frontend developer with React and TypeScript",
        "I completed a machine learning certification from Stanford last month",
    ]

    LOW_VALUE_MESSAGES = [
        "Thanks!",
        "Hello",
        "Ok",
        "Yes",
        "Goodbye",
    ]

    def test_high_value_accuracy(self):
        correct = 0
        for msg in self.HIGH_VALUE_MESSAGES:
            result = classify_message_heuristic(msg)
            if result.worthiness == MessageWorthiness.HIGH:
                correct += 1

        accuracy = correct / len(self.HIGH_VALUE_MESSAGES)
        assert accuracy >= 0.8, f"High-value accuracy {accuracy:.0%} below 80% threshold"

    def test_low_value_accuracy(self):
        correct = 0
        for msg in self.LOW_VALUE_MESSAGES:
            result = classify_message_heuristic(msg)
            if result.worthiness == MessageWorthiness.LOW:
                correct += 1

        accuracy = correct / len(self.LOW_VALUE_MESSAGES)
        assert accuracy >= 0.9, f"Low-value accuracy {accuracy:.0%} below 90% threshold"
