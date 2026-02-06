"""
DistilBERT Intent Classifier for LangGraph Workflow

Provides a fine-tuned DistilBERT model alternative to the OpenAI-based
intent classifier. Returns the same IntentClassification format for
seamless integration with the workflow.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from src.agents.langgraph_workflow import IntentClassification, MessageCategory


def _load_from_local_path(model_path: Path, tokenizer_cls, model_cls):
    """Load from local path, bypassing HF repo ID validation on Windows."""
    path_str = str(model_path)
    try:
        return (
            tokenizer_cls.from_pretrained(path_str, local_files_only=True),
            model_cls.from_pretrained(path_str, local_files_only=True),
        )
    except Exception as e:
        err_msg = str(e)
        if "Repo id must use alphanumeric" in err_msg or "repo_id" in err_msg.lower():
            import huggingface_hub.utils._validators as _validators

            _orig = getattr(_validators, "validate_repo_id", None)
            if _orig:

                def _noop(_):
                    pass

                _validators.validate_repo_id = _noop
                try:
                    return (
                        tokenizer_cls.from_pretrained(path_str, local_files_only=True),
                        model_cls.from_pretrained(path_str, local_files_only=True),
                    )
                finally:
                    _validators.validate_repo_id = _orig
        raise


class DistilBertIntentClassifier:
    """
    Intent classifier using fine-tuned DistilBERT model.

    Lazy loads the model on first prediction to avoid loading if not configured.
    """

    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize the DistilBERT intent classifier.

        Args:
            model_path: Path to fine-tuned model directory
            device: Device to run on ('cuda' or 'cpu'). Auto-detected if None.
        """
        self.model_path = model_path
        self.device = device
        self._model = None
        self._tokenizer = None

        # Category mapping: model outputs "chitchat" which isn't in workflow
        # We'll map it to EMOTIONAL (highest weight context)
        self.category_mapping = {
            "rag_query": MessageCategory.RAG_QUERY,
            "professional": MessageCategory.PROFESSIONAL,
            "psychological": MessageCategory.PSYCHOLOGICAL,
            "learning": MessageCategory.LEARNING,
            "social": MessageCategory.SOCIAL,
            "emotional": MessageCategory.EMOTIONAL,
            "aspirational": MessageCategory.ASPIRATIONAL,
            "chitchat": MessageCategory.CHITCHAT,
            "off_topic": MessageCategory.OFF_TOPIC,
        }

    def _load_model(self):
        """Lazy load the model and tokenizer."""
        if self._model is not None:
            return

        # Import here to avoid requiring transformers/torch for OpenAI classifier
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError:
            raise ImportError("DistilBERT classifier requires 'transformers' and 'torch'. Install with: pip install transformers torch")

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_path_resolved = Path(self.model_path).resolve()

        print(f"[DistilBERT] Loading intent classifier from {model_path_resolved} on {device}...")

        self._tokenizer, self._model = _load_from_local_path(model_path_resolved, AutoTokenizer, AutoModelForSequenceClassification)
        self._model.to(device)
        self._model.eval()

        print(f"[DistilBERT] Model loaded. Categories: {list(self._model.config.id2label.values())}")

    async def classify(self, message: str) -> IntentClassification:
        """
        Classify a message into one of 7 categories.

        Args:
            message: User message to classify

        Returns:
            IntentClassification with category, confidence, and metadata
        """
        # Lazy load model
        self._load_model()

        # Import torch here (already loaded in _load_model)
        import torch

        # Tokenize input
        inputs = self._tokenizer(message, return_tensors="pt", truncation=True, max_length=512, padding=True).to(self._model.device)

        # Get predictions
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]

        # Get predicted class and probabilities
        pred_class_id = torch.argmax(probs).item()
        model_category = self._model.config.id2label[pred_class_id]
        confidence = probs[pred_class_id].item()

        # Map to workflow category (handles chitchat -> EMOTIONAL)
        category = self.category_mapping.get(model_category, MessageCategory.EMOTIONAL)

        # Get all probabilities for secondary categories
        all_probs = {self._model.config.id2label[i]: float(probs[i].item()) for i in range(len(probs))}

        # Get top 3 categories (excluding the primary)
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)

        # Find secondary categories (excluding primary, only include if > 0.15 confidence)
        secondary_categories = []
        for cat_name, prob in sorted_probs[1:]:  # Skip first (primary)
            if prob > 0.15 and cat_name != model_category:
                mapped_cat = self.category_mapping.get(cat_name)
                if mapped_cat and mapped_cat not in secondary_categories:
                    secondary_categories.append(mapped_cat)
                if len(secondary_categories) >= 2:
                    break

        # Extract key entities based on category
        key_entities = self._extract_entities(message, category)

        # Build reasoning
        reasoning = f"DistilBERT classified as {category.value} with {confidence:.2%} confidence"
        if model_category == "chitchat" and category == MessageCategory.EMOTIONAL:
            reasoning += " (chitchat mapped to emotional)"

        return IntentClassification(
            category=category, confidence=float(confidence), reasoning=reasoning, key_entities=key_entities, secondary_categories=secondary_categories
        )

    def _extract_entities(self, message: str, category: MessageCategory) -> dict:
        """
        Extract basic entities from message based on category.

        This is a simple heuristic extraction. For production use,
        consider using a separate NER model.

        Args:
            message: User message
            category: Classified category

        Returns:
            Dictionary of extracted entities
        """
        entities = {}
        message_lower = message.lower()

        # Simple keyword-based entity extraction
        if category in [MessageCategory.PROFESSIONAL, MessageCategory.ASPIRATIONAL]:
            skills = []
            common_skills = [
                "python",
                "javascript",
                "java",
                "react",
                "node",
                "sql",
                "machine learning",
                "ai",
                "data science",
                "api",
                "rest",
                "docker",
                "kubernetes",
                "aws",
                "cloud",
                "leadership",
            ]
            for skill in common_skills:
                if skill in message_lower:
                    skills.append(skill)
            if skills:
                entities["skills"] = skills

        if category == MessageCategory.ASPIRATIONAL:
            goals = []
            goal_keywords = ["want to", "goal", "dream", "aspire", "become", "achieve", "hope to", "plan to"]
            for keyword in goal_keywords:
                if keyword in message_lower:
                    goals.append(message[:100])  # Use message snippet
                    break
            if goals:
                entities["goals"] = goals

        if category == MessageCategory.EMOTIONAL:
            emotions = []
            emotion_keywords = [
                "stressed",
                "anxious",
                "overwhelmed",
                "frustrated",
                "burned out",
                "exhausted",
                "worried",
                "confident",
                "excited",
                "motivated",
                "discouraged",
            ]
            for emotion in emotion_keywords:
                if emotion in message_lower:
                    emotions.append(emotion)
            if emotions:
                entities["emotions"] = emotions

        if category == MessageCategory.PSYCHOLOGICAL:
            values = []
            value_keywords = ["value", "important", "believe", "principle", "prefer", "care about", "matters to me"]
            for keyword in value_keywords:
                if keyword in message_lower:
                    values.append(message[:100])
                    break
            if values:
                entities["values"] = values

        return entities


@lru_cache(maxsize=1)
def get_distilbert_classifier(model_path: str) -> DistilBertIntentClassifier:
    """
    Get or create cached DistilBERT classifier.

    Args:
        model_path: Path to fine-tuned model directory

    Returns:
        Cached DistilBertIntentClassifier instance
    """
    return DistilBertIntentClassifier(model_path)


# Example usage
if __name__ == "__main__":
    import asyncio

    from src.config import INTENT_CLASSIFIER_MODEL_PATH

    async def test_classifier():
        classifier = get_distilbert_classifier(INTENT_CLASSIFIER_MODEL_PATH)

        test_messages = [
            "What is machine learning?",
            "I have 5 years of Python experience",
            "I want to become a data scientist",
            "I'm feeling overwhelmed with work",
            "My mentor helped me navigate my career",
            "I learn best through hands-on projects",
        ]

        print("=" * 80)
        print("Testing DistilBERT Intent Classifier")
        print("=" * 80)

        for msg in test_messages:
            result = await classifier.classify(msg)
            print(f"\nMessage: {msg}")
            print(f"Category: {result.category.value}")
            print(f"Confidence: {result.confidence:.2%}")
            print(f"Reasoning: {result.reasoning}")
            if result.secondary_categories:
                print(f"Secondary: {[c.value for c in result.secondary_categories]}")
            if result.key_entities:
                print(f"Entities: {result.key_entities}")

    asyncio.run(test_classifier())
