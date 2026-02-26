"""
Intent Classifier for LangGraph Workflow

Provides a fine-tuned transformer model alternative to the OpenAI-based
intent classifier. Returns the same IntentClassification format for
seamless integration with the workflow.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from src.agents.langgraph_workflow import ActiveClassification, IntentClassification, MessageCategory


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
    Intent classifier using fine-tuned transformer model.

    Lazy loads the model on first prediction to avoid loading if not configured.
    """

    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize the intent classifier.

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
            "personal": MessageCategory.PERSONAL,
            "chitchat": MessageCategory.CHITCHAT,
            "meta": MessageCategory.META,
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
            raise ImportError("Intent classifier requires 'transformers' and 'torch'.Install with: pip install transformers torch")

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_path_resolved = Path(self.model_path).resolve()

        print(f"[Intent Classifier] Loading intent classifier from {model_path_resolved} on {device}...")

        self._tokenizer, self._model = _load_from_local_path(model_path_resolved, AutoTokenizer, AutoModelForSequenceClassification)
        self._model.to(device)
        self._model.eval()

        print(f"[Intent Classifier] Model loaded. Categories: {list(self._model.config.id2label.values())}")

    async def classify(self, message: str, multilabel_threshold: float = 0.5) -> IntentClassification:
        """
        Classify a message using multi-label classification (sigmoid activations).

        The primary classifier uses sigmoid + BCE, so multiple categories can be
        active simultaneously.  All categories whose sigmoid probability exceeds
        ``multilabel_threshold`` are returned as active classifications.  At least
        one category is always returned (the highest-probability one).

        Args:
            message: User message to classify
            multilabel_threshold: Minimum sigmoid probability to consider a
                category active (default 0.5).

        Returns:
            IntentClassification where:
            - ``category`` / ``confidence`` describe the highest-probability active category
            - ``active_classifications`` lists ALL active (category, confidence) pairs
            - ``secondary_categories`` lists additional active categories (backward-compat)
        """
        # Lazy load model
        self._load_model()

        import torch

        # Tokenize input
        inputs = self._tokenizer(
            message, return_tensors="pt", truncation=True, max_length=512, padding=True
        ).to(self._model.device)

        # Multi-label forward pass: sigmoid instead of softmax
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            probs = torch.sigmoid(logits)[0]  # shape: (num_labels,)

        # Determine active categories (above threshold)
        active_mask = probs >= multilabel_threshold
        if not active_mask.any():
            # Always activate at least the top-1
            active_mask[torch.argmax(probs)] = True

        active_indices: list[int] = active_mask.nonzero(as_tuple=True)[0].tolist()
        # Sort by probability descending
        active_indices.sort(key=lambda i: probs[i].item(), reverse=True)

        # Build ActiveClassification list (one per active label)
        active_classifications: list[ActiveClassification] = []
        for idx in active_indices:
            model_cat = self._model.config.id2label[idx]
            mapped_cat = self.category_mapping.get(model_cat, MessageCategory.PSYCHOLOGICAL)
            conf = float(probs[idx].item())
            active_classifications.append(ActiveClassification(category=mapped_cat, confidence=conf))

        # Primary = highest-probability active category
        primary_cat = active_classifications[0].category
        primary_conf = active_classifications[0].confidence
        primary_model_cat = self._model.config.id2label[active_indices[0]]

        # Build reasoning
        active_vals = [ac.category.value for ac in active_classifications]
        reasoning = (
            f"Multi-label classifier: active={active_vals} "
            f"(primary={primary_cat.value} @ {primary_conf:.2%})"
        )
        if primary_model_cat == "chitchat" and primary_cat == MessageCategory.PSYCHOLOGICAL:
            reasoning += " (chitchat mapped to psychological)"

        # Extract key entities for primary category
        key_entities = self._extract_entities(message, primary_cat)

        # secondary_categories kept for backward compatibility
        secondary_categories = [ac.category for ac in active_classifications[1:]]

        return IntentClassification(
            category=primary_cat,
            confidence=primary_conf,
            reasoning=reasoning,
            key_entities=key_entities,
            secondary_categories=secondary_categories,
            active_classifications=active_classifications,
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
        if category == MessageCategory.PROFESSIONAL:
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

        if category == MessageCategory.PROFESSIONAL:
            goals = []
            goal_keywords = ["want to", "goal", "dream", "aspire", "become", "achieve", "hope to", "plan to"]
            for keyword in goal_keywords:
                if keyword in message_lower:
                    goals.append(message[:100])
                    break
            if goals:
                entities["goals"] = goals

        if category == MessageCategory.PSYCHOLOGICAL:
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


class SpanClassifier:
    """
    Token-level span classifier for hierarchical intent classification.

    Trained on inline-tagged multiclass data where each clause ends with a
    [category, subcategory] label.  At inference time the model assigns a
    (category, subcategory) label to every token; consecutive tokens sharing
    the same label are merged into a single span.

    Supports BIO-prefixed labels ("B-professional.work_history") and plain
    labels ("professional.work_history").
    """

    def __init__(self, model_path: str, device: Optional[str] = None):
        self.model_path = model_path
        self.device = device
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer
        except ImportError:
            raise ImportError(
                "SpanClassifier requires 'transformers' and 'torch'. "
                "Install with: pip install transformers torch"
            )

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_path_resolved = Path(self.model_path).resolve()

        print(f"[Span Classifier] Loading from {model_path_resolved} on {device}...")
        self._tokenizer, self._model = _load_from_local_path(
            model_path_resolved, AutoTokenizer, AutoModelForTokenClassification
        )
        self._model.to(device)
        self._model.eval()
        print(f"[Span Classifier] Model loaded. Labels: {list(self._model.config.id2label.values())}")

    @staticmethod
    def _parse_label(label: str) -> "tuple[str, str | None] | None":
        """
        Parse a token-classification label into (category, subcategory).

        Handles:
          "O"                           → None  (non-entity)
          "professional"                → ("professional", None)  (primary classifier)
          "professional.work_history"   → ("professional", "work_history")
          "B-professional.work_history" → ("professional", "work_history")
          "I-professional.work_history" → ("professional", "work_history")
        """
        if label in ("O", "o"):
            return None
        # Strip BIO prefix
        if label.startswith(("B-", "I-")):
            label = label[2:]
        if "." in label:
            cat, sub = label.split(".", 1)
            return (cat.strip(), sub.strip())
        # Plain single-word label (e.g. primary classifier outputs "professional")
        return (label.strip(), None)

    def classify_spans(self, message: str) -> "list[dict]":
        """
        Run token classification and return labeled text spans.

        Returns:
            List of {"text": str, "category": str, "subcategory": str,
                     "start": int, "end": int}, in order of appearance.
        """
        self._load_model()

        import torch

        encoding = self._tokenizer(
            message,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
        )
        offset_mapping = encoding.pop("offset_mapping")[0].tolist()
        inputs = {k: v.to(self._model.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            pred_ids = torch.argmax(outputs.logits[0], dim=-1).tolist()

        spans: list[dict] = []
        current: dict | None = None

        for pred_id, (char_start, char_end) in zip(pred_ids, offset_mapping):
            if char_start == char_end == 0:  # special token (CLS/SEP/PAD)
                continue

            label = self._model.config.id2label[pred_id]
            parsed = self._parse_label(label)

            if parsed is None:
                if current is not None:
                    spans.append(current)
                    current = None
            else:
                cat, sub = parsed
                if current is None or current["category"] != cat or current["subcategory"] != sub:
                    if current is not None:
                        spans.append(current)
                    current = {
                        "text": message[char_start:char_end],
                        "category": cat,
                        "subcategory": sub,
                        "start": char_start,
                        "end": char_end,
                    }
                else:
                    # Extend current span
                    current["end"] = char_end
                    current["text"] = message[current["start"]:char_end]

        if current is not None:
            spans.append(current)

        return spans


@lru_cache(maxsize=None)
def get_span_classifier(model_path: str) -> SpanClassifier:
    """
    Get or create cached span classifier.

    Args:
        model_path: Path to fine-tuned token classification model directory

    Returns:
        Cached SpanClassifier instance
    """
    return SpanClassifier(model_path)


@lru_cache(maxsize=1)
def get_intent_classifier(model_path: str) -> DistilBertIntentClassifier:
    """
    Get or create cached intent classifier.

    Args:
        model_path: Path to fine-tuned model directory

    Returns:
        Cached intent classifier instance
    """
    return DistilBertIntentClassifier(model_path)


# Example usage
if __name__ == "__main__":
    import asyncio

    from src.config import INTENT_CLASSIFIER_MODEL_PATH

    async def test_classifier():
        classifier = get_intent_classifier(INTENT_CLASSIFIER_MODEL_PATH)

        test_messages = [
            "What is machine learning?",
            "I have 5 years of Python experience",
            "I want to become a data scientist",
            "I'm feeling overwhelmed with work",
            "My mentor helped me navigate my career",
            "I learn best through hands-on projects",
        ]

        print("=" * 80)
        print("Testing Intent Classifier")
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
