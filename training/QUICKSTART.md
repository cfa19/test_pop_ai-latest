# Quick Start Guide

Get started training models to replace LangGraph workflow nodes in 5 steps.

## Step 1: Install Dependencies

```bash
cd training
pip install -r requirements.txt
```

## Step 2: Get Training Data

You have two options:

### Option A: Export Real Conversations (Recommended if available)

Export messages from Supabase:

```bash
python scripts/prepare_data.py \
    --task export \
    --output data/raw/conversations.csv \
    --limit 2000
```

Then generate labels:

```bash
python scripts/prepare_data.py \
    --task intent_labels \
    --input data/raw/conversations.csv \
    --output data/processed/intent_labeled.csv \
    --use-llm
```

### Option B: Generate Synthetic Data (If conversation_history is empty)

Use GPT-3.5-turbo to generate diverse synthetic training data:

```bash
python scripts/generate_data.py \
    --output data/processed/synthetic_labeled.csv \
    --rag-count 100 \
    --context-count 80 \
    --chitchat-count 60
```

This generates:
- 100 RAG query messages (based on your knowledge base)
- 80 messages per Store A context (6 contexts = 480 total)
- 60 conversational chitchat messages
- **Total: 640 labeled examples**

**Cost:** ~$0.20-0.30 for GPT-3.5-turbo API calls

## Step 3: Train Model

Train the intent classifier directly from the generated category files:

```bash
python scripts/train_intent_classifier.py \
    --model distilbert-base-uncased \
    --train-data data/processed \
    --output models/checkpoints/intent_classifier \
    --epochs 3 \
    --batch-size 16 \
    --val-split 0.2
```

**Note:** The training script automatically:
- Loads all category CSV files from the directory
- Combines them into a training dataset
- Splits 80% for training, 20% for validation
- Uses stratified splitting to maintain class balance

**Training time:** ~10-30 minutes depending on data size and hardware.

**GPU recommended** but not required. CPU training will be slower.

## Step 4: Evaluate

Evaluate on test set:

```bash
python scripts/evaluate.py \
    --task intent \
    --model models/checkpoints/intent_classifier/final \
    --test-data data/processed/intent/test.csv \
    --output results/intent_classifier/
```

Review the results:
- `results/intent_classifier/confusion_matrix.png` - Visual confusion matrix
- `results/intent_classifier/evaluation_report.txt` - Detailed metrics
- `results/intent_classifier/predictions.csv` - All predictions

## Integration into Workflow

Once you're satisfied with the model performance:

### 1. Update src/config.py

```python
# Add at the end of config.py
INTENT_CLASSIFIER_MODEL = os.getenv(
    "INTENT_CLASSIFIER_MODEL",
    "training/models/checkpoints/intent_classifier/final"
)
USE_FINETUNED_INTENT_CLASSIFIER = os.getenv(
    "USE_FINETUNED_INTENT_CLASSIFIER",
    "false"
).lower() == "true"
```

### 2. Update src/agents/langgraph_workflow.py

At the top of the file:

```python
from src.config import INTENT_CLASSIFIER_MODEL, USE_FINETUNED_INTENT_CLASSIFIER

# Load model once at startup
if USE_FINETUNED_INTENT_CLASSIFIER:
    from training.inference import IntentClassifierModel
    intent_classifier_model = IntentClassifierModel(INTENT_CLASSIFIER_MODEL)
```

In the `intent_classifier_node` function:

```python
def intent_classifier_node(state: WorkflowState) -> WorkflowState:
    """Classify message intent"""

    if USE_FINETUNED_INTENT_CLASSIFIER:
        # Use fine-tuned BERT model
        result = intent_classifier_model.predict(state["message"])

        state["unified_classification"] = IntentClassification(
            category=MessageCategory(result["category"]),
            confidence=result["confidence"],
            reasoning="Classified by fine-tuned BERT model",
            key_entities={},
            secondary_categories=[]
        )
    else:
        # Use existing LLM-based classification
        # ... existing code ...

    return state
```

### 3. Update .env

```env
# Enable fine-tuned model
USE_FINETUNED_INTENT_CLASSIFIER=true
INTENT_CLASSIFIER_MODEL=training/models/checkpoints/intent_classifier/final
```

### 4. Test

Restart the server and send test requests:

```bash
python main.py
```

```bash
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is machine learning?"}'
```

Check the classification result in the response.

## Tips for Better Models

### 1. Get More Data
- **Minimum:** 500 labeled examples (70 per category)
- **Good:** 1000+ examples (140+ per category)
- **Great:** 2000+ examples (285+ per category)

### 2. Balance Classes
If some categories have far fewer examples:

```python
# In train_intent_classifier.py, set:
use_class_weights=True  # Already default
```

### 3. Try Different Models

**Fast (recommended for production):**
```bash
python scripts/train_intent_classifier.py \
    --model distilbert-base-uncased \
    ...
```

**Accurate:**
```bash
python scripts/train_intent_classifier.py \
    --model roberta-base \
    ...
```

**Multilingual (if needed):**
```bash
python scripts/train_intent_classifier.py \
    --model xlm-roberta-base \
    ...
```

### 4. Tune Hyperparameters

```bash
python scripts/train_intent_classifier.py \
    --model distilbert-base-uncased \
    --train-data data/processed/intent/train.csv \
    --val-data data/processed/intent/val.csv \
    --epochs 5 \              # Try 3-5 epochs
    --batch-size 32 \         # Try 16 or 32
    --lr 3e-5 \              # Try 2e-5 to 5e-5
    --output models/checkpoints/intent_v2
```

## Monitoring in Production

### Log Predictions

In `src/agents/langgraph_workflow.py`:

```python
import logging
logger = logging.getLogger(__name__)

def intent_classifier_node(state: WorkflowState) -> WorkflowState:
    result = intent_classifier_model.predict(state["message"])

    # Log for monitoring
    logger.info(
        f"Intent prediction: category={result['category']}, "
        f"confidence={result['confidence']:.3f}"
    )

    # Alert on low confidence
    if result["confidence"] < 0.6:
        logger.warning(
            f"Low confidence prediction: {result['confidence']:.3f} "
            f"for message: {state['message'][:50]}"
        )

    # ... rest of code
```

### A/B Testing

Run both models and compare:

```python
import random

# 50% traffic to fine-tuned model, 50% to LLM
use_finetuned = random.random() < 0.5

if use_finetuned:
    result = intent_classifier_model.predict(message)
    model_used = "finetuned"
else:
    result = classify_with_llm(message)
    model_used = "llm"

# Log both for comparison
logger.info(f"AB_TEST: model={model_used}, result={result}")
```

### Retraining Schedule

1. **Weekly:** Review low-confidence predictions
2. **Monthly:** Collect human-reviewed corrections
3. **Quarterly:** Retrain with new data
4. **Always:** A/B test before full rollout

## Troubleshooting

### Low Accuracy (<80%)

**Solutions:**
1. Collect more training data (aim for 200+ per class)
2. Try a larger model (roberta-base instead of distilbert)
3. Check label quality - review some examples manually
4. Increase training epochs to 5

### Model Too Slow

**Solutions:**
1. Use DistilBERT instead of BERT (2x faster)
2. Reduce max_length to 256 tokens
3. Use ONNX runtime (see README for instructions)
4. Batch predictions when possible

### Out of Memory

**Solutions:**
1. Reduce batch size to 8 or 4
2. Use gradient accumulation
3. Use a smaller model (distilbert or albert)
4. Use CPU if GPU memory is insufficient

## Next Steps

- Try training a **worthiness classifier** following the same steps
- Experiment with **data augmentation** for small datasets
- Set up **continuous training** pipeline with new data
- Deploy with **ONNX** for production speed
- Add **monitoring dashboard** to track model performance

## Questions?

See the main README.md for detailed documentation on:
- Model architectures
- Training strategies
- Data augmentation
- Deployment options
