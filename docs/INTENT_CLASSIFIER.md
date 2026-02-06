# Intent Classifier Configuration

The LangGraph workflow supports two intent classification backends for classifying user messages into 7 categories (RAG_QUERY or 6 Store A contexts).

## Available Classifiers

### 1. OpenAI LLM Classifier (Default)

Uses GPT models with structured prompts to classify messages.

**Pros:**
- High accuracy with contextual understanding
- No local model dependencies
- Works out of the box

**Cons:**
- API costs per classification
- Slower (network latency)
- Requires OpenAI API key

**Configuration:**
```env
INTENT_CLASSIFIER_TYPE=openai
CHAT_MODEL=gpt-4o-mini
```

### 2. DistilBERT Fine-Tuned Model

Uses a locally fine-tuned DistilBERT model for classification.

**Pros:**
- No API costs (free after training)
- Fast inference (local GPU/CPU)
- Offline capable
- Consistent predictions

**Cons:**
- Requires model training first
- Needs transformers + torch dependencies
- Slightly lower accuracy on edge cases

**Configuration:**
```env
INTENT_CLASSIFIER_TYPE=distilbert
INTENT_CLASSIFIER_MODEL_PATH=training/models/20260201_203858/final
```

## Environment Variables

Add these to your `.env` file:

```env
# Intent Classifier Configuration
INTENT_CLASSIFIER_TYPE=distilbert  # "openai" or "distilbert" (default: "openai")
INTENT_CLASSIFIER_MODEL_PATH=training/models/20260201_203858/final  # Path to fine-tuned model

# OpenAI Configuration (required for OpenAI classifier)
OPENAI_API_KEY=sk-...
CHAT_MODEL=gpt-4o-mini
```

## Installation

### For OpenAI Classifier (Default)
No additional dependencies required. Just set your OpenAI API key.

### For DistilBERT Classifier
Install PyTorch and Transformers:

```bash
# CPU only
pip install transformers torch

# GPU support (CUDA)
pip install transformers torch --index-url https://download.pytorch.org/whl/cu118
```

## Usage

The classifier is automatically selected based on `INTENT_CLASSIFIER_TYPE`. No code changes needed.

### Testing the Classifier

Run the test script to verify both classifiers:

```bash
# Test both classifiers and compare
python tests/test_intent_classifiers.py

# Test only OpenAI
python tests/test_intent_classifiers.py --classifier openai

# Test only DistilBERT
python tests/test_intent_classifiers.py --classifier distilbert
```

### Switching Between Classifiers

Simply change the environment variable and restart the server:

```bash
# Use OpenAI
export INTENT_CLASSIFIER_TYPE=openai
python main.py

# Use DistilBERT
export INTENT_CLASSIFIER_TYPE=distilbert
python main.py
```

## The 7 Categories

Both classifiers output one of these categories:

1. **rag_query** - Factual questions needing information lookup
2. **professional** - Skills, experience, technical abilities
3. **psychological** - Personality, values, motivations
4. **learning** - Learning preferences, educational background
5. **social** - Network, mentors, community
6. **emotional** - Confidence, stress, wellbeing (highest weight)
7. **aspirational** - Career goals, dreams, ambitions

**Note:** The DistilBERT model also outputs "chitchat" but this is automatically mapped to "emotional" for workflow compatibility.

## Model Details

### DistilBERT Model

- **Architecture:** DistilBertForSequenceClassification
- **Base Model:** distilbert-base-uncased
- **Output:** 8 classes (including chitchat)
- **Input:** Max 512 tokens
- **Location:** `training/models/20260201_203858/final/`

### Model Files

```
training/models/20260201_203858/final/
├── config.json              # Model configuration and label mappings
├── model.safetensors        # Model weights
├── tokenizer.json           # Tokenizer vocabulary
├── tokenizer_config.json    # Tokenizer configuration
├── training_args.bin        # Training hyperparameters
├── centroids.pkl            # Centroid embeddings (for hybrid approach)
└── centroid_metadata.json   # Centroid metadata
```

## Performance Comparison

Based on test results:

| Metric | OpenAI (gpt-4o-mini) | DistilBERT |
|--------|---------------------|------------|
| Accuracy | ~95%+ | ~85-90% |
| Latency | ~500-1000ms | ~50-100ms |
| Cost | $0.0002/request | Free |
| Offline | ❌ | ✓ |

## Fallback Behavior

If the DistilBERT classifier fails (model not found, import error, etc.), the system automatically falls back to the OpenAI classifier.

```python
# Automatic fallback on error
try:
    classification = await distilbert_classifier.classify(message)
except Exception as e:
    print(f"DistilBERT failed: {e}")
    classification = await openai_classifier.classify(message)
```

## Troubleshooting

### "Model not found" error

Ensure the model path is correct and the model files exist:

```bash
ls -la training/models/20260201_203858/final/
```

Expected files: `config.json`, `model.safetensors`, `tokenizer.json`

### "ImportError: No module named 'transformers'"

Install the required dependencies:

```bash
pip install transformers torch
```

### Poor classification accuracy

1. Check if you're using the right model path
2. Verify the model was trained on similar data
3. Try increasing the confidence threshold
4. Consider retraining the model with more data

### Slow inference on CPU

The DistilBERT model runs faster on GPU:

```bash
# Install GPU-enabled PyTorch (CUDA 11.8)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Verify GPU is detected:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Training Your Own Model

See `training/scripts/train_intent_classifier.py` for training a custom model:

```bash
# Generate training data
python training/scripts/generate_data.py

# Train the model
python training/scripts/train_intent_classifier.py

# Evaluate performance
python training/scripts/evaluate.py
```

## API Response Metadata

The classifier type is included in the response metadata:

```json
{
  "response": "...",
  "metadata": {
    "category": "professional",
    "classification_confidence": 0.89,
    "classifier_type": "distilbert"
  }
}
```

This helps track which classifier was used for each request.
