# Model Training for LangGraph Workflow Nodes

This directory contains training scripts, data, and models for fine-tuning Hugging Face transformers to replace LLM-based nodes in the LangGraph workflow.

## Overview

Instead of using expensive OpenAI API calls for classification tasks, we can fine-tune lightweight models like BERT, DistilBERT, or RoBERTa for specific nodes in our workflow:

### Replaceable Nodes

1. **Intent Classifier Node** - Classify messages into 7 categories (RAG_QUERY or 6 Store A contexts)
2. **Context Classifier Node** - Classify Store A contexts with confidence scores
3. **Message Worthiness Classifier** - Determine if messages are worthy of long-term storage
4. **Sentiment/Emotion Analyzer** - Extract emotional dimensions from messages

## Directory Structure

```
training/
├── README.md                       # This file
├── requirements.txt                # Training dependencies
│
├── data/                           # Training data
│   ├── raw/                        # Raw conversation logs, exports
│   ├── processed/                  # Cleaned, labeled datasets
│   └── validation/                 # Validation/test sets
│
├── configs/                        # Model configurations
│   ├── intent_classifier.yaml      # Intent classifier config
│   ├── context_classifier.yaml     # Context classifier config
│   └── worthiness_classifier.yaml  # Message worthiness config
│
├── scripts/                        # Training scripts
│   ├── prepare_data.py             # Data preparation & labeling
│   ├── train_intent_classifier.py  # Train intent classifier
│   ├── train_context_classifier.py # Train context classifier
│   ├── train_worthiness_classifier.py # Train worthiness classifier
│   ├── evaluate.py                 # Model evaluation
│   └── export_model.py             # Export for production
│
├── notebooks/                      # Jupyter notebooks
│   ├── data_exploration.ipynb      # EDA on conversation data
│   ├── model_comparison.ipynb      # Compare model architectures
│   └── error_analysis.ipynb        # Analyze model errors
│
└── models/                         # Trained models
    ├── checkpoints/                # Training checkpoints
    └── final/                      # Production-ready models
        ├── intent_classifier/      # Fine-tuned intent model
        ├── context_classifier/     # Fine-tuned context model
        └── worthiness_classifier/  # Fine-tuned worthiness model
```

## Quick Start

### 1. Install Dependencies

```bash
cd training
pip install -r requirements.txt
```

### 2. Prepare Training Data

Export conversation data from Supabase and prepare training datasets:

```bash
python scripts/prepare_data.py \
    --task intent_classification \
    --output data/processed/intent_train.csv
```

### 3. Train a Model

Train the intent classifier:

```bash
python scripts/train_intent_classifier.py \
    --config configs/intent_classifier.yaml \
    --data data/processed/intent_train.csv \
    --output models/checkpoints/intent_classifier
```

### 4. Evaluate

```bash
python scripts/evaluate.py \
    --model models/checkpoints/intent_classifier \
    --test-data data/validation/intent_test.csv
```

### 5. Export for Production

```bash
python scripts/export_model.py \
    --model models/checkpoints/intent_classifier/checkpoint-best \
    --output models/final/intent_classifier
```

### 6. Integrate into Workflow

Replace the LangGraph node with the fine-tuned model:

```python
from training.inference import IntentClassifierModel

# Load fine-tuned model
intent_model = IntentClassifierModel("training/models/final/intent_classifier")

# Replace in workflow
def intent_classifier_node(state: WorkflowState) -> WorkflowState:
    result = intent_model.predict(state["message"])
    state["unified_classification"] = result
    return state
```

## Recommended Models

### For Intent/Context Classification (7 categories)

**Option 1: DistilBERT (Recommended for speed)**
- Model: `distilbert-base-uncased`
- Size: 66M parameters
- Inference: ~20ms per message
- Accuracy: 85-90% (with 1000+ labeled examples)
- Best for: Production deployment, low latency

**Option 2: BERT Base**
- Model: `bert-base-uncased`
- Size: 110M parameters
- Inference: ~40ms per message
- Accuracy: 88-92% (with 1000+ labeled examples)
- Best for: Higher accuracy requirements

**Option 3: RoBERTa Base**
- Model: `roberta-base`
- Size: 125M parameters
- Inference: ~45ms per message
- Accuracy: 90-94% (with 1000+ labeled examples)
- Best for: Maximum accuracy

### For Message Worthiness (Binary classification)

**Option 1: DistilBERT (Recommended)**
- Model: `distilbert-base-uncased`
- Best for: Fast filtering of worthy messages

**Option 2: ALBERT Base**
- Model: `albert-base-v2`
- Size: 12M parameters
- Inference: ~15ms per message
- Best for: Extremely fast inference

## Data Requirements

### Intent Classifier
- **Minimum**: 500 labeled examples (70 per category)
- **Recommended**: 2000+ labeled examples (285+ per category)
- **Format**: `message,category,confidence`

### Context Classifier
- **Minimum**: 300 labeled examples (50 per context)
- **Recommended**: 1200+ labeled examples (200+ per context)
- **Format**: `message,context,confidence`

### Worthiness Classifier
- **Minimum**: 500 labeled examples (balanced worthy/not-worthy)
- **Recommended**: 2000+ labeled examples
- **Format**: `message,is_worthy,score`

## Labeling Strategy

### Option 1: Generate Synthetic Data (Fastest - No existing conversations needed)
Use GPT-3.5-turbo to generate diverse, labeled training data from scratch:

```bash
python scripts/generate_data.py \
    --output data/processed/synthetic_labeled.csv \
    --rag-count 100 \
    --context-count 80 \
    --chitchat-count 60
```

**Advantages:**
- Works immediately (no need for existing conversations)
- Generates balanced dataset across all 7 categories
- Uses knowledge base content for realistic RAG queries
- Fast (5-10 minutes for 600+ examples)
- Cheap (~$0.20-0.30 in API costs)

**How it works:**
1. Fetches content from `general_embeddings_1024` table
2. Generates RAG queries based on your knowledge base
3. Generates realistic messages for each Store A context
4. Includes conversational chitchat for negative examples
5. Outputs labeled CSV ready for training

**Generated categories:**
- `rag_query`: Factual questions based on your knowledge base
- `professional`: Skills, experience, work history
- `psychological`: Values, motivations, personality
- `learning`: Learning preferences, education
- `social`: Network, mentors, community
- `emotional`: Feelings, confidence, wellbeing
- `aspirational`: Career goals, ambitions
- `chitchat`: Greetings, chitchat, unclear messages

### Option 2: Use Current LLM for Labeling
Export conversations and use the current OpenAI-based classifier to generate labels:

```bash
python scripts/prepare_data.py \
    --task generate_labels \
    --input data/raw/conversations.csv \
    --output data/processed/labeled.csv \
    --use-llm
```

### Option 3: Human Labeling
Use a labeling tool like Label Studio or Prodigy:

```bash
# Start Label Studio
label-studio start --data-dir data/raw
```

### Option 4: Hybrid Approach
1. Generate synthetic data for initial training
2. Collect real conversations in production
3. LLM labels high-confidence predictions
4. Human reviews disagreements
5. Retrain with real + synthetic data

## Training Tips

### 1. Class Imbalance
If some categories have fewer examples, use class weights:

```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(labels),
    y=labels
)
```

### 2. Data Augmentation
For small datasets, use back-translation or paraphrasing:

```python
from nlpaug.augmenter.word import SynonymAug

aug = SynonymAug(aug_src='wordnet')
augmented = aug.augment(original_text)
```

### 3. Hyperparameter Tuning
Key hyperparameters to tune:
- Learning rate: 2e-5 to 5e-5
- Batch size: 16 or 32
- Epochs: 3-5
- Warmup steps: 10% of total steps

### 4. Early Stopping
Monitor validation loss and stop when it stops improving:

```python
from transformers import EarlyStoppingCallback

trainer = Trainer(
    ...
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)
```

## Performance Comparison

### Current (OpenAI API)
- **Latency**: 200-500ms per classification
- **Cost**: $0.0001-0.0003 per message
- **Accuracy**: 92-95%

### Fine-tuned Model (DistilBERT)
- **Latency**: 20-40ms per classification (5-10x faster)
- **Cost**: ~$0 after training (no API calls)
- **Accuracy**: 85-90% (with good training data)

### Break-even Analysis
With 10,000 messages/day:
- OpenAI cost: $3-9/day = $90-270/month
- Fine-tuned model: $0/month + initial training cost
- **Break-even**: ~1-2 months

## Integration Guide

### Step 1: Update Config

Add model paths to `src/config.py`:

```python
# Fine-tuned model paths
INTENT_CLASSIFIER_MODEL = os.getenv(
    "INTENT_CLASSIFIER_MODEL",
    "training/models/final/intent_classifier"
)
CONTEXT_CLASSIFIER_MODEL = os.getenv(
    "CONTEXT_CLASSIFIER_MODEL",
    "training/models/final/context_classifier"
)
```

### Step 2: Create Inference Module

Create `training/inference.py`:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class IntentClassifierModel:
    def __init__(self, model_path: str):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()

    def predict(self, text: str) -> dict:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_class].item()

        return {
            "category": self.model.config.id2label[pred_class],
            "confidence": confidence
        }
```

### Step 3: Update Workflow

Modify `src/agents/langgraph_workflow.py`:

```python
# At the top
from training.inference import IntentClassifierModel

# Load model once at startup
intent_classifier = IntentClassifierModel(INTENT_CLASSIFIER_MODEL)

# Replace the node
def intent_classifier_node(state: WorkflowState) -> WorkflowState:
    """Classify message intent using fine-tuned BERT"""
    result = intent_classifier.predict(state["message"])

    state["unified_classification"] = IntentClassification(
        category=MessageCategory(result["category"]),
        confidence=result["confidence"],
        reasoning="Classified by fine-tuned BERT model",
        key_entities={},
        secondary_categories=[]
    )
    return state
```

### Step 4: A/B Testing

Run both models in parallel and compare:

```python
# Use fine-tuned model for 50% of traffic
use_finetuned = random.random() < 0.5

if use_finetuned:
    result = intent_classifier.predict(message)
else:
    result = classify_with_openai(message)

# Log both results for comparison
logger.info(f"Model comparison: finetuned={finetuned_result}, llm={llm_result}")
```

## Monitoring

### Metrics to Track

1. **Accuracy**: Compare with LLM baseline
2. **Latency**: Measure inference time
3. **Disagreement Rate**: How often does model disagree with LLM?
4. **User Feedback**: Track implicit signals (conversation quality)

### Retraining Strategy

1. **Collect**: Save misclassified examples
2. **Review**: Human reviews disagreements
3. **Retrain**: Fine-tune on corrected data every quarter
4. **Deploy**: A/B test before full rollout

## Troubleshooting

### Low Accuracy
- **Solution**: Increase training data (aim for 200+ examples per class)
- **Solution**: Use a larger model (BERT instead of DistilBERT)
- **Solution**: Try different learning rates

### Slow Inference
- **Solution**: Use ONNX runtime for 2-3x speedup
- **Solution**: Quantize model to INT8
- **Solution**: Use DistilBERT or ALBERT

### Class Imbalance
- **Solution**: Use class weights during training
- **Solution**: Oversample minority classes
- **Solution**: Use focal loss instead of cross-entropy

## Resources

- [Hugging Face Course](https://huggingface.co/course)
- [Fine-tuning Guide](https://huggingface.co/docs/transformers/training)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
