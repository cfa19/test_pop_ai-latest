# Hierarchical Intent Classification System

## Overview

The hierarchical classification system uses a two-level approach for intent classification:

1. **Primary Classifier**: Classifies messages into 9 main categories
2. **Secondary Classifiers**: For each category, classifies into specific subcategories

This approach provides both high-level intent detection and fine-grained subcategory classification.

## Architecture

```
User Message
    ↓
[Primary Classifier]
    ↓
Category (9 options):
- rag_query
- professional
- psychological
- learning
- social
- emotional
- aspirational
- chitchat
- off_topic
    ↓
[Secondary Classifier for Category]
    ↓
Subcategory (category-specific)
```

### Primary Categories (9)

1. **rag_query** - Questions about products, services, company information
2. **professional** - Skills, experience, technical abilities
3. **psychological** - Personality, values, motivations
4. **learning** - Learning preferences, educational background
5. **social** - Network, mentors, community
6. **emotional** - Confidence, stress, wellbeing
7. **aspirational** - Career goals, dreams, ambitions
8. **chitchat** - Casual conversation, greetings
9. **off_topic** - Non-career related topics

### Secondary Categories (Subcategories)

Each primary category has multiple subcategories. Examples:

**aspirational** subcategories:
- dream_roles
- career_goals
- next_steps
- timeline
- etc.

**professional** subcategories:
- skills_technical
- skills_soft
- experience
- achievements
- etc.

**chitchat** subcategories:
- greetings
- thanks
- farewells
- acknowledgments
- small_talk

## Data Format

Training data should be CSV files with three columns:

```csv
message,category,subcategory
"I want to be a manager",aspirational,dream_roles
"hi there",chitchat,greetings
"What is PopCoach?",rag_query,products
```

### Column Definitions

- **message**: The text message to classify
- **category**: Primary category (one of the 9 main categories)
- **subcategory**: Specific subcategory within that category

## Training

### Step 1: Generate Training Data

Use the data generation script to create training data:

```bash
# Generate data for all categories
python training/scripts/generate_data.py \
  --category-type all \
  --batch-size 50 \
  --num-batches 10
```

This will create CSV files in `training/data/` with the required format.

### Step 2: Train Hierarchical Classifiers

Train both primary and secondary classifiers:

```bash
python training/scripts/train_hierarchical_classifier.py \
  --data-dir training/data \
  --model distilbert-base-uncased \
  --output-dir training/models/hierarchical \
  --epochs 5 \
  --batch-size 20 \
  --val-split 0.2
```

**Arguments:**

- `--data-dir`: Directory containing CSV files with training data
- `--model`: Hugging Face model name (default: distilbert-base-uncased)
- `--output-dir`: Output directory for trained models
- `--epochs`: Number of training epochs (default: 5)
- `--batch-size`: Training batch size (default: 20)
- `--val-split`: Validation split fraction (default: 0.2)
- `--embedding-model`: Sentence transformer for centroids (default: all-MiniLM-L6-v2)
- `--no-class-weights`: Disable class weights for imbalanced data
- `--skip-secondary`: Only train primary classifier (skip secondary classifiers)

### Training Process

The training script will:

1. **Load and validate data** from all CSV files in the data directory
2. **Build category hierarchy** by analyzing the data
3. **Split into train/validation** sets (stratified by primary category)
4. **Train primary classifier** on 9 main categories
5. **Compute centroids** for semantic gate (off-topic detection)
6. **Train secondary classifiers** for each category's subcategories
   - Only trains if category has 2+ subcategories
   - Only trains if sufficient training data (10+ examples)
7. **Save all models** in organized directory structure

### Output Directory Structure

```
training/models/hierarchical/
└── 20260206_123456/                    # Timestamped run
    ├── hierarchy_metadata.json         # Category structure info
    ├── primary/                        # Primary classifier
    │   └── final/
    │       ├── pytorch_model.bin
    │       ├── config.json
    │       ├── tokenizer_config.json
    │       ├── label_mappings.json     # Category mappings
    │       ├── centroids.pkl           # Semantic gate centroids
    │       └── centroid_metadata.json
    └── secondary/                      # Secondary classifiers
        ├── secondary_metadata.json     # Summary of trained classifiers
        ├── aspirational/
        │   └── final/
        │       ├── pytorch_model.bin
        │       ├── config.json
        │       ├── tokenizer_config.json
        │       └── label_mappings.json
        ├── professional/
        │   └── final/
        │       └── ...
        └── ...
```

## Inference

### Using the Test Script

Test the trained hierarchical classifier:

```bash
# Single message
python training/scripts/test_hierarchical_classifier.py \
  --model-dir training/models/hierarchical/20260206_123456 \
  --message "I want to become a data scientist"

# File with messages
python training/scripts/test_hierarchical_classifier.py \
  --model-dir training/models/hierarchical/20260206_123456 \
  --file test_messages.txt

# Interactive mode
python training/scripts/test_hierarchical_classifier.py \
  --model-dir training/models/hierarchical/20260206_123456 \
  --interactive
```

### Using in Code

```python
from training.scripts.test_hierarchical_classifier import HierarchicalClassifier

# Load classifier
classifier = HierarchicalClassifier(
    model_dir="training/models/hierarchical/20260206_123456"
)

# Predict single message
result = classifier.predict("I want to be a manager")
print(f"Category: {result['category']}")          # aspirational
print(f"Subcategory: {result['subcategory']}")    # dream_roles
print(f"Confidence: {result['category_confidence']:.3f}")

# Predict batch
messages = [
    "hello there",
    "I want to improve my Python skills",
    "What is PopCoach?"
]
results = classifier.predict_batch(messages)
for result in results:
    print(f"{result['message']}: {result['category']} -> {result['subcategory']}")
```

### Output Format

```python
{
    "message": "I want to be a manager",
    "category": "aspirational",
    "category_confidence": 0.987,
    "subcategory": "dream_roles",
    "subcategory_confidence": 0.945
}
```

If no secondary classifier exists for the category:
```python
{
    "message": "hello",
    "category": "chitchat",
    "category_confidence": 0.998,
    "subcategory": None,          # No secondary classifier
    "subcategory_confidence": None
}
```

## Integration with Existing Workflow

### Option 1: Replace Existing Classifier

Update `src/agents/distilbert_classifier.py` to use the hierarchical classifier:

```python
class DistilBERTClassifier:
    def __init__(self, model_path: str):
        # Load hierarchical classifier instead
        self.hierarchical = HierarchicalClassifier(model_path)

    def classify(self, message: str) -> str:
        result = self.hierarchical.predict(message)
        # Return primary category for workflow routing
        return result['category']

    def classify_detailed(self, message: str) -> dict:
        # Return full hierarchical result
        return self.hierarchical.predict(message)
```

### Option 2: Add Subcategory Node

Add a new LangGraph node after intent classification:

```python
def subcategory_classification_node(state: WorkflowState):
    """Classify subcategory within the detected category."""
    category = state["intent"]
    message = state["original_message"]

    # Load secondary classifier for this category
    subcategory, confidence = secondary_classifier.predict(message, category)

    state["subcategory"] = subcategory
    state["subcategory_confidence"] = confidence

    return state
```

### Option 3: Use for Analytics

Use subcategories for detailed analytics and insights:

```python
# Log detailed classifications
result = classifier.predict(message)
analytics.log({
    "category": result["category"],
    "subcategory": result["subcategory"],
    "category_conf": result["category_confidence"],
    "subcategory_conf": result["subcategory_confidence"]
})

# Generate insights
subcategory_counts = analytics.count_by_subcategory()
print(f"Most common aspirational subcategory: {subcategory_counts['aspirational'].most_common(1)}")
```

## Semantic Gate Integration

The primary classifier training also computes **centroids** for the semantic gate (first-stage off-topic detection).

### Centroids File

Located at: `primary/final/centroids.pkl`

Contains:
- Embedding centroids for each primary category (except off_topic)
- Computed using SentenceTransformer (all-MiniLM-L6-v2 by default)
- Used to filter out off-topic messages before classification

### Using Centroids

```python
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load centroids
with open("primary/final/centroids.pkl", "rb") as f:
    centroids = pickle.load(f)

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Compute message embedding
message = "I like chocolate cake"
embedding = model.encode([message])[0]

# Compare to centroids
from sklearn.metrics.pairwise import cosine_similarity
similarities = {
    cat: cosine_similarity([embedding], [centroid])[0][0]
    for cat, centroid in centroids.items()
}

# Get best matching category
best_category = max(similarities, key=similarities.get)
best_similarity = similarities[best_category]

# Check threshold (from tuning results)
if best_similarity < threshold:
    print("OFF_TOPIC: Message does not match any domain category")
else:
    print(f"Likely category: {best_category} (similarity: {best_similarity:.3f})")
```

## Performance Considerations

### Training Time

- **Primary classifier**: ~5-10 minutes (depends on dataset size)
- **Secondary classifiers**: ~2-5 minutes each
- **Total training time**: ~30-60 minutes for all classifiers

### Inference Time

- **Primary prediction**: ~50-100ms (DistilBERT on CPU)
- **Secondary prediction**: ~50-100ms (DistilBERT on CPU)
- **Total inference**: ~100-200ms per message

### Model Size

- **DistilBERT model**: ~250MB per classifier
- **Primary + 8 secondary classifiers**: ~2.25GB total
- **Centroids**: ~1MB

### Optimization Tips

1. **Use smaller models** for faster inference:
   - `distilbert-base-uncased` (default, 66M params)
   - `albert-base-v2` (12M params, smaller)
   - `google/electra-small-discriminator` (14M params)

2. **Quantization**: Use int8 quantization to reduce model size:
   ```python
   from transformers import AutoModelForSequenceClassification
   model = AutoModelForSequenceClassification.from_pretrained(
       model_path,
       load_in_8bit=True
   )
   ```

3. **Batch inference**: Process multiple messages at once:
   ```python
   results = classifier.predict_batch(messages)
   ```

4. **GPU acceleration**: Enable CUDA for faster inference:
   ```python
   model.to("cuda")
   ```

## Evaluation Metrics

The training script outputs detailed metrics for each classifier:

### Per-Classifier Metrics

- **Accuracy**: Overall classification accuracy
- **Precision**: Precision score (weighted average)
- **Recall**: Recall score (weighted average)
- **F1 Score**: F1 score (weighted average)

### Per-Category Metrics

The classification report shows:
- Precision, Recall, F1 for each category/subcategory
- Support (number of examples)
- Macro and weighted averages

### Example Output

```
Classification Report:
                   precision    recall  f1-score   support

      aspirational       0.95      0.93      0.94       120
        chitchat       0.98      0.99      0.98        85
       emotional       0.91      0.89      0.90       110
        learning       0.88      0.90      0.89        95
    professional       0.93      0.94      0.94       130
   psychological       0.87      0.86      0.87       100
       rag_query       0.96      0.97      0.97       140
          social       0.89      0.88      0.89       105
       off_topic       0.94      0.95      0.95       115

        accuracy                           0.93      1000
       macro avg       0.92      0.92      0.92      1000
    weighted avg       0.93      0.93      0.93      1000
```

## Troubleshooting

### Issue: "No valid CSV files found"

**Solution**: Ensure CSV files have required columns: `message`, `category`, `subcategory`

### Issue: "Insufficient training data for category X"

**Solution**: Generate more training data for that category or lower the minimum threshold (currently 10 examples)

### Issue: "Out of memory during training"

**Solution**:
- Reduce batch size: `--batch-size 8`
- Use smaller model: `--model distilbert-base-uncased`
- Enable gradient checkpointing in code

### Issue: "Secondary classifier not found for category X"

**Solution**: Check if secondary classifier was trained:
- Category must have 2+ subcategories
- Category must have 10+ training examples
- Check `secondary_metadata.json` for trained categories

## Best Practices

1. **Balanced Data**: Ensure each category has sufficient examples (100+ recommended)
2. **Clear Boundaries**: Subcategories within a category should be distinct
3. **Validation**: Always use validation split to monitor overfitting
4. **Iterative Training**: Start with primary classifier, then add secondary classifiers
5. **Regular Retraining**: Retrain models periodically with new data
6. **Version Control**: Use timestamps to track different model versions
7. **A/B Testing**: Test new models against production before deployment

## Future Enhancements

1. **Multi-label Classification**: Allow messages to belong to multiple categories
2. **Confidence Thresholds**: Add configurable thresholds for low-confidence predictions
3. **Active Learning**: Identify uncertain predictions for human review
4. **Few-Shot Learning**: Use few-shot techniques for new subcategories
5. **Cross-Category Learning**: Share knowledge between related categories
6. **Ensemble Methods**: Combine multiple models for better performance

## References

- Hugging Face Transformers: https://huggingface.co/docs/transformers
- DistilBERT Paper: https://arxiv.org/abs/1910.01108
- Sentence Transformers: https://www.sbert.net/
- LangGraph: https://langchain-ai.github.io/langgraph/
