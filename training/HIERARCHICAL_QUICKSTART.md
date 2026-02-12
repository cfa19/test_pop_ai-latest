# Hierarchical Classifier - Quick Start Guide

## 5-Minute Setup

### 1. Generate Training Data

```bash
# Generate data for all 9 categories
python training/scripts/generate_data.py \
  --category-type all \
  --batch-size 50 \
  --num-batches 10
```

This creates CSV files in `training/data/` with format:
```csv
message,category,subcategory
"I want to be a manager",aspirational,dream_roles
"hello",chitchat,greetings
"What is PopCoach?",rag_query,products
```

### 2. Train Hierarchical Classifiers

```bash
# Train primary + secondary classifiers
python training/scripts/train_hierarchical_classifier.py \
  --data-dir training/data \
  --epochs 5 \
  --batch-size 20
```

This will:
- Train primary classifier (9 categories)
- Train secondary classifiers (subcategories for each category)
- Compute semantic gate centroids
- Save everything to `training/models/hierarchical/TIMESTAMP/`

Training takes ~30-60 minutes depending on your hardware.

### 3. Test the Classifier

```bash
# Interactive mode
python training/scripts/test_hierarchical_classifier.py \
  --model-dir training/models/hierarchical/TIMESTAMP \
  --interactive
```

Or test a single message:
```bash
python training/scripts/test_hierarchical_classifier.py \
  --model-dir training/models/hierarchical/TIMESTAMP \
  --message "I want to become a data scientist"
```

Output:
```
Message: I want to become a data scientist
Category: aspirational (confidence: 0.987)
Subcategory: dream_roles (confidence: 0.945)
```

## Usage in Code

```python
from training.scripts.test_hierarchical_classifier import HierarchicalClassifier

# Load classifier
classifier = HierarchicalClassifier(
    model_dir="training/models/hierarchical/TIMESTAMP"
)

# Classify a message
result = classifier.predict("I want to improve my Python skills")

print(f"Category: {result['category']}")        # professional
print(f"Subcategory: {result['subcategory']}")  # skills_technical
print(f"Confidence: {result['category_confidence']:.2%}")
```

## Understanding the Output

### Primary Categories (9)

| Category | Description | Example |
|----------|-------------|---------|
| `rag_query` | Questions about products/services | "What is PopCoach?" |
| `professional` | Skills, experience | "I have 5 years of Python experience" |
| `psychological` | Personality, values | "I'm an introvert who values autonomy" |
| `learning` | Learning preferences | "I learn best through hands-on practice" |
| `social` | Network, mentors | "I have a mentor who helps me" |
| `emotional` | Feelings, wellbeing | "I feel burned out from work" |
| `aspirational` | Career goals | "I want to become a manager" |
| `chitchat` | Greetings, small talk | "hello there" |
| `off_topic` | Non-career topics | "I like chocolate cake" |

### Secondary Categories (Examples)

**aspirational** → dream_roles, career_goals, next_steps, timeline, etc.

**professional** → skills_technical, skills_soft, experience, achievements, etc.

**chitchat** → greetings, thanks, farewells, acknowledgments, small_talk

## Key Files

After training, you'll have:

```
training/models/hierarchical/TIMESTAMP/
├── primary/final/              # Primary classifier (9 categories)
│   ├── pytorch_model.bin
│   ├── centroids.pkl          # For semantic gate
│   └── label_mappings.json
└── secondary/                  # Secondary classifiers
    ├── aspirational/final/
    ├── professional/final/
    └── ...
```

## Common Issues

**"No valid CSV files found"**
- Check that CSV files have columns: `message`, `category`, `subcategory`
- Make sure files are in the `--data-dir` directory

**"Insufficient training data"**
- Generate more data: increase `--num-batches`
- Each category needs 10+ examples minimum

**"Out of memory"**
- Reduce batch size: `--batch-size 8`
- Use CPU: don't install CUDA/PyTorch-GPU

## Next Steps

1. **Integrate with workflow**: See `HIERARCHICAL_CLASSIFIER.md` for integration options
2. **Tune performance**: Adjust epochs, batch size, learning rate
3. **Add more data**: Generate more training examples for better accuracy
4. **Monitor metrics**: Check validation accuracy and F1 scores
5. **Deploy**: Use the classifier in your production workflow

## Help

Full documentation: `training/HIERARCHICAL_CLASSIFIER.md`

For questions or issues, check the troubleshooting section in the main docs.
