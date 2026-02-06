# Synthetic Data Generation Guide

When your `conversation_history` table is empty, you can use GPT-3.5-turbo to generate high-quality synthetic training data.

## Why Synthetic Data?

**Advantages:**
- ✅ No need for existing conversations
- ✅ Perfectly balanced across all categories
- ✅ Based on your actual knowledge base content
- ✅ Fast (5-10 minutes for 600+ examples)
- ✅ Cheap (~$0.20-0.30 in API costs)
- ✅ Consistent labeling (no human error)

**When to use:**
- Just starting out with no conversation data
- Need to bootstrap your training dataset quickly
- Want a balanced baseline model before real data collection

## How It Works

The `generate_data.py` script uses GPT-3.5-turbo to create realistic messages:

### 1. RAG Query Generation
```
Knowledge Base → GPT-3.5 → Factual Questions
```

Fetches content from your `general_embeddings_1024` table and generates relevant questions:

**Example:**
```
Knowledge base: "Python is a high-level programming language..."
Generated: "What are the main features of Python?", "How do I learn Python?"
```

### 2. Store A Context Generation
```
Category Description + Examples → GPT-3.5 → Diverse Messages
```

For each of the 6 contexts, generates realistic user messages:

**Professional:**
- "I have 5 years of Python development experience"
- "I'm certified in AWS solutions architecture"

**Psychological:**
- "I value work-life balance above all else"
- "I'm motivated by solving complex problems"

**Learning:**
- "I learn best through hands-on projects"
- "I prefer video tutorials over reading documentation"

**Social:**
- "My mentor helped me navigate my career transition"
- "I'm part of the local Python developer community"

**Emotional:**
- "I'm feeling burned out from work"
- "I lack confidence in my technical abilities"

**Aspirational:**
- "I want to become a CTO in 5 years"
- "My goal is to work at a FAANG company"

### 3. chitchat Generation
```
Conversational Patterns → GPT-3.5 → Short, Non-informative Messages
```

Generates greetings, acknowledgments, and unclear messages:
- "hi", "thanks", "ok cool", "hmm", "not sure"

## Usage

### Basic Usage

Generate 640 labeled examples (default):

```bash
python scripts/generate_data.py \
    --output data/processed/synthetic_labeled.csv
```

This creates:
- 100 RAG queries
- 480 Store A messages (80 per category × 6)
- 60 chitchat messages

### Custom Counts

Adjust the number of messages per category:

```bash
python scripts/generate_data.py \
    --output data/processed/synthetic_labeled.csv \
    --rag-count 150 \
    --context-count 100 \
    --chitchat-count 80
```

This would generate:
- 150 RAG queries
- 600 Store A messages (100 per category × 6)
- 80 chitchat messages
- **Total: 830 examples**

### Without Knowledge Base

If you don't have documents in `general_embeddings_1024` yet:

```bash
python scripts/generate_data.py \
    --output data/processed/synthetic_labeled.csv \
    --no-knowledge-base
```

This generates generic career coaching questions instead of knowledge base-specific queries.

## Output Format

The script generates a CSV file with these columns:

| Column | Description | Example |
|--------|-------------|---------|
| `message` | The generated message | "What is machine learning?" |
| `category` | Intent category | "rag_query" |
| `confidence` | Label confidence (always 1.0 for synthetic) | 1.0 |

## Quality Considerations

### Strengths
- ✅ Diverse and natural-sounding messages
- ✅ Covers all 7 categories evenly
- ✅ Based on your actual knowledge base
- ✅ Includes realistic variations in tone and length

### Limitations
- ⚠️ May not capture all nuances of real user messages
- ⚠️ Less "messy" than real conversations (typos, grammar)
- ⚠️ Should be supplemented with real data when available

## Improving Quality

### 1. Generate More Data
More training data = better model:

```bash
# Generate 1500+ examples
python scripts/generate_data.py \
    --output data/processed/synthetic_large.csv \
    --rag-count 200 \
    --context-count 150 \
    --chitchat-count 100
```

### 2. Manual Review & Refinement

Review and edit generated messages:

```bash
# After generation
cat data/processed/synthetic_labeled.csv

# Edit in Excel, VS Code, or any CSV editor
# Fix any unrealistic or off-topic messages
```

### 3. Mix with Real Data

Once you have real conversations, combine them:

```bash
# Generate synthetic baseline
python scripts/generate_data.py --output data/processed/synthetic.csv

# Export real conversations
python scripts/prepare_data.py --task export --output data/raw/real.csv

# Label real conversations
python scripts/prepare_data.py --task intent_labels \
    --input data/raw/real.csv \
    --output data/processed/real_labeled.csv

# Combine both datasets
cat data/processed/synthetic.csv data/processed/real_labeled.csv > data/processed/combined.csv
```

### 4. Iterative Improvement

1. Train model on synthetic data
2. Deploy and collect real conversations
3. Find misclassifications
4. Generate more synthetic data for weak categories
5. Retrain with combined data

## Cost Estimation

GPT-3.5-turbo pricing (as of 2024):
- Input: $0.50 per 1M tokens
- Output: $1.50 per 1M tokens

Typical costs for generating data:

| Examples | API Calls | Estimated Cost |
|----------|-----------|----------------|
| 640 (default) | 8 calls | $0.20-0.30 |
| 1,000 | 10 calls | $0.30-0.50 |
| 2,000 | 15 calls | $0.50-0.80 |

**Very affordable** compared to human labeling or using GPT-4.

## Example Output

Here's what generated data looks like:

```csv
message,category,confidence
"What are the best practices for Python development?",rag_query,1.0
"I have 7 years of experience in data engineering",professional,1.0
"I value creative freedom in my work",psychological,1.0
"I learn best through pair programming",learning,1.0
"My manager is my biggest advocate",social,1.0
"I feel overwhelmed by the fast pace of tech",emotional,1.0
"I want to transition into AI research",aspirational,1.0
"thanks!",chitchat,1.0
```

## Next Steps After Generation

1. **Split the dataset:**
   ```bash
   python scripts/prepare_data.py --task split \
       --input data/processed/synthetic_labeled.csv \
       --output data/processed/intent/
   ```

2. **Train the model:**
   ```bash
   python scripts/train_intent_classifier.py \
       --train-data data/processed/intent/train.csv \
       --val-data data/processed/intent/val.csv
   ```

3. **Evaluate:**
   ```bash
   python scripts/evaluate.py --task intent \
       --model models/checkpoints/intent_classifier/final \
       --test-data data/processed/intent/test.csv
   ```

4. **Deploy and collect real data for future retraining**

## Complete Pipeline Script

For convenience, use the provided pipeline script:

**Linux/Mac:**
```bash
bash scripts/example_generate_and_train.sh
```

**Windows:**
```cmd
scripts\example_generate_and_train.bat
```

This runs the entire pipeline automatically:
1. Generate synthetic data
2. Split into train/val/test
3. Train model
4. Evaluate on test set

## Troubleshooting

### Error: "Could not fetch knowledge base"

**Solution:** Run with `--no-knowledge-base` flag:
```bash
python scripts/generate_data.py --no-knowledge-base --output data/processed/synthetic.csv
```

### Error: "OpenAI API error"

**Solution:** Check your `OPENAI_API_KEY` in `.env`:
```bash
# Verify key is set
echo $OPENAI_API_KEY  # Linux/Mac
echo %OPENAI_API_KEY%  # Windows
```

### Low diversity in generated messages

**Solution:** Generate more data (increases variety):
```bash
python scripts/generate_data.py \
    --rag-count 200 \
    --context-count 150
```

### Rate limit errors

**Solution:** Script includes automatic 1-second pauses between API calls. If you still hit limits, wait a minute and re-run.

## FAQ

**Q: Is synthetic data as good as real data?**

A: Synthetic data is a great starting point and typically achieves 80-85% of the accuracy of models trained on real data. For production, plan to collect real conversations and retrain.

**Q: How many examples do I need?**

A: Minimum 500-600 total (like the default), but 1000-2000 is better for production quality.

**Q: Can I generate data for other languages?**

A: Yes! Modify the prompts in `generate_data.py` to request messages in Spanish, French, etc.

**Q: Should I review the generated data?**

A: Quick review is recommended, but GPT-3.5 is generally reliable. Focus on checking a sample from each category.

**Q: How often should I regenerate?**

A: Once is usually enough for initial training. Use real conversation data for subsequent retraining.

## Summary

Synthetic data generation is the fastest way to get started when you don't have existing conversations:

1. ✅ **Run once:** `python scripts/generate_data.py`
2. ✅ **Train model:** Within minutes, not days
3. ✅ **Deploy:** Start collecting real data
4. ✅ **Improve:** Retrain with real data quarterly

Perfect for rapid prototyping and getting your first model into production!
