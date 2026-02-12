# Data-Driven Entity Extraction

## Overview

Replaced hardcoded if-statement logic with a **flexible, data-driven approach** using an OpenAI-generated verb-to-aspiration database.

## Problem with Old Approach

**Hardcoded if-statements** are:
- ❌ **Limited coverage**: Only handles roles we explicitly code
- ❌ **Brittle**: Breaks with new role combinations
- ❌ **Hard to maintain**: Every new role requires code changes
- ❌ **Not scalable**: Can't cover 1000s of role variations

```python
# OLD APPROACH (bad)
if ACTION == "lead" and FUNCTION == "marketing":
    role = "marketing director"
elif ACTION == "lead" and FUNCTION == "engineering":
    role = "engineering director"
# ... 100s more if-statements ...
```

## New Approach: Verb Database

**Data-driven matching** using OpenAI-generated database:
- ✅ **Comprehensive**: Covers 100s of verb-object combinations
- ✅ **Flexible**: Easy to add new combinations (just regenerate)
- ✅ **Maintainable**: Data, not code
- ✅ **Scalable**: Can handle 1000s of aspirations

### Database Structure

```json
{
  "lead": [
    "a marketing team",
    "an engineering organization",
    "product development",
    "a global sales division",
    "a cross-functional team",
    ...
  ],
  "earn": [
    "$150k",
    "$200,000 per year",
    "six figures",
    "a competitive salary",
    "$250k+ in total comp",
    ...
  ],
  "become": [
    "a senior data scientist",
    "CTO",
    "engineering manager",
    "staff engineer",
    "product director",
    ...
  ],
  "build": [
    "ML systems",
    "a product from scratch",
    "a high-performing team",
    "scalable infrastructure",
    ...
  ]
}
```

## How It Works

### Step 1: Generate Database

```bash
# Generate verb-to-aspiration database using OpenAI
python training/scripts/generate_aspiration_verb_database.py \
  --output training/data/aspiration_verb_database.json \
  --num-verbs 30 \
  --objects-per-verb 20

# Or use predefined common verbs
python training/scripts/generate_aspiration_verb_database.py \
  --use-predefined \
  --objects-per-verb 25
```

This creates a JSON file with ~30 verbs × ~20 objects = **600+ aspiration combinations**.

### Step 2: Extract Components (spaCy)

```python
# Input: "I want to lead a global marketing team"

# spaCy extracts:
{
  "actions": ["lead"],
  "functions": ["marketing"],
  "scopes": ["global"],
  "org_units": ["team"]
}
```

### Step 3: Match to Database

```python
# Combine components: "global marketing team"

# Find in database:
verb_database["lead"] = [
  "a marketing team",  # ← MATCH!
  "an engineering org",
  ...
]

# Best match: "a marketing team" (score: 0.75)
# Add scope: "global" + "a marketing team"
# Result: "global marketing director"
```

### Step 4: Infer Seniority

```python
# "lead" → director level (from action)
# "global" → +1 seniority boost

# Final:
{
  "inferred_role": "global marketing director",
  "inferred_seniority": "director"
}
```

## Usage

### Generate Database (First Time)

```bash
# Generate comprehensive database
python training/scripts/generate_aspiration_verb_database.py \
  --use-predefined \
  --objects-per-verb 25 \
  --output training/data/aspiration_verb_database.json
```

### Entity Extraction (Automatic)

```python
from src.agents.entity_extraction_spacy import extract_entities

# Automatically loads and uses verb database
entities = extract_entities("I want to lead a marketing team at Google")

print(entities)
# {
#   "components": {
#     "actions": ["lead"],
#     "functions": ["marketing"],
#     "org_units": ["team"]
#   },
#   "inferred_role": "marketing director",
#   "inferred_seniority": "director",
#   "organizations": ["Google"]
# }
```

## Customization

### Add More Verbs

Edit `generate_aspiration_verb_database.py`:

```python
verbs = [
    "lead", "manage", "build", "create",
    # Add your custom verbs
    "master", "pioneer", "revolutionize"
]
```

### Add More Objects per Verb

```bash
python training/scripts/generate_aspiration_verb_database.py \
  --objects-per-verb 50  # More diversity
```

### Regenerate Specific Verbs

Load existing database, regenerate specific verbs, merge back.

## Architecture

```
User Message
    ↓
[spaCy] Extract components (ACTION, FUNCTION, SCOPE, ORG_UNIT)
    ↓
[Matcher] Match components to verb database
    ↓
[Inference] Combine matched object + seniority
    ↓
Canonical role + seniority
```

## Benefits

1. **No Code Changes for New Roles**
   - Add "become a prompt engineer" → just regenerate database
   - No need to modify inference logic

2. **Comprehensive Coverage**
   - 30 verbs × 25 objects = 750 combinations
   - Covers most common career aspirations

3. **Easy to Audit**
   - Database is JSON (human-readable)
   - Can review and edit manually

4. **Transparent**
   - Matching logic is simple (substring overlap)
   - No black-box LLM inference at runtime

5. **Fast**
   - Database loaded once on startup
   - Matching is simple string operations (~1-5ms)

## Comparison

| Approach | Coverage | Maintainability | Speed | Cost |
|----------|----------|-----------------|-------|------|
| **Hardcoded if-statements** | ❌ Limited | ❌ High effort | ✅ Fast | ✅ Free |
| **LLM at runtime** | ✅ Unlimited | ✅ Zero effort | ❌ Slow (200ms) | ❌ $$ |
| **Verb database (NEW)** | ✅ High | ✅ Low effort | ✅ Fast | ✅ One-time cost |

## Next Steps

1. **Generate the database**:
   ```bash
   python training/scripts/generate_aspiration_verb_database.py --use-predefined
   ```

2. **Test extraction**:
   ```bash
   python src/agents/entity_extraction_spacy.py
   ```

3. **Review database**:
   - Check `training/data/aspiration_verb_database.json`
   - Add/remove objects manually if needed

4. **Integrate with labeling**:
   - Use in `generate_aspirational_data.py` with `--label-entities`

## Example Output

```bash
$ python training/scripts/generate_aspiration_verb_database.py --use-predefined

ASPIRATION VERB DATABASE GENERATION
===============================================================================

Configuration:
  Output: training/data/aspiration_verb_database.json
  Number of verbs: predefined (~35)
  Objects per verb: 20

[1/35] Processing verb: 'lead'
  Generating 20 objects for 'lead'...
    ✓ Generated 20 objects

[2/35] Processing verb: 'earn'
  Generating 20 objects for 'earn'...
    ✓ Generated 20 objects

...

✓ Saved database to: training/data/aspiration_verb_database.json

Database statistics:
  Total verbs: 35
  Total aspiration objects: 700
  Average objects per verb: 20.0

Sample entries:

  lead:
    - a marketing team
    - an engineering organization
    - product development
    - a global sales division
    - a cross-functional team
    ... and 15 more

  earn:
    - $150k
    - $200,000 per year
    - six figures
    - a competitive salary
    - $250k+ in total comp
    ... and 15 more
```

## Troubleshooting

### Database Not Found

```
[ENTITY EXTRACTION] Verb database not found at training/data/aspiration_verb_database.json
  Run: python training/scripts/generate_aspiration_verb_database.py
```

**Solution**: Generate the database first.

### Low Match Quality

If `inferred_role` is often `None`:
- Increase `objects_per_verb` (more diversity)
- Lower match threshold in `infer_role_from_components()` (default: 0.3)
- Add more verbs to cover edge cases

### Wrong Role Inferred

- Review database entries for that verb
- Manually edit JSON to fix specific mappings
- Regenerate with better prompt engineering
