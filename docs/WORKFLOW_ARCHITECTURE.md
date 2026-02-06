# LangGraph Workflow Architecture

## Overview

The Pop Skills AI chatbot uses a LangGraph-based multi-agent workflow to process user messages. The workflow classifies messages into 9 categories and routes them to specialized response nodes for personalized career coaching.

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Message                              │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Intent Classifier Node                          │
│  (OpenAI LLM or DistilBERT Model)                               │
│  Classifies into 1 of 9 categories                              │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Conditional Router                             │
│  Routes to appropriate response node based on category           │
└─────────┬───────────────────────────────────────────────────────┘
          │
          ├──[RAG_QUERY]──────► RAG Retrieval Node ──► RAG Query Response Node
          │                     (Hybrid Search)        (Educational, temp=0.3)
          │
          ├──[PROFESSIONAL]────► Professional Context Node
          │                      (Skills/Experience, temp=0.7)
          │
          ├──[PSYCHOLOGICAL]───► Psychological Context Node
          │                      (Values/Motivation, temp=0.7)
          │
          ├──[LEARNING]────────► Learning Context Node
          │                      (Learning/Development, temp=0.7)
          │
          ├──[SOCIAL]──────────► Social Context Node
          │                      (Networking/Mentorship, temp=0.7)
          │
          ├──[EMOTIONAL]───────► Emotional Context Node ⭐
          │                      (Wellbeing/Resilience, temp=0.7)
          │                      **Highest Priority Context**
          │
          ├──[ASPIRATIONAL]────► Aspirational Context Node
          │                      (Goal-Setting/Vision, temp=0.7)
          │
          ├──[CHITCHAT]────────► Chitchat Response Node
          │                      (Friendly Conversation, temp=0.8)
          │
          └──[OFF_TOPIC]───────► Off-Topic Response Node
                                 (Boundary-Setting, temp=0.5)
```

## The 9 Categories

### 1. RAG_QUERY - Information/Knowledge Seeking
**Examples:**
- "What is machine learning?"
- "How do I write a resume?"
- "What skills do I need for data science?"

**Response Style:** Educational, factual, uses knowledge base retrieval
**Temperature:** 0.3 (deterministic)
**RAG:** Yes (hybrid search on documents + conversation history)

---

### 2. PROFESSIONAL - Professional Context
**Examples:**
- "I have 5 years of Python experience"
- "I've built REST APIs with FastAPI"
- "I'm certified in AWS"

**Response Style:** Professional development coaching, acknowledges skills, suggests growth opportunities
**Temperature:** 0.7
**RAG:** Knowledge base context provided

---

### 3. PSYCHOLOGICAL - Psychological Context
**Examples:**
- "I value work-life balance above everything"
- "I'm naturally introverted"
- "I'm driven by making impact"

**Response Style:** Values alignment, self-awareness coaching, deep empathy
**Temperature:** 0.7
**RAG:** Knowledge base context provided

---

### 4. LEARNING - Learning Context
**Examples:**
- "I learn best through hands-on projects"
- "I completed a data science bootcamp"
- "I want to learn cloud computing"

**Response Style:** Learning strategy coaching, resource recommendations
**Temperature:** 0.7
**RAG:** Knowledge base context provided

---

### 5. SOCIAL - Social Context
**Examples:**
- "My mentor helped me navigate my career"
- "I'm not good at networking"
- "I prefer working in teams"

**Response Style:** Networking and relationship-building coaching
**Temperature:** 0.7
**RAG:** Knowledge base context provided

---

### 6. EMOTIONAL - Emotional Context ⭐
**Examples:**
- "I'm feeling burned out"
- "I'm not confident in my abilities"
- "I'm stressed about my job search"

**Response Style:** Deep empathy, validation, gentle wellbeing strategies
**Temperature:** 0.7
**RAG:** Knowledge base context provided
**Priority:** **HIGHEST** - Emotional wellbeing takes precedence over career advancement

---

### 7. ASPIRATIONAL - Aspirational Context
**Examples:**
- "I want to become a CTO in 5 years"
- "My dream is to work at a FAANG company"
- "I want to make $200k per year"

**Response Style:** Goal-setting, inspiring, breaks down aspirations into actionable steps
**Temperature:** 0.7
**RAG:** Knowledge base context provided

---

### 8. CHITCHAT - Casual Conversation
**Examples:**
- "Hey! How are you?"
- "What's up?"
- "Good morning!"

**Response Style:** Friendly, brief (1-3 sentences), gently redirects to career topics
**Temperature:** 0.8 (natural conversation)
**RAG:** No (uses conversation history only)

---

### 9. OFF_TOPIC - Out of Scope
**Examples:**
- "What's the weather like?"
- "Can you help me with my taxes?"
- "Tell me a joke"

**Response Style:** Polite boundary-setting, explains scope, redirects to career topics
**Temperature:** 0.5
**RAG:** No

---

## Node Specifications

### Intent Classifier Node

**Input:** User message
**Output:** IntentClassification object with:
- `category`: One of 9 MessageCategory enums
- `confidence`: 0.0-1.0
- `reasoning`: Explanation for classification
- `key_entities`: Extracted entities (skills, goals, emotions, values)
- `secondary_categories`: Additional relevant categories

**Two Backends:**
1. **OpenAI LLM** (default)
   - Uses GPT models with structured prompts
   - High accuracy (~95%+)
   - API costs (~$0.0002/request)
   - Latency: 500-1000ms

2. **DistilBERT Fine-Tuned Model**
   - Uses local fine-tuned model
   - Good accuracy (~85-90%)
   - No API costs
   - Latency: 50-100ms
   - Requires transformers + torch

**Configuration:** `INTENT_CLASSIFIER_TYPE` environment variable

---

### RAG Retrieval Node

**Triggered By:** RAG_QUERY category only

**Process:**
1. Hybrid search on knowledge base (semantic + keyword)
   - Top k=3 documents
   - RRF (Reciprocal Rank Fusion) scoring
2. Conversation history search (semantic only)
   - Top k=5 messages
   - Similarity threshold=0.6
3. Format contexts for response generation

**Output:**
- `document_results`: Raw document search results
- `conversation_history`: Raw conversation history
- `document_context`: Formatted document context string
- `conversation_context`: Formatted conversation context string
- `sources`: Source references for API response

---

### Context-Specific Response Nodes

Each Store A context (PROFESSIONAL, PSYCHOLOGICAL, LEARNING, SOCIAL, EMOTIONAL, ASPIRATIONAL) has a dedicated response node with:

**Specialized System Prompts:**
- Tailored to the specific context
- Empathetic and coaching-oriented
- Action-focused (2-3 concrete recommendations)
- Context-aware (uses classification reasoning)

**Access to:**
- User's message
- Classification reasoning
- Knowledge base context
- Conversation history

**Response Style:**
- Warm, empathetic, supportive
- 2-3 actionable recommendations
- Connects to knowledge base when relevant
- Natural ending (no forced summaries)

**Temperature:** 0.7 (creative yet coherent)

---

### Special Response Nodes

**Chitchat Response Node:**
- Brief, friendly responses (1-3 sentences)
- Gentle redirection to career topics
- Higher temperature (0.8) for natural conversation

**Off-Topic Response Node:**
- Polite boundary-setting
- Clear explanation of scope
- Redirection to career topics
- Lower temperature (0.5) for consistent messaging

---

## Classification Rules

### Priority Rules
1. Pure factual questions → RAG_QUERY
2. Greetings/small talk → CHITCHAT
3. Unrelated topics → OFF_TOPIC
4. Personal experiences/statements → Store A context
5. When uncertain between Store A contexts → EMOTIONAL (highest priority)
6. When in doubt between CHITCHAT and a context → Choose the context

### Disambiguation Examples
- "I'm frustrated with learning X" → EMOTIONAL (focus is frustration, not learning)
- "I want to learn Y to get a better job" → ASPIRATIONAL (focus is career goal, not learning)
- Career goals → ASPIRATIONAL (not PROFESSIONAL)
- Stress/burnout → EMOTIONAL (not PROFESSIONAL)

---

## State Management

### WorkflowState (TypedDict)

**Input:**
- `message`: User message
- `user_id`: User ID
- `conversation_id`: Conversation ID

**RAG Parameters:**
- `supabase`: Supabase client
- `chat_client`: OpenAI client
- `embed_client`: Embedding client (Voyage or OpenAI)
- `embed_model`: Embedding model name
- `embed_dimensions`: Embedding dimensions
- `chat_model`: Chat model name

**RAG Results:**
- `document_results`: Raw document search results
- `conversation_history`: Raw conversation history
- `document_context`: Formatted document context
- `conversation_context`: Formatted conversation context
- `sources`: Source references

**Classification:**
- `unified_classification`: IntentClassification object

**Output:**
- `response`: Generated response text
- `metadata`: Response metadata (category, confidence, classifier_type)

---

## Performance Characteristics

### Latency Breakdown (Typical)

| Component | OpenAI Classifier | DistilBERT Classifier |
|-----------|------------------|----------------------|
| Intent Classification | 500-1000ms | 50-100ms |
| RAG Retrieval (if RAG_QUERY) | 200-500ms | 200-500ms |
| Response Generation | 500-1500ms | 500-1500ms |
| **Total (RAG path)** | **1200-3000ms** | **750-2100ms** |
| **Total (Context path)** | **1000-2500ms** | **550-1600ms** |

### Costs (per request)

| Component | Cost |
|-----------|------|
| OpenAI Classification | ~$0.0002 |
| DistilBERT Classification | $0 (free) |
| RAG Retrieval | $0 (free, local DB) |
| Response Generation (gpt-4o-mini) | ~$0.0003-0.0008 |
| **Total (OpenAI)** | **~$0.0005-0.001** |
| **Total (DistilBERT)** | **~$0.0003-0.0008** |

---

## Testing

### Test All Categories
```bash
python tests/test_workflow_nodes.py
```

### Test Specific Category
```bash
python tests/test_workflow_nodes.py --category emotional
python tests/test_workflow_nodes.py --category chitchat
python tests/test_workflow_nodes.py --category off_topic
```

### Expected Results
- Accuracy: >85% (DistilBERT) or >95% (OpenAI)
- Response generated for all categories
- Appropriate tone/style for each context
- No errors or fallback triggers

---

## Configuration

### Environment Variables

```env
# Intent Classifier
INTENT_CLASSIFIER_TYPE=openai  # or "distilbert"
INTENT_CLASSIFIER_MODEL_PATH=training/models/20260201_203858/final

# OpenAI
OPENAI_API_KEY=sk-...
CHAT_MODEL=gpt-4o-mini

# Voyage AI (embeddings)
VOYAGE_API_KEY=pa-...
EMBED_MODEL=voyage-3-large
EMBED_DIMENSIONS=1024
```

---

## Best Practices

### When to Use Each Category

**RAG_QUERY:** Questions starting with "What is...", "How do I...", "Can you explain..."
**PROFESSIONAL:** Statements about skills, experience, or technical abilities
**PSYCHOLOGICAL:** Statements about values, motivations, or personality
**LEARNING:** Information about learning style or educational background
**SOCIAL:** Information about network, mentors, or collaboration
**EMOTIONAL:** Expressions of stress, burnout, confidence issues, or wellbeing concerns
**ASPIRATIONAL:** Statements about career goals, dreams, or future vision
**CHITCHAT:** Greetings, small talk, casual conversation
**OFF_TOPIC:** Anything unrelated to career development

### Response Guidelines

1. **Always validate emotions first** (especially EMOTIONAL context)
2. **Provide 2-3 actionable recommendations** (not just empathy)
3. **Draw from knowledge base when relevant** (but don't force it)
4. **Keep responses conversational** (avoid academic tone)
5. **End naturally** (no forced summaries or conclusions)
6. **For CHITCHAT:** Keep brief, redirect gently
7. **For OFF_TOPIC:** Be polite but clear about boundaries

---

## Troubleshooting

### Classification Issues
- Check classifier type in metadata
- Review classification reasoning
- Verify test cases match expected categories
- Consider retraining DistilBERT model if accuracy is low

### Response Quality Issues
- Verify RAG retrieval is finding relevant documents
- Check conversation history search results
- Review system prompts for each context
- Adjust temperature if responses are too generic/random

### Performance Issues
- Use DistilBERT for faster classification
- Monitor RAG retrieval latency
- Consider caching for common queries
- Profile workflow with LangSmith or similar tools
