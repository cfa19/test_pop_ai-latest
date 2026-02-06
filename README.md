# Pop Skills AI Chatbot Server

An intelligent career coaching chatbot API server built with FastAPI that uses LangGraph workflows to orchestrate multi-agent message processing, combining RAG (Retrieval-Augmented Generation) with conversation memory and intelligent intent classification powered by either OpenAI LLMs or fine-tuned DistilBERT models.

## Features

### LangGraph Multi-Agent Workflow

- **Intent Classification**: Automatically classifies messages into 9 categories using OpenAI LLM or fine-tuned DistilBERT model
- **Specialized Response Nodes**: 9 dedicated nodes, each optimized for specific message types
- **Conditional Routing**: RAG queries trigger document retrieval, coaching messages go to specialized context nodes
- **Store A Context Types**: Professional, Psychological, Learning, Social, Emotional, Aspirational
- **Special Categories**: Chitchat (casual conversation) and Off-Topic (boundary setting)
- **Adaptive Response Generation**: Educational responses for queries, empathetic coaching for personal contexts

### Dual Intent Classification System

- **OpenAI LLM Classifier** (default): High accuracy contextual understanding, ~95%+ accuracy, 500-1000ms latency
- **DistilBERT Fine-Tuned Model**: No API costs, fast local inference (~50-100ms), offline capable, ~85-90% accuracy
- **Configuration-Based Switching**: Change classifier via environment variable without code changes
- **Automatic Fallback**: Falls back to OpenAI if DistilBERT fails to load
- **Synthetic Training Data Generation**: GPT-3.5-turbo generates labeled training data for model fine-tuning
- **Two-Stage Classification**: Semantic gate (Stage 1) filters off-topic, BERT classifier (Stage 2) classifies into 8 in-domain categories

### Multi-Provider Support

- **Flexible Embedding Providers**: OpenAI or Voyage AI (configurable per request)
- **Flexible Chat Providers**: OpenAI (configurable per request)
- **Dynamic Model Selection**: Specify models per request or use environment defaults
- **Provider Validation**: Automatic validation of model/provider compatibility

### Dual RAG System

- **Document RAG**: Searches company knowledge base using hybrid search (semantic + keyword)
- **Conversation Memory RAG**: Retrieves relevant past messages from conversation history using vector similarity

### Secure Authentication

- JWT-based authentication via Supabase
- Row-Level Security (RLS) for data isolation
- User-specific conversation history

### Intelligent Message Filtering

- Automatic classification of high-value user messages (career goals, skills, preferences, challenges)
- Hybrid classifier: heuristic scoring + LLM validation
- Stores worthy messages in `retrieval_chunks` for long-term user context
- Multilingual support with automatic translation to English for consistent RAG search

### User Context Management

- Search indexed worthy messages per user
- Reindex past conversations to extract valuable context (backfill)
- View statistics on stored user context

### Intelligent Conversation Management

- Persistent conversation sessions with unique IDs
- Vector-based semantic search on conversation history
- Handles long conversations efficiently (100+ messages)
- Only retrieves contextually relevant past messages

### Context-Aware Responses

- Combines document knowledge with conversation context
- Returns sources for transparency
- Shows which past messages influenced the response
- Includes intent classification metadata

### Security Hardening

- Input sanitization: message length limits (1-5000 chars) and whitespace stripping
- Prompt injection protection: XML tag delimiters for context data, security rules in system prompt
- Generic error messages to clients (no internal details leaked)
- API client timeouts (30s) to prevent hanging requests

## Architecture

```
┌─────────────────────┐
│      Client         │
└──────────┬──────────┘
           │ POST /chat
           │ + JWT Token
           │ + Provider/Model Config
           ▼
┌───────────────────────────────────────────────────────────┐
│              FastAPI Server (main.py)                     │
├───────────────────────────────────────────────────────────┤
│  1. Authenticate User (Supabase JWT)                      │
│  2. Select Embed & Chat Clients (Provider-based)          │
│  3. Store User Message + Embedding                        │
│                                                            │
│  4. ┌─────────────────────────────────────────────────┐  │
│     │         LangGraph Workflow                      │  │
│     ├─────────────────────────────────────────────────┤  │
│     │  A. Intent Classifier Node                      │  │
│     │     → OpenAI LLM or DistilBERT model            │  │
│     │     → Classify into 9 categories                │  │
│     │                                                  │  │
│     │  B. Conditional Router                          │  │
│     │     → Route to specialized response node        │  │
│     │                                                  │  │
│     │  C. RAG Retrieval Node (for RAG_QUERY only)     │  │
│     │     → Hybrid search on documents                │  │
│     │     → Semantic search on conversation history   │  │
│     │                                                  │  │
│     │  D. Specialized Response Nodes (9 nodes)        │  │
│     │     → RAG Query: Educational (temp=0.3)         │  │
│     │     → 6 Store A Contexts: Coaching (temp=0.7)   │  │
│     │     → Chitchat: Friendly (temp=0.8)             │  │
│     │     → Off-Topic: Boundary setting (temp=0.5)    │  │
│     └─────────────────────────────────────────────────┘  │
│                                                            │
│  5. Store Assistant Response + Embedding                  │
│  6. Return Response + Classification + Sources            │
└──────────┬─────────────────────┬─────────────────────────┘
           │                     │
           ├──────────┬──────────┼──────────┐
           ▼          ▼          ▼          ▼
    ┌──────────┐ ┌─────────┐ ┌─────────┐ ┌────────┐
    │ Supabase │ │ OpenAI  │ │ Voyage  │ │Vector  │
    │   Auth   │ │   API   │ │   AI    │ │ Store  │
    └──────────┘ └─────────┘ └─────────┘ └────────┘
```

## Tech Stack

- **Framework**: FastAPI
- **Workflow Orchestration**: LangGraph
- **AI/ML Providers**:
  - OpenAI (GPT-4o-mini, GPT-4o for chat + classification; text-embedding-3 for embeddings; GPT-3.5-turbo for synthetic data generation)
  - Voyage AI (voyage-3-large for embeddings)
- **Intent Classification**:
  - OpenAI LLM classifier (default)
  - DistilBERT fine-tuned model (optional, local inference)
- **ML Training Stack**:
  - Hugging Face Transformers (DistilBERT fine-tuning)
  - Sentence Transformers (centroid computation for semantic gate)
  - PyTorch (training backend)
- **Database**: Supabase (PostgreSQL + pgvector)
- **Authentication**: Supabase JWT
- **Language**: Python 3.10+

## Installation

### Prerequisites

- Python 3.10 or higher
- Supabase account with a project
- OpenAI API key (required for chat, optional for LLM-based intent classification and embeddings)
- Voyage AI API key (optional for embeddings)
- (Optional) PyTorch and Transformers for DistilBERT classifier training/inference

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd pop-ai
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   Create a `.env` file in the root directory:
   ```env
   # Supabase Configuration
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_SERVICE_KEY=your-service-role-key
   SUPABASE_JWT_SECRET=your-jwt-secret-key

   # OpenAI (Chat + LLM Classifier + Optional Embeddings)
   OPENAI_API_KEY=your-openai-api-key
   CHAT_MODEL=gpt-4o-mini

   # Voyage AI (Optional Embeddings - Default)
   VOYAGE_API_KEY=pa-your-voyage-api-key
   EMBED_MODEL=voyage-3-large
   EMBED_DIMENSIONS=1024

   # Message Classifier Settings (optional, shown with defaults)
   MESSAGE_WORTHINESS_THRESHOLD=70
   MESSAGE_CLASSIFIER_TYPE=hybrid
   MESSAGE_LLM_THRESHOLD=60

   # Intent Classifier Settings (optional, shown with defaults)
   INTENT_CLASSIFIER_TYPE=openai  # "openai" or "distilbert"
   INTENT_CLASSIFIER_MODEL_PATH=training/models/20260201_203858/final
   ```

5. **Set up database**

   Run the SQL migrations in your Supabase SQL Editor:
   ```bash
   # 1. Create user sessions table
   sql/create_user_sessions_table.sql

   # 2. Create conversation history table
   sql/create_conversation_history_table.sql
   ```

6. **Load your knowledge base**

   Use the RAG utilities in `rag/load_embeddings.py` to load your documents into the vector store.

## Running the Server

### Development Mode

```bash
python main.py
```

The server will start at `http://localhost:8000`

### Development Mode with Verbose Logging

Enable verbose mode to append workflow process information to the response text:

```bash
python main.py -v
# or
python main.py --verbose
```

When enabled, the `/chat` endpoint will append workflow details to the response:
- Intent classification steps and confidence scores
- Routing decisions (RAG retrieval vs direct response)
- Document and conversation history search results
- Response generation details (model, temperature, etc.)

The workflow information is appended to the response text after a `\n\n` separator.

See [VERBOSE_MODE.md](VERBOSE_MODE.md) for detailed documentation.

### Production Mode

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Quick Start with Intent Classifiers

The system supports two intent classification backends. Choose based on your needs:

### Option 1: OpenAI LLM Classifier (Default)

**Best for:** Quick setup, highest accuracy, low-volume use

```bash
# Already configured if you have OPENAI_API_KEY set
python main.py
```

No additional setup required. The system uses OpenAI by default.

### Option 2: DistilBERT Classifier (Cost-Free)

**Best for:** High-volume use, offline deployments, cost optimization

**Step 1: Install dependencies**
```bash
pip install transformers torch
```

**Step 2: Generate synthetic training data**
```bash
python training/scripts/generate_data.py \
  --rag-count 1000 \
  --context-count 600 \
  --chitchat-count 800 \
  --offtopic-count 5000 \
  --output training/data/
```

**Step 3: Train the model**
```bash
python training/scripts/train_intent_classifier.py \
  --train-data training/data/ \
  --output training/models/ \
  --epochs 5
```

**Step 4: Configure and run**
```bash
# Update .env
INTENT_CLASSIFIER_TYPE=distilbert
INTENT_CLASSIFIER_MODEL_PATH=training/models/20260201_203858/final

# Start server
python main.py
```

**Quick comparison:**
- **OpenAI**: ~95% accuracy, 500-1000ms, ~$0.0002/request
- **DistilBERT**: ~85-90% accuracy, 50-100ms, free after training

See [docs/INTENT_CLASSIFIER.md](docs/INTENT_CLASSIFIER.md) for detailed configuration and performance comparison.

## API Documentation

### Chat Endpoint

**POST** `/chat`

Send a message to the chatbot and receive an AI-generated response with intent classification.

#### Headers

```
Authorization: Bearer <supabase-jwt-token>
Content-Type: application/json
```

#### Request Body

```json
{
  "message": "What services does Pop Skills offer?",
  "conversation_id": "optional-uuid",
  "embed_provider": "voyage",
  "embed_model": "voyage-3-large",
  "chat_provider": "openai",
  "chat_model": "gpt-4o-mini",
  "message_worth_method": "hybrid"
}
```

**Fields:**

- `message` (string, required): The user's message (1-5000 characters)
- `conversation_id` (string, optional): Conversation ID for continuity. Omit to start a new conversation
- `embed_provider` (string, optional): Embedding provider ("openai" or "voyage"). Defaults to "voyage"
- `embed_model` (string, optional): Embedding model name. Defaults to env `EMBED_MODEL`
- `chat_provider` (string, optional): Chat provider ("openai"). Defaults to "openai"
- `chat_model` (string, optional): Chat model name (e.g., "gpt-4o-mini", "gpt-4o"). Defaults to env `CHAT_MODEL`
- `message_worth_method` (string, optional): Message classification method ("heuristic", "llm", "hybrid"). Defaults to env `MESSAGE_CLASSIFIER_TYPE`

#### Response

```json
{
  "response": "Pop Skills offers comprehensive training in AI, data science, web development...",
  "user_id": "user-uuid",
  "conversation_id": "conversation-uuid",
  "classification": {
    "category": "rag_query",
    "confidence": 0.95,
    "secondary_categories": [],
    "reasoning": "User is asking a factual question about services",
    "key_entities": {
      "topics": ["services", "training"]
    }
  },
  "sources": [
    {
      "content": "Pop Skills provides training programs in artificial intelligence...",
      "score": 0.87
    }
  ],
  "conversation_context": [
    {
      "role": "user",
      "message": "Tell me about your company...",
      "similarity": 0.75
    }
  ]
}
```

**Fields:**

- `response` (string): The AI-generated response
- `user_id` (string): Authenticated user's ID
- `conversation_id` (string): Conversation session ID (save this for follow-up messages)
- `classification` (object): Intent classification results
  - `category` (string): Message category (rag_query, professional, psychological, learning, social, emotional, aspirational, chitchat, off_topic)
  - `confidence` (float): Classification confidence (0-1)
  - `classifier_type` (string): Classifier used ("openai" or "distilbert")
  - `secondary_categories` (array): Additional relevant categories
  - `reasoning` (string): Explanation for classification
  - `key_entities` (object): Extracted entities from the message
- `sources` (array): Relevant documents from knowledge base (when RAG query)
- `conversation_context` (array): Relevant past messages that influenced the response

#### Example: Information Query (RAG Path)

```bash
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer eyJhbGc..." \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is machine learning?",
    "embed_provider": "voyage",
    "chat_model": "gpt-4o-mini"
  }'
```

Response will include:
- Educational, factual response (temperature=0.3)
- `classification.category` = "rag_query"
- `sources` array with relevant documents

#### Example: Career Coaching (Coaching Path)

```bash
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer eyJhbGc..." \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I want to become a data scientist in 2 years",
    "chat_model": "gpt-4o"
  }'
```

Response will include:
- Empathetic, actionable coaching response (temperature=0.7)
- `classification.category` = "aspirational"
- No document sources (direct coaching)

#### Example: Using OpenAI Embeddings

```bash
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer eyJhbGc..." \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How do I improve my Python skills?",
    "embed_provider": "openai",
    "embed_model": "text-embedding-3-small"
  }'
```

#### Example: Continuing a Conversation

```bash
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer eyJhbGc..." \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tell me more about that",
    "conversation_id": "abc-123-def-456"
  }'
```

### User Context Endpoints

#### Search User Context

**GET** `/user/context/search?query=<text>&top_k=5`

Search the user's own indexed worthy messages (retrieval_chunks).

```bash
curl -X GET "http://localhost:8000/user/context/search?query=leadership&top_k=5" \
  -H "Authorization: Bearer eyJhbGc..."
```

#### Reindex Conversation

**POST** `/user/context/reindex?conversation_id=<uuid>`

Reprocess a past conversation to extract worthy messages (backfill).

```bash
curl -X POST "http://localhost:8000/user/context/reindex?conversation_id=abc-123" \
  -H "Authorization: Bearer eyJhbGc..."
```

#### User Context Stats

**GET** `/user/context/stats`

Get the total number of indexed worthy messages for the authenticated user.

```bash
curl -X GET "http://localhost:8000/user/context/stats" \
  -H "Authorization: Bearer eyJhbGc..."
```

### Interactive API Documentation

Once the server is running, visit:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Project Structure

```
pop-ai/
├── main.py                              # FastAPI application entry point
├── requirements.txt                     # Python dependencies
├── .env                                 # Environment variables (not in git)
│
├── src/
│   ├── config.py                        # Centralized configuration & client factories
│   │
│   ├── api/
│   │   ├── __init__.py                  # Router exports
│   │   ├── chat.py                      # Chat endpoint with LangGraph workflow
│   │   └── user_context.py             # User context search, reindex & stats
│   │
│   ├── models/
│   │   ├── chat.py                      # Pydantic models for chat (with provider parameters)
│   │   ├── chunk.py                     # Document chunk models
│   │   └── embeddings.py               # Embedding models
│   │
│   ├── agents/
│   │   ├── langgraph_workflow.py       # LangGraph workflow orchestration (9 response nodes)
│   │   ├── context_classifier.py       # Store A context classification
│   │   └── distilbert_classifier.py    # DistilBERT intent classifier (optional)
│   │
│   └── utils/
│       ├── auth.py                      # JWT authentication
│       ├── chat.py                      # Response generation utilities
│       ├── conversation_memory.py       # Conversation RAG utilities
│       ├── message_classifier.py        # Heuristic + LLM message worthiness classifier
│       ├── rag.py                       # Document RAG utilities (hybrid search)
│       ├── user_context_search.py       # Search user's worthy messages
│       └── worthy_message_storage.py    # Classify & store high-value messages
│
├── rag/
│   ├── load_embeddings.py               # Batch embed & load documents into Supabase
│   ├── test_rag.py                      # Interactive RAG testing CLI
│   └── benchmarks.py                    # Search benchmarking utilities
│
├── training/
│   ├── scripts/
│   │   ├── generate_data.py            # Generate synthetic training data with GPT-3.5
│   │   ├── train_intent_classifier.py  # Fine-tune DistilBERT on synthetic data
│   │   └── evaluate.py                 # Evaluate trained model performance
│   │
│   ├── utils/
│   │   ├── generation.py               # Synthetic data generation utilities
│   │   ├── supabase.py                 # Knowledge base fetching utilities
│   │   ├── processing.py               # Data processing utilities
│   │   └── ai.py                       # AI utilities
│   │
│   ├── constants/
│   │   ├── categories.py               # Intent category definitions
│   │   └── prompts.py                  # Synthetic data generation prompts
│   │
│   └── models/                         # Trained model checkpoints
│       └── 20260201_203858/
│           └── final/                  # Production-ready model
│               ├── config.json         # Model configuration
│               ├── model.safetensors   # Model weights
│               ├── tokenizer.json      # Tokenizer
│               ├── centroids.pkl       # Centroid embeddings
│               └── centroid_metadata.json  # Centroid metadata
│
├── tests/
│   ├── test_message_classifier.py       # Message classifier unit tests
│   ├── test_worthy_message_storage.py   # Worthy message storage unit tests
│   ├── test_intent_classifiers.py       # Intent classifier comparison tests
│   └── test_workflow_nodes.py           # Workflow node tests (9 categories)
│
├── docs/
│   ├── INTENT_CLASSIFIER.md            # Intent classifier configuration guide
│   ├── WORKFLOW_ARCHITECTURE.md        # Workflow architecture documentation
│   └── RESPONSE_STYLE_GUIDE.md         # Response style guidelines
│
├── sql/
│   ├── 01_setup_supabase.sql           # General embeddings table + search functions
│   ├── 02_create_user_embeddings_table.sql  # User-specific embeddings table
│   ├── 03_create_conversation_history_table.sql  # Conversation history
│   └── 04_switch_to_english_language.sql  # (Optional) Switch to English text search
│
├── .env.example                        # Example environment configuration
└── README.md                            # This file
```

## How It Works

### 1. User Authentication

The server validates the JWT token from the `Authorization` header using Supabase Auth.

### 2. Provider & Model Selection

Based on request parameters or environment defaults, the system selects:
- **Embedding client**: OpenAI or Voyage AI
- **Chat client**: OpenAI
- **Models**: Specific models for embeddings and chat

### 3. Message Storage

Each user message is stored in the database with a vector embedding for future retrieval.

### 4. LangGraph Workflow Execution

The workflow orchestrates multiple agents:

#### A. Intent Classifier Node
Uses OpenAI LLM (default) or fine-tuned DistilBERT model to classify messages into one of 9 categories:

**Core Categories (7):**
- **RAG_QUERY**: Factual questions needing information lookup
- **PROFESSIONAL**: Skills, experience, technical abilities
- **PSYCHOLOGICAL**: Personality, values, motivations
- **LEARNING**: Learning preferences, educational background
- **SOCIAL**: Network, mentors, community
- **EMOTIONAL**: Confidence, stress, wellbeing (highest priority)
- **ASPIRATIONAL**: Career goals, dreams, ambitions

**Special Categories (2):**
- **CHITCHAT**: Casual conversation, greetings, small talk
- **OFF_TOPIC**: Topics unrelated to career coaching

**Classifier Options:**
- **OpenAI LLM**: High accuracy (~95%+), 500-1000ms latency, API costs
- **DistilBERT Model**: Fast (~50-100ms), no API costs, offline capable, ~85-90% accuracy

#### B. Conditional Router
Routes to specialized response nodes based on classification:
- **RAG_QUERY** → RAG Retrieval Node → RAG Query Response Node
- **Store A Contexts** (6 categories) → Specialized Context Response Node
- **CHITCHAT** → Chitchat Response Node
- **OFF_TOPIC** → Off-Topic Response Node

#### C. RAG Retrieval Node (for queries only)
- Hybrid search on documents (semantic + keyword via RRF fusion)
- Semantic search on conversation history
- Formats contexts for LLM

#### D. Specialized Response Nodes (9 nodes)
Each category has a dedicated response node with optimized prompts and temperature settings:

**RAG Query Response Node** (temp=0.3):
- Educational, factual responses
- Uses document retrieval results
- Deterministic and reliable

**Store A Context Response Nodes** (temp=0.7):
- **Professional**: Skills/experience coaching, acknowledges strengths
- **Psychological**: Values alignment coaching, deep empathy
- **Learning**: Learning strategy coaching, resource recommendations
- **Social**: Networking/mentorship coaching, relationship strategies
- **Emotional**: Wellbeing/resilience coaching, **highest priority**, deep validation
- **Aspirational**: Goal-setting/vision coaching, actionable steps

**Chitchat Response Node** (temp=0.8):
- Friendly, conversational tone
- Brief responses (1-3 sentences)
- Gentle redirection to career topics

**Off-Topic Response Node** (temp=0.5):
- Polite boundary setting
- Explains scope as career coach
- Consistent messaging

### 5. Response Storage

The assistant's response is stored with an embedding for future context retrieval.

### 6. Worthy Message Classification (Background)

After the response is sent, a background task classifies the user's message:

1. **Heuristic scoring**: Pattern matching for career goals, skills, preferences, challenges
2. **LLM validation** (if score >= threshold): OpenAI confirms worthiness and extracts entities
3. **Storage**: High-value messages are embedded and stored in `retrieval_chunks` for long-term user context

## Configuration

### Provider & Model Selection

The system supports flexible provider and model selection:

#### Embedding Providers
- **Voyage AI** (default): `voyage-3-large`, `voyage-3`
- **OpenAI**: `text-embedding-3-small`, `text-embedding-3-large`

#### Chat Providers
- **OpenAI**: `gpt-4o-mini`, `gpt-4o`, `gpt-4-turbo`

#### Validation
The system validates that models match their providers:
- OpenAI embeddings must start with `text-embedding`
- Voyage embeddings must start with `voyage`

### RAG Parameters

Adjust in `src/api/chat.py` or `src/agents/langgraph_workflow.py`:

```python
# Conversation memory search
relevant_history = search_conversation_history(
    ...
    top_k=5,                      # Number of past messages to retrieve
    similarity_threshold=0.6      # Minimum similarity (0-1)
)

# Document search
document_results = hybrid_search(
    ...
    top_k=3                       # Number of documents to retrieve
)
```

### Model Configuration

Update in `.env` or pass per request:

```env
# Default Embedding (Voyage AI)
EMBED_MODEL=voyage-3-large
EMBED_DIMENSIONS=1024

# Default Chat (OpenAI)
CHAT_MODEL=gpt-4o-mini
```

### Message Classifier Settings

Update in `.env`:

```env
# Minimum score (0-100) to consider a message worthy of storage
MESSAGE_WORTHINESS_THRESHOLD=70

# Classifier type: "heuristic", "llm", or "hybrid"
MESSAGE_CLASSIFIER_TYPE=hybrid

# Score threshold for LLM validation in hybrid mode
MESSAGE_LLM_THRESHOLD=60
```

## Intent Classification System

The system classifies every message to determine the appropriate response strategy using either OpenAI LLMs or a fine-tuned DistilBERT model.

### 9 Categories

**Core Categories (7):**

1. **RAG_QUERY** - Information seeking
   - "What is a REST API?"
   - "How do I write a resume?"
   - Triggers document retrieval
   - Educational response style (temp=0.3)

2. **PROFESSIONAL** - Skills & experience
   - "I have 5 years of Python experience"
   - Store for user profile
   - Professional coaching response (temp=0.7)

3. **PSYCHOLOGICAL** - Values & personality
   - "I value work-life balance"
   - Store for personalization
   - Values alignment coaching (temp=0.7)

4. **LEARNING** - Learning preferences
   - "I learn best through hands-on projects"
   - Store for adaptive teaching
   - Learning strategy coaching (temp=0.7)

5. **SOCIAL** - Network & relationships
   - "My mentor helped me navigate my career"
   - Store for social context
   - Networking coaching response (temp=0.7)

6. **EMOTIONAL** - Wellbeing & confidence (highest priority)
   - "I'm feeling burned out"
   - Immediate empathetic response
   - Wellbeing/resilience coaching (temp=0.7)

7. **ASPIRATIONAL** - Goals & dreams
   - "I want to become a CTO in 5 years"
   - Store for goal tracking
   - Goal-setting coaching response (temp=0.7)

**Special Categories (2):**

8. **CHITCHAT** - Casual conversation
   - "Hey! How are you?"
   - "Good morning!"
   - Brief, friendly responses (temp=0.8)
   - Gentle redirection to career topics

9. **OFF_TOPIC** - Unrelated topics
   - "What's the weather like?"
   - "Tell me a joke"
   - Polite boundary setting (temp=0.5)
   - Explains scope as career coach

### Dual Classifier System

#### OpenAI LLM Classifier (Default)
- **Accuracy**: ~95%+ with contextual understanding
- **Latency**: 500-1000ms (network latency)
- **Cost**: ~$0.0002 per classification
- **Setup**: Works out of box with OpenAI API key
- **Best for**: High accuracy requirements, cloud-based deployments

#### DistilBERT Fine-Tuned Model
- **Accuracy**: ~85-90% (excellent for most use cases)
- **Latency**: 50-100ms (local inference)
- **Cost**: Free after training
- **Setup**: Requires PyTorch + Transformers
- **Best for**: Cost optimization, offline deployments, high-volume use
- **Two-Stage Approach**:
  - Stage 1: Semantic gate (filters off-topic using centroid similarity)
  - Stage 2: BERT classifier (fine-grained classification of 8 in-domain categories)

**Switching Classifiers:**
```bash
# Use OpenAI (default)
export INTENT_CLASSIFIER_TYPE=openai

# Use DistilBERT
export INTENT_CLASSIFIER_TYPE=distilbert
export INTENT_CLASSIFIER_MODEL_PATH=training/models/20260201_203858/final
```

### Response Strategies

#### RAG Query Path
```
User Query → Intent Classifier → RAG Retrieval → RAG Query Response Node
            (category=rag_query)  (docs+history)   (educational, temp=0.3)
```

#### Coaching Path (6 Store A Contexts)
```
User Message → Intent Classifier → Specialized Context Response Node
              (Store A context)   (empathetic coaching, temp=0.7)
```

#### Chitchat Path
```
User Message → Intent Classifier → Chitchat Response Node
              (category=chitchat) (friendly, brief, temp=0.8)
```

#### Off-Topic Path
```
User Message → Intent Classifier → Off-Topic Response Node
              (category=off_topic) (boundary setting, temp=0.5)
```

## Training Your Own Intent Classifier

The system supports training a custom DistilBERT model for intent classification, eliminating OpenAI API costs and enabling offline deployments.

### Step 1: Generate Synthetic Training Data

Use GPT-3.5-turbo to generate labeled training data for all 9 categories:

```bash
# Generate balanced dataset
python training/scripts/generate_data.py \
  --output training/data/synthetic_labeled.csv \
  --rag-count 1000 \
  --context-count 600 \
  --chitchat-count 800 \
  --offtopic-count 5000 \
  --use-knowledge-base
```

**Features:**
- Generates messages for 8 categories (rag_query, 6 Store A contexts, chitchat)
- Fetches content from your knowledge base for realistic RAG queries
- Creates diverse off-topic messages to improve boundary detection
- Saves category-specific files (e.g., `rag_queries.txt`, `professional.txt`)

**Example Output:**
```
training/data/
├── rag_queries.txt      # 1000 RAG query messages
├── professional.txt     # 600 professional context messages
├── psychological.txt    # 600 psychological context messages
├── learning.txt         # 600 learning context messages
├── social.txt           # 600 social context messages
├── emotional.txt        # 600 emotional context messages
├── aspirational.txt     # 600 aspirational context messages
├── chitchat.txt         # 800 chitchat messages
└── off_topic.txt        # 5000 off-topic messages
```

### Step 2: Train the DistilBERT Model

Fine-tune DistilBERT on the synthetic data:

```bash
# Train with directory of category files
python training/scripts/train_intent_classifier.py \
  --train-data training/data/ \
  --output training/models/ \
  --model distilbert-base-uncased \
  --epochs 5 \
  --batch-size 20 \
  --val-split 0.2
```

**Training Features:**
- **Two-Stage Architecture**: Trains both semantic gate (Stage 1) and BERT classifier (Stage 2)
- **Centroid Computation**: Automatically computes embedding centroids for off-topic detection
- **Class Weights**: Handles imbalanced data with automatic class weighting
- **Early Stopping**: Prevents overfitting with validation-based early stopping
- **GPU Acceleration**: Automatic GPU detection and mixed-precision training

**Training Output:**
```
training/models/20260201_203858/
├── checkpoint-100/          # Training checkpoints
├── checkpoint-200/
└── final/
    ├── config.json          # Model configuration
    ├── model.safetensors    # Model weights
    ├── tokenizer.json       # Tokenizer
    ├── centroids.pkl        # Centroid embeddings for semantic gate
    └── centroid_metadata.json  # Centroid metadata
```

### Step 3: Evaluate the Model

Test the trained model's accuracy:

```bash
# Evaluate on test set
python training/scripts/evaluate.py \
  --model-path training/models/20260201_203858/final \
  --test-data training/data/test.csv
```

### Step 4: Use the Trained Model

Update your `.env` file:

```env
INTENT_CLASSIFIER_TYPE=distilbert
INTENT_CLASSIFIER_MODEL_PATH=training/models/20260201_203858/final
```

Restart the server:

```bash
python main.py
```

### Two-Stage Classification Architecture

The DistilBERT classifier uses a two-stage approach for robust off-topic detection:

**Stage 1: Semantic Gate (Gross Filter)**
- Computes cosine similarity between input and category centroids
- Filters out messages with low similarity to all real intent categories
- Fast and efficient (no model inference)
- Classifies as OFF_TOPIC if below threshold

**Stage 2: BERT Classifier (Fine-Grained)**
- Only runs if Stage 1 passes (message is in-domain)
- Fine-tuned DistilBERT classifies into 8 categories:
  - 7 core categories (rag_query, 6 Store A contexts)
  - chitchat
- Does NOT predict off_topic (handled by Stage 1)

**Benefits:**
- Reduces false positives for off-topic detection
- Fast rejection of irrelevant messages
- More accurate in-domain classification
- Lower computational cost

### Performance Comparison

| Metric | OpenAI LLM | DistilBERT |
|--------|------------|------------|
| Accuracy | ~95%+ | ~85-90% |
| Latency | 500-1000ms | 50-100ms |
| Cost per request | ~$0.0002 | Free |
| Offline capable | ❌ | ✅ |
| Setup complexity | Low (API key) | Medium (training required) |
| Training time | N/A | ~10-30 minutes |
| Model size | Cloud-based | ~270MB |

**When to use DistilBERT:**
- High-volume deployments (>10,000 requests/day)
- Offline or air-gapped environments
- Cost-sensitive applications
- Consistent, predictable latency requirements

**When to use OpenAI LLM:**
- Low-volume deployments (<1,000 requests/day)
- Prototyping and development
- Highest accuracy requirements
- Minimal setup time

## Conversation Memory Features

### Smart Context Retrieval

Instead of sending the entire conversation history, the system:

1. Embeds the current query
2. Searches for semantically similar past messages
3. Only includes the most relevant messages in the context

### Benefits

- Handles conversations with 100+ messages
- Reduces token costs
- Maintains context even after long gaps
- References specific past topics when relevant

### Example

```
User: "What training programs do you offer?"
Bot: "We offer AI, data science, and web development training..."

[50 messages later about different topics]

User: "How long are those programs?"
Bot: "The AI, data science, and web development programs I mentioned are 12 weeks each..."
```

The bot retrieves the relevant message from 50 turns ago using vector search.

## Security

### Authentication

- All endpoints require valid Supabase JWT tokens
- Tokens are verified on each request

### Input Validation

- Message length enforced (1-5000 characters) via Pydantic Field validation
- Whitespace stripping to prevent empty-looking messages
- Provider/model validation to prevent mismatches

### Prompt Injection Protection

- User-provided context wrapped in XML tags (`<documents>`, `<history>`) to separate data from instructions
- System prompt includes explicit security rules preventing instruction injection
- LLM instructed to never reveal system prompt or raw context data

### Error Handling

- Internal errors logged with full stack traces (`logger.exception`)
- Generic error messages returned to clients (no internal details leaked)

### Data Isolation

- Row-Level Security (RLS) ensures users only access their own data
- Conversation history is user-scoped
- Service role has full access for backend operations

### Environment Variables

- Sensitive credentials stored in `.env` (excluded from git)
- Never commit API keys or secrets

## Performance

### Latency

- Average response time: 2-4 seconds
- Breakdown:
  - Authentication: ~50ms
  - Intent Classification: ~200ms
  - Embedding generation: ~200ms
  - Vector search: ~100ms
  - Response generation: 1-3s

### Scaling

- Stateless design allows horizontal scaling
- Database handles concurrent requests efficiently
- HNSW indexes provide fast approximate nearest neighbor search
- Multiple provider support for load distribution

## Testing

### Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_message_classifier.py -v
pytest tests/test_worthy_message_storage.py -v
```

### Intent Classifier Tests

Compare OpenAI and DistilBERT classifiers:

```bash
# Test both classifiers side-by-side
python tests/test_intent_classifiers.py

# Test only OpenAI classifier
python tests/test_intent_classifiers.py --classifier openai

# Test only DistilBERT classifier
python tests/test_intent_classifiers.py --classifier distilbert
```

### Workflow Node Tests

Test all 9 specialized response nodes:

```bash
# Test all 9 categories
python tests/test_workflow_nodes.py

# Test specific category
python tests/test_workflow_nodes.py --category chitchat
python tests/test_workflow_nodes.py --category off_topic
python tests/test_workflow_nodes.py --category emotional
python tests/test_workflow_nodes.py --category rag_query
```

### Training Pipeline Tests

Test synthetic data generation and model training:

```bash
# Generate small test dataset
python training/scripts/generate_data.py \
  --rag-count 10 \
  --context-count 10 \
  --chitchat-count 10 \
  --offtopic-count 10 \
  --output training/data/test/

# Train on test dataset (quick validation)
python training/scripts/train_intent_classifier.py \
  --train-data training/data/test/ \
  --epochs 1 \
  --batch-size 8 \
  --output training/models/test/

# Evaluate test model
python training/scripts/evaluate.py \
  --model-path training/models/test/final \
  --test-data training/data/test/
```

## Troubleshooting

### Common Issues

**Issue**: "Authentication failed"

- **Solution**: Verify JWT token is valid and not expired
- Check Supabase service key in `.env`

**Issue**: "Unsupported embed provider"

- **Solution**: Ensure `embed_provider` is "openai" or "voyage"
- Verify model matches provider (text-embedding-* for OpenAI, voyage-* for Voyage)

**Issue**: "No conversation context retrieved"

- **Solution**: Lower `similarity_threshold` from 0.6 to 0.5
- Ensure embeddings are being stored (check database)

**Issue**: "OpenAI API error"

- **Solution**: Verify `OPENAI_API_KEY` in `.env`
- Check API quota and billing

**Issue**: "Model validation error"

- **Solution**: Ensure model name matches provider
  - OpenAI: use `text-embedding-3-small`, `text-embedding-3-large`
  - Voyage: use `voyage-3-large`, `voyage-3`

**Issue**: "DistilBERT model not found"

- **Solution**: Verify `INTENT_CLASSIFIER_MODEL_PATH` points to valid model directory
- Check model files exist: `config.json`, `model.safetensors`, `tokenizer.json`
- System automatically falls back to OpenAI if DistilBERT fails

**Issue**: "ImportError: No module named 'transformers'"

- **Solution**: Install PyTorch and Transformers:
  ```bash
  pip install transformers torch
  ```
- Or continue using OpenAI classifier (no additional dependencies)

**Issue**: "Poor classification accuracy with DistilBERT"

- **Solution**: Retrain model with more synthetic data
- Increase training epochs (default: 5)
- Check class balance in training data
- Consider using OpenAI classifier for highest accuracy

**Issue**: "Synthetic data generation too slow"

- **Solution**: GPT-3.5-turbo has rate limits (3500 RPM)
- Generation automatically handles rate limiting
- Reduce data counts for faster generation
- Consider generating data in batches over time

### Logs

Enable debug logging in `main.py`:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Deployment

### Docker (Recommended)

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Cloud Platforms

- **Render**: Connect GitHub repo, set environment variables
- **Railway**: One-click deploy from GitHub
- **AWS/GCP**: Use container services (ECS, Cloud Run)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is proprietary and confidential to Pop Skills (HARMONIA).

## Support

For questions or issues, contact the development team.

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Workflow orchestration by [LangGraph](https://langchain-ai.github.io/langgraph/)
- Chat powered by [OpenAI](https://openai.com/)
- Embeddings by [Voyage AI](https://voyageai.com/) and [OpenAI](https://openai.com/)
- Database by [Supabase](https://supabase.com/)
- Vector search via [pgvector](https://github.com/pgvector/pgvector)
- ML training with [Hugging Face Transformers](https://huggingface.co/transformers/) and [PyTorch](https://pytorch.org/)
- Sentence embeddings by [Sentence Transformers](https://www.sbert.net/)
- Synthetic data generation powered by GPT-3.5-turbo
