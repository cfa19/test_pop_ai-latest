# RAG General - Hybrid Search with Supabase

Hybrid search system (semantic + full-text) using pgvector and Supabase with Reciprocal Rank Fusion (RRF).

## Features

- **Hybrid Search**: Combines semantic (embeddings) and full-text (keywords) search
- **HNSW Index**: Fast approximate nearest neighbor search with O(log n) complexity
- **GIN Index**: PostgreSQL full-text search with tsvector
- **RRF Fusion**: Reciprocal Rank Fusion to combine ranking results
- **Voyage AI Embeddings**: `voyage-3-large` model with 1024-dimensional vectors

## Project Structure

```
rag/
├── load_embeddings.py     # Load documents and generate embeddings
├── test_rag.py            # Interactive CLI for searches
├── benchmarks.py          # Performance benchmarks
└── README.md

# Related files at project root
sql/01_setup_supabase.sql          # Tables and SQL functions
info/chunks/general_info_chunks.md # Pre-chunked source document (50 chunks)
src/utils/rag.py                   # Shared search & embedding functions
```

## Configuration

### 1. Environment variables

The scripts use the project root `.env` file. Required variables:

```env
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_SERVICE_KEY=eyJxxx...

# Voyage AI (embeddings)
VOYAGE_API_KEY=pa-xxx...
EMBED_MODEL=voyage-3-large
EMBED_DIMENSIONS=1024

# OpenAI (chat in test_rag.py)
OPENAI_API_KEY=sk-xxx...
CHAT_MODEL=gpt-4o-mini
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Execute SQL in Supabase

- Go to Supabase Dashboard > SQL Editor
- Copy and execute `sql/01_setup_supabase.sql`

### 4. Load embeddings

```bash
python rag/load_embeddings.py
```

Select option 1 to load `info/chunks/general_info_chunks.md` into Supabase.

#### Chunk File Format

The chunks file uses `## CHUNK X:` headers parsed by `src/utils/rag.py:parse_chunks_file()`:

```markdown
## CHUNK 1: Title of First Chunk

**Section:** Main Section Name
**Subsection:** Subsection Name (optional)
**Chunk ID:** 1

This is the content of the first semantic unit or chunk.
It can contain multiple paragraphs, lists, tables, or any markdown content.

---

## CHUNK 2: Title of Second Chunk

**Section:** Main Section Name
**Subsection:** Another Subsection (optional)
**Chunk ID:** 2

Content of the second chunk. Each chunk should be
semantically coherent and self-contained for optimal retrieval.

---
```

**Key points:**
- Each chunk starts with `## CHUNK X:` (two hash marks + chunk number)
- Chunks are separated by `---` (three dashes)
- Metadata fields (`**Section:**`, `**Subsection:**`, `**Chunk ID:**`) are optional but recommended
- Content follows the metadata and continues until the next `---` separator

## Usage

### Interactive CLI

```bash
python rag/test_rag.py
```

Options:
1. Chat with RAG (hybrid search + LLM answer)
2. Compare search types (debug)
3. Adjust semantic weight
4. Exit

### Performance Benchmarks

```bash
python rag/benchmarks.py
```

Generates `doc/BENCHMARK_RESULTS.md` with detailed analysis.

## SQL Functions

| Function | Description |
|----------|-------------|
| `rag_search_semantic()` | Search by embedding similarity (cosine) |
| `rag_search_fulltext()` | Search by keywords (tsvector) |
| `rag_hybrid_search_user_context()` | Hybrid search with RRF |

## Adjustable Parameters

### semantic_weight (0.0 to 1.0)

Controls the balance between semantic and full-text search:

| Value | Description |
|-------|-------------|
| 1.0 | 100% semantic (embeddings only) |
| 0.7 | 70% semantic, 30% full-text |
| 0.5 | 50/50 balanced (recommended) |
| 0.3 | 30% semantic, 70% full-text |
| 0.0 | 100% full-text (keywords only) |

### rrf_k (default: 60)

RRF smoothing parameter. Higher values give more weight to documents with good ranking in both searches.

### top_k

Number of results to return:
- 3-5 for chatbot
- 10-20 for exhaustive search

## Embedding Configuration

| Model | Provider | Dimensions | Vector Index |
|-------|----------|------------|--------------|
| `voyage-3-large` | Voyage AI | **1024** | HNSW (cosine) |

Voyage AI's `voyage-3-large` produces 1024-dimensional embeddings natively, well within pgvector's HNSW limit of 2000 dimensions.

## Database Schema

### Table: `general_embeddings_1024`

| Column | Type | Description |
|--------|------|-------------|
| id | BIGSERIAL | Primary key |
| content | TEXT | Chunk text content |
| embedding | VECTOR(1024) | Voyage AI embedding |
| content_tsvector | TSVECTOR | Full-text search vector |
| metadata | JSONB | Source, chunk index, model info |
| created_at | TIMESTAMPTZ | Creation timestamp |

### Indexes

1. **HNSW Index**: `vector_cosine_ops` for semantic search
2. **GIN Index**: On `content_tsvector` for full-text search

## Requirements Checklist

### Hybrid Search
- [x] PostgreSQL function `rag_hybrid_search_user_context`
- [x] Reciprocal Rank Fusion (RRF) algorithm
- [x] Adjustable `semantic_weight` parameter

### Semantic Search
- [x] HNSW index for fast vector search
- [x] pgvector implementation
- [x] Cosine similarity function
- [x] Voyage AI voyage-3-large (1024 dims)

### Full-text Search
- [x] tsvector and ts_rank implementation
- [x] GIN index for fast keyword search
- [x] Language configuration (Spanish/English)

### Benchmarks
- [x] Performance benchmarks (`benchmarks.py`)
- [x] Search type comparison
- [x] top_k performance tests
- [x] semantic_weight tests
- [x] Optimal parameter recommendations
- [x] Markdown report generation

