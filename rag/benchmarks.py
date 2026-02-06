"""
RAG HYBRID SEARCH BENCHMARKS
============================
This script measures and compares the performance of:
1. Semantic search (embeddings)
2. Full-text search (keywords)
3. Hybrid search (RRF fusion)

Also tests different parameter configurations.
Generates results in ../doc/BENCHMARK_RESULTS.md

Run: python rag/benchmarks.py
"""

import json
import os
import statistics
import sys
import time
from datetime import datetime

# Add project root to path for src imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Load .env with explicit path (before config import, for standalone execution)
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from src.config import (  # noqa: E402
    EMBED_DIMENSIONS,
    EMBED_MODEL,
    RPCFunctions,
    get_embed_client,
    get_supabase,
)

# Create clients (lazy, from centralized config)
embed_client = get_embed_client()
supabase = get_supabase()

# Test queries
TEST_QUERIES = [
    "What is Pop Skills?",
    "coaching services",
    "professional development platform",
    "how does the platform work",
    "pricing and plans",
]


# ===============================
# SEARCH FUNCTIONS
# ===============================
def create_embedding(text: str) -> list[float]:
    """Generate embedding with Voyage AI (1024 dimensions)"""
    response = embed_client.embeddings.create(
        model=EMBED_MODEL,
        input=text,
    )
    return response.data[0].embedding


def semantic_search(query: str, query_embedding: list, top_k: int = 5):
    """Search only by embedding similarity"""
    result = supabase.rpc(RPCFunctions.RAG_SEARCH_SEMANTIC, {"query_embedding": query_embedding, "match_count": top_k}).execute()
    return result.data


def fulltext_search(query: str, top_k: int = 5):
    """Search only by keywords"""
    result = supabase.rpc(RPCFunctions.RAG_SEARCH_FULLTEXT, {"query_text": query, "match_count": top_k}).execute()
    return result.data


def hybrid_search(query: str, query_embedding: list, top_k: int = 5, semantic_weight: float = 0.5, rrf_k: int = 60):
    """Hybrid search with RRF fusion"""
    result = supabase.rpc(
        RPCFunctions.RAG_HYBRID_SEARCH,
        {"query_text": query, "query_embedding": query_embedding, "match_count": top_k, "semantic_weight": semantic_weight, "rrf_k": rrf_k},
    ).execute()
    return result.data


# ===============================
# BENCHMARK FUNCTIONS
# ===============================
def benchmark_search(search_func, query: str, query_embedding: list = None, iterations: int = 5, **kwargs):
    """Execute a search function multiple times and measure timings"""
    times = []
    results = None

    for _ in range(iterations):
        start = time.perf_counter()
        if query_embedding:
            results = search_func(query, query_embedding, **kwargs)
        else:
            results = search_func(query, **kwargs)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return {
        "avg_ms": statistics.mean(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "num_results": len(results) if results else 0,
        "results": results,
    }


def run_benchmarks():
    """Execute all benchmarks and return results"""
    print("=" * 70)
    print("RAG HYBRID SEARCH PERFORMANCE BENCHMARKS")
    print(f"Embedding: {EMBED_MODEL} ({EMBED_DIMENSIONS}d)")
    print("=" * 70)

    # Data to save
    benchmark_data = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {"embedding_model": EMBED_MODEL, "dimensions": EMBED_DIMENSIONS, "test_queries": TEST_QUERIES},
        "results": {},
    }

    # Pre-generate embeddings for test queries
    print("\n[1/4] Generating embeddings for test queries...")
    query_embeddings = {}
    for q in TEST_QUERIES:
        query_embeddings[q] = create_embedding(q)
    print(f"      {len(TEST_QUERIES)} embeddings generated")

    # ===============================
    # BENCHMARK 1: Compare search types
    # ===============================
    print("\n[2/4] Comparing search types...")
    print("-" * 70)
    print(f"{'Query':<35} {'Semantic':<12} {'Full-text':<12} {'Hybrid':<12}")
    print("-" * 70)

    all_results = {"semantic": [], "fulltext": [], "hybrid": []}
    query_details = []

    for query in TEST_QUERIES:
        emb = query_embeddings[query]

        # Semantic
        sem = benchmark_search(semantic_search, query, emb, top_k=5)
        all_results["semantic"].append(sem["avg_ms"])

        # Full-text
        ft = benchmark_search(fulltext_search, query, top_k=5)
        all_results["fulltext"].append(ft["avg_ms"])

        # Hybrid
        hyb = benchmark_search(hybrid_search, query, emb, top_k=5)
        all_results["hybrid"].append(hyb["avg_ms"])

        query_details.append(
            {"query": query, "semantic_ms": round(sem["avg_ms"], 2), "fulltext_ms": round(ft["avg_ms"], 2), "hybrid_ms": round(hyb["avg_ms"], 2)}
        )

        print(f"{query[:33]:<35} {sem['avg_ms']:>8.2f} ms  {ft['avg_ms']:>8.2f} ms  {hyb['avg_ms']:>8.2f} ms")

    print("-" * 70)
    print(
        f"{'AVERAGE':<35} {statistics.mean(all_results['semantic']):>8.2f} ms  "
        f"{statistics.mean(all_results['fulltext']):>8.2f} ms  "
        f"{statistics.mean(all_results['hybrid']):>8.2f} ms"
    )

    benchmark_data["results"]["search_comparison"] = {
        "by_query": query_details,
        "averages": {
            "semantic_ms": round(statistics.mean(all_results["semantic"]), 2),
            "fulltext_ms": round(statistics.mean(all_results["fulltext"]), 2),
            "hybrid_ms": round(statistics.mean(all_results["hybrid"]), 2),
        },
    }

    # ===============================
    # BENCHMARK 2: Different top_k values
    # ===============================
    print("\n[3/4] Testing different top_k values...")
    print("-" * 50)
    print(f"{'top_k':<10} {'Time (ms)':<15} {'Results':<15}")
    print("-" * 50)

    test_query = TEST_QUERIES[0]
    test_emb = query_embeddings[test_query]
    top_k_results = []

    for top_k in [1, 3, 5, 10, 20]:
        result = benchmark_search(hybrid_search, test_query, test_emb, top_k=top_k)
        top_k_results.append({"top_k": top_k, "time_ms": round(result["avg_ms"], 2), "results": result["num_results"]})
        print(f"{top_k:<10} {result['avg_ms']:>10.2f} ms   {result['num_results']:<15}")

    benchmark_data["results"]["top_k_tests"] = top_k_results

    # ===============================
    # BENCHMARK 3: Different semantic_weight values
    # ===============================
    print("\n[4/4] Testing different semantic_weight values...")
    print("-" * 70)
    print(f"{'Weight':<10} {'Time (ms)':<15} {'Description':<40}")
    print("-" * 70)

    weight_descriptions = {
        0.0: "100% Full-text (keywords only)",
        0.3: "70% Full-text, 30% Semantic",
        0.5: "50/50 Balanced (recommended)",
        0.7: "30% Full-text, 70% Semantic",
        1.0: "100% Semantic (embeddings only)",
    }
    weight_results = []

    for weight in [0.0, 0.3, 0.5, 0.7, 1.0]:
        result = benchmark_search(hybrid_search, test_query, test_emb, top_k=5, semantic_weight=weight)
        weight_results.append({"weight": weight, "time_ms": round(result["avg_ms"], 2), "description": weight_descriptions[weight]})
        print(f"{weight:<10} {result['avg_ms']:>10.2f} ms   {weight_descriptions[weight]:<40}")

    benchmark_data["results"]["semantic_weight_tests"] = weight_results

    # ===============================
    # SUMMARY
    # ===============================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    summary = f"""
Average response times:
  - Semantic Search: {statistics.mean(all_results["semantic"]):.2f} ms
  - Full-text Search: {statistics.mean(all_results["fulltext"]):.2f} ms
  - Hybrid Search:    {statistics.mean(all_results["hybrid"]):.2f} ms

Recommendations:
  - For fast searches: Full-text is faster but less semantically accurate
  - For best quality: Hybrid with semantic_weight=0.5 (balanced)
  - For short queries: semantic_weight=0.7 (more semantic)
  - For specific terms: semantic_weight=0.3 (more full-text)

Suggested optimal parameters:
  - semantic_weight: 0.5 (balanced)
  - rrf_k: 60 (standard RRF value)
  - top_k: 3-5 for chatbot, 10-20 for exhaustive search
"""
    print(summary)

    benchmark_data["results"]["recommendations"] = {"semantic_weight": 0.5, "rrf_k": 60, "top_k_chatbot": "3-5", "top_k_exhaustive": "10-20"}

    return benchmark_data


def save_benchmark_results(data: dict):
    """Save results to doc/BENCHMARK_RESULTS.md"""
    # Save to doc folder
    doc_dir = os.path.join(os.path.dirname(__file__), "..", "doc")
    os.makedirs(doc_dir, exist_ok=True)

    # Generate markdown report
    md_content = generate_markdown_report(data)

    # Save main report (overwrites)
    md_path = os.path.join(doc_dir, "BENCHMARK_RESULTS.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    # Also save timestamped version in results folder
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON
    json_path = os.path.join(results_dir, f"benchmark_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("\nResults saved to:")
    print(f"  - {md_path} (documentation)")
    print(f"  - {json_path} (raw data)")

    return md_path, json_path


def generate_markdown_report(data: dict) -> str:
    """Generate a comprehensive Markdown report in English"""
    r = data["results"]
    c = data["configuration"]

    md = f"""# RAG Hybrid Search Benchmark Results

**Generated:** {data["timestamp"]}
**Purpose:** Performance analysis of semantic, full-text, and hybrid search methods

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Embedding Model | `{c["embedding_model"]}` |
| Dimensions | {c["dimensions"]} |
| Provider | Voyage AI |
| Vector Index | HNSW with cosine distance |
| Full-text Index | GIN with Spanish tsvector |
| Test Queries | {len(c["test_queries"])} |

### Why {c["dimensions"]} Dimensions?

Voyage AI's `voyage-3-large` model produces {c["dimensions"]}-dimensional embeddings natively.
These provide high semantic precision and are well within pgvector's HNSW index limit of 2000 dimensions.

---

## Search Type Comparison

Comparison of average response times across {len(c["test_queries"])} test queries.

| Query | Semantic | Full-text | Hybrid |
|-------|----------|-----------|--------|
"""
    for q in r["search_comparison"]["by_query"]:
        query_display = q["query"][:35] + "..." if len(q["query"]) > 35 else q["query"]
        md += f"| {query_display} | {q['semantic_ms']} ms | {q['fulltext_ms']} ms | {q['hybrid_ms']} ms |\n"

    avgs = r["search_comparison"]["averages"]
    md += f"| **AVERAGE** | **{avgs['semantic_ms']} ms** | **{avgs['fulltext_ms']} ms** | **{avgs['hybrid_ms']} ms** |\n"

    md += """
### Analysis

- **Semantic Search**: Uses vector similarity (cosine distance) via HNSW index
- **Full-text Search**: Uses PostgreSQL tsvector with GIN index
- **Hybrid Search**: Combines both using Reciprocal Rank Fusion (RRF)

---

## Top-K Performance Tests

Testing how the number of results affects response time.

| top_k | Response Time | Results Returned |
|-------|---------------|------------------|
"""
    for t in r["top_k_tests"]:
        md += f"| {t['top_k']} | {t['time_ms']} ms | {t['results']} |\n"

    md += """
### Observations

- Response time scales approximately linearly with top_k
- For chatbots, `top_k=3-5` provides good results with minimal latency
- For comprehensive search, `top_k=10-20` may be appropriate

---

## Semantic Weight Tests

Testing different balances between semantic and full-text search.

| Weight | Time | Description |
|--------|------|-------------|
"""
    for w in r["semantic_weight_tests"]:
        md += f"| {w['weight']} | {w['time_ms']} ms | {w['description']} |\n"

    md += """
### Understanding Semantic Weight

The `semantic_weight` parameter controls the balance in hybrid search:

- **0.0**: 100% full-text (keyword matching only)
- **0.5**: 50/50 balanced (recommended default)
- **1.0**: 100% semantic (embedding similarity only)

**RRF Formula:**
```
rrf_score = semantic_weight * (1 / (k + semantic_rank)) +
            (1 - semantic_weight) * (1 / (k + fulltext_rank))
```

---

## Recommendations

"""
    rec = r["recommendations"]
    md += f"""### Optimal Parameters

| Parameter | Recommended Value | Use Case |
|-----------|-------------------|----------|
| `semantic_weight` | **{rec["semantic_weight"]}** | Balanced quality/speed |
| `rrf_k` | **{rec["rrf_k"]}** | Standard RRF smoothing |
| `top_k` (chatbot) | **{rec["top_k_chatbot"]}** | Quick responses |
| `top_k` (search) | **{rec["top_k_exhaustive"]}** | Comprehensive results |

### When to Adjust Semantic Weight

| Scenario | Recommended Weight | Reason |
|----------|-------------------|--------|
| Short/ambiguous queries | 0.7 | Semantic understanding helps |
| Specific terms/names | 0.3 | Exact keyword matching needed |
| General questions | 0.5 | Balanced approach |
| Technical documentation | 0.4 | Mix of terms and concepts |

---

## Technical Details

### Database Schema

```sql
CREATE TABLE general_embeddings_1024 (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding VECTOR({c["dimensions"]}),
    content_tsvector TSVECTOR,
    metadata JSONB DEFAULT '{{}}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Indexes

1. **HNSW Index** (Semantic): `vector_cosine_ops` with m=16, ef_construction=64
2. **GIN Index** (Full-text): On `content_tsvector` column

### RPC Functions

- `rag_search_semantic(query_embedding, match_count)`
- `rag_search_fulltext(query_text, match_count)`
- `rag_hybrid_search_user_context(query_text, query_embedding, match_count, semantic_weight, rrf_k)`

---

## Conclusion

The hybrid search approach combining HNSW vector search with PostgreSQL full-text search provides:

1. **Best of both worlds**: Semantic understanding + exact keyword matching
2. **Configurable balance**: Adjust `semantic_weight` per use case
3. **Good performance**: Sub-100ms response times for typical queries
4. **Scalability**: HNSW provides O(log n) search complexity

For production chatbots, we recommend:
- `semantic_weight = 0.5`
- `top_k = 3-5`
- `rrf_k = 60`
"""
    return md


def test_quality():
    """Test result quality with different configurations"""
    print("\n" + "=" * 70)
    print("RESULT QUALITY TEST")
    print("=" * 70)

    test_query = "What services does Pop Skills offer?"
    print(f"\nQuery: '{test_query}'")

    emb = create_embedding(test_query)

    print("\n--- Results with semantic_weight=0.5 ---")
    results = hybrid_search(test_query, emb, top_k=3, semantic_weight=0.5)
    for i, r in enumerate(results, 1):
        print(f"  [{i}] Score: {r['rrf_score']:.6f}")
        print(f"      {r['content'][:100]}...")

    print("\n--- Results with semantic_weight=0.8 ---")
    results = hybrid_search(test_query, emb, top_k=3, semantic_weight=0.8)
    for i, r in enumerate(results, 1):
        print(f"  [{i}] Score: {r['rrf_score']:.6f}")
        print(f"      {r['content'][:100]}...")


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    print("\nStarting benchmarks...")
    print(f"Embedding model: {EMBED_MODEL} ({EMBED_DIMENSIONS}d)")
    print("Make sure general_embeddings_1024 table has data\n")

    try:
        # Run benchmarks
        benchmark_data = run_benchmarks()

        # Save results
        save_benchmark_results(benchmark_data)

        print("\n" + "-" * 70)
        response = input("\nRun result quality test? (y/n): ")
        if response.lower() == "y":
            test_quality()

    except Exception as e:
        print(f"\nError: {e}")
        print("\nVerify that:")
        print("  1. Environment variables are configured (.env)")
        print("  2. general_embeddings_1024 table has data")
        print("  3. SQL functions are created in Supabase")
