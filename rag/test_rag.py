"""
HYBRID SEARCH IN SUPABASE
=========================
This script:
1. Receives a question
2. Generates the question embedding
3. Calls the Supabase hybrid function
4. Displays results with scores

Run: python rag/test_rag.py
"""

import os
import sys

# Add project root to path for src imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Load .env with explicit path (before config import, for standalone execution)
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from src.config import (  # noqa: E402
    CHAT_MODEL,
    EMBED_DIMENSIONS,
    EMBED_MODEL,
    get_embed_client,
    get_openai,
)
from src.utils.rag import fulltext_search, hybrid_search, semantic_search  # noqa: E402

# Create clients (lazy, from centralized config)
embed_client = get_embed_client()
openai_client = get_openai()


# ===============================
# LLM QUESTION FUNCTION
# ===============================
def ask_question(question: str, semantic_weight: float = 0.5):
    """Search context and answer with LLM"""

    # 1. Search relevant chunks
    results = hybrid_search(question, embed_client, EMBED_MODEL, EMBED_DIMENSIONS, top_k=3, semantic_weight=semantic_weight)

    if not results:
        return "No relevant information found in the database.", []

    # 2. Build context
    context = "\n\n".join([r["content"] for r in results])

    # 3. Ask LLM
    prompt = f"""
Use the following information to answer the user's question.
If the information is related to the question, use it to provide a helpful answer.

Context:
{context}

Question:
{question}
"""

    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are an assistant that answers questions about the company Pop Skills."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )

    return response.choices[0].message.content, results


# ===============================
# DEBUG FUNCTIONS
# ===============================
def compare_searches(query: str):
    """Compare the 3 search types to see differences"""

    print(f"\n{'=' * 60}")
    print(f"COMPARING SEARCHES FOR: '{query}'")
    print(f"{'=' * 60}")

    # Semantic
    print("\n--- SEMANTIC SEARCH (embeddings) ---")
    sem_results = semantic_search(query, embed_client, EMBED_MODEL, EMBED_DIMENSIONS, 3)
    for i, r in enumerate(sem_results, 1):
        print(f"  [{i}] Similarity: {r.get('similarity', 'N/A'):.4f}")
        print(f"      {r['content'][:80]}...")

    # Full-text
    print("\n--- FULL-TEXT SEARCH (keywords) ---")
    ft_results = fulltext_search(query, 3)
    if ft_results:
        for i, r in enumerate(ft_results, 1):
            print(f"  [{i}] Rank: {r.get('rank', 'N/A'):.4f}")
            print(f"      {r['content'][:80]}...")
    else:
        print("  No results found with exact keywords")

    # Hybrid
    print("\n--- HYBRID SEARCH (RRF 50/50) ---")
    hyb_results = hybrid_search(query, embed_client, EMBED_MODEL, EMBED_DIMENSIONS, top_k=3, semantic_weight=0.5)
    for i, r in enumerate(hyb_results, 1):
        print(f"  [{i}] RRF Score: {r['rrf_score']:.6f}")
        print(f"      Semantic Rank: {r['semantic_rank']} | Full-text Rank: {r['fulltext_rank']}")
        print(f"      {r['content'][:80]}...")


# ===============================
# MAIN - INTERACTIVE CLI
# ===============================
if __name__ == "__main__":
    print("=" * 50)
    print("HYBRID SEARCH IN SUPABASE")
    print(f"Embedding: {EMBED_MODEL} ({EMBED_DIMENSIONS}d)")
    print("=" * 50)

    print("\nOptions:")
    print("1. Chat with RAG (hybrid search)")
    print("2. Compare search types (debug)")
    print("3. Adjust semantic weight")
    print("4. Exit")

    option = input("\nChoose an option (1-4): ").strip()

    if option == "1":
        print("\n--- CHAT MODE ---")
        print("Type 'exit' to quit.\n")

        while True:
            q = input("Question: ").strip()
            if q.lower() in ("exit", "quit"):
                break
            if not q:
                continue

            answer, results = ask_question(q)

            print("\n--- Retrieved chunks ---")
            for i, r in enumerate(results, 1):
                print(f"  [{i}] Score: {r['rrf_score']:.4f} | {r['content'][:60]}...")

            print("\n--- Answer ---")
            print(answer)
            print("\n" + "-" * 60 + "\n")

    elif option == "2":
        print("\n--- DEBUG MODE ---")
        print("Type 'exit' to quit.\n")

        while True:
            q = input("Query to compare: ").strip()
            if q.lower() in ("exit", "quit"):
                break
            if not q:
                continue

            compare_searches(q)

    elif option == "3":
        print("\n--- ADJUST SEMANTIC WEIGHT ---")
        print("Weight controls which search type dominates:")
        print("  1.0 = 100% semantic (embeddings only)")
        print("  0.5 = 50/50 balanced")
        print("  0.0 = 100% full-text (keywords only)")

        weight = input("\nEnter weight (0.0 to 1.0): ").strip()
        try:
            weight = float(weight)
            if 0 <= weight <= 1:
                print(f"\nTesting with semantic_weight = {weight}")
                q = input("Query: ").strip()
                results = hybrid_search(q, embed_client, EMBED_MODEL, EMBED_DIMENSIONS, top_k=5, semantic_weight=weight)
                for i, r in enumerate(results, 1):
                    print(f"  [{i}] RRF: {r['rrf_score']:.6f} | Sem: {r['semantic_rank']} | FT: {r['fulltext_rank']}")
                    print(f"      {r['content'][:80]}...")
            else:
                print("Weight must be between 0.0 and 1.0")
        except ValueError:
            print("Invalid value")

    elif option == "4":
        print("Exiting...")
