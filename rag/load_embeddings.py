"""
LOAD DOCUMENTS WITH EMBEDDINGS TO SUPABASE
===========================================
This script:
1. Reads pre-chunked .md files
2. Parses chunks with metadata
3. Generates embeddings with Voyage AI (1024 dimensions)
4. Processes in batch with retries
5. Uploads to Supabase

Run: python load_embeddings.py
"""

import os
import sys

# Add project root to path for src imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Load .env with explicit path (before config import, for standalone execution)
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from src.config import (  # noqa: E402
    EMBED_DIMENSIONS,
    EMBED_MODEL,
    SUPABASE_KEY,
    SUPABASE_URL,
    Tables,
    get_client_by_provider,
    get_supabase,
)
from src.utils.rag import create_embeddings_batch, parse_chunks_file  # noqa: E402

# Batch configuration
BATCH_SIZE = 20  # Number of texts per batch
MAX_RETRIES = 5  # Maximum number of retries
INITIAL_BACKOFF = 1.0  # Initial wait time in seconds
MAX_BACKOFF = 60.0  # Maximum wait time

# Verify configuration
if not SUPABASE_URL or "YOUR_PROJECT" in str(SUPABASE_URL):
    print("ERROR: Configure SUPABASE_URL in the .env file")
    exit(1)

if not SUPABASE_KEY or "YOUR_SERVICE_KEY" in str(SUPABASE_KEY):
    print("ERROR: Configure SUPABASE_SERVICE_KEY in the .env file")
    exit(1)

print("Connecting to Voyage AI and Supabase...")
embed_client = get_client_by_provider("voyage")
supabase = get_supabase()
print("Connection successful")


# ===============================
# MAIN LOADING FUNCTION
# ===============================
def load_document(file_path: str, batch_size: int = BATCH_SIZE) -> bool:
    """
    Load a pre-chunked document to Supabase with batch embeddings.

    Args:
        file_path: Path to the pre-chunked .md file
        batch_size: Batch size for embeddings

    Returns:
        True if successful
    """
    # 1. Parse chunks from file
    print(f"\n1. Parsing chunks from file: {file_path}")
    parsed_chunks = parse_chunks_file(file_path)

    if not parsed_chunks:
        print("   ERROR: No chunks found in file")
        return False

    print(f"   Chunks parsed: {len(parsed_chunks)}")

    # 2. Generate embeddings in batch
    print(f"\n2. Generating embeddings in batches of {batch_size}...")
    print(f"   Model: {EMBED_MODEL}")
    print(f"   Dimensions: {EMBED_DIMENSIONS}")

    all_embeddings = []
    total_batches = (len(parsed_chunks) + batch_size - 1) // batch_size

    for i in range(0, len(parsed_chunks), batch_size):
        batch_num = i // batch_size + 1
        batch_chunks = parsed_chunks[i : i + batch_size]

        print(f"   Processing batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)...", end=" ")

        try:
            batch_embeddings = create_embeddings_batch(
                batch_chunks, embed_client, EMBED_MODEL, EMBED_DIMENSIONS, MAX_RETRIES, INITIAL_BACKOFF, MAX_BACKOFF
            )
            all_embeddings.extend(batch_embeddings)
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")
            return False

    print(f"   Total embeddings generated: {len(all_embeddings)}")

    # 3. Upload to Supabase in batch
    print("\n3. Uploading to Supabase...")

    records = []
    for i, (parsed_chunk, embedding) in enumerate(zip(parsed_chunks, all_embeddings)):
        # Build metadata from parsed chunk
        chunk_metadata = parsed_chunk.metadata or {}
        metadata = {
            "source": os.path.basename(file_path),
            "chunk_index": i,
            "chunk_id": chunk_metadata.get("chunk_id", i + 1),
            "total_chunks": len(parsed_chunks),
            "dimensions": EMBED_DIMENSIONS,
            "model": EMBED_MODEL,
        }

        # Add section and subsection if available
        if chunk_metadata.get("section"):
            metadata["section"] = chunk_metadata["section"]
        if chunk_metadata.get("subsection"):
            metadata["subsection"] = chunk_metadata["subsection"]
        if chunk_metadata.get("title"):
            metadata["title"] = chunk_metadata["title"]

        records.append({"content": parsed_chunk.content, "embedding": embedding.embedding, "metadata": metadata})

    # Insert in batches of 100 (Supabase limit)
    supabase_batch_size = 100
    total_supabase_batches = (len(records) + supabase_batch_size - 1) // supabase_batch_size

    for i in range(0, len(records), supabase_batch_size):
        batch_num = i // supabase_batch_size + 1
        batch_records = records[i : i + supabase_batch_size]

        print(f"   Inserting batch {batch_num}/{total_supabase_batches}...", end=" ")

        try:
            supabase.table(Tables.GENERAL_EMBEDDINGS_1024).insert(batch_records).execute()
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")
            return False

    print(f"\n4. COMPLETED: {len(parsed_chunks)} chunks uploaded to Supabase")
    print(f"   - Model: {EMBED_MODEL}")
    print(f"   - Dimensions: {EMBED_DIMENSIONS}")
    print(f"   - File: {os.path.basename(file_path)}")

    return True


def verify_data():
    """Verify how many documents are in the table"""
    result = supabase.table(Tables.GENERAL_EMBEDDINGS_1024).select("id", count="exact").execute()
    count = result.count or 0
    print(f"\nDocuments in Supabase: {count}")

    if count > 0:
        # Show preview
        preview = supabase.table(Tables.GENERAL_EMBEDDINGS_1024).select("id, content, metadata").limit(3).execute()
        print("\nPreview of first documents:")
        for doc in preview.data:
            meta = doc.get("metadata", {})
            dims = meta.get("dimensions", "N/A")
            print(f"  ID {doc['id']}: [{dims} dims] {doc['content'][:60]}...")


def clear_table():
    """Delete all documents (use with caution)"""
    result = supabase.table(Tables.GENERAL_EMBEDDINGS_1024).select("id", count="exact").execute()
    count = result.count or 0

    if count == 0:
        print("\nTable is already empty.")
        return

    response = input(f"\nThis will delete {count} documents. Continue? (y/n): ")
    if response.lower() == "y":
        supabase.table(Tables.GENERAL_EMBEDDINGS_1024).delete().neq("id", 0).execute()
        print("Table cleared")


def show_model_info():
    """Display information about the embedding model"""
    print("\n" + "=" * 50)
    print("EMBEDDING MODEL INFORMATION")
    print("=" * 50)
    print(f"""
Model: {EMBED_MODEL}
Dimensions: {EMBED_DIMENSIONS}

Advantages of voyage-3-large:
  - High semantic precision
  - Better capture of language nuances
  - Ideal for technical and business documents
  - Cost-effective embedding generation

Batch configuration:
  - Batch size: {BATCH_SIZE} texts per request
  - Max retries: {MAX_RETRIES}
  - Initial backoff: {INITIAL_BACKOFF}s
  - Max backoff: {MAX_BACKOFF}s
""")


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    print("=" * 50)
    print("EMBEDDINGS LOADER TO SUPABASE")
    print(f"Model: {EMBED_MODEL} | Dimensions: {EMBED_DIMENSIONS}")
    print("=" * 50)

    print("\nOptions:")
    print("1. Load document")
    print("2. Verify existing data")
    print("3. Clear table (delete all)")
    print("4. Model info")
    print("5. Exit")

    option = input("\nChoose an option (1-5): ").strip()

    if option == "1":
        data_path = os.path.join(os.path.dirname(__file__), "..", "info", "general_info_chunks.md")
        load_document(data_path)
        verify_data()
    elif option == "2":
        verify_data()
    elif option == "3":
        clear_table()
    elif option == "4":
        show_model_info()
    elif option == "5":
        print("Exiting...")
    else:
        print("Invalid option")
