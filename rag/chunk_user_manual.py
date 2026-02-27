"""
CHUNK USER MANUAL FOR RAG EMBEDDINGS
=====================================
Reads all markdown files from trajectoire/docs/user_docs/
and produces a single chunked file in the format expected by
parse_chunks_file() in src/utils/rag.py.

Each markdown section (split by --- or ## headers) becomes one chunk.
For large files like runners.md, each ### runner becomes its own chunk.

Output: info/user_manual_chunks.md

Run: python rag/chunk_user_manual.py
"""

import re
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
USER_DOCS_DIR = Path(r"D:\VIT\Work\Harmonia\trajectoire\docs\user_docs")
OUTPUT_FILE = PROJECT_ROOT / "info" / "user_manual_chunks.md"

# Max chunk size (characters). If a section exceeds this, split further.
MAX_CHUNK_CHARS = 2000


def clean_text(text: str) -> str:
    """Remove excessive blank lines and trim."""
    text = text.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def extract_title_from_content(content: str) -> str:
    """Extract the first heading or first line as title."""
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("#"):
            return re.sub(r"^#+\s*", "", line).strip()
        if line and not line.startswith("**") and not line.startswith("---"):
            return line[:80]
    return "Untitled"


def split_by_sections(content: str) -> list[str]:
    """Split markdown content by --- separators, keeping headers with their content."""
    # Split by horizontal rules
    sections = re.split(r"\n---\n", content)
    return [s.strip() for s in sections if s.strip()]


def split_runners_file(content: str) -> list[tuple[str, str, str]]:
    """
    Special handling for runners.md: split by ### headers.
    Returns list of (category, runner_name, content).
    """
    chunks = []
    current_category = "Runners"
    current_runner = None
    current_lines = []

    for line in content.split("\n"):
        if line.startswith("## ") and not line.startswith("## CHUNK"):
            # Category header (e.g., ## CV Analysis)
            if current_runner and current_lines:
                chunks.append((current_category, current_runner, "\n".join(current_lines).strip()))
                current_lines = []
            current_category = line.lstrip("#").strip()
            current_runner = None
        elif line.startswith("### "):
            # Runner header
            if current_runner and current_lines:
                chunks.append((current_category, current_runner, "\n".join(current_lines).strip()))
                current_lines = []
            current_runner = line.lstrip("#").strip()
            current_lines = [line]
        elif line.strip() == "---":
            continue  # Skip separators
        else:
            if current_runner:
                current_lines.append(line)

    # Last runner
    if current_runner and current_lines:
        chunks.append((current_category, current_runner, "\n".join(current_lines).strip()))

    return chunks


def process_regular_file(filepath: Path, section_name: str, subsection_name: str) -> list[dict]:
    """Process a regular markdown file into chunks."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    sections = split_by_sections(content)
    chunks = []

    for section_content in sections:
        cleaned = clean_text(section_content)
        if not cleaned or len(cleaned) < 20:
            continue

        title = extract_title_from_content(cleaned)

        # If chunk is too large, split by ## headers
        if len(cleaned) > MAX_CHUNK_CHARS:
            sub_parts = re.split(r"\n(?=## )", cleaned)
            for part in sub_parts:
                part = clean_text(part)
                if part and len(part) >= 20:
                    sub_title = extract_title_from_content(part)
                    chunks.append({
                        "section": section_name,
                        "subsection": f"{subsection_name} - {sub_title}",
                        "title": sub_title,
                        "content": part,
                    })
        else:
            chunks.append({
                "section": section_name,
                "subsection": subsection_name,
                "title": title,
                "content": cleaned,
            })

    return chunks


def process_runners_file(filepath: Path) -> list[dict]:
    """Process runners.md with special handling."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    runner_chunks = split_runners_file(content)
    chunks = []

    for category, runner_name, runner_content in runner_chunks:
        cleaned = clean_text(runner_content)
        if not cleaned or len(cleaned) < 20:
            continue

        chunks.append({
            "section": f"Runners - {category}",
            "subsection": runner_name,
            "title": runner_name,
            "content": cleaned,
        })

    return chunks


def format_output(all_chunks: list[dict]) -> str:
    """Format chunks into the expected markdown format for parse_chunks_file()."""
    lines = [
        "# POPskills User Manual - Chunked Document for Embeddings",
        "",
        "**Source:** trajectoire/docs/user_docs/",
        "**Version:** 1.0",
        "**Purpose:** Semantic chunks of the user manual optimized for RAG retrieval",
        "",
        "---",
        "",
    ]

    for i, chunk in enumerate(all_chunks, 1):
        lines.append(f"## CHUNK {i}: {chunk['title']}")
        lines.append("")
        lines.append(f"**Section:** {chunk['section']}")
        if chunk.get("subsection"):
            lines.append(f"**Subsection:** {chunk['subsection']}")
        lines.append(f"**Chunk ID:** {i}")
        lines.append("")
        lines.append(chunk["content"])
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def main():
    if not USER_DOCS_DIR.exists():
        print(f"ERROR: User docs directory not found: {USER_DOCS_DIR}")
        return

    all_chunks = []

    # Section name mapping from folder names
    section_names = {
        "getting-started": "Getting Started",
        "activities": "Activities",
        "programs": "Programs",
        "credits": "Credits",
        "users": "Users & Organizations",
        "results": "Results & Validation",
        "analytics": "Analytics",
        "runners": "Runners",
        "admin": "Administration",
    }

    # Process index.md first
    index_file = USER_DOCS_DIR / "index.md"
    if index_file.exists():
        chunks = process_regular_file(index_file, "Overview", "Platform Overview")
        all_chunks.extend(chunks)
        print(f"  index.md -> {len(chunks)} chunks")

    # Process each subdirectory in order
    ordered_dirs = [
        "getting-started", "activities", "programs", "credits",
        "users", "results", "analytics", "runners", "admin",
    ]

    for dir_name in ordered_dirs:
        dir_path = USER_DOCS_DIR / dir_name
        if not dir_path.exists():
            continue

        section = section_names.get(dir_name, dir_name.title())

        # Get all .md files sorted alphabetically
        md_files = sorted(dir_path.glob("*.md"))

        for md_file in md_files:
            subsection = md_file.stem.replace("-", " ").title()

            # Special handling for runners.md (large file with many ### sections)
            if dir_name == "runners" and md_file.name == "runners.md":
                chunks = process_runners_file(md_file)
                all_chunks.extend(chunks)
                print(f"  runners/{md_file.name} -> {len(chunks)} chunks (runners split)")
            else:
                chunks = process_regular_file(md_file, section, subsection)
                all_chunks.extend(chunks)
                print(f"  {dir_name}/{md_file.name} -> {len(chunks)} chunks")

    # Write output
    output = format_output(all_chunks)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"\n{'=' * 50}")
    print(f"DONE: {len(all_chunks)} total chunks")
    print(f"Output: {OUTPUT_FILE}")
    print(f"{'=' * 50}")

    # Show stats
    sizes = [len(c["content"]) for c in all_chunks]
    print("\nChunk stats:")
    print(f"  Min size: {min(sizes)} chars")
    print(f"  Max size: {max(sizes)} chars")
    print(f"  Avg size: {sum(sizes) // len(sizes)} chars")


if __name__ == "__main__":
    main()
