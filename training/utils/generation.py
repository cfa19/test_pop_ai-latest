import random
from typing import Dict, List

from training.constants.categories import INTENT_CATEGORIES
from training.constants.prompts import (
    SYSTEM_MESSAGE,
    format_chitchat_message,
    format_context_message,
    format_offtopic_message,
    format_rag_query_generic,
    format_rag_query_knowledge_base,
)
from training.utils.ai import call_openai_and_extract_messages


def generate_rag_queries(openai_client, knowledge_base: List[str], count: int = 50, batch_size: int = 25) -> List[Dict]:
    """
    Generate RAG query messages based on knowledge base content.
    Randomly selects a document for each iteration to ensure diverse queries.

    Args:
        openai_client: OpenAI client
        knowledge_base: List of knowledge base documents
        count: Number of queries to generate

    Returns:
        List of dicts with 'message' and 'category'
    """
    print(f"\nGenerating {count} RAG query messages...")

    all_results = []
    queries_per_batch = 5  # Generate 5 queries per API call

    if not knowledge_base:
        # Generate generic queries without context
        try:
            for i in range(0, int(count / batch_size)):
                print(f"  Generic batch {i + 1}/{int(count / batch_size)}")

                # Format prompt using imported template
                prompt = format_rag_query_generic(batch_size)

                # Use common API call function
                questions = call_openai_and_extract_messages(
                    openai_client=openai_client, prompt=prompt, system_message=SYSTEM_MESSAGE, temperature=0.9, max_tokens=2000
                )

                # Format as training data
                all_results.extend(questions)
                print(f"  Generated {(i + 1) * batch_size}/{count} RAG queries")

        except Exception as e:
            print(f"Error generating generic RAG queries: {e}")
            return []

    else:
        # Generate queries based on random documents from knowledge base
        for i in range(0, int(count / batch_size)):
            try:
                # Randomly select a document for this batch
                random_doc = random.choice(knowledge_base)

                # Truncate document if too long (keep first 1000 chars)
                if len(random_doc) > 1000:
                    random_doc = random_doc[:1000] + "..."

                # Format prompt using imported template
                prompt = format_rag_query_knowledge_base(batch_size, random_doc)

                # Use common API call function
                questions = call_openai_and_extract_messages(
                    openai_client=openai_client, prompt=prompt, system_message=SYSTEM_MESSAGE, temperature=0.9, max_tokens=1000
                )

                # Format as training data
                all_results.extend(questions)
                print(f"  Generated {(i + 1) * batch_size}/{count} RAG queries")

            except Exception as e:
                print(f"  Warning: Error in batch {i + 1}: {e}")
                continue

    return all_results[:count]


def generate_messages(openai_client, category: str, count: int = 50, batch_size: int = 25) -> List[Dict]:
    """
    Generate messages for a specific Store A context, chitchat, or off-topic.

    Args:
        openai_client: OpenAI client
        category: One of the 6 Store A contexts, chitchat, or off_topic
        count: Number of messages to generate
        batch_size: Number of messages per API call

    Returns:
        List of dicts with 'message' and 'category'

    Note:
        off_topic messages are for testing the semantic gate only,
        NOT for training the BERT classifier.
    """
    print(f"\nGenerating {count} {category} messages...")

    cat_info = INTENT_CATEGORIES[category]

    all_results = []

    try:
        for i in range(0, int(count / batch_size)):
            # Format prompt using imported template
            if category in ["professional", "psychological", "learning", "social", "emotional", "aspirational"]:
                prompt = format_context_message(
                    batch_size=batch_size, category=category, description=cat_info["description"], examples=cat_info["examples"]
                )
            elif category == "chitchat":
                prompt = format_chitchat_message(batch_size)
            elif category == "off_topic":
                prompt = format_offtopic_message(batch_size)
            else:
                raise ValueError(f"Invalid category: {category}")

            # Use common API call function
            messages = call_openai_and_extract_messages(
                openai_client=openai_client, prompt=prompt, system_message=SYSTEM_MESSAGE, temperature=0.9, max_tokens=2000
            )

            # Format as training data
            all_results.extend(messages)
            print(f"  Generated {(i + 1) * batch_size}/{count} {category} messages")

        return all_results

    except Exception as e:
        print(f"Error generating {category} messages: {e}")
        return []
