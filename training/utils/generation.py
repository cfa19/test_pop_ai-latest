import json
from typing import List, Tuple
import time

from training.constants import *

# ==============================================================================
# SHARED GENERATION LOGIC
# ==============================================================================

def _batch_generate_with_openai(
    client,
    category_key: str,
    total_count: int,
    batch_size: int,
    temperature: float,
    prompt_builder_fn,
    system_prompt: str,
    result_key: str,
    item_name: str,
    validator_fn=None
) -> List[str]:
    """
    Generic function to batch generate items using OpenAI API.

    Args:
        client: OpenAI client
        category_key: Key from ASPIRATION_CATEGORIES
        total_count: Total number of items to generate
        batch_size: Number of items per API call
        temperature: Creativity level (0.0-1.0)
        prompt_builder_fn: Function(category_key, batch_size) -> prompt string
        system_prompt: System prompt for the API call
        result_key: Key to extract from JSON response (e.g., "patterns", "aspirations")
        item_name: Display name for progress messages (e.g., "patterns", "aspirations")
        validator_fn: Optional function(item) -> bool to validate each item

    Returns:
        List of generated items
    """
    try:
        all_items = []
        num_batches = max(1, int(total_count / batch_size))

        for i in range(num_batches):
            # Build prompt using the provided builder function
            prompt = prompt_builder_fn(category_key, batch_size)

            # Call OpenAI API
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                response_format={"type": "json_object"}
            )

            # Parse JSON response
            result = json.loads(response.choices[0].message.content)
            items = result.get(result_key, [])

            # Apply validation if provided
            if validator_fn:
                items = [item for item in items if validator_fn(item)]

            all_items.extend(items)
            print(f"  ✓ Generated {(i + 1) * batch_size}/{total_count} valid {item_name}")

        # Warn if generation was significantly below target
        if len(all_items) < total_count * 0.8:
            print(f"⚠ Warning: Only {len(all_items)} valid {item_name} (expected {total_count})")

        return all_items

    except Exception as e:
        print(f"✗ Error generating {item_name}: {e}")
        return []

# ==============================================================================
# MESSAGE GENERATION
# ==============================================================================

def generate_messages_for_category(
    client,
    category_key: str,
    category_type: str,
    num_messages: int = 50,
    temperature: float = 0.8,
    batch_size: int = 20
) -> List[str]:
    """
    Generate messages for a specific category (aspirational, professional, psychological, learning, social, emotional, rag_query, chitchat, or off_topic).

    Args:
        client: OpenAI client
        category_key: Category key (e.g., "dream_roles", "experiences", "personality_profile", "knowledge", "mentors", "confidence", "company_overview", "greetings", or "random_topics")
        category_type: "aspirational", "professional", "psychological", "learning", "social", "emotional", "rag_query", "chitchat", or "off_topic"
        num_messages: Number of messages to generate
        temperature: Creativity level (0.0-1.0)
        batch_size: Number of messages per API call

    Returns:
        List of message strings
    """
    if category_type == "aspirational":
        categories_dict = ASPIRATION_CATEGORIES
        prompt_builder = build_aspirational_prompt
        system_prompt = ASPIRATIONAL_SYSTEM_PROMPT
    elif category_type == "professional":
        categories_dict = PROFESSIONAL_CATEGORIES
        prompt_builder = build_professional_prompt
        system_prompt = PROFESSIONAL_SYSTEM_PROMPT
    elif category_type == "psychological":
        categories_dict = PSYCHOLOGICAL_CATEGORIES
        prompt_builder = build_psychological_prompt
        system_prompt = PSYCHOLOGICAL_SYSTEM_PROMPT
    elif category_type == "learning":
        categories_dict = LEARNING_CATEGORIES
        prompt_builder = build_learning_prompt
        system_prompt = LEARNING_SYSTEM_PROMPT
    elif category_type == "social":
        categories_dict = SOCIAL_CATEGORIES
        prompt_builder = build_social_prompt
        system_prompt = SOCIAL_SYSTEM_PROMPT
    elif category_type == "emotional":
        categories_dict = EMOTIONAL_CATEGORIES
        prompt_builder = build_emotional_prompt
        system_prompt = EMOTIONAL_SYSTEM_PROMPT
    elif category_type == "rag_query":
        categories_dict = RAG_QUERY_CATEGORIES
        prompt_builder = build_rag_query_prompt
        system_prompt = RAG_QUERY_SYSTEM_PROMPT
    elif category_type == "chitchat":
        categories_dict = CHITCHAT_CATEGORIES
        prompt_builder = build_chitchat_prompt
        system_prompt = CHITCHAT_SYSTEM_PROMPT
    elif category_type == "off_topic":
        categories_dict = OFF_TOPIC_CATEGORIES
        prompt_builder = build_off_topic_prompt
        system_prompt = OFF_TOPIC_SYSTEM_PROMPT
    else:
        raise ValueError(f"Invalid category_type: {category_type}")

    category = categories_dict[category_key]
    print(f"\nGenerating {num_messages} {category_type} messages for {category['name']}...")

    return _batch_generate_with_openai(
        client=client,
        category_key=category_key,
        total_count=num_messages,
        batch_size=batch_size,
        temperature=temperature,
        prompt_builder_fn=prompt_builder,
        system_prompt=system_prompt,
        result_key="messages",
        item_name="messages",
        validator_fn=None
    )


def generate_messages_by_type(
    client,
    category_type: str,
    messages_per_category: int = 50,
    temperature: float = 0.8,
    categories: List[str] = None,
    batch_size: int = 20
) -> List[Tuple[str, str, str]]:
    """
    Generate messages for all (or selected) categories of a given type.

    Args:
        client: OpenAI client
        category_type: "aspirational", "professional", "psychological", "learning", "social", "emotional", "rag_query", "chitchat", or "off_topic"
        messages_per_category: Messages per subcategory
        temperature: Creativity level
        categories: Specific subcategories to generate (None = all)
        batch_size: Batch size for API calls

    Returns:
        List of (message, category_type, subcategory) tuples
    """
    if category_type == "aspirational":
        categories_dict = ASPIRATION_CATEGORIES
    elif category_type == "professional":
        categories_dict = PROFESSIONAL_CATEGORIES
    elif category_type == "psychological":
        categories_dict = PSYCHOLOGICAL_CATEGORIES
    elif category_type == "learning":
        categories_dict = LEARNING_CATEGORIES
    elif category_type == "social":
        categories_dict = SOCIAL_CATEGORIES
    elif category_type == "emotional":
        categories_dict = EMOTIONAL_CATEGORIES
    elif category_type == "rag_query":
        categories_dict = RAG_QUERY_CATEGORIES
    elif category_type == "chitchat":
        categories_dict = CHITCHAT_CATEGORIES
    elif category_type == "off_topic":
        categories_dict = OFF_TOPIC_CATEGORIES
    else:
        raise ValueError(f"Invalid category_type: {category_type}")

    if categories is None:
        categories = list(categories_dict.keys())

    all_messages = []

    for i, category_key in enumerate(categories, 1):
        category_name = categories_dict[category_key]["name"]
        print(f"\n{'='*80}")
        print(f"[{category_type.upper()}] Category {i}/{len(categories)}: {category_name}")
        print(f"{'='*80}")

        messages = generate_messages_for_category(
            client,
            category_key,
            category_type,
            num_messages=messages_per_category,
            temperature=temperature,
            batch_size=batch_size
        )

        for msg in messages:
            all_messages.append((msg, category_type, category_key))

        print(f"  ✓ Generated {len(messages)} messages")

        if i < len(categories):
            time.sleep(2)

    print(f"\n{'='*80}")
    print(f"Total {category_type} messages: {len(all_messages):,}")
    print(f"{'='*80}")

    return all_messages
