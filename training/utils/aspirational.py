import json
from typing import List

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
    validator_fn=None,
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
        num_batches = int(total_count / batch_size)

        for i in range(num_batches):
            # Build prompt using the provided builder function
            prompt = prompt_builder_fn(category_key, batch_size)

            # Call OpenAI API
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                temperature=temperature,
                response_format={"type": "json_object"},
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
