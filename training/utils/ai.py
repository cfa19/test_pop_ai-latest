import json
from typing import List

from training.utils.processing import _extract_quoted_strings, clean_message, extract_json_array


def call_openai_and_extract_messages(
    openai_client,
    prompt: str,
    system_message: str = "You are a helpful assistant that generates realistic training data. Always respond with valid JSON. Escape all quotes inside strings with backslash. Your response must be a JSON array of strings.",
    temperature: float = 0.9,
    max_tokens: int = 2000,
) -> List[str]:
    """
    Call OpenAI API and extract messages from JSON response.

    This is a common function used by all generate_* functions to:
    1. Make the API call
    2. Parse JSON response with robust error handling
    3. Return list of message strings

    Args:
        openai_client: OpenAI client
        prompt: User prompt for generation
        system_message: System message for the API
        temperature: Sampling temperature (0-2)
        max_tokens: Maximum tokens in response

    Returns:
        List of extracted message strings

    Raises:
        Exception: If API call fails or no messages can be extracted
    """
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": system_message}, {"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Parse JSON response with robust error handling
    content = response.choices[0].message.content.strip()
    messages = None

    try:
        # Try direct JSON parsing first
        messages = json.loads(content)
        # Clean each message to remove surrounding double quotes
        if isinstance(messages, list):
            messages = [clean_message(msg) if isinstance(msg, str) else msg for msg in messages]
    except (json.JSONDecodeError, ValueError):
        # JSON parsing failed, try to extract manually
        try:
            messages = extract_json_array(content)
            if messages:
                print(f"    Extracted {len(messages)} messages using fallback parser")
        except Exception:
            # Last resort: try to find any quoted strings
            messages = _extract_quoted_strings(content)
            if messages:
                print(f"    Extracted {len(messages)} messages using string extraction")

    if not messages or not isinstance(messages, list):
        raise ValueError(f"Could not extract messages from response. Preview: {content[:200]}...")

    return messages
