import json
from typing import List


def clean_message(message: str) -> str:
    """
    Clean a message by removing surrounding double quotes and extra whitespace.

    Args:
        message: Raw message string

    Returns:
        Cleaned message string
    """
    if not message:
        return message

    # Strip whitespace
    message = message.strip()

    # Remove surrounding double quotes if present
    if len(message) >= 2 and message[0] == '"' and message[-1] == '"':
        message = message[1:-1]

    # Strip whitespace again after removing quotes
    message = message.strip()

    return message


def extract_json_array(text: str) -> List[str]:
    """
    Extract JSON array from text, handling unescaped quotes and unterminated strings.
    Manually parses the array structure to avoid JSON parsing errors.

    Args:
        text: Text that may contain a JSON array

    Returns:
        List of strings from the JSON array

    Raises:
        ValueError: If no valid strings can be extracted
    """
    # First, try standard JSON parsing
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            # Clean each message to remove surrounding double quotes
            return [clean_message(msg) if isinstance(msg, str) else msg for msg in parsed]
    except (json.JSONDecodeError, ValueError):
        pass

    # Find the array boundaries
    array_start = text.find("[")
    if array_start == -1:
        # No array found, try to extract all quoted strings
        fallback = _extract_quoted_strings(text)
        if not fallback:
            raise ValueError("No array found and no quoted strings extracted")
        return fallback

    # Extract content between brackets (handle nested brackets and unterminated strings)
    # Find the end of the array content (either closing bracket or end of text)
    bracket_count = 1
    array_end = array_start + 1

    while array_end < len(text) and bracket_count > 0:
        if text[array_end] == "[":
            bracket_count += 1
        elif text[array_end] == "]":
            bracket_count -= 1
        array_end += 1

    # Extract array content (even if unterminated)
    array_content = text[array_start + 1 : array_end - 1] if bracket_count == 0 else text[array_start + 1 :]

    messages = []
    i = 0
    while i < len(array_content):
        # Skip whitespace and commas
        while i < len(array_content) and array_content[i] in " \n\t\r,":
            i += 1

        if i >= len(array_content):
            break

        # Check if we hit the closing bracket
        if array_content[i] == "]":
            break

        # Look for start of string
        if array_content[i] == '"':
            # Found a string, extract it manually
            i += 1  # Skip opening quote
            string_content = []
            found_closing_quote = False

            # Parse until we find the closing quote (handling escapes)
            while i < len(array_content):
                if array_content[i] == "\\" and i + 1 < len(array_content):
                    # Escaped character - preserve it
                    string_content.append(array_content[i])
                    string_content.append(array_content[i + 1])
                    i += 2
                elif array_content[i] == '"':
                    # Found closing quote
                    found_closing_quote = True
                    break
                else:
                    string_content.append(array_content[i])
                    i += 1

            # Extract the string (even if unterminated)
            msg = "".join(string_content)
            # Only process if we have content
            if msg.strip():
                # Unescape common sequences (do this carefully to avoid double unescaping)
                # First handle escaped backslashes, then other escapes
                msg = msg.replace("\\\\", "\x00")  # Temporary marker
                msg = msg.replace('\\"', '"')
                msg = msg.replace("\\n", "\n")
                msg = msg.replace("\\t", "\t")
                msg = msg.replace("\x00", "\\")  # Restore escaped backslashes
                # Clean message to remove any surrounding double quotes
                msg = clean_message(msg)
                messages.append(msg)

            # Skip the closing quote if found
            if found_closing_quote:
                i += 1
        else:
            # Skip unknown characters (might be part of malformed JSON)
            i += 1

    if messages:
        return messages

    # Fallback: extract all quoted strings from the text
    fallback = _extract_quoted_strings(text)
    return fallback if fallback else []


def _extract_quoted_strings(text: str) -> List[str]:
    """
    Extract all quoted strings from text as a fallback method.

    Args:
        text: Text containing quoted strings

    Returns:
        List of extracted strings
    """
    messages = []
    i = 0

    while i < len(text):
        # Find opening quote
        if text[i] == '"':
            i += 1
            # string_start = i
            string_content = []

            # Extract until closing quote (handling escapes)
            while i < len(text):
                if text[i] == "\\" and i + 1 < len(text):
                    # Escaped character
                    string_content.append(text[i])
                    string_content.append(text[i + 1])
                    i += 2
                elif text[i] == '"':
                    # Found closing quote
                    break
                else:
                    string_content.append(text[i])
                    i += 1

            # Extract the string (even if unterminated)
            msg = "".join(string_content)
            # Only add non-empty strings that look like messages (not too short, not just punctuation)
            if len(msg.strip()) > 2:
                # Unescape common sequences
                msg = msg.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\")
                # Clean message to remove any surrounding double quotes
                msg = clean_message(msg)
                messages.append(msg)

            # Skip the closing quote if found
            if i < len(text) and text[i] == '"':
                i += 1
        else:
            i += 1

    return messages
