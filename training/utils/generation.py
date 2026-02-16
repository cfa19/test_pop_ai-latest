"""
Training data generation utilities.

Supports two generation modes:
1. Single-label: one message → one context/entity/sub_entity
2. Multi-label: one message → multiple contexts/entities/sub_entities (compound messages)

New hierarchical taxonomy: context > entity > sub_entity
"""

import json
import itertools
import random
import time
from typing import Dict, List, Tuple, Optional

from training.constants import CONTEXT_REGISTRY, NON_CONTEXT_REGISTRY

# ==============================================================================
# SHARED GENERATION LOGIC
# ==============================================================================

def _batch_generate_with_openai(
    client,
    total_count: int,
    batch_size: int,
    temperature: float,
    prompt: str,
    system_prompt: str,
    result_key: str,
    item_name: str,
    validator_fn=None,
    model: str = "gpt-4o-mini"
) -> List:
    """
    Generic batch generation using OpenAI API.

    Returns:
        List of generated items (strings or dicts depending on result_key structure)
    """
    try:
        all_items = []
        num_batches = max(1, int(total_count / batch_size))

        for i in range(num_batches):
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            items = result.get(result_key, [])

            if validator_fn:
                items = [item for item in items if validator_fn(item)]

            all_items.extend(items)
            print(f"  > Generated {(i + 1) * batch_size}/{total_count} {item_name}")

        if len(all_items) < total_count * 0.8:
            print(f"  ! Warning: Only {len(all_items)} valid {item_name} (expected {total_count})")

        return all_items

    except Exception as e:
        print(f"  x Error generating {item_name}: {e}")
        return []


# ==============================================================================
# SINGLE-LABEL GENERATION (context > entity > sub_entity)
# ==============================================================================

def generate_single_label_messages(
    client,
    context: str,
    entity_key: str,
    sub_entity_key: str = None,
    num_messages: int = 25,
    temperature: float = 0.8,
    batch_size: int = 25,
    model: str = "gpt-4o-mini"
) -> List[Tuple[str, str, str, str]]:
    """
    Generate single-label messages for a specific context/entity/sub_entity.

    Returns:
        List of (message, context, entity, sub_entities) tuples
        sub_entities is a single value string for single-label
    """
    if context not in CONTEXT_REGISTRY:
        raise ValueError(f"Invalid context: {context}. Valid: {list(CONTEXT_REGISTRY.keys())}")

    reg = CONTEXT_REGISTRY[context]
    entities = reg["entities"]

    if entity_key not in entities:
        raise ValueError(f"Invalid entity: {entity_key}. Valid: {list(entities.keys())}")

    entity = entities[entity_key]
    build_prompt = reg["build_prompt"]
    system_prompt = reg["system_prompt"]

    # Build prompt for specific sub_entity or entity-level
    prompt = build_prompt(entity_key, batch_size, sub_entity_key)

    label = sub_entity_key or entity_key
    entity_name = entity["name"]
    sub_name = sub_entity_key or entity_key

    print(f"\n  Generating {num_messages} single-label: {context} > {entity_key} > {sub_name}")

    messages = _batch_generate_with_openai(
        client=client,
        total_count=num_messages,
        batch_size=batch_size,
        temperature=temperature,
        prompt=prompt,
        system_prompt=system_prompt,
        result_key="messages",
        item_name="messages",
        model=model,
    )

    return [(msg, context, entity_key, label) for msg in messages]


# ==============================================================================
# MULTI-LABEL GENERATION (compound messages)
# ==============================================================================

def generate_multilabel_messages(
    client,
    context: str,
    entity_keys: List[str] = None,
    num_messages: int = 25,
    num_labels: int = 3,
    temperature: float = 0.9,
    batch_size: int = 15,
    model: str = "gpt-4o-mini"
) -> List[Tuple[str, str, str, str]]:
    """
    Generate multi-label compound messages for a context.

    Returns:
        List of (message, context, entities_pipe, sub_entities_pipe) tuples
        entities_pipe and sub_entities_pipe use | as separator
    """
    if context not in CONTEXT_REGISTRY:
        raise ValueError(f"Invalid context: {context}")

    reg = CONTEXT_REGISTRY[context]
    entities = reg["entities"]
    build_multilabel = reg["build_multilabel_prompt"]

    if entity_keys is None:
        entity_keys = list(entities.keys())

    prompt = build_multilabel(entity_keys, batch_size, num_labels)
    system_prompt = reg["system_prompt"]

    print(f"\n  Generating {num_messages} multi-label ({num_labels}+ labels): {context}")

    items = _batch_generate_with_openai(
        client=client,
        total_count=num_messages,
        batch_size=batch_size,
        temperature=temperature,
        prompt=prompt,
        system_prompt=system_prompt,
        result_key="messages",
        item_name="multi-label messages",
        model=model,
    )

    results = []
    for item in items:
        if isinstance(item, dict):
            text = item.get("text", "")
            sub_entities = item.get("sub_entities", [])
            # Determine which entities these sub_entities belong to
            detected_entities = set()
            for se in sub_entities:
                for ek, ev in entities.items():
                    if se in ev["sub_entities"]:
                        detected_entities.add(ek)
                        break
            entities_str = "|".join(sorted(detected_entities)) if detected_entities else entity_keys[0]
            sub_entities_str = "|".join(sub_entities) if sub_entities else ""
            results.append((text, context, entities_str, sub_entities_str))
        elif isinstance(item, str):
            # Fallback: if GPT returned plain strings instead of objects
            results.append((item, context, "|".join(entity_keys[:2]), ""))

    return results


# ==============================================================================
# CROSS-CONTEXT MULTI-LABEL GENERATION
# ==============================================================================

CROSS_CONTEXT_PROMPT_TEMPLATE = """Generate {batch_size} natural messages for career coaching that span MULTIPLE life contexts in a single message.

Each message should naturally touch on topics from {num_contexts} or more of these contexts: {context_list}

Available sub-entities per context:
{context_details}

These are messages where someone naturally talks about multiple aspects of their life at once. For example:
- "I want to be a VP (professional) but I can't relocate because of my kids (personal) and the imposter syndrome (psychological) is holding me back"
- "I'm learning Python (learning) and building a side project (personal) while networking with CTOs (social)"

Requirements:
1. Each message MUST span at least {num_contexts} different contexts
2. Length: 40-70 words (rich paragraphs)
3. Natural and conversational
4. Cover DIVERSE professions and life situations
5. Include REALISTIC TYPOS in ~15%

Return valid JSON:
{{"messages": [
  {{"text": "the message", "contexts": ["ctx1", "ctx2"], "sub_entities": ["sub1", "sub2", "sub3"]}},
  ...
]}}"""


def generate_cross_context_messages(
    client,
    contexts: List[str] = None,
    num_messages: int = 25,
    num_contexts: int = 2,
    temperature: float = 0.9,
    batch_size: int = 10,
    model: str = "gpt-4o-mini"
) -> List[Tuple[str, str, str, str]]:
    """
    Generate messages that span multiple contexts (e.g., professional + personal).

    Returns:
        List of (message, contexts_pipe, entities_pipe, sub_entities_pipe) tuples
    """
    if contexts is None:
        contexts = list(CONTEXT_REGISTRY.keys())

    # Build context details for the prompt
    context_details = []
    for ctx in contexts:
        entities = CONTEXT_REGISTRY[ctx]["entities"]
        subs = []
        for ek, ev in entities.items():
            for sk in ev["sub_entities"]:
                subs.append(sk)
        context_details.append(f"  {ctx}: {', '.join(subs[:10])}...")

    prompt = CROSS_CONTEXT_PROMPT_TEMPLATE.format(
        batch_size=batch_size,
        num_contexts=num_contexts,
        context_list=", ".join(contexts),
        context_details="\n".join(context_details),
    )

    system_prompt = "You are an expert at generating natural career coaching messages that span multiple life contexts. Always respond with valid JSON."

    print(f"\n  Generating {num_messages} cross-context ({num_contexts}+ contexts)")

    items = _batch_generate_with_openai(
        client=client,
        total_count=num_messages,
        batch_size=batch_size,
        temperature=temperature,
        prompt=prompt,
        system_prompt=system_prompt,
        result_key="messages",
        item_name="cross-context messages",
        model=model,
    )

    results = []
    for item in items:
        if isinstance(item, dict):
            text = item.get("text", "")
            ctxs = item.get("contexts", contexts[:2])
            sub_entities = item.get("sub_entities", [])
            # Detect entities from sub_entities
            detected_entities = set()
            for se in sub_entities:
                for ctx in ctxs:
                    if ctx in CONTEXT_REGISTRY:
                        for ek, ev in CONTEXT_REGISTRY[ctx]["entities"].items():
                            if se in ev["sub_entities"]:
                                detected_entities.add(ek)
            contexts_str = "|".join(ctxs)
            entities_str = "|".join(sorted(detected_entities)) if detected_entities else ""
            sub_entities_str = "|".join(sub_entities) if sub_entities else ""
            results.append((text, contexts_str, entities_str, sub_entities_str))
        elif isinstance(item, str):
            results.append((item, "|".join(contexts[:2]), "", ""))

    return results


# ==============================================================================
# NON-CONTEXT GENERATION (rag_query, chitchat, off_topic)
# ==============================================================================

def generate_non_context_messages(
    client,
    category_type: str,
    category_key: str,
    num_messages: int = 25,
    temperature: float = 0.8,
    batch_size: int = 25,
    model: str = "gpt-4o-mini"
) -> List[Tuple[str, str, str]]:
    """
    Generate messages for non-context types (rag_query, chitchat, off_topic).
    These keep the old flat structure: (message, category_type, subcategory).
    """
    if category_type not in NON_CONTEXT_REGISTRY:
        raise ValueError(f"Invalid type: {category_type}. Valid: {list(NON_CONTEXT_REGISTRY.keys())}")

    reg = NON_CONTEXT_REGISTRY[category_type]
    categories_dict = reg["categories"]
    build_prompt = reg["build_prompt"]
    system_prompt = reg["system_prompt"]

    if category_key not in categories_dict:
        raise ValueError(f"Invalid category: {category_key}")

    category = categories_dict[category_key]
    prompt = build_prompt(category_key, batch_size)

    print(f"\n  Generating {num_messages} {category_type}: {category['name']}")

    messages = _batch_generate_with_openai(
        client=client,
        total_count=num_messages,
        batch_size=batch_size,
        temperature=temperature,
        prompt=prompt,
        system_prompt=system_prompt,
        result_key="messages",
        item_name="messages",
        model=model,
    )

    return [(msg, category_type, category_key) for msg in messages]


# ==============================================================================
# FULL CONTEXT GENERATION (all entities + multi-label for one context)
# ==============================================================================

def generate_full_context(
    client,
    context: str,
    messages_per_sub_entity: int = 25,
    multilabel_messages: int = 50,
    cross_context_messages: int = 25,
    temperature: float = 0.8,
    batch_size: int = 25,
    model: str = "gpt-4o-mini"
) -> Dict[str, List]:
    """
    Generate complete training data for one context:
    - Single-label messages for each entity/sub_entity
    - Multi-label compound messages within the context
    - Cross-context messages (optional)

    Returns dict with keys: single_label, multi_label, cross_context
    """
    if context not in CONTEXT_REGISTRY:
        raise ValueError(f"Invalid context: {context}")

    entities = CONTEXT_REGISTRY[context]["entities"]
    results = {"single_label": [], "multi_label": [], "cross_context": []}

    print(f"\n{'='*80}")
    print(f"GENERATING: {context.upper()}")
    print(f"{'='*80}")

    # 1. Single-label: for each entity, generate for each sub_entity
    for entity_key, entity_info in entities.items():
        for sub_entity_key in entity_info["sub_entities"]:
            msgs = generate_single_label_messages(
                client, context, entity_key, sub_entity_key,
                num_messages=messages_per_sub_entity,
                temperature=temperature,
                batch_size=batch_size,
                model=model,
            )
            results["single_label"].extend(msgs)
            time.sleep(1)  # Rate limiting

    # 2. Multi-label within context (2-7+ labels)
    #    Distribution: 25% 2-label, 25% 3-label, 20% 4-label, 15% 5-label, 10% 6-label, 5% 7+
    if multilabel_messages > 0:
        n_entities = len(entities)
        label_distribution = [
            (2, 0.25),
            (3, 0.25),
            (4, 0.20),
            (5, 0.15),
            (6, 0.10),
            (7, 0.05),
        ]
        # Filter out label counts that exceed available entities
        label_distribution = [(n, pct) for n, pct in label_distribution if n <= n_entities]
        # Redistribute percentages proportionally
        total_pct = sum(pct for _, pct in label_distribution)
        label_distribution = [(n, pct / total_pct) for n, pct in label_distribution]

        for num_labels, pct in label_distribution:
            count = max(1, int(multilabel_messages * pct))
            msgs = generate_multilabel_messages(
                client, context,
                num_messages=count,
                num_labels=num_labels,
                temperature=temperature + 0.1,
                batch_size=min(batch_size, 15),
                model=model,
            )
            results["multi_label"].extend(msgs)
            time.sleep(1)

    # 3. Cross-context (this context + others) — weighted by realistic co-occurrence
    if cross_context_messages > 0:
        # Realistic pair weights: which contexts naturally appear together?
        # Higher weight = more likely to be mentioned in the same message
        CROSS_CONTEXT_WEIGHTS = {
            ("professional", "learning"):      5,  # "I know Python and want to be CTO"
            ("professional", "psychological"):  4,  # "I'm a manager but burnout is killing me"
            ("professional", "personal"):       3,  # "I want a promotion but can't relocate (kids)"
            ("professional", "social"):         4,  # "My mentor says I should switch companies"
            ("learning", "psychological"):      3,  # "I want to learn leadership but lack confidence"
            ("learning", "personal"):           2,  # "I'm studying an MBA while raising kids"
            ("learning", "social"):             2,  # "My network recommended this certification"
            ("psychological", "personal"):      4,  # "Work stress is affecting my family life"
            ("psychological", "social"):        2,  # "I feel isolated at work, no mentors"
            ("personal", "social"):             2,  # "My family situation limits my networking"
        }

        # Get weighted pairs for this context
        weighted_pairs = []
        for (c1, c2), weight in CROSS_CONTEXT_WEIGHTS.items():
            if c1 == context:
                weighted_pairs.extend([(c2, weight)])
            elif c2 == context:
                weighted_pairs.extend([(c1, weight)])

        if not weighted_pairs:
            # Fallback: all other contexts with equal weight
            weighted_pairs = [(c, 1) for c in CONTEXT_REGISTRY if c != context]

        # Sample proportionally to weights
        total_weight = sum(w for _, w in weighted_pairs)
        for other, weight in weighted_pairs:
            count = max(1, int(cross_context_messages * weight / total_weight))
            msgs = generate_cross_context_messages(
                client,
                contexts=[context, other],
                num_messages=count,
                num_contexts=2,
                temperature=temperature + 0.1,
                batch_size=min(batch_size, 10),
                model=model,
            )
            results["cross_context"].extend(msgs)
            time.sleep(1)

    total = sum(len(v) for v in results.values())
    print(f"\n  Total for {context}: {total} messages")
    print(f"    Single-label: {len(results['single_label'])}")
    print(f"    Multi-label:  {len(results['multi_label'])}")
    print(f"    Cross-context: {len(results['cross_context'])}")

    return results


# ==============================================================================
# BACKWARD COMPATIBILITY
# ==============================================================================

def generate_messages_for_category(
    client,
    category_key: str,
    category_type: str,
    num_messages: int = 50,
    temperature: float = 0.8,
    batch_size: int = 20
) -> List[str]:
    """Backward-compatible: generate messages for non-context types."""
    if category_type in NON_CONTEXT_REGISTRY:
        tuples = generate_non_context_messages(
            client, category_type, category_key,
            num_messages=num_messages,
            temperature=temperature,
            batch_size=batch_size,
        )
        return [t[0] for t in tuples]
    else:
        raise ValueError(f"Use generate_single_label_messages for context types. Got: {category_type}")


def generate_messages_by_type(
    client,
    category_type: str,
    messages_per_category: int = 50,
    temperature: float = 0.8,
    categories: List[str] = None,
    batch_size: int = 20
) -> List[Tuple[str, str, str]]:
    """Backward-compatible: generate for non-context types (rag_query, chitchat, off_topic)."""
    if category_type not in NON_CONTEXT_REGISTRY:
        raise ValueError(f"Use generate_full_context for context types. Got: {category_type}")

    reg = NON_CONTEXT_REGISTRY[category_type]
    categories_dict = reg["categories"]

    if categories is None:
        categories = list(categories_dict.keys())

    all_messages = []
    for i, category_key in enumerate(categories, 1):
        category_name = categories_dict[category_key]["name"]
        print(f"\n{'='*80}")
        print(f"[{category_type.upper()}] Category {i}/{len(categories)}: {category_name}")
        print(f"{'='*80}")

        tuples = generate_non_context_messages(
            client, category_type, category_key,
            num_messages=messages_per_category,
            temperature=temperature,
            batch_size=batch_size,
        )
        all_messages.extend(tuples)

        if i < len(categories):
            time.sleep(2)

    print(f"\nTotal {category_type} messages: {len(all_messages):,}")
    return all_messages
