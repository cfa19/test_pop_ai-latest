"""
Entity Extraction using spaCy for Career Coaching

Two-stage approach (as per docs/spacy_strategy.md):
1. Extract components: ACTION, FUNCTION, SCOPE, ORG_UNIT (what spaCy can do)
2. Infer roles: Map components → canonical job titles (data-driven, not hardcoded)

Uses aspiration_verb_database.json for flexible verb-to-object matching.

Performance: ~10-50ms per message (en_core_web_sm model)
"""

import os
import json
import spacy
from spacy.matcher import Matcher
from typing import Dict, List, Optional, Tuple
from functools import lru_cache
from pathlib import Path


# =============================================================================
# Model Loading (singleton pattern)
# =============================================================================

_nlp_instance = None
_matcher_instance = None
_verb_database = None


def load_verb_database(database_path: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Load aspiration verb-to-object database.

    Args:
        database_path: Path to verb database JSON (default: training/data/aspiration_verb_database.json)

    Returns:
        Dictionary mapping verbs to aspiration objects
    """
    global _verb_database

    if _verb_database is not None:
        return _verb_database

    if database_path is None:
        # Default path relative to project root
        project_root = Path(__file__).parent.parent.parent
        database_path = project_root / "training" / "data" / "aspiration_verb_database.json"

    if not os.path.exists(database_path):
        print(f"[ENTITY EXTRACTION] Verb database not found at {database_path}")
        print(f"  Run: python training/scripts/generate_aspiration_verb_database.py")
        return {}

    try:
        with open(database_path, 'r', encoding='utf-8') as f:
            _verb_database = json.load(f)
        print(f"[ENTITY EXTRACTION] Loaded {len(_verb_database)} verbs from database")
        return _verb_database
    except Exception as e:
        print(f"[ENTITY EXTRACTION] Error loading verb database: {e}")
        return {}


@lru_cache(maxsize=1)
def get_nlp():
    """Load and cache spaCy model."""
    global _nlp_instance

    if _nlp_instance is not None:
        return _nlp_instance

    print("[ENTITY EXTRACTION] Loading spaCy model (en_core_web_sm)...")
    nlp = spacy.load("en_core_web_sm")
    _nlp_instance = nlp
    return nlp


def _get_matcher(nlp):
    """Create and cache Matcher with career entity patterns."""
    global _matcher_instance

    if _matcher_instance is not None:
        return _matcher_instance

    matcher = Matcher(nlp.vocab)
    patterns = _get_entity_patterns()

    for p in patterns:
        matcher.add(p["label"], [p["pattern"]])

    print(f"[ENTITY EXTRACTION] Loaded {len(patterns)} matcher patterns")
    _matcher_instance = matcher
    return matcher


# =============================================================================
# Component Extraction Patterns (Task A: Extract what's explicitly there)
# =============================================================================

def _get_entity_patterns() -> List[Dict]:
    """
    Define token-based patterns for career COMPONENTS.

    Following spacy_strategy.md:
    - Extract ACTION (lead, manage, build)
    - Extract FUNCTION (marketing, engineering, sales)
    - Extract SCOPE (global, regional, senior, junior)
    - Extract ORG_UNIT (team, department, org)
    - Extract INDUSTRY, COMPANY_TYPE, LOCATION (same as before)
    """
    patterns = []

    # =============================================================================
    # ACTION - Verbs indicating responsibility
    # =============================================================================

    action_verbs = [
        "lead", "manage", "oversee", "direct", "run", "head",
        "build", "create", "develop", "establish", "grow",
        "own", "drive", "execute", "deliver",
        "advise", "consult", "mentor", "coach",
        "architect", "design", "engineer",
    ]

    for verb in action_verbs:
        patterns.append({
            "label": "ACTION",
            "pattern": [{"LOWER": verb}]
        })

    # =============================================================================
    # FUNCTION - Functional domains
    # =============================================================================

    functions = [
        # Engineering
        "engineering", "software", "data", "ml", "machine learning",
        "backend", "frontend", "full stack", "fullstack", "devops",
        "infrastructure", "platform", "security", "qa", "testing",
        # Data & AI
        "data science", "analytics", "ai", "artificial intelligence",
        "research",
        # Product & Design
        "product", "design", "ux", "ui", "user experience",
        # Business
        "marketing", "sales", "business development", "partnerships",
        "operations", "finance", "hr", "recruiting", "people",
        "customer success", "support",
    ]

    for function in functions:
        tokens = function.split()
        pattern = [{"LOWER": token} for token in tokens]
        patterns.append({
            "label": "FUNCTION",
            "pattern": pattern
        })

    # =============================================================================
    # SCOPE - Scale, reach, seniority indicators
    # =============================================================================

    scope_terms = [
        # Geographic/organizational scale
        "global", "international", "regional", "local",
        "enterprise", "large", "small", "startup",
        # Seniority indicators (explicit)
        "senior", "junior", "mid-level", "entry-level",
        "staff", "principal", "lead", "chief",
        # Size indicators
        "large-scale", "small-scale", "cross-functional",
    ]

    for term in scope_terms:
        tokens = term.split("-") if "-" in term else term.split()
        pattern = [{"LOWER": token} for token in tokens]
        patterns.append({
            "label": "SCOPE",
            "pattern": pattern
        })

    # =============================================================================
    # ORG_UNIT - Organizational units
    # =============================================================================

    org_units = [
        "team", "department", "division", "group", "unit",
        "org", "organization", "company", "business",
        "function", "practice", "studio",
    ]

    for unit in org_units:
        patterns.append({
            "label": "ORG_UNIT",
            "pattern": [{"LOWER": unit}]
        })

    # =============================================================================
    # INDUSTRIES (unchanged)
    # =============================================================================

    industries = [
        "fintech", "finance", "healthcare", "edtech", "education",
        "ecommerce", "retail", "gaming", "saas", "consulting",
        "cybersecurity", "blockchain", "crypto", "ai", "startup",
    ]

    for industry in industries:
        patterns.append({
            "label": "INDUSTRY",
            "pattern": [{"LOWER": industry}]
        })

    # =============================================================================
    # COMPANY_TYPE (unchanged)
    # =============================================================================

    company_types = [
        (["faang"], "COMPANY_TYPE_FAANG"),
        (["fang"], "COMPANY_TYPE_FAANG"),
        (["big", "tech"], "COMPANY_TYPE_BIG_TECH"),
        (["startup"], "COMPANY_TYPE_STARTUP"),
        (["early", "stage"], "COMPANY_TYPE_STARTUP"),
        (["scale", "-", "up"], "COMPANY_TYPE_SCALEUP"),
        (["scaleup"], "COMPANY_TYPE_SCALEUP"),
        (["enterprise"], "COMPANY_TYPE_ENTERPRISE"),
    ]

    for terms, label in company_types:
        pattern = [{"LOWER": term} for term in terms]
        patterns.append({"label": label, "pattern": pattern})

    # =============================================================================
    # LOCATION (unchanged)
    # =============================================================================

    location_prefs = [
        (["remote"], "LOCATION_REMOTE"),
        (["fully", "remote"], "LOCATION_REMOTE"),
        (["work", "from", "home"], "LOCATION_REMOTE"),
        (["wfh"], "LOCATION_REMOTE"),
        (["hybrid"], "LOCATION_HYBRID"),
        (["onsite"], "LOCATION_ONSITE"),
        (["on", "-", "site"], "LOCATION_ONSITE"),
        (["in", "office"], "LOCATION_ONSITE"),
    ]

    for terms, label in location_prefs:
        pattern = [{"LOWER": term} for term in terms]
        patterns.append({"label": label, "pattern": pattern})

    return patterns


# =============================================================================
# Role Inference Layer (Task B: Infer canonical roles from components)
# =============================================================================

def infer_role_from_components(
    actions: List[str],
    functions: List[str],
    scopes: List[str],
    org_units: List[str],
    verb_database: Optional[Dict[str, List[str]]] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Infer canonical job title and seniority from extracted components.

    DATA-DRIVEN APPROACH (replaces hardcoded if-statements):
    - Uses verb database to match ACTION + object combinations
    - Flexible and comprehensive (covers all verb-object pairs in database)

    Args:
        actions: ACTION components (e.g., ["lead", "manage"])
        functions: FUNCTION components (e.g., ["marketing"])
        scopes: SCOPE components (e.g., ["global", "senior"])
        org_units: ORG_UNIT components (e.g., ["team"])
        verb_database: Optional preloaded verb database (will load if None)

    Returns:
        (job_title, seniority) tuple, or (None, None) if can't infer
    """
    # Load verb database if not provided
    if verb_database is None:
        verb_database = load_verb_database()

    if not actions and not functions:
        return None, None

    # =============================================================================
    # Step 1: Extract seniority from SCOPE
    # =============================================================================

    seniority = None
    scope_lower = [s.lower() for s in scopes]

    # Seniority hierarchy (explicit mentions in SCOPE)
    seniority_map = {
        "c-level": ["chief", "c-level", "cxo"],
        "director": ["director", "head of"],
        "principal": ["principal"],
        "staff": ["staff"],
        "senior": ["senior", "sr"],
        "lead": ["lead", "tech lead"],
        "mid": ["mid-level", "mid", "intermediate"],
        "junior": ["junior", "entry-level", "entry", "jr"],
    }

    for level, keywords in seniority_map.items():
        if any(keyword in scope_lower for keyword in keywords):
            seniority = level
            break

    # =============================================================================
    # Step 2: Match ACTION + components to verb database
    # =============================================================================

    job_title = None

    if actions and verb_database:
        # Build aspiration object from components
        # Example: functions=["marketing"] + org_units=["team"] → "a marketing team"
        aspiration_object_parts = []

        # Add scopes (non-seniority)
        non_seniority_scopes = [
            s for s in scopes
            if s.lower() not in sum(seniority_map.values(), [])
        ]
        aspiration_object_parts.extend(non_seniority_scopes)

        # Add functions
        aspiration_object_parts.extend(functions)

        # Add org units
        aspiration_object_parts.extend(org_units)

        aspiration_object = " ".join(aspiration_object_parts).strip()

        # Try to match ACTION + aspiration_object to database
        best_match = None
        best_match_score = 0

        for action in actions:
            action_lower = action.lower()

            # Check if this action is in the database
            if action_lower in verb_database:
                valid_objects = verb_database[action_lower]

                # Find best matching object in database
                for valid_object in valid_objects:
                    # Simple substring matching (can be improved with fuzzy matching)
                    score = 0

                    # Check if aspiration_object contains key terms from valid_object
                    valid_terms = set(valid_object.lower().split())
                    aspiration_terms = set(aspiration_object.lower().split())

                    # Calculate overlap
                    overlap = valid_terms & aspiration_terms
                    if overlap:
                        score = len(overlap) / max(len(valid_terms), len(aspiration_terms))

                    if score > best_match_score:
                        best_match_score = score
                        best_match = valid_object

        # If we found a good match in the database, use it
        if best_match and best_match_score > 0.3:  # Threshold for match quality
            # Add seniority prefix if available
            if seniority:
                job_title = f"{seniority} {best_match}".strip()
            else:
                job_title = best_match

        # Infer seniority from ACTION if not explicitly in SCOPE
        if not seniority and actions:
            action_lower = [a.lower() for a in actions]

            # Action-based seniority inference
            if any(a in action_lower for a in ["lead", "head", "direct", "oversee"]):
                seniority = "director"
            elif any(a in action_lower for a in ["manage", "run"]):
                seniority = "manager"
            elif any(a in action_lower for a in ["build", "own", "drive"]):
                seniority = "senior"

    # =============================================================================
    # Step 3: Fallback to simple combination if no database match
    # =============================================================================

    if not job_title and (functions or org_units):
        # Simple fallback: combine components
        parts = []
        if seniority:
            parts.append(seniority)
        if functions:
            parts.append(functions[0])
        if org_units and "manager" in (actions + [seniority or ""]):
            parts.append("manager")

        if parts:
            job_title = " ".join(parts)

    return job_title, seniority


# =============================================================================
# Main Extraction Function
# =============================================================================

def extract_entities(text: str, nlp=None) -> Dict[str, any]:
    """
    Extract career entities from text using two-stage approach:

    Stage 1: Extract components (ACTION, FUNCTION, SCOPE, ORG_UNIT)
    Stage 2: Infer canonical role from components

    Args:
        text: User message
        nlp: Optional spaCy model (will load default if None)

    Returns:
        Dictionary with:
        - components: Raw extracted components
        - inferred_role: Canonical job title (inferred)
        - inferred_seniority: Seniority level (inferred)
        - salary, timeline, organizations: From built-in NER
        - industries, company_type, location_preference: From patterns
    """
    if nlp is None:
        nlp = get_nlp()

    doc = nlp(text)
    matcher = _get_matcher(nlp)

    # Stage 1: Extract components
    components = {
        "actions": [],
        "functions": [],
        "scopes": [],
        "org_units": [],
    }

    entities = {
        "industries": [],
        "company_type": None,
        "location_preference": None,
        "salary": None,
        "timeline": None,
        "organizations": [],
    }

    # Extract custom entities using Matcher
    for match_id, start, end in matcher(doc):
        label = nlp.vocab.strings[match_id]
        span_text = doc[start:end].text

        # Extract components
        if label == "ACTION":
            if span_text not in components["actions"]:
                components["actions"].append(span_text)
        elif label == "FUNCTION":
            if span_text not in components["functions"]:
                components["functions"].append(span_text)
        elif label == "SCOPE":
            if span_text not in components["scopes"]:
                components["scopes"].append(span_text)
        elif label == "ORG_UNIT":
            if span_text not in components["org_units"]:
                components["org_units"].append(span_text)

        # Extract other entities (same as before)
        elif label == "INDUSTRY":
            if span_text not in entities["industries"]:
                entities["industries"].append(span_text)
        elif label.startswith("COMPANY_TYPE_"):
            company_type = label.replace("COMPANY_TYPE_", "").lower()
            entities["company_type"] = company_type
        elif label.startswith("LOCATION_"):
            location = label.replace("LOCATION_", "").lower()
            entities["location_preference"] = location

    # Extract built-in NER entities (MONEY, DATE, ORG)
    for ent in doc.ents:
        if ent.label_ == "MONEY":
            entities["salary"] = ent.text
        elif ent.label_ == "DATE":
            entities["timeline"] = ent.text
        elif ent.label_ == "ORG":
            if ent.text not in entities["organizations"]:
                entities["organizations"].append(ent.text)

    # Stage 2: Infer role from components (using verb database)
    verb_database = load_verb_database()
    inferred_role, inferred_seniority = infer_role_from_components(
        components["actions"],
        components["functions"],
        components["scopes"],
        components["org_units"],
        verb_database=verb_database
    )

    # Build final result
    result = {
        "components": components,  # Raw components
        "inferred_role": inferred_role,  # Canonical job title
        "inferred_seniority": inferred_seniority,  # Seniority level
    }

    # Add other entities
    result.update(entities)

    # Clean up empty lists
    result = {k: v for k, v in result.items() if v}

    # Clean up empty component lists
    if "components" in result:
        result["components"] = {k: v for k, v in result["components"].items() if v}
        if not result["components"]:
            del result["components"]

    return result


def extract_aspirational_entities(message: str) -> Dict[str, any]:
    """Extract aspirational career entities from a message."""
    return extract_entities(message)


def extract_professional_entities(message: str) -> Dict[str, any]:
    """Extract professional context entities from a message."""
    entities = extract_entities(message)
    # For professional context, filter out timeline (focus on current state)
    if "timeline" in entities:
        del entities["timeline"]
    return entities


# =============================================================================
# Example Usage & Benchmarking
# =============================================================================

if __name__ == "__main__":
    import time

    # Test messages
    test_messages = [
        "I want to lead a global marketing team",  # Implicit role
        "I wish to become a Senior Data Scientist making $150k in 2 years",  # Explicit + salary
        "My goal is to manage an engineering team at Google",  # Implicit role + org
        "I'd love to build ML systems at a startup",  # Implicit role
        "Looking to transition to a Staff Engineer role, remote only",  # Explicit role
        "I want to own the product strategy for a SaaS company",  # Implicit role
        "Aiming for a director of engineering position in fintech",  # Explicit role
        "I aspire to advise Fortune 500 companies on AI strategy",  # Implicit role
    ]

    print("=" * 80)
    print("Testing spaCy Entity Extraction (Two-Stage Approach)")
    print("=" * 80)

    # Load model once
    nlp = get_nlp()

    # Warm-up run
    _ = extract_entities(test_messages[0], nlp)

    print("\nExtracting entities...\n")

    total_time = 0
    for msg in test_messages:
        t0 = time.perf_counter()
        entities = extract_entities(msg, nlp)
        elapsed = (time.perf_counter() - t0) * 1000  # Convert to ms
        total_time += elapsed

        print(f"Message: {msg}")
        print(f"Entities: {entities}")
        print(f"Time: {elapsed:.2f}ms\n")

    avg_time = total_time / len(test_messages)
    print("=" * 80)
    print(f"Average extraction time: {avg_time:.2f}ms per message")
    print("=" * 80)
