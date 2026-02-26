"""
Fast Entity Extraction for Career Coaching Messages

Uses regex and pattern matching for low-latency local extraction.
No external dependencies beyond standard library.
"""

import re
from typing import Dict, List, Optional

# =============================================================================
# Entity Patterns
# =============================================================================

# Job titles (common patterns)
JOB_TITLES = [
    # Engineering
    "software engineer", "data engineer", "ml engineer", "machine learning engineer",
    "devops engineer", "site reliability engineer", "sre", "backend engineer",
    "frontend engineer", "full stack engineer", "fullstack engineer",
    # Data & AI
    "data scientist", "data analyst", "machine learning scientist", "ai researcher",
    "research scientist", "applied scientist",
    # Product & Design
    "product manager", "product owner", "ux designer", "ui designer",
    "product designer", "user researcher",
    # Leadership
    "engineering manager", "tech lead", "technical lead", "architect",
    "principal engineer", "staff engineer", "director of engineering",
    "vp of engineering", "cto", "chief technology officer",
    # Other tech roles
    "security engineer", "qa engineer", "test engineer", "technical writer",
    "developer advocate", "solutions architect",
]

# Seniority levels
SENIORITY_LEVELS = {
    "intern": ["intern", "internship", "co-op", "coop"],
    "junior": ["junior", "jr", "entry level", "entry-level", "associate"],
    "mid": ["mid level", "mid-level", "intermediate"],
    "senior": ["senior", "sr"],
    "staff": ["staff"],
    "principal": ["principal"],
    "lead": ["lead", "tech lead", "technical lead"],
    "manager": ["manager", "engineering manager", "em"],
    "director": ["director", "head of"],
    "vp": ["vp", "vice president", "vp of"],
    "c-level": ["cto", "ceo", "coo", "cfo", "chief"],
}

# Industries
INDUSTRIES = [
    "fintech", "finance", "healthcare", "health tech", "edtech", "education",
    "e-commerce", "ecommerce", "retail", "gaming", "entertainment",
    "social media", "saas", "b2b", "b2c", "enterprise", "startup",
    "consulting", "cybersecurity", "security", "ai", "artificial intelligence",
    "blockchain", "crypto", "web3", "climate tech", "clean tech",
]

# Company types
COMPANY_TYPES = {
    "faang": ["faang", "fang", "mamaa"],
    "big_tech": ["google", "microsoft", "apple", "amazon", "meta", "facebook"],
    "startup": ["startup", "early stage", "seed stage"],
    "scale_up": ["scale-up", "scaleup", "series a", "series b", "series c"],
    "enterprise": ["enterprise", "large company", "big company", "corporation"],
}

# Location preferences
LOCATION_PREFERENCES = {
    "remote": ["remote", "work from home", "wfh", "fully remote", "100% remote"],
    "hybrid": ["hybrid", "flexible", "part remote"],
    "onsite": ["onsite", "on-site", "in office", "in-office", "office based"],
}


# =============================================================================
# Extraction Functions
# =============================================================================

def extract_salary(text: str) -> Optional[Dict[str, any]]:
    """
    Extract salary expectations from text.

    Examples:
    - "$150k" → {"amount": 150000, "currency": "USD"}
    - "$150,000 per year" → {"amount": 150000, "currency": "USD", "period": "year"}
    - "120-150k" → {"min": 120000, "max": 150000, "currency": "USD"}
    """
    text_lower = text.lower()

    # Pattern 1: $XXXk or $XXX,XXX
    pattern1 = r'\$\s*(\d{1,3})(?:,(\d{3}))*k?'
    matches = re.findall(pattern1, text)

    if matches:
        amounts = []
        for match in matches:
            if match[1]:  # Has comma (e.g., $150,000)
                amount = int(match[0]) * 1000 + int(match[1])
            else:  # No comma (e.g., $150k)
                amount = int(match[0]) * 1000
            amounts.append(amount)

        if len(amounts) == 1:
            return {"amount": amounts[0], "currency": "USD"}
        elif len(amounts) == 2:
            return {"min": min(amounts), "max": max(amounts), "currency": "USD"}

    # Pattern 2: XXXk or XXX thousand
    pattern2 = r'(\d{2,3})k'
    match = re.search(pattern2, text_lower)
    if match:
        return {"amount": int(match.group(1)) * 1000, "currency": "USD"}

    return None


def extract_timeline(text: str) -> Optional[str]:
    """
    Extract timeline/timeframe from text.

    Examples:
    - "in 2 years" → "2 years"
    - "within 6 months" → "6 months"
    - "by 2026" → "by 2026"
    """
    text_lower = text.lower()

    # Pattern: "in X years/months"
    pattern1 = r'in (\d+)\s*(year|month|week)s?'
    match = re.search(pattern1, text_lower)
    if match:
        return f"{match.group(1)} {match.group(2)}s"

    # Pattern: "within X years/months"
    pattern2 = r'within\s+(\d+)\s*(year|month|week)s?'
    match = re.search(pattern2, text_lower)
    if match:
        return f"{match.group(1)} {match.group(2)}s"

    # Pattern: "by YYYY"
    pattern3 = r'by\s+(20\d{2})'
    match = re.search(pattern3, text)
    if match:
        return f"by {match.group(1)}"

    # Pattern: "next year/month"
    pattern4 = r'next\s+(year|month|quarter)'
    match = re.search(pattern4, text_lower)
    if match:
        return f"next {match.group(1)}"

    return None


def extract_job_title(text: str) -> List[str]:
    """
    Extract job titles from text.

    Returns list of matches (may be multiple if user mentions options).
    """
    text_lower = text.lower()
    matches = []

    # Check for seniority + title combinations
    for seniority_level, keywords in SENIORITY_LEVELS.items():
        for keyword in keywords:
            for title in JOB_TITLES:
                # Pattern: "senior data scientist"
                pattern = rf'\b{keyword}\s+{title}\b'
                if re.search(pattern, text_lower):
                    full_title = f"{keyword} {title}".title()
                    if full_title not in matches:
                        matches.append(full_title)

    # Check for standalone titles
    for title in JOB_TITLES:
        pattern = rf'\b{title}\b'
        if re.search(pattern, text_lower):
            title_formatted = title.title()
            # Only add if not already captured with seniority
            if not any(title_formatted in m for m in matches):
                matches.append(title_formatted)

    return matches


def extract_seniority(text: str) -> Optional[str]:
    """Extract seniority level from text."""
    text_lower = text.lower()

    for level, keywords in SENIORITY_LEVELS.items():
        for keyword in keywords:
            # Match whole words only
            pattern = rf'\b{keyword}\b'
            if re.search(pattern, text_lower):
                return level.title()

    return None


def extract_industry(text: str) -> List[str]:
    """Extract industry mentions from text."""
    text_lower = text.lower()
    matches = []

    for industry in INDUSTRIES:
        pattern = rf'\b{industry}\b'
        if re.search(pattern, text_lower):
            matches.append(industry.title())

    return matches


def extract_company_type(text: str) -> Optional[str]:
    """Extract company type preference from text."""
    text_lower = text.lower()

    for company_type, keywords in COMPANY_TYPES.items():
        for keyword in keywords:
            pattern = rf'\b{keyword}\b'
            if re.search(pattern, text_lower):
                return company_type

    return None


def extract_location_preference(text: str) -> Optional[str]:
    """Extract location/work arrangement preference from text."""
    text_lower = text.lower()

    for pref_type, keywords in LOCATION_PREFERENCES.items():
        for keyword in keywords:
            pattern = rf'\b{keyword}\b'
            if re.search(pattern, text_lower):
                return pref_type

    return None


# =============================================================================
# Main Extraction Function
# =============================================================================

def extract_aspirational_entities(message: str) -> Dict[str, any]:
    """
    Extract all aspirational career entities from a message.

    Args:
        message: User message (aspirational category)

    Returns:
        Dictionary with extracted entities:
        {
            "job_titles": ["Senior Data Scientist", "ML Engineer"],
            "seniority": "Senior",
            "salary": {"min": 150000, "max": 200000, "currency": "USD"},
            "timeline": "2 years",
            "industries": ["Fintech", "Healthcare"],
            "company_type": "startup",
            "location_preference": "remote"
        }
    """
    entities = {}

    # Extract all entity types
    job_titles = extract_job_title(message)
    if job_titles:
        entities["job_titles"] = job_titles

    seniority = extract_seniority(message)
    if seniority:
        entities["seniority"] = seniority

    salary = extract_salary(message)
    if salary:
        entities["salary"] = salary

    timeline = extract_timeline(message)
    if timeline:
        entities["timeline"] = timeline

    industries = extract_industry(message)
    if industries:
        entities["industries"] = industries

    company_type = extract_company_type(message)
    if company_type:
        entities["company_type"] = company_type

    location_preference = extract_location_preference(message)
    if location_preference:
        entities["location_preference"] = location_preference

    return entities


def extract_professional_entities(message: str) -> Dict[str, any]:
    """
    Extract professional context entities (skills, experience).

    Returns:
        {
            "job_titles": [...],
            "seniority": "...",
            "industries": [...]
        }
    """
    entities = {}

    job_titles = extract_job_title(message)
    if job_titles:
        entities["job_titles"] = job_titles

    seniority = extract_seniority(message)
    if seniority:
        entities["seniority"] = seniority

    industries = extract_industry(message)
    if industries:
        entities["industries"] = industries

    return entities


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Test aspirational extraction
    test_messages = [
        "I want to become a senior data scientist making $150k in 2 years",
        "My goal is to work as a Staff ML Engineer at a FAANG company, preferably remote, earning 200-250k",
        "I'm aiming for a product manager role in fintech within the next 6 months",
        "I'd love to be a CTO of a startup by 2026",
        "Looking to transition to engineering manager, hybrid work, at a scale-up"
    ]

    print("=" * 80)
    print("Testing Aspirational Entity Extraction")
    print("=" * 80)

    for msg in test_messages:
        print(f"\nMessage: {msg}")
        entities = extract_aspirational_entities(msg)
        print(f"Entities: {entities}")
