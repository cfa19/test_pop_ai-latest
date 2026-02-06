"""
ESCO Dataset Loader

Utilities for loading and processing the European Skills, Competences,
Qualifications and Occupations (ESCO) dataset.

ESCO provides standardized terminology for:
- Occupations (job titles and roles)
- Skills and competences
- Qualifications

This module loads ESCO data to use as realistic, standardized aspirational objects.
"""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# ==============================================================================
# FILE PATHS
# ==============================================================================

ESCO_DIR = Path(__file__).parent.parent / "data" / "ESCO dataset - v1.2.1 - classification - en - csv"

ESCO_FILES = {
    "occupations": ESCO_DIR / "occupations_en.csv",
    "skills": ESCO_DIR / "skills_en.csv",
    "skill_groups": ESCO_DIR / "skillGroups_en.csv",
    "occupation_skills": ESCO_DIR / "occupationSkillRelations_en.csv",
}


# ==============================================================================
# DATA LOADING FUNCTIONS
# ==============================================================================


def load_occupations(limit: Optional[int] = None) -> pd.DataFrame:
    """
    Load ESCO occupations data.

    Args:
        limit: Maximum number of occupations to load (None = all)

    Returns:
        DataFrame with columns: preferredLabel, altLabels, description, iscoGroup, code
    """
    if not ESCO_FILES["occupations"].exists():
        raise FileNotFoundError(f"ESCO occupations file not found: {ESCO_FILES['occupations']}")

    df = pd.read_csv(ESCO_FILES["occupations"])

    # Select relevant columns
    df = df[["preferredLabel", "altLabels", "description", "iscoGroup", "code"]]

    # Remove rows with missing preferred labels
    df = df.dropna(subset=["preferredLabel"])

    if limit:
        df = df.head(limit)

    print(f"Loaded {len(df)} occupations from ESCO dataset")
    return df


def load_skills(limit: Optional[int] = None, skill_types: List[str] = None) -> pd.DataFrame:
    """
    Load ESCO skills data.

    Args:
        limit: Maximum number of skills to load (None = all)
        skill_types: Filter by skill type (e.g., ['skill/competence', 'knowledge'])

    Returns:
        DataFrame with columns: preferredLabel, altLabels, description, skillType, reuseLevel
    """
    if not ESCO_FILES["skills"].exists():
        raise FileNotFoundError(f"ESCO skills file not found: {ESCO_FILES['skills']}")

    df = pd.read_csv(ESCO_FILES["skills"])

    # Select relevant columns
    df = df[["preferredLabel", "altLabels", "description", "skillType", "reuseLevel"]]

    # Remove rows with missing preferred labels
    df = df.dropna(subset=["preferredLabel"])

    # Filter by skill type if specified
    if skill_types:
        df = df[df["skillType"].isin(skill_types)]

    if limit:
        df = df.head(limit)

    print(f"Loaded {len(df)} skills from ESCO dataset")
    return df


# ==============================================================================
# EXTRACTION FUNCTIONS
# ==============================================================================


def extract_occupation_titles(df: pd.DataFrame, include_alternatives: bool = True) -> List[str]:
    """
    Extract occupation titles from ESCO dataframe.

    Args:
        df: ESCO occupations dataframe
        include_alternatives: Include alternative labels (synonyms)

    Returns:
        List of occupation title strings
    """
    titles = df["preferredLabel"].tolist()

    if include_alternatives:
        # Parse alternative labels (comma or newline separated)
        for alt_labels in df["altLabels"].dropna():
            if isinstance(alt_labels, str):
                # Split by comma or newline
                alternatives = [label.strip() for label in alt_labels.replace("\n", ",").split(",") if label.strip()]
                titles.extend(alternatives)

    # Clean and deduplicate
    titles = list(set([title.strip() for title in titles if title and title.strip()]))

    print(f"Extracted {len(titles)} occupation titles (including alternatives)")
    return titles


def extract_skill_descriptions(df: pd.DataFrame, include_alternatives: bool = True) -> List[str]:
    """
    Extract skill descriptions from ESCO dataframe.

    Args:
        df: ESCO skills dataframe
        include_alternatives: Include alternative labels (synonyms)

    Returns:
        List of skill description strings
    """
    skills = df["preferredLabel"].tolist()

    if include_alternatives:
        # Parse alternative labels
        for alt_labels in df["altLabels"].dropna():
            if isinstance(alt_labels, str):
                alternatives = [label.strip() for label in alt_labels.replace("\n", ",").split(",") if label.strip()]
                skills.extend(alternatives)

    # Clean and deduplicate
    skills = list(set([skill.strip() for skill in skills if skill and skill.strip()]))

    print(f"Extracted {len(skills)} skill descriptions (including alternatives)")
    return skills


# ==============================================================================
# FILTERING FUNCTIONS
# ==============================================================================


def filter_occupations_by_category(df: pd.DataFrame, isco_groups: List[str] = None) -> pd.DataFrame:
    """
    Filter occupations by ISCO group (International Standard Classification of Occupations).

    Common ISCO groups:
    - 1: Managers
    - 2: Professionals
    - 3: Technicians and Associate Professionals
    - 4: Clerical Support Workers
    - 5: Service and Sales Workers
    - 6: Skilled Agricultural, Forestry and Fishery Workers
    - 7: Craft and Related Trades Workers
    - 8: Plant and Machine Operators and Assemblers
    - 9: Elementary Occupations

    Args:
        df: ESCO occupations dataframe
        isco_groups: List of ISCO group codes (e.g., ['1', '2', '3'])

    Returns:
        Filtered dataframe
    """
    if not isco_groups:
        return df

    # Filter by first digit of iscoGroup
    mask = df["iscoGroup"].astype(str).str[0].isin(isco_groups)
    filtered = df[mask]

    print(f"Filtered to {len(filtered)} occupations in ISCO groups: {isco_groups}")
    return filtered


def filter_skills_by_reuse_level(df: pd.DataFrame, reuse_levels: List[str] = None) -> pd.DataFrame:
    """
    Filter skills by reuse level.

    Reuse levels:
    - transversal: Skills applicable across all sectors and occupations
    - cross-sectoral: Skills used across multiple sectors
    - sector-specific: Skills specific to one sector
    - occupation-specific: Skills specific to one occupation

    Args:
        df: ESCO skills dataframe
        reuse_levels: List of reuse levels to include

    Returns:
        Filtered dataframe
    """
    if not reuse_levels:
        return df

    filtered = df[df["reuseLevel"].isin(reuse_levels)]

    print(f"Filtered to {len(filtered)} skills with reuse levels: {reuse_levels}")
    return filtered


# ==============================================================================
# ASPIRATION CONVERSION FUNCTIONS
# ==============================================================================


def convert_occupations_to_aspirations(occupations: List[str], verb_forms: List[str] = None) -> List[str]:
    """
    Convert occupation titles to aspiration verb phrases.

    Args:
        occupations: List of occupation titles
        verb_forms: List of verb forms to use (default: ["become a", "work as a"])

    Returns:
        List of aspiration verb phrases

    Example:
        ["data scientist"] -> ["become a data scientist", "work as a data scientist"]
    """
    if verb_forms is None:
        verb_forms = ["become a", "become an", "work as a", "work as an", "be a", "be an"]

    aspirations = []

    for occupation in occupations:
        # Determine article (a/an)
        article = "an" if occupation[0].lower() in "aeiou" else "a"

        # Generate variations
        aspirations.append(f"become {article} {occupation}")
        aspirations.append(f"work as {article} {occupation}")

    print(f"Converted {len(occupations)} occupations to {len(aspirations)} aspirations")
    return aspirations


def convert_skills_to_aspirations(skills: List[str], verb_forms: List[str] = None) -> List[str]:
    """
    Convert skill descriptions to aspiration verb phrases.

    Args:
        skills: List of skill descriptions
        verb_forms: List of verb forms to use

    Returns:
        List of aspiration verb phrases

    Example:
        ["manage teams"] -> ["master managing teams", "develop expertise in managing teams"]
    """
    if verb_forms is None:
        verb_forms = ["master", "develop expertise in", "become proficient in", "learn", "improve my skills in"]

    aspirations = []

    for skill in skills:
        # Convert skill to gerund form if needed (heuristic)
        skill_lower = skill.lower()

        # Check if already a verb phrase
        if any(skill_lower.startswith(v) for v in ["manage", "develop", "create", "lead", "design"]):
            # Convert to gerund: "manage teams" -> "managing teams"
            gerund = skill_lower + "ing" if not skill_lower.endswith("e") else skill_lower[:-1] + "ing"
        else:
            gerund = skill_lower

        # Generate variations
        for verb_form in verb_forms[:3]:  # Limit variations
            aspirations.append(f"{verb_form} {gerund}")

    print(f"Converted {len(skills)} skills to {len(aspirations)} aspirations")
    return aspirations


# ==============================================================================
# MAIN LOADING FUNCTION
# ==============================================================================


def load_esco_aspirations(
    occupation_limit: int = 500,
    skill_limit: int = 500,
    include_alternatives: bool = True,
    skill_reuse_levels: List[str] = None,
    isco_groups: List[str] = None,
) -> Dict[str, List[str]]:
    """
    Load ESCO data and convert to aspirational verb phrases.

    Args:
        occupation_limit: Max occupations to load (None = all)
        skill_limit: Max skills to load (None = all)
        include_alternatives: Include alternative labels
        skill_reuse_levels: Filter skills by reuse level
        isco_groups: Filter occupations by ISCO group

    Returns:
        Dictionary with keys:
        - 'occupations': List of occupation-based aspirations
        - 'skills': List of skill-based aspirations
        - 'all': Combined list
    """
    print("=" * 80)
    print("LOADING ESCO ASPIRATIONS")
    print("=" * 80)

    # Load occupations
    print("\n1. Loading occupations...")
    occupations_df = load_occupations(limit=occupation_limit)

    # Filter if needed
    if isco_groups:
        occupations_df = filter_occupations_by_category(occupations_df, isco_groups)

    # Extract titles
    occupation_titles = extract_occupation_titles(occupations_df, include_alternatives)

    # Convert to aspirations
    occupation_aspirations = convert_occupations_to_aspirations(occupation_titles[:occupation_limit])

    # Load skills
    print("\n2. Loading skills...")
    skills_df = load_skills(limit=skill_limit)

    # Filter if needed
    if skill_reuse_levels:
        skills_df = filter_skills_by_reuse_level(skills_df, skill_reuse_levels)

    # Extract skill descriptions
    skill_descriptions = extract_skill_descriptions(skills_df, include_alternatives)

    # Convert to aspirations
    skill_aspirations = convert_skills_to_aspirations(skill_descriptions[:skill_limit])

    # Combine
    all_aspirations = occupation_aspirations + skill_aspirations

    print("\n" + "=" * 80)
    print("ESCO ASPIRATIONS LOADED")
    print("=" * 80)
    print(f"  Occupation-based: {len(occupation_aspirations)}")
    print(f"  Skill-based: {len(skill_aspirations)}")
    print(f"  Total: {len(all_aspirations)}")

    return {"occupations": occupation_aspirations, "skills": skill_aspirations, "all": all_aspirations}


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def print_esco_statistics():
    """Print statistics about the ESCO dataset."""
    print("=" * 80)
    print("ESCO DATASET STATISTICS")
    print("=" * 80)

    # Occupations
    try:
        occupations_df = load_occupations()
        print(f"\nOccupations: {len(occupations_df)}")
        print(f"ISCO Groups: {occupations_df['iscoGroup'].nunique()}")
        print("\nTop 10 ISCO Groups:")
        print(occupations_df["iscoGroup"].value_counts().head(10))
    except FileNotFoundError as e:
        print(f"\nOccupations: ERROR - {e}")

    # Skills
    try:
        skills_df = load_skills()
        print(f"\nSkills: {len(skills_df)}")
        print("\nSkill Types:")
        print(skills_df["skillType"].value_counts())
        print("\nReuse Levels:")
        print(skills_df["reuseLevel"].value_counts())
    except FileNotFoundError as e:
        print(f"\nSkills: ERROR - {e}")


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    # Print statistics
    print_esco_statistics()

    print("\n\n")

    # Load sample aspirations
    aspirations = load_esco_aspirations(occupation_limit=10, skill_limit=10, skill_reuse_levels=["transversal", "cross-sectoral"])

    # Print samples
    print("\n" + "=" * 80)
    print("SAMPLE ASPIRATIONS")
    print("=" * 80)

    print("\nOccupation-based (sample):")
    for asp in aspirations["occupations"][:5]:
        print(f"  - {asp}")

    print("\nSkill-based (sample):")
    for asp in aspirations["skills"][:5]:
        print(f"  - {asp}")
