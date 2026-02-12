"""
Aspirational Category Constants and Prompt Templates

Contains category definitions and prompt templates for generating
aspirational training data.
"""

from typing import Dict, List

# ==============================================================================
# ASPIRATION CATEGORIES
# ==============================================================================

ASPIRATION_CATEGORIES = {
    "dream_roles": {
        "name": "Dream Roles",
        "description": "Career positions, titles, professional roles. Focus on the JOB/PROFESSION itself, not the lifestyle it enables.",
        "examples": [
            "Would love to run my own bakery and become known for my artisan sourdough.",
            "My goal is to become a licensed therapist within three years",
            "Hoping to move from line cook to executive chef—maybe open my own place and get a Michelin star",
            "I want to become a principal investigator and lead my own lab in cancer research",
            "Dream role: school principal leading a transformative educational program.",
            "Eventually want to be a regional manager—have the district reporting to me",
            "Aiming for nurse practitioner or clinical nurse specialist.",
            "Would love to run my own organic farm and promote sustainable agriculture practices in the community"
        ],
    },
    "salary_expectations": {
        "name": "Salary Expectations",
        "description": "Compensation, financial goals, monetary aspirations. Focus ONLY on money/compensation—do NOT mention specific job titles or roles.",
        "examples": [
            "Six figures within five years.",
            "Hoping to double my current salary in my next position",
            "Would love to earn enough to support my family without a second job",
            "My goal is to reach $150k total comp—base plus bonus—within the next three years",
            "Need to earn more—kids are starting college soon.",
            "Want to be in the top 10% of earners in my field",
            "Financial independence by fifty.",
            "At least match what the union guys make, with benefits"
        ],
    },
    "life_goals": {
        "name": "Life Goals Beyond Career",
        "description": "Personal life, family, travel, hobbies, and lifestyle aspirations. Focus on HOW YOU WANT TO LIVE, not the career/job itself.",
        "examples": [
            "More time for my kids.",
            "Want to travel while I'm still young—maybe remote work or seasonal gigs",
            "Goal: work four days a week so I can care for my parents",
            "Would love to start a small farm, grow organic produce, and sell at local markets—simple living, close to nature",
            "I'd love to achieve financial independence so I can volunteer more and pursue my hobbies without stress",
            "Flexibility to pick up my grandkids from school.",
            "Want a job that lets me train for marathons—not killing myself with overtime",
            "My dream is to have a little cabin in the mountains where I can write and enjoy the quiet"
        ],
    },
    "values": {
        "name": "Values",
        "description": "What matters most to you in your career and life: work-life balance, financial security, social impact, personal growth, creativity, autonomy, stability, recognition, helping others, etc. What you prioritize and value.",
        "examples": [
            "Work-life balance is my top priority—I need time for my family.",
            "Financial security matters most to me right now—I'm the sole provider.",
            "I value social impact over salary—want to make a real difference in people's lives.",
            "Personal growth is what drives me—always learning, always improving.",
            "I value creativity and the freedom to try new ideas.",
            "Autonomy is huge for me—I need flexibility in how I work.",
            "I prioritize stability and job security over everything else—I have kids to support.",
            "Recognition and advancement opportunities are important to my career satisfaction.",
            "Helping others is what matters most—that's why I went into nursing.",
            "I value a supportive work environment where I feel respected and valued."
        ],
    },
    "impact_legacy": {
        "name": "Impact & Legacy",
        "description": "Making a difference, helping others, leaving a mark on the world",
        "examples": [
            "Want to help people directly.",
            "My goal is to mentor the next generation—pay it forward",
            "Would love to build something that actually helps communities",
            "I hope to leave the trade better than I found it—train apprentices, raise standards",
            "Make a real impact in education—not just push papers.",
            "Want to work in nonprofit—something that matters.",
            "Leave a mark in healthcare—improve outcomes for underserved populations.",
            "Build a business that treats employees well and gives back"
        ],
    },
    "skill_expertise": {
        "name": "Skill & Expertise",
        "description": "Mastering skills, becoming an expert, developing capabilities",
        "examples": [
            "Master the craft.",
            "Want to become expert in project management—PMP, then lead bigger initiatives",
            "Hoping to deepen my clinical skills—maybe specialize in wound care or palliative",
            "Learn the business side—financials, operations—so I can run my own shop",
            "Get really good at public speaking—keynotes, trainings.",
            "Become the go-to person for HVAC design in commercial builds.",
            "Develop my teaching skills—curriculum design, adult learners.",
            "Master data analysis—SQL, visualization—to make better decisions"
        ],
    },
}


# ==============================================================================
# PROMPT TEMPLATES
# ==============================================================================

MESSAGE_GENERATION_SYSTEM_PROMPT = "You are an expert at generating natural career aspiration messages. Always respond with valid JSON."

# Aliases for backward compatibility with generate_aspirational_data.py
PATTERN_GENERATION_SYSTEM_PROMPT = MESSAGE_GENERATION_SYSTEM_PROMPT
ASPIRATION_GENERATION_SYSTEM_PROMPT = MESSAGE_GENERATION_SYSTEM_PROMPT

# Category-specific guidance to prevent confusion between subcategories
CATEGORY_GUIDANCE = {
    "dream_roles": """
CRITICAL: DREAM ROLES focus on PROFESSIONAL/CAREER POSITIONS, titles, business roles.
Focus: What you want to BE or DO professionally (the job/career itself)

✓ CORRECT: Career positions with PROFESSIONAL focus
  - "I want to become a nurse practitioner" (professional title)
  - "Goal: school principal" (career position)
  - "Would love to run my own organic farm and promote sustainable agriculture practices" (business/professional mission)
  - "My dream is to open a restaurant and become known for my farm-to-table cuisine" (professional achievement)

✗ WRONG: Lifestyle focus (that's life_goals!)
  - "Would love to start a small farm, grow organic produce, and sell at local markets" ← NO! Focus is on simple lifestyle, not professional role
  - "I value creativity" ← NO! That's values subcategory
  - "More time with family" ← NO! That's life_goals

Key distinction: Is the focus on the PROFESSIONAL ROLE/BUSINESS or the LIFESTYLE it enables?""",

    "salary_expectations": """
CRITICAL: SALARY EXPECTATIONS are PURELY about MONEY/COMPENSATION—NO job titles or roles!
Focus: Dollar amounts, compensation levels, financial goals (NOTHING about what job/title)

✓ CORRECT: Pure financial goals (no job titles mentioned)
  - "Six figures within five years"
  - "Want to earn $150k total comp"
  - "Need to double my current salary"
  - "Financial independence by fifty"
  - "Hoping to reach the $200k range within three years"
  - "Want to earn enough to support my family comfortably"

✗ WRONG: Mentions ANY job title/role (that's dream_roles, even if salary is mentioned!)
  - "Creative Director with a salary that reflects my expertise" ← NO! Contains job title "Creative Director"
  - "Want to become a manager earning six figures" ← NO! Contains "manager" role
  - "Goal: VP-level position with $200k+ comp" ← NO! Contains "VP" title
  - "I value financial security" ← NO! That's values subcategory

Rule: If you mention a SPECIFIC JOB TITLE or ROLE, it's dream_roles (even if salary is mentioned). Salary_expectations = PURE money talk.""",

    "life_goals": """
CRITICAL: LIFE GOALS focus on LIFESTYLE, PERSONAL LIFE, and how you want to LIVE.
Focus: Quality of life, personal fulfillment, work-life balance (NOT the career/job itself)

✓ CORRECT: Lifestyle aspirations (focus on HOW YOU LIVE, not what job you have)
  - "More time for my kids"
  - "Want to travel while I'm young"
  - "Would love to start a small farm, grow organic produce, and sell at local markets" (focus: simple living, not professional farming)
  - "My dream is to own a bed and breakfast in the countryside, enjoying a slower pace of life" (focus: lifestyle quality)
  - "Work four days a week so I can care for my parents"
  - "Goal: work remotely so I can live near the beach"
  - "Have more freedom and flexibility in my daily life"

✗ WRONG: Professional/career focus (that's dream_roles!)
  - "Would love to run my own organic farm and promote sustainable agriculture practices" ← NO! Focus is on professional mission, that's dream_roles
  - "Want to become a therapist" ← NO! That's a job title (dream_roles)
  - "Open a restaurant and become known for my cuisine" ← NO! Professional achievement (dream_roles)
  - "Dream role: owning a bed and breakfast" ← NO! Career framing (dream_roles)

Key distinction: Is the focus on HOW YOU WANT TO LIVE (lifestyle) or the PROFESSIONAL ROLE (career)? Same activity can be either depending on emphasis!""",

    "values": """
CRITICAL: VALUES are ABSTRACT PRINCIPLES/PRIORITIES, not specific roles or goals.
✓ CORRECT: What you prioritize/value (abstract)
  - "Work-life balance is my top priority"
  - "I value financial security above everything"
  - "Social impact matters more than salary to me"
  - "Personal growth is what drives me"
  - "I value creativity and autonomy in my work"

✗ WRONG: Specific jobs, roles, or career goals (that's dream_roles!)
  - "Dream role: opening my own restaurant" ← NO! That's dream_roles!
  - "I want to become a manager" ← NO! That's dream_roles!
  - "Goal is to earn six figures" ← NO! That's salary_expectations!

Rule: Values are WHAT MATTERS TO YOU (principles), not WHAT YOU WANT TO DO (roles/goals).""",

    "impact_legacy": """
CRITICAL: IMPACT & LEGACY are about MAKING A DIFFERENCE, helping others, leaving a mark.
✓ CORRECT: Impact on others/world
  - "Want to help people directly"
  - "My goal is to mentor the next generation"
  - "Leave a mark in healthcare—improve outcomes"
  - "Build something that helps communities"

✗ WRONG: Personal values or specific roles
  - "I value helping others" ← NO! That's values subcategory
  - "Want to become a social worker" ← NO! That's dream_roles""",

    "skill_expertise": """
CRITICAL: SKILL & EXPERTISE are about MASTERING ABILITIES, becoming expert.
✓ CORRECT: Skill development goals
  - "Master the craft"
  - "Become expert in project management"
  - "Deepen my clinical skills"
  - "Get really good at public speaking"

✗ WRONG: Job roles or values
  - "Want to become a master electrician" ← NO! That's dream_roles (it's a title)
  - "I value personal growth" ← NO! That's values subcategory""",
}

MESSAGE_GENERATION_PROMPT_TEMPLATE = """Generate {batch_size} diverse, natural aspirational messages for career coaching specifically about {category_name}.

Category: {category_name}
Description: {category_description}
{category_guidance}

Each message should be a complete, natural sentence expressing a career aspiration. Vary widely in length and formulation. Be conversational as if spoken by a real person.

Requirements:
1. Be specifically relevant to {category_name_lower}
2. Vary WIDELY in length: mix very short (5-10 words), short (11-15 words), medium (16-25 words), and long (26-40 words)
3. Cover DIVERSE industries: healthcare, education, trades, creative, business, service, law, government, nonprofit, etc. - NOT just tech
4. Include REALISTIC TYPOS in some messages: "i wnat to", "managment", "become a manger", "teh", "carrer", "oppurtunity", "resturant"
5. Vary FORMULATION: not all "I want to...". Use "My goal is...", "Would love to...", "Hoping to...", "Dream role:", "Eventually...", fragments, etc.

Example messages for {category_name} (note varied length and formulation):
{category_examples}

More examples—vary length and formulation (not all "I..."):
- Very short: "Want my own bakery." / "Six figures within five years." / "More time for my kids." / "Work-life balance matters."
- Short, different starts: "My goal is to become a licensed therapist." / "Hoping to run my own shop someday." / "Dream role: school principal."
- Medium: "I'm working towards transitioning from retail management into HR so I can help employees develop." / "i wnat to become a manger" (typos ok) / "Would love to move from line cook to executive chef—maybe open my own place."
- Long: "I hope to eventually move from teaching into educational administration, maybe start as assistant principal and work my way up to leading my own school." / "My goal is to reach $150k total comp within three years—base plus bonus—so we can buy a house and stop renting."
- Alternative formulations: "Eventually want to be regional manager." / "Aiming for nurse practitioner." / "Financial independence by fifty." / "Want to mentor the next generation—pay it forward."

Generate {batch_size} unique, complete messages as a JSON array. Return ONLY valid JSON with no additional text:
{{"messages": ["message1", "message2", ...]}}"""


# ==============================================================================
# PROMPT BUILDER FUNCTIONS
# ==============================================================================


def build_pattern_generation_prompt(category_key: str, batch_size: int) -> str:
    """Build a prompt for generating aspirational messages (delegates to build_message_generation_prompt)."""
    return build_message_generation_prompt(category_key, batch_size)


def build_aspiration_generation_prompt(category_key: str, num_aspirations: int) -> str:
    """Build a prompt for generating aspirational messages (delegates to build_message_generation_prompt)."""
    return build_message_generation_prompt(category_key, num_aspirations)


def build_message_generation_prompt(
    category_key: str,
    batch_size: int
) -> str:
    """
    Build a prompt for generating complete aspirational messages (no patterns/objects).

    Args:
        category_key: Key from ASPIRATION_CATEGORIES
        batch_size: Number of messages to generate in this batch

    Returns:
        Formatted prompt string
    """
    if category_key not in ASPIRATION_CATEGORIES:
        raise ValueError(f"Unknown category: {category_key}")

    category = ASPIRATION_CATEGORIES[category_key]
    examples_str = "\n".join(f"- {ex}" for ex in category["examples"])

    # Get category-specific guidance if available
    guidance = CATEGORY_GUIDANCE.get(category_key, "")
    category_guidance = f"\n{guidance}" if guidance else ""

    return MESSAGE_GENERATION_PROMPT_TEMPLATE.format(
        batch_size=batch_size,
        category_name=category["name"],
        category_description=category["description"],
        category_name_lower=category["name"].lower(),
        category_examples=examples_str,
        category_guidance=category_guidance
    )


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def get_category_info(category_key: str) -> Dict:
    """
    Get category information by key.

    Args:
        category_key: Key from ASPIRATION_CATEGORIES

    Returns:
        Category dictionary with name, description, examples

    Raises:
        ValueError: If category key is invalid
    """
    if category_key not in ASPIRATION_CATEGORIES:
        raise ValueError(f"Unknown category: {category_key}. Valid categories: {list(ASPIRATION_CATEGORIES.keys())}")

    return ASPIRATION_CATEGORIES[category_key]


def get_all_category_keys() -> List[str]:
    """
    Get list of all category keys.

    Returns:
        List of category keys
    """
    return list(ASPIRATION_CATEGORIES.keys())


def get_category_name(category_key: str) -> str:
    """
    Get category display name.

    Args:
        category_key: Key from ASPIRATION_CATEGORIES

    Returns:
        Category display name
    """
    return get_category_info(category_key)["name"]
