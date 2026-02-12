"""
Professional Category Constants and Prompt Templates

Contains category definitions and prompt templates for generating
professional context training data.
"""

from typing import Dict, List


# ==============================================================================
# PROFESSIONAL CATEGORIES
# ==============================================================================

PROFESSIONAL_CATEGORIES = {
    "experiences": {
        "name": "Experiences",
        "description": "What you did, where, and in what context. Roles, responsibilities, achievements tied to a job, project, or real-world setting. Answers: What have you actually done? MUST use ACTION VERBS describing real work (Managed..., Led..., Oversaw..., Built..., Taught..., Worked as..., Spent X years doing...). Experience tells the story of your work history.",
        "examples": [
            "Managed a team of five technicians at Memorial Hospital, overseeing medical equipment maintenance and repairs.",
            "Led preventive maintenance operations in a hospital environment for five years, ensuring zero equipment failures during critical procedures.",
            "Oversaw compliance audits and safety protocol implementation at three nursing homes across the region.",
            "Started as a line cook at a bistro, worked up to sous chef over six years, handled all seafood prep and menu development.",
            "Ran my own landscaping business for a decade—designed residential gardens and maintained commercial properties.",
            "Spent five years doing child protective services casework, managing a caseload of 30-40 families at any given time.",
            "Built custom cabinets and did finish carpentry on commercial builds for eight years before transitioning into project management.",
            "Taught third grade for three years, then moved to middle school math and took over the robotics program, grew it from 8 to 45 students."
        ]
    },
    "skills": {
        "name": "Skills",
        "description": "What you're capable of doing. Abilities, competencies, knowledge areas—often transferable across jobs. Answers: What can you do well? Use NATURAL CONVERSATIONAL FRAMES wrapping SHORT ABILITY PHRASES. Examples: 'I'm good at team leadership', 'My skills include Python and SQL', 'I have strong patient care abilities'. Skills summarize your capabilities in natural language.",
        "examples": [
            "I'm good at team leadership and coordinating staff across departments.",
            "My skills include patient care, triage, and bedside manner.",
            "I'm skilled in maintenance planning and equipment troubleshooting.",
            "My abilities include curriculum design, classroom management, and parent communication.",
            "I have strong skills in HVAC installation, repair, and system diagnostics.",
            "I'm proficient in Python, SQL, data analysis, and visualization.",
            "I'm good at conflict resolution, crisis intervention, and documentation.",
            "My skills are QuickBooks, payroll processing, and invoicing."
        ]
    },
    "certifications": {
        "name": "Certifications",
        "description": "Certifications, degrees, courses, licenses, formal qualifications. VARY THE LANGUAGE—avoid overusing 'certification' or 'certified'. Use alternatives: 'I have my [credential]', 'licensed [profession]', 'earned my [degree]', 'passed [exam]', 'completed [program]', 'I hold [credential]', or just state it directly.",
        "examples": [
            "I'm a licensed master electrician.",
            "I have my RN license and BSN degree, plus wound care specialty training.",
            "I hold a PMP credential and earned my MBA from State U.",
            "I completed my cosmetology license last year after finishing the 1500-hour program.",
            "I have my CDL Class A with hazmat endorsement.",
            "I hold a teaching credential in single-subject math.",
            "I passed the CPA exam, just need to finish the experience hours now.",
            "I have my real estate license for both residential and commercial properties."
        ]
    },
    "current_position": {
        "name": "Current Position",
        "description": "What you're doing RIGHT NOW professionally. Current job title, role, employer, responsibilities. MUST use PRESENT TENSE or present perfect continuous (I'm working as..., I work at..., I currently..., I've been... for X time and still am). Focus on ONGOING status, not past history.",
        "examples": [
            "I'm currently a senior software engineer at a mid-size tech company.",
            "I work as an LPN at a community health center, mostly doing geriatric care.",
            "I run a small HVAC business—I have three employees and we do residential and light commercial work.",
            "I teach high school math and lead the department; I've been here eight years now.",
            "I'm a social worker at a nonprofit, currently handling housing case management.",
            "I'm working as a line supervisor at a packaging plant on the second shift.",
            "I'm a freelance photographer—I do weddings, events, and some corporate work.",
            "I work as a receptionist and admin at a dental office, part-time while I finish my degree."
        ]
    }
}


# ==============================================================================
# PROMPT TEMPLATES
# ==============================================================================

MESSAGE_GENERATION_SYSTEM_PROMPT = """You are an expert at generating natural professional context messages. Always respond with valid JSON."""


# Category-specific guidance for Experience vs Skill distinction (see experience_vs_skill.md)
CATEGORY_GUIDANCE = {
    "experiences": """
CRITICAL: EXPERIENCES describe ACTIONS you took in REAL JOB CONTEXTS.
✓ CORRECT format: ACTION VERB + specific context/details
  - "Managed a team of 12 nurses on the night shift at City Hospital"
  - "Built custom furniture for residential clients over 8 years"
  - "Spent 5 years teaching high school chemistry and coaching debate"

✗ WRONG format: "X years of experience in [ability/skill]"
  - "Ten years of experience in team leadership" ← NO! This is listing a skill, not describing what you DID
  - "Experience in project management" ← NO! This is vague skill mention, not real work
  - "5 years of technical supervision" ← NO! What did you actually supervise? Where? Be specific!

Rule: If you can't picture the person doing the actual work, it's TOO VAGUE and belongs in Skills instead.""",

    "skills": """
CRITICAL: SKILLS use NATURAL CONVERSATIONAL FRAMES + SHORT ABILITY PHRASES.
✓ CORRECT format: Natural frame + ability phrases
  - "I'm good at team leadership and staff development"
  - "My skills include Python, SQL, and data visualization"
  - "I have strong patient care and triage abilities"
  - "I'm proficient in project management and budgeting"
  - "My strengths are communication and problem-solving"

Acceptable frames: "I'm good at...", "My skills include...", "I have...", "I'm proficient in...", "My abilities include...", "I'm skilled in...", "My strengths are..."

✗ WRONG format: Action verbs + past context (that's Experience!)
  - "Managed a team of engineers" ← NO! This is experience
  - "Led projects from concept to delivery" ← NO! This is what you DID, not what you CAN DO
  - "Oversaw maintenance operations" ← NO! This describes past work, not transferable ability

Rule: Use conversational frame + ability phrases. Keep ability phrases concise and focused on capabilities.""",

    "certifications": """
CRITICAL: VARY YOUR LANGUAGE! Don't overuse "certification" or "certified".
✓ CORRECT format: Use VARIED phrases for credentials
  - "I'm a licensed master electrician"
  - "I have my RN license and BSN degree"
  - "I hold a PMP credential"
  - "I earned my MBA from State University"
  - "I passed the CPA exam"
  - "I completed my cosmetology license last year"
  - "I have my CDL Class A with hazmat endorsement"
  - Just state it: "RN, BSN, and wound care specialty"

Varied vocabulary: "licensed", "I have my", "I hold", "I earned", "I passed", "I completed", "trained in", "credentialed"

✗ WRONG format: Overusing "certification/certified"
  - "I'm certified in nursing" ← Better: "I have my RN license" or "I'm a licensed nurse"
  - "I have a certification in project management" ← Better: "I hold a PMP credential"
  - "I'm certified as an electrician" ← Better: "I'm a licensed electrician"
  - "My certifications include..." repeated too often ← Vary the phrasing!

Rule: Use natural, varied language. Think how people actually talk about their credentials in casual conversation.""",

    "current_position": """
CRITICAL: CURRENT POSITION describes what you're doing RIGHT NOW (ongoing).
✓ CORRECT format: PRESENT TENSE or present perfect continuous
  - "I'm currently a project manager at ABC Construction"
  - "I work as a nurse at County Hospital on the night shift"
  - "I run my own landscaping business with four employees"
  - "I've been teaching third grade at Lincoln Elementary for five years"
  - "I'm working as a line cook at a downtown restaurant"

Present tense indicators: "I'm...", "I work as...", "I currently...", "I run...", "I teach..."
Present perfect continuous (ongoing): "I've been working as... for X years" (implies still there)

✗ WRONG format: Past tense or "my background includes" (that's Experience!)
  - "My background includes 5 years as a project manager" ← NO! That's past experience
  - "I was a nurse at County Hospital" ← NO! Past tense = experience
  - "I worked as a line cook" ← NO! "Worked" is past, use "I work" or "I'm working"
  - "Spent five years teaching" ← NO! That's completed experience, not current

Rule: Use present tense to describe what you're doing NOW. If using time duration, make it clear you're STILL in that role.""",
}

MESSAGE_GENERATION_PROMPT_TEMPLATE = """Generate {batch_size} diverse, natural professional messages for career coaching specifically about {category_name}.

Category: {category_name}
Description: {category_description}
{category_guidance}

Each message should be a complete, natural sentence as if spoken by a real person to a career coach. Use appropriate tense (past for experiences, present for current skills/position). Vary widely in length and formality.

Requirements:
1. Be specifically relevant to {category_name_lower}
2. Vary WIDELY in length: mix very short (5-10 words), short (11-15 words), medium (16-25 words), and long (26-40 words)
3. Cover DIVERSE industries: healthcare, education, trades, creative, business, service, law, social work, nonprofit, agriculture, manufacturing, etc. - NOT just tech
4. Include REALISTIC TYPOS in some messages: "experiance", "managment", "certfied", "teh", "recieve"
5. Vary FORMULATION: not all "I...". Use "My background...", "Started as...", "Worked at...", "Licensed...", fragments, etc.

Example messages for {category_name}:
{category_examples}

CRITICAL DISTINCTION - Experiences vs Skills vs Current Position:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IF GENERATING EXPERIENCES (past work):
  ✓ DO: Use PAST TENSE action verbs + specific completed context
     "Managed a team of 10 nurses at County Hospital for three years"
     "Built custom homes for 15 years in the residential sector"
     "Taught AP calculus and coached the math team for a decade"

  ✗ DON'T: Use "X years of experience in [skill]" format
     "Ten years of experience in team management" ← WRONG! Too vague!
     "Experience in patient care" ← WRONG! What did you actually DO?

IF GENERATING SKILLS (abilities/capabilities):
  ✓ DO: Use NATURAL CONVERSATIONAL FRAME + ability phrases
     "I'm good at team leadership and staff training"
     "My skills include custom carpentry and finish work"
     "I'm proficient in advanced mathematics and curriculum design"
     "I have strong communication and problem-solving abilities"
     "My strengths are Python, SQL, and data visualization"

  ✗ DON'T: Use action verbs + past context (that's experience!)
     "Managed teams of nurses" ← WRONG! That's experience!
     "Built homes for clients" ← WRONG! That's what you DID, not what you CAN DO!

  ✗ DON'T: List raw abilities without conversational frame
     "Team leadership and staff training" ← WRONG! Too raw, add "I'm good at..."
     "Python, SQL, data analysis" ← WRONG! Too raw, add "My skills include..."

IF GENERATING CURRENT POSITION (what you're doing NOW):
  ✓ DO: Use PRESENT TENSE (ongoing status)
     "I'm currently a project manager at ABC Construction"
     "I work as a nurse at County Hospital on the night shift"
     "I've been teaching third grade for five years" (implies still teaching)
     "I run my own landscaping business with four employees"

  ✗ DON'T: Use past tense or "background" language (that's experience!)
     "My background includes 5 years as a project manager" ← WRONG! Past experience!
     "I was a nurse at County Hospital" ← WRONG! Past tense = past experience!
     "Worked as a project manager" ← WRONG! Use "I work" or "I'm working"

IF GENERATING CERTIFICATIONS (credentials/qualifications):
  ✓ DO: Use VARIED language (avoid overusing "certification/certified")
     "I'm a licensed master electrician"
     "I have my RN license and BSN degree"
     "I hold a PMP credential"
     "I earned my MBA from State University"
     "I passed the CPA exam"
     "I completed my teaching credential last year"

  ✗ DON'T: Overuse "certification" or "certified"
     "I'm certified in nursing" ← Better: "I have my RN license"
     "I have a certification in project management" ← Better: "I hold a PMP"
     "My certifications include..." (repeated) ← Vary your phrasing!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Generate {batch_size} unique, complete messages as a JSON array. Return ONLY valid JSON with no additional text:
{{"messages": ["message1", "message2", ...]}}"""


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_category_name(category_key: str) -> str:
    """Get display name for category."""
    return PROFESSIONAL_CATEGORIES[category_key]["name"]


def get_category_description(category_key: str) -> str:
    """Get description for category."""
    return PROFESSIONAL_CATEGORIES[category_key]["description"]


def get_category_examples(category_key: str) -> List[str]:
    """Get examples for category."""
    return PROFESSIONAL_CATEGORIES[category_key]["examples"]


def build_message_generation_prompt(category_key: str, batch_size: int) -> str:
    """
    Build a prompt for generating complete professional messages (no patterns/objects).

    Args:
        category_key: Key from PROFESSIONAL_CATEGORIES
        batch_size: Number of messages to generate in this batch

    Returns:
        Formatted prompt string
    """
    if category_key not in PROFESSIONAL_CATEGORIES:
        raise ValueError(f"Unknown category: {category_key}")

    category = PROFESSIONAL_CATEGORIES[category_key]
    examples_str = "\n".join(f"- {ex}" for ex in category["examples"])
    guidance = CATEGORY_GUIDANCE.get(category_key, "")
    category_guidance = f"Important: {guidance}" if guidance else ""

    return MESSAGE_GENERATION_PROMPT_TEMPLATE.format(
        batch_size=batch_size,
        category_name=category["name"],
        category_description=category["description"],
        category_name_lower=category["name"].lower(),
        category_examples=examples_str,
        category_guidance=category_guidance
    )
