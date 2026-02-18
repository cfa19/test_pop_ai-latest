"""
Learning Context Constants for Hierarchical Multi-Label Classification

Taxonomy: learning > entity > sub_entity
6 entities: current_skills, education_history, learning_gaps, learning_aspirations, certifications, learning_history
"""

CONTEXT_NAME = "learning"

ENTITIES = {
    "current_skills": {
        "name": "Current Skills",
        "description": "Skills they currently have, proficiency level, years of experience, verification.",
        "sub_entities": {
            "skills": "Skills they currently have, proficiency level, years of experience, and verification"
        },
        "examples": [
            "I'm an expert in product strategy with 8 years experience",
            "I know Python at an intermediate level",
            "I have 8 years of data analysis experience",
            "I'm proficient in SQL and Tableau for data visualization",
            "I'm a skilled welder certified in TIG and MIG welding",
            "I speak fluent JavaScript and TypeScript been coding for 6 years",
            "I'm decent at public speaking maybe a 6 out of 10",
            "I'm a beginner at machine learning just started learning",
            "I'm very strong in financial modeling and Excel",
            "I have advanced skills in Adobe Creative Suite been using it for 12 years"
        ]
    },
    "education_history": {
        "name": "Education History",
        "description": "Degrees earned, schools attended, field of study, GPA, graduation dates.",
        "sub_entities": {
            "education": "Degrees earned, schools attended, field of study, GPA, graduation dates"
        },
        "examples": [
            "I have a BS in Computer Science from UC Berkeley",
            "I graduated with a 3.7 GPA from Stanford",
            "I studied Software Engineering for my masters at MIT",
            "I have an MBA from Wharton graduated in 2020",
            "I went to community college then transferred to state university",
            "I dropped out of college to start my business",
            "I have a nursing degree from Johns Hopkins",
            "I studied electrical engineering at Georgia Tech",
            "I have a PhD in biochemistry from Columbia",
            "I got my associates degree in culinary arts"
        ]
    },
    "learning_gaps": {
        "name": "Learning Gaps",
        "description": "Missing skills and knowledge that block career goals. Includes both skill gaps and knowledge gaps.",
        "sub_entities": {
            "skill_gaps": "Missing skills blocking career goals, impact, what aspiration it blocks",
            "knowledge_gaps": "Missing knowledge blocking career goals, what aspiration it blocks"
        },
        "examples": [
            "I need to improve my executive presence to become a VP",
            "I lack experience managing people for the Director role",
            "I don't have enough technical skills for that Senior Engineer position",
            "I need better public speaking skills to get promoted",
            "I'm missing Python experience to transition into data science",
            "I don't understand blockchain well enough to work at Coinbase",
            "I need to learn more about AI/ML to transition into machine learning",
            "I lack fintech knowledge for the payments role I want",
            "I need deeper understanding of cloud architecture to become a Solutions Architect",
            "I need better executive presence and deeper fintech knowledge to become a VP at a payments company"
        ]
    },
    "learning_aspirations": {
        "name": "Learning Aspirations",
        "description": "Future learning goals: skills they want to learn, degrees they want to pursue, certifications they want to earn.",
        "sub_entities": {
            "skill_aspirations": "Skills they want to learn, learning plan, timeline, progress",
            "education_aspirations": "Degrees they want to pursue, target schools, timeline, funding",
            "certification_aspirations": "Certifications they want to earn, study plan, exam date"
        },
        "examples": [
            "I want to learn machine learning this year",
            "I'm learning public speaking through Toastmasters",
            "I'm 25 percent done with my AI course on Coursera",
            "I want to get an MBA from Stanford in 2 years",
            "I'm planning to pursue a masters in AI next fall",
            "I'll apply to business school in 2027",
            "I'm studying for the Google Cloud certification",
            "I want to get my PMP next year",
            "I'm taking a prep course for the AWS Solutions Architect exam",
            "I want to learn Python and get AWS certified while pursuing an MBA"
        ]
    },
    "certifications": {
        "name": "Certifications",
        "description": "Certifications already earned, issue date, expiry date, active status.",
        "sub_entities": {
            "certifications": "Certifications already earned, issue date, expiry date, active status"
        },
        "examples": [
            "I'm AWS Solutions Architect certified got it last year",
            "I have my CSPO certification its still active",
            "My PMP expires next year need to renew",
            "I'm a certified Scrum Master since 2021",
            "I have my Google Analytics certification",
            "I got the Salesforce Admin cert in January",
            "I'm a certified nursing assistant CNA",
            "I have 3 active Azure certifications"
        ]
    },
    "learning_history": {
        "name": "Learning History",
        "description": "Past courses taken, books read, bootcamps completed, learning outcomes.",
        "sub_entities": {
            "past_courses": "Courses, bootcamps, and training programs completed",
            "books_and_resources": "Books and resources read for professional development"
        },
        "examples": [
            "I took Andrew Ng's ML course on Coursera it was excellent",
            "I read High Output Management it changed how I think about leadership",
            "I completed a 12-week data science bootcamp at General Assembly",
            "I finished the Google UX Design certificate on Coursera",
            "I read about 20 product management books last year",
            "I took a leadership development program at my company",
            "I completed a coding bootcamp and it helped me transition to tech",
            "I just finished reading Cracking the PM Interview super helpful"
        ]
    }

}

ENTITY_GUIDANCE = {
    "current_skills": """
CRITICAL: Skills they HAVE right now. Uses skills schema with fields: name, level, validationDate, validationSource.
✓ CORRECT: "I know Python", "I'm proficient in SQL", "I have 8 years of data analysis"
✗ WRONG: "I want to learn Python" (that's learning_aspirations!)
✗ WRONG: "I need to improve my public speaking" (that's learning_gaps!)""",

    "learning_gaps": """
CRITICAL: Skills/knowledge they're MISSING that BLOCK career goals.
✓ CORRECT: "I need executive presence for the VP role", "I lack Python for data science"
✗ WRONG: "I want to learn Python" (that's learning_aspirations—a goal, not a gap!)
✗ WRONG: "I know Python at intermediate level" (that's current_skills!)""",

    "learning_aspirations": """
CRITICAL: MULTI-LABEL entity. Skills, degrees, certs they want to PURSUE.
- skill_aspirations: "I want to learn ML", "I'm taking a Python course"
- education_aspirations: "I want an MBA from Stanford"
- certification_aspirations: "Studying for AWS cert"
A message like "I want to learn Python and get AWS certified while doing an MBA" touches all three.""",

    "certifications": """
CRITICAL: Certs they ALREADY HAVE. Uses certifications schema with fields: name, issuer, date, expiryDate.
✓ CORRECT: "I'm AWS certified", "My PMP expires next year"
✗ WRONG: "I want to get PMP certified" (that's learning_aspirations > certification_aspirations!)""",

}

MULTI_LABEL_EXAMPLES = [
    {
        "message": (
            "I know Python at an intermediate level and I have 5 years of data "
            "analysis experience, but I need to learn machine learning to transition "
            "into a data science role, I'm taking Andrew Ng's course on Coursera "
            "and I'm about 40 percent done"
        ),
        "entities": ["current_skills", "learning_gaps", "learning_aspirations", "learning_history"],
        "sub_entities": ["skills", "skill_gaps", "skill_aspirations", "past_courses"]
    },
    {
        "message": (
            "I have a BS in Computer Science from Berkeley with a 3.8 GPA, "
            "now I want to get an MBA from Wharton to transition into product management"
        ),
        "entities": ["education_history", "learning_aspirations"],
        "sub_entities": ["education", "education_aspirations"]
    },
    {
        "message": (
            "I'm AWS Solutions Architect certified and studying for the Google Cloud "
            "cert, I have deep expertise in cloud architecture and distributed systems"
        ),
        "entities": ["certifications", "learning_aspirations"],
        "sub_entities": ["certifications", "certification_aspirations"]
    },
    {
        "message": (
            "I completed a 12-week bootcamp at General Assembly and got the Google "
            "UX Design certificate, now I'm studying for the CSPO certification "
            "and want to learn Figma at an advanced level"
        ),
        "entities": ["learning_history", "certifications", "learning_aspirations"],
        "sub_entities": ["past_courses", "certifications", "certification_aspirations", "skill_aspirations"]
    },
    {
        "message": (
            "My biggest knowledge gap right now is cloud architecture which I need "
            "for the Solutions Architect role, I also need better public speaking skills"
        ),
        "entities": ["learning_gaps"],
        "sub_entities": ["knowledge_gaps", "skill_gaps"]
    },
    {
        "message": (
            "I have expert-level Python and SQL skills verified by my 10 years of "
            "experience, but I lack experience with blockchain technology which is "
            "blocking me from a role at Coinbase"
        ),
        "entities": ["current_skills", "learning_gaps"],
        "sub_entities": ["skills", "knowledge_gaps"]
    }
]

MESSAGE_GENERATION_SYSTEM_PROMPT = (
    "You are an expert at generating natural learning context messages for career "
    "coaching. Generate messages that sound like real people talking about their "
    "skills, education, learning, and knowledge. Cover DIVERSE fields and learning "
    "styles. Always respond with valid JSON."
)

SINGLE_LABEL_PROMPT_TEMPLATE = """\
Generate {batch_size} diverse, natural learning messages for career \
coaching specifically about {entity_name} > {sub_entity_name}.

Context: Learning
Entity: {entity_name}
Sub-entity: {sub_entity_name} - {sub_entity_description}
{entity_guidance}

Requirements:
1. Each message MUST be specifically about {sub_entity_name}
2. Vary WIDELY in length:
   - 20% very short (5-12 words): "I know Python well"
   - 30% short-medium (13-25 words)
   - 30% medium-long (26-50 words)
   - 20% long paragraphs (51-70 words)
3. Cover DIVERSE fields: tech, healthcare, trades, business, creative, science
4. Include REALISTIC TYPOS in ~15%: "im", "becuase", "knowlege", "certifcation"
5. Vary formulation: "I...", "My...", "I've...", fragments
6. Be authentic—real learning talk

Example messages for {entity_name}:
{entity_examples}

Generate {batch_size} unique messages as JSON:
{{"messages": ["message1", "message2", ...]}}"""

MULTI_LABEL_PROMPT_TEMPLATE = """\
Generate {batch_size} natural, compound messages for career coaching \
that COMBINE multiple learning topics in a single message.

Each message should naturally touch on {num_labels} or more of these sub-entities: {sub_entity_list}

Requirements:
1. Each message MUST mention at least {num_labels} different sub-entities
2. Length: 30-70 words (paragraphs)
3. Natural and conversational
4. Cover DIVERSE fields and learning journeys
5. Include REALISTIC TYPOS in ~15%

Example compound messages:
{multi_label_examples}

Return valid JSON:
{{"messages": [
  {{"text": "the message", "sub_entities": ["sub1", "sub2", "sub3"]}},
  ...
]}}"""


def build_message_generation_prompt(entity_key: str, batch_size: int, sub_entity_key: str = None) -> str:
    if entity_key not in ENTITIES:
        raise ValueError(f"Unknown entity: {entity_key}")

    entity = ENTITIES[entity_key]
    examples_str = "\n".join(f"- {ex}" for ex in entity["examples"])
    guidance = ENTITY_GUIDANCE.get(entity_key, "")
    entity_guidance = f"\nImportant:\n{guidance}" if guidance else ""

    if sub_entity_key and sub_entity_key in entity["sub_entities"]:
        sub_entity_desc = entity["sub_entities"][sub_entity_key]
        sub_entity_name = sub_entity_key.replace("_", " ").title()
    else:
        sub_entity_desc = entity["description"]
        sub_entity_key = entity_key
        sub_entity_name = entity["name"]

    return SINGLE_LABEL_PROMPT_TEMPLATE.format(
        batch_size=batch_size,
        entity_name=entity["name"],
        sub_entity_name=sub_entity_name,
        sub_entity_description=sub_entity_desc,
        entity_guidance=entity_guidance,
        entity_examples=examples_str
    )


def build_multilabel_generation_prompt(entity_keys: list, batch_size: int, num_labels: int = 3) -> str:
    all_sub_entities = []
    for ek in entity_keys:
        if ek in ENTITIES:
            for sk, desc in ENTITIES[ek]["sub_entities"].items():
                all_sub_entities.append(f"{sk} ({desc})")

    examples_str = "\n".join(
        f"- \"{ex['message'][:100]}...\" -> {ex['sub_entities']}"
        for ex in MULTI_LABEL_EXAMPLES[:5]
    )

    return MULTI_LABEL_PROMPT_TEMPLATE.format(
        batch_size=batch_size,
        num_labels=num_labels,
        sub_entity_list=", ".join(all_sub_entities),
        multi_label_examples=examples_str
    )
