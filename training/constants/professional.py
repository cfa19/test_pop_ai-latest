"""
Professional Context Constants for Hierarchical Multi-Label Classification

Taxonomy: professional > entity > sub_entity
4 entities: current_position, professional_experience, awards, professional_aspirations
"""

CONTEXT_NAME = "professional"

ENTITIES = {
    "current_position": {
        "name": "Current Position",
        "description": "Current job title, employer, compensation, and tenure.",
        "sub_entities": {
            "current_position": "Current job title, employer, compensation, and tenure"
        },
        "examples": [
            "I'm currently a Senior Product Manager at Google",
            "I make 150k base salary plus equity",
            "I've been in this role for 2 years now",
            "I work at a Series B startup as a frontend engineer",
            "My total comp is about 280k including stock options",
            "I just started as a marketing director at a mid-size agency",
            "I'm a staff nurse at Memorial Hospital been here since 2022",
            "I'm an electrician foreman at a construction company making about 85k",
            "I'm a high school math teacher in my fourth year",
            "I work as a sous chef at a fine dining restaurant downtown"
        ]
    },
    "professional_experience": {
        "name": "Professional Experience",
        "description": "Past roles, employers, responsibilities, achievements, and duration. MUST use action verbs describing real work.",
        "sub_entities": {
            "experiences": "Past roles, employers, responsibilities, achievements, and duration"
        },
        "examples": [
            "I worked at Microsoft for 3 years as a PM",
            "At my last job I launched a feature that increased revenue by 20 percent",
            "I led a team of 5 engineers building the payments platform",
            "Managed a team of five technicians at Memorial Hospital overseeing equipment maintenance",
            "Started as a line cook at a bistro worked up to sous chef over six years",
            "Ran my own landscaping business for a decade designing residential gardens",
            "Spent five years doing child protective services casework managing 30-40 families",
            "Built custom cabinets and did finish carpentry on commercial builds for eight years",
            "Taught third grade for three years then moved to middle school math",
            "I was a junior developer at a startup before it got acquihired by Amazon"
        ]
    },
    "awards": {
        "name": "Awards",
        "description": "Professional awards and recognitions from employers or industry bodies.",
        "sub_entities": {
            "awards": "Professional awards and recognitions"
        },
        "examples": [
            "I won Product Manager of the Year at our company",
            "I received Employee of the Month three times this year",
            "I was recognized as Top Performer in my division",
            "I got the Presidents Club award for exceeding sales targets",
            "I won the innovation award at our annual company hackathon",
            "I was named Nurse of the Year at my hospital",
            "I received a teaching excellence award from the school district",
            "My team won the best project award at the company offsite"
        ]
    },
    "professional_aspirations": {
        "name": "Professional Aspirations",
        "description": (
            "Career goals: dream roles, target companies/industries, salary "
            "expectations, desired work environment, career change "
            "considerations, job search status."
        ),
        "sub_entities": {
            "dream_roles": "Desired roles, target companies, target industries, career goals",
            "compensation_expectations": "Target salary, minimum acceptable, total comp goals",
            "desired_work_environment": "Remote/hybrid, company size, culture, deal-breakers",
            "career_change_considerations": "Considering change, risk tolerance, obstacles",
            "job_search_status": "Currently searching, applications, interviews, offers"
        },
        "examples": [
            "I want to become a VP of Product in 2 years",
            "My dream is to be a CPO someday",
            "My goal is to earn 120k next year",
            "I need at least 180k base salary to make a move",
            "I want to work hybrid not fully remote",
            "I'm looking for a Series C startup with good culture",
            "I'm thinking about switching from engineering to PM",
            "I'm casually looking for new opportunities right now",
            "I have 2 interviews next week at fintech companies",
            "I'd love to work at Stripe or a similar payments company",
            "I want to transition into the AI/ML industry",
            "I got an offer from Google but I'm not sure if I should take it"
        ]
    }
}

ENTITY_GUIDANCE = {
    "current_position": """
CRITICAL: Current position = what you do RIGHT NOW. Uses current_position schema with fields: title, company, compensation, startDate, department.
✓ CORRECT: "I'm a PM at Google", "I make 150k base"
✗ WRONG: "I worked at Microsoft" (that's professional_experience—PAST)
✗ WRONG: "I want to become a VP" (that's professional_aspirations!)""",

    "professional_experience": """
CRITICAL: PAST work. Uses experiences schema with fields: role, company, description, startDate, endDate, achievements.
Must use ACTION VERBS: Managed, Led, Built, Oversaw, Worked as, Spent X years...
✓ CORRECT: "I worked at Amazon for 3 years", "I led a team of 10"
✗ WRONG: "I currently work at..." (that's current_position!)""",

    "professional_aspirations": """
CRITICAL: MULTI-LABEL entity. Dream roles, salary goals, work environment preferences, career change, job search.
- dream_roles: "I want to be a VP", "I'd love to work at Stripe"
- compensation_expectations: "I want 200k+", "Need at least 150k base"
- desired_work_environment: "I want remote work", "Looking for a startup"
- career_change_considerations: "Thinking of switching to PM"
- job_search_status: "I'm actively interviewing", "Got 2 offers"
A message like "I want to be a VP at a fintech startup making 300k" touches dream_roles + compensation_expectations + desired_work_environment.""",

    "awards": """
CRITICAL: Professional awards from EMPLOYERS or INDUSTRY.
✓ CORRECT: "Employee of the Month", "Presidents Club", "Innovation Award"
✗ WRONG: "Dean's List" (that's learning > academic_awards!)
✗ WRONG: "AWS Certified" (that's learning > certifications!)""",
}

MULTI_LABEL_EXAMPLES = [
    {
        "message": (
            "I'm currently a Senior PM at Google making about 280k total comp, "
            "but I want to become a VP of Product at a startup in the next 2 years, "
            "I'm actively interviewing at a few Series B companies right now"
        ),
        "entities": ["current_position", "professional_aspirations"],
        "sub_entities": ["current_position", "dream_roles", "desired_work_environment", "job_search_status"]
    },
    {
        "message": (
            "I worked at Amazon for 5 years leading a team of 12 engineers and "
            "launched a feature that saved 2M in costs, now I'm at a startup as CTO "
            "but honestly I'm thinking about going back to big tech the pay cut is brutal"
        ),
        "entities": ["professional_experience", "current_position", "professional_aspirations"],
        "sub_entities": ["experiences", "current_position", "career_change_considerations"]
    },
    {
        "message": (
            "I want to earn at least 200k base in my next role, I'm looking for a "
            "hybrid setup at a Series C or later company, I won't work at a place "
            "without good work-life balance that's a deal-breaker for me"
        ),
        "entities": ["professional_aspirations"],
        "sub_entities": ["compensation_expectations", "desired_work_environment", "dream_roles"]
    },
    {
        "message": (
            "I spent 10 years at Microsoft going from junior dev to senior engineer, "
            "my biggest achievement was architecting the new billing system that handles "
            "50M transactions, I got promoted 4 times and won the engineering excellence award"
        ),
        "entities": ["professional_experience", "awards"],
        "sub_entities": ["experiences", "awards"]
    },
    {
        "message": (
            "My dream is to be CPO at a mission-driven company ideally in edtech or "
            "healthtech, I need at least 250k total comp and remote work is non-negotiable, "
            "I'm not in a rush though I'm giving myself 2 years to make the move"
        ),
        "entities": ["professional_aspirations"],
        "sub_entities": ["dream_roles", "compensation_expectations", "desired_work_environment", "job_search_status"]
    }
]

MESSAGE_GENERATION_SYSTEM_PROMPT = (
    "You are an expert at generating natural professional context messages for "
    "career coaching. Generate messages that sound like real people talking about "
    "their career. Cover DIVERSE professions: tech, healthcare, trades, education, "
    "finance, creative, service industry, military, government, etc. "
    "Always respond with valid JSON."
)

SINGLE_LABEL_PROMPT_TEMPLATE = """\
Generate {batch_size} diverse, natural professional messages for career \
coaching specifically about {entity_name} > {sub_entity_name}.

Context: Professional
Entity: {entity_name}
Sub-entity: {sub_entity_name} - {sub_entity_description}
{entity_guidance}

Requirements:
1. Each message MUST be specifically about {sub_entity_name}
2. Vary WIDELY in length:
   - 20% very short (5-12 words): "I'm a PM at Google"
   - 30% short-medium (13-25 words)
   - 30% medium-long (26-50 words)
   - 20% long paragraphs (51-70 words)
3. Cover DIVERSE professions: tech, healthcare, trades, education, finance, creative, service, military
4. Include REALISTIC TYPOS in ~15%: "im", "becuase", "managment", "recieved"
5. Vary formulation: "I...", "My...", "Currently...", "Been...", fragments
6. Be authentic—real career talk

Example messages for {entity_name}:
{entity_examples}

Generate {batch_size} unique messages as JSON:
{{"messages": ["message1", "message2", ...]}}"""

MULTI_LABEL_PROMPT_TEMPLATE = """\
Generate {batch_size} natural, compound messages for career coaching \
that COMBINE multiple professional topics in a single message.

Each message should naturally touch on {num_labels} or more of these sub-entities: {sub_entity_list}

Requirements:
1. Each message MUST mention at least {num_labels} different sub-entities
2. Length: 30-70 words (paragraphs)
3. Natural and conversational
4. Cover DIVERSE professions and career stages
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
