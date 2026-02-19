"""
Professional Context Constants for Hierarchical Multi-Label Classification

Taxonomy: professional > entity > sub_entity
6 entities: current_position, professional_experience, awards,
            professional_aspirations, workplace_challenges, job_search_status
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
            "considerations."
        ),
        "sub_entities": {
            "dream_roles": "Desired roles, target companies, target industries, career goals",
            "compensation_expectations": "Target salary, minimum acceptable, total comp goals",
            "desired_work_environment": "Remote/hybrid, company size, culture, deal-breakers",
            "career_change_considerations": "Considering change, risk tolerance, obstacles",
        },
        "examples": [
            "I want to become a VP of Product in 2 years",
            "My dream is to be a CPO someday",
            "My goal is to earn 120k next year",
            "I need at least 180k base salary to make a move",
            "I want to work hybrid not fully remote",
            "I'm looking for a Series C startup with good culture",
            "I'm thinking about switching from engineering to PM",
            "I'd love to work at Stripe or a similar payments company",
            "I want to transition into the AI/ML industry",
        ]
    },
    "workplace_challenges": {
        "name": "Workplace Challenges",
        "description": (
            "Workplace issues: toxic culture, bad management, conflicts, "
            "dead-end roles, office politics, discrimination, overwork."
        ),
        "sub_entities": {
            "workplace_challenges": (
                "Workplace issues, conflicts, toxic culture, "
                "management problems, and career blockers"
            ),
        },
        "examples": [
            "My manager micromanages everything I do",
            "The team culture is really toxic nobody trusts each other",
            "I'm stuck in a dead-end role with no growth opportunities",
            "There's so much office politics it's exhausting",
            "I keep getting passed over for promotions despite good reviews",
            "My boss takes credit for my work all the time",
            "The workload is insane I'm doing the job of three people",
            "I'm dealing with a difficult coworker who undermines me",
            "There's no clear career path at my company",
            "I feel like I'm being discriminated against at work",
        ]
    },
    "job_search_status": {
        "name": "Job Search Status",
        "description": (
            "Active job search: applications sent, interviews scheduled, "
            "offers received, search urgency, desired start date."
        ),
        "sub_entities": {
            "job_search_status": (
                "Currently searching, applications, interviews, "
                "offers, urgency, desired start date"
            ),
        },
        "examples": [
            "I'm casually looking for new opportunities right now",
            "I have 2 interviews next week at fintech companies",
            "I got an offer from Google but I'm not sure if I should take it",
            "I've applied to about 15 companies in the last month",
            "I'm actively interviewing at 3 different startups",
            "I just started my job search this week",
            "I have a final round at Amazon next Monday",
            "I got two offers and need to decide by Friday",
            "I'm not looking yet but keeping my options open",
            "I need to find something within the next 2 months",
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
CRITICAL: MULTI-LABEL entity. Dream roles, salary goals, work environment preferences, career change.
- dream_roles: "I want to be a VP", "I'd love to work at Stripe"
- compensation_expectations: "I want 200k+", "Need at least 150k base"
- desired_work_environment: "I want remote work", "Looking for a startup"
- career_change_considerations: "Thinking of switching to PM"
A message like "I want to be a VP at a fintech startup making 300k" touches dream_roles + compensation_expectations + desired_work_environment.
✗ WRONG: "I'm actively interviewing" (that's job_search_status!)""",

    "workplace_challenges": """
CRITICAL: WORKPLACE ISSUES affecting the person at their current job.
✓ CORRECT: "My boss micromanages me", "The culture is toxic", "I'm stuck with no growth"
✗ WRONG: "I'm stressed about work" (that's psychological > stress_and_coping!)
✗ WRONG: "I want to leave my job" (that's professional_aspirations or job_search_status!)""",

    "job_search_status": """
CRITICAL: ACTIVE job search activities ONLY. Must mention searching, applying, interviewing, or offers.
✓ CORRECT: "I'm interviewing at 3 companies", "I got an offer from Google", "Applied to 15 jobs"
✗ WRONG: "I want to be a CEO" (that's professional_aspirations > dream_roles!)
✗ WRONG: "I'm thinking about leaving" (that's career_change_considerations!)""",

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
            "but I want to become a VP of Product at a startup in the next "
            "2 years and I've started interviewing at a few Series B companies"
        ),
        "entities": [
            "current_position", "professional_aspirations", "job_search_status",
        ],
        "sub_entities": [
            "current_position", "dream_roles",
            "desired_work_environment", "job_search_status",
        ]
    },
    {
        "message": (
            "I worked at Amazon for 5 years leading a team of 12 engineers "
            "and launched a feature that saved 2M in costs, now I'm at a "
            "startup as CTO but honestly I'm thinking about going back to "
            "big tech the pay cut is brutal"
        ),
        "entities": [
            "professional_experience", "current_position",
            "professional_aspirations",
        ],
        "sub_entities": [
            "experiences", "current_position",
            "career_change_considerations",
        ]
    },
    {
        "message": (
            "My boss micromanages everything and takes credit for my work, "
            "I'm stuck with no growth path here so I've applied to about "
            "15 companies and have 2 interviews next week"
        ),
        "entities": ["workplace_challenges", "job_search_status"],
        "sub_entities": ["workplace_challenges", "job_search_status"]
    },
    {
        "message": (
            "I spent 10 years at Microsoft going from junior dev to senior "
            "engineer, my biggest achievement was architecting the new "
            "billing system that handles 50M transactions, I got promoted "
            "4 times and won the engineering excellence award"
        ),
        "entities": ["professional_experience", "awards"],
        "sub_entities": ["experiences", "awards"]
    },
    {
        "message": (
            "The team culture is toxic and there's constant office politics "
            "but I want to earn at least 200k in my next role at a company "
            "with good work-life balance, I'm giving myself 2 months to "
            "find something better"
        ),
        "entities": [
            "workplace_challenges", "professional_aspirations",
            "job_search_status",
        ],
        "sub_entities": [
            "workplace_challenges", "compensation_expectations",
            "desired_work_environment", "job_search_status",
        ]
    },
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
