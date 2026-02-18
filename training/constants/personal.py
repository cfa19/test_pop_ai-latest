"""
Personal Context Constants for Hierarchical Multi-Label Classification

Taxonomy: personal > entity > sub_entity
6 entities: personal_life, health_and_wellbeing, financial_situation, personal_goals, lifestyle_preferences, life_constraints
"""

CONTEXT_NAME = "personal"

ENTITIES = {
    "personal_life": {
        "name": "Personal Life",
        "description": "Life stage, family situation, relationship status, children, dependents, life transitions and priorities.",
        "sub_entities": {
            "demographics": "Age range, life stage (early career, mid-career, settling down)",
            "family_situation": "Relationship status, partner, children, dependents, support system, transitions, priorities"
        },
        "examples": [
            "I'm in my early 30s with a young family just had our second kid",
            "I just got married last summer and we're settling into our new life",
            "I'm prioritizing family time right now my kids are still small",
            "I'm married with a 1.5 year old and another on the way",
            "My spouse is a teacher so we have opposite schedules which helps with childcare",
            "My in-laws help with childcare three days a week which is a lifesaver",
            "I'm going through a divorce right now so my life is pretty unstable",
            "I'm single no kids so I have a lot of flexibility in my career choices",
            "I take care of my elderly mother she lives with us and needs daily help",
            "I'm planning to have another child in 2-3 years so timing matters"
        ]
    },
    "health_and_wellbeing": {
        "name": "Health & Wellbeing",
        "description": "Physical health, mental health, addictions/recovery, overall wellbeing. MULTI-LABEL: these often appear together.",
        "sub_entities": {
            "physical_health": "General health, chronic conditions, energy levels, limitations",
            "mental_health": "Mental health conditions, severity, treatment, work impact",
            "addictions_or_recovery": "Addiction type, recovery status, clean time, support system",
            "overall_wellbeing": "Overall stress level, wellbeing score"
        },
        "examples": [
            "I have a chronic back condition that limits my physical work",
            "I'm sleep-deprived from new parent life barely functioning some days",
            "I have anxiety and it's managed with therapy and medication",
            "I struggle with depression and it affects my work performance sometimes",
            "I'm on medication for ADHD which helps a lot with focus",
            "Burnout is really affecting my work performance right now",
            "I'm 9 months sober from alcohol and my sobriety is my top priority",
            "I attend AA meetings 3 times a week and I can't miss them",
            "I'm feeling pretty good overall maybe a 7 out of 10 on wellbeing",
            "I have low energy most days probably need to see a doctor about it"
        ]
    },
    "financial_situation": {
        "name": "Financial Situation",
        "description": "Financial stability, debt, savings, dependents, risk tolerance, financial stress.",
        "sub_entities": {
            "financial_situation": "Financial stability, debt, savings, dependents, income dependency, risk tolerance, financial stress"
        },
        "examples": [
            "I have 45k in student loan debt still paying it off",
            "I can't afford career risks right now I'm the sole provider",
            "I have 3-6 months emergency fund so I feel relatively secure",
            "I'm financially stressed about the mortgage and daycare costs",
            "We're a dual income household so I have some flexibility",
            "I have zero debt and 50k in savings so I can take risks",
            "I'm paycheck to paycheck right now any career move needs to pay more",
            "I'm not in a position to take a pay cut even for a dream job"
        ]
    },
    "personal_goals": {
        "name": "Personal Goals",
        "description": "Non-career life goals: health, family, relationships, personal growth, lifestyle.",
        "sub_entities": {
            "life_goals": "Non-career life goals: health, family, relationships, personal growth, lifestyle, relocation"
        },
        "examples": [
            "I want to maintain sobriety that's my highest priority above everything",
            "I want to lose 20 pounds in 6 months been going to the gym regularly",
            "I want to be more present with my family and stop checking work email",
            "I want to take my partner on a nice anniversary trip to Italy",
            "I want to get a dog in 1-2 years once we have a bigger place",
            "My goal is to run a marathon next year I'm training 4 days a week",
            "I want to learn to play guitar just for fun not career related",
            "I want to spend more quality time with my aging parents while I can"
        ]
    },
    "lifestyle_preferences": {
        "name": "Lifestyle Preferences",
        "description": "Work-life balance importance, ideal schedule, flexibility needs, non-negotiables.",
        "sub_entities": {
            "lifestyle_preferences": "Work-life balance importance, ideal schedule, flexibility needs, non-negotiables"
        },
        "examples": [
            "Work-life balance is critical for me 10 out of 10 importance",
            "I need flexibility for my AA meetings on Tuesday and Thursday evenings",
            "I won't work more than 45 hours a week I've done burnout before",
            "Remote work is non-negotiable for me at this point in my life",
            "I prefer a 9-5 schedule so I can pick up my kids from school",
            "I need summers off which is why I stay in teaching",
            "I'm fine working long hours right now building my career while young",
            "Flexible hours matter more to me than salary I need to manage my health"
        ]
    },
    "life_constraints": {
        "name": "Life Constraints",
        "description": "Limitations that affect career options: family, health, location, financial, time.",
        "sub_entities": {
            "life_constraints": "Constraint type, description, career impact, severity, and duration"
        },
        "examples": [
            "I can't travel for work because of childcare responsibilities",
            "I need to stay near my mother for her medical appointments",
            "I can't afford to take a pay cut right now even for a better role",
            "My recovery meetings limit my evening availability 3 nights a week",
            "My visa situation limits which companies I can work for",
            "I have a non-compete preventing me from working in my industry for a year",
            "My health condition means I can't do physically demanding work anymore",
            "I can't relocate for the next 3 years until my kid finishes high school"
        ]
    }

}

ENTITY_GUIDANCE = {
    "personal_life": """
CRITICAL: Two sub_entities: demographics (age, life stage) and family_situation (relationships, children, support, transitions).
✓ CORRECT: "I'm married with 2 kids", "I'm in my early 30s", "I take care of my elderly mother"
✗ WRONG: "I value family" (that's personal_values!)
✗ WRONG: "I want more family time" (that's personal_goals or lifestyle_preferences!)""",

    "health_and_wellbeing": """
CRITICAL: MULTI-LABEL entity. Physical, mental, addictions, overall wellbeing.
- physical_health: chronic conditions, energy, limitations
- mental_health: anxiety, depression, ADHD, treatment
- addictions_or_recovery: sobriety, meetings, triggers
- overall_wellbeing: general score, stress level
A message like "I have chronic pain AND anxiety AND I'm sober, overall 5/10" touches all four.""",

    "financial_situation": """
CRITICAL: Current financial STATE. Uses financial_situation schema with fields:
stability, debt, savings, incomeDependency, riskTolerance, financialStress.
✓ CORRECT: "I have 45k in student debt", "I'm paycheck to paycheck"
✗ WRONG: "I want to earn 200k" (that's professional > professional_aspirations!)""",

    "personal_goals": """
CRITICAL: NON-CAREER life goals. Uses life_goals schema with fields: title, description, targetDate, progress.
✓ CORRECT: "I want to run a marathon", "I want more family time", "I'm planning to move to Barcelona"
✗ WRONG: "I want to become a VP" (that's professional > professional_aspirations!)""",

    "lifestyle_preferences": """
CRITICAL: How you want to LIVE—schedule, balance, flexibility.
✓ CORRECT: "I need flexible hours", "Remote work is non-negotiable"
✗ WRONG: "I prefer collaborative environments" (that's psychological > working_style_preferences!)""",

    "life_constraints": """
CRITICAL: Things that RESTRICT your career options.
✓ CORRECT: "I can't travel", "I can't relocate", "I can't take a pay cut"
✗ WRONG: "My in-laws help with childcare" (that's life_enablers, a positive!)""",

}

MULTI_LABEL_EXAMPLES = [
    {
        "message": (
            "I'm married with a 1.5 year old and another on the way, my in-laws "
            "help with childcare 3 days a week which is a lifesaver, but I have "
            "about 40k in student debt and I'm the primary earner so I can't "
            "afford career risks right now"
        ),
        "entities": ["personal_life", "financial_situation", "life_constraints"],
        "sub_entities": ["family_situation", "financial_situation", "life_constraints"]
    },
    {
        "message": (
            "I have chronic back pain and anxiety that's managed with medication, "
            "I'm also 9 months sober and attend AA three times a week, overall my "
            "wellbeing is about a 5 out of 10 because the physical pain really "
            "gets me down some days"
        ),
        "entities": ["health_and_wellbeing"],
        "sub_entities": [
            "physical_health", "mental_health",
            "addictions_or_recovery", "overall_wellbeing"
        ]
    },
    {
        "message": (
            "My biggest goal right now is maintaining sobriety and being more "
            "present with my family, I won't work more than 45 hours a week and "
            "remote work is non-negotiable because I need to be near my AA "
            "meetings and my kids school"
        ),
        "entities": ["personal_goals", "lifestyle_preferences", "life_constraints"],
        "sub_entities": ["life_goals", "lifestyle_preferences", "life_constraints"]
    },
    {
        "message": (
            "I just got married and we bought our first house in Austin, "
            "family is my top priority but I need more fulfillment at work"
        ),
        "entities": ["personal_life"],
        "sub_entities": ["family_situation"]
    },
    {
        "message": (
            "I'm in my early 30s with no kids so I have a lot of flexibility "
            "in my career choices, my main personal goal is to maintain "
            "work-life balance"
        ),
        "entities": ["personal_life", "personal_goals"],
        "sub_entities": ["demographics", "family_situation", "life_goals"]
    },
    {
        "message": (
            "I can't relocate because my partner has tenure at the university, "
            "but my parents help with our mortgage so expenses are manageable "
            "even on my salary"
        ),
        "entities": ["life_constraints", "financial_situation"],
        "sub_entities": ["life_constraints", "financial_situation"]
    },
    {
        "message": (
            "My goal is to run a marathon next year and I've been training "
            "4 days a week, health is important to me"
        ),
        "entities": ["personal_goals"],
        "sub_entities": ["life_goals"]
    },
    {
        "message": (
            "I'm dealing with financial stress because I burned through most "
            "of my savings, my spouse is supportive but we're now a single "
            "income family with two kids"
        ),
        "entities": ["financial_situation", "personal_life"],
        "sub_entities": ["financial_situation", "family_situation"]
    }
]

MESSAGE_GENERATION_SYSTEM_PROMPT = (
    "You are an expert at generating natural personal context messages "
    "for career coaching. Always respond with valid JSON."
)

SINGLE_LABEL_PROMPT_TEMPLATE = """\
Generate {batch_size} diverse, natural personal messages for career \
coaching specifically about {entity_name} > {sub_entity_name}.

Context: Personal
Entity: {entity_name}
Sub-entity: {sub_entity_name} - {sub_entity_description}
{entity_guidance}

Requirements:
1. Each message MUST be specifically about {sub_entity_name}
2. Vary WIDELY in length:
   - 20% very short (5-12 words): "I'm married with two kids"
   - 30% short-medium (13-25 words)
   - 30% medium-long (26-50 words)
   - 20% long paragraphs (51-70 words)
3. Cover DIVERSE life situations, ages, family structures
4. Include REALISTIC TYPOS in ~15%: "im", "becuase", "famly", "finacial"
5. Vary formulation: "I...", "My...", "We...", fragments
6. Be authentic—these are personal, sometimes vulnerable messages

Example messages for {entity_name}:
{entity_examples}

Generate {batch_size} unique messages as JSON:
{{"messages": ["message1", "message2", ...]}}"""

MULTI_LABEL_PROMPT_TEMPLATE = """\
Generate {batch_size} natural, compound messages for career coaching \
that COMBINE multiple personal topics in a single message.

Each message should naturally touch on {num_labels} or more of these sub-entities: {sub_entity_list}

Requirements:
1. Each message MUST mention at least {num_labels} different sub-entities
2. Length: 30-70 words (paragraphs)
3. Natural and conversational, authentic and personal
4. Cover DIVERSE life situations
5. Include REALISTIC TYPOS in ~15%
6. Be appropriately vulnerable and honest

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
