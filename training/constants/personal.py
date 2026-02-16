"""
Personal Context Constants for Hierarchical Multi-Label Classification

Taxonomy: personal > entity > sub_entity
12 entities: personal_life, health_and_wellbeing, living_situation, financial_situation,
             personal_goals, personal_projects, lifestyle_preferences, life_constraints,
             life_enablers, major_life_events, personal_values, life_satisfaction
"""

CONTEXT_NAME = "personal"

ENTITIES = {
    "personal_life": {
        "name": "Personal Life",
        "description": "Life stage, family situation, relationship status, children, dependents, life transitions and priorities.",
        "sub_entities": {
            "life_stage": "Life stage (early career, mid-career, settling down, etc.)",
            "age_range": "Age bracket",
            "relationship_status": "Single, married, partnered, divorced",
            "partner": "Partner's situation and career",
            "children": "Kids and ages",
            "dependents": "Other dependents (parents, family members)",
            "childcare": "Childcare arrangements",
            "family_support": "Support system (in-laws, relatives, friends)",
            "life_transitions": "Recent or upcoming transitions",
            "life_priorities": "Current life priorities"
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
    "living_situation": {
        "name": "Living Situation",
        "description": "Housing, location, who they live with, willingness to relocate, home office setup.",
        "sub_entities": {
            "housing_type": "Own, rent, etc.",
            "location": "Where they live",
            "living_with": "Who they live with",
            "relocation_openness": "Willing to move or not",
            "constraints": "What prevents relocation",
            "home_office": "Remote work setup"
        },
        "examples": [
            "I own a house in Austin Texas with my family",
            "I can't relocate because my partner has tenure at the university",
            "I have a great home office setup with dual monitors and standing desk",
            "I rent an apartment in Brooklyn and my lease is up in 6 months",
            "I live with my parents to save money while I pay off student loans",
            "I'd be open to relocating for the right opportunity anywhere in the US",
            "I live in a rural area so remote work is really my only option",
            "We just bought our first house so I need to stay in this area"
        ]
    },
    "financial_situation": {
        "name": "Financial Situation",
        "description": "Financial stability, debt, savings, dependents, risk tolerance, financial stress.",
        "sub_entities": {
            "stability": "Financial stability level",
            "debt": "Debt situation",
            "emergency_fund": "Savings cushion",
            "dependents": "Financial dependents",
            "income_dependency": "Single or dual income",
            "risk_tolerance": "Financial risk tolerance",
            "stress_level": "Financial stress"
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
            "non_career_goals": "Personal life goals",
            "category": "Health, family, relationship, hobby, etc.",
            "priority": "Importance level",
            "timeframe": "When they want to achieve it",
            "progress": "Current progress"
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
    "personal_projects": {
        "name": "Personal Projects",
        "description": "Side projects, hobbies, creative work, both career-relevant and purely personal.",
        "sub_entities": {
            "project_name": "Name or description of the project",
            "project_description": "What the project does or is about",
            "project_type": "Career-related, hobby, creative, etc.",
            "project_role": "Their role in the project",
            "project_skills": "Skills used if applicable",
            "time_commitment": "Hours per week",
            "motivation": "Why they do it"
        },
        "examples": [
            "I built an open-source analytics dashboard that has 500 GitHub stars",
            "I maintain a product management blog with 10k monthly readers",
            "I'm restoring a vintage motorcycle with my dad it's our weekend project",
            "I have a vegetable garden in my backyard it's very therapeutic",
            "I spend about 3 hours a week on my side project building an app",
            "I do woodworking as a hobby I make furniture for friends and family",
            "I have an Etsy shop selling handmade jewelry brings in 500 a month",
            "I volunteer coach a kids soccer team every Saturday morning"
        ]
    },
    "lifestyle_preferences": {
        "name": "Lifestyle Preferences",
        "description": "Work-life balance importance, ideal schedule, flexibility needs, non-negotiables.",
        "sub_entities": {
            "work_life_balance": "How important balance is",
            "ideal_schedule": "Preferred work schedule",
            "flexibility_needs": "What flexibility they need",
            "non_negotiables": "What they won't compromise on"
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
            "constraint_type": "Family, health, location, financial",
            "description": "What the constraint is",
            "impact_on_career": "How it affects career choices",
            "severity": "How limiting it is",
            "timeframe": "How long it will last"
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
    },
    "life_enablers": {
        "name": "Life Enablers",
        "description": "Things that help and support their career: family support, financial cushion, location.",
        "sub_entities": {
            "enabler_type": "Family, support, location, financial, etc.",
            "description": "What helps them",
            "benefit_to_career": "How it helps their career",
            "strength": "How strong the enabler is"
        },
        "examples": [
            "My in-laws provide free childcare 3 days a week which is huge",
            "My spouse is very supportive of my career and picks up slack at home",
            "My AA community keeps me accountable and grounded",
            "Living in San Francisco gives me access to tons of tech companies",
            "I have no debt and savings so I can afford to take career risks",
            "My partner makes good money so I can take a lower paying job I love",
            "My parents help with the mortgage so my housing costs are low",
            "I live near a major university so there are always networking events"
        ]
    },
    "major_life_events": {
        "name": "Major Life Events",
        "description": "Significant events: marriage, birth, health events, moves, losses, career changes.",
        "sub_entities": {
            "event_type": "Marriage, birth, move, health, loss, etc.",
            "date": "When it happened",
            "description": "What happened",
            "impact": "How it affected them"
        },
        "examples": [
            "I got married last year and it shifted my priorities completely",
            "My first child was born in 2023 and everything changed",
            "I started recovery 9 months ago hardest and best thing I've done",
            "We bought our first house last month huge milestone for us",
            "I got laid off 6 months ago and I'm still processing it emotionally",
            "My parent passed away last year and it made me rethink my career",
            "I survived a serious health scare that made me reprioritize everything",
            "I moved to a new city for my partner's job had to start over professionally"
        ]
    },
    "personal_values": {
        "name": "Personal Values",
        "description": "What matters in LIFE (not just work): family, health, authenticity, community, faith.",
        "sub_entities": {
            "life_values": "What matters most in life",
            "importance": "Priority level"
        },
        "examples": [
            "Family is my top priority above any career achievement",
            "Health and sobriety are the most important things in my life",
            "I value authenticity and honesty in all my relationships",
            "Community and giving back matter deeply to me",
            "Experiences matter more to me than material things",
            "I value freedom and independence in how I live my life",
            "Faith and spirituality are central to who I am",
            "I believe in leaving the world better than I found it"
        ]
    },
    "life_satisfaction": {
        "name": "Life Satisfaction",
        "description": "Overall life satisfaction, satisfaction by area, areas they want to improve.",
        "sub_entities": {
            "overall_satisfaction": "Overall life satisfaction score",
            "satisfaction_by_area": "Breakdown by area (career, family, health)",
            "areas_to_improve": "What they want to improve"
        },
        "examples": [
            "I'm satisfied with life overall about a 7 out of 10",
            "I'm very happy with my family life 9 out of 10 but work is a 4",
            "I want to improve my work fulfillment that's the biggest gap right now",
            "I'm dissatisfied with my social connections only a 5 out of 10",
            "My health and fitness are where I want them but career is lagging",
            "Overall I'd rate my life satisfaction at about 6 out of 10",
            "I'm content with most areas but financial stress brings everything down",
            "My relationships are strong but I feel unfulfilled professionally"
        ]
    }
}

ENTITY_GUIDANCE = {
    "personal_life": """
CRITICAL: Personal life = family situation, life stage, relationships, dependents.
✓ CORRECT: "I'm married with 2 kids", "I take care of my elderly mother"
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
CRITICAL: Current financial STATE, not career salary goals.
✓ CORRECT: "I have 45k in student debt", "I'm paycheck to paycheck"
✗ WRONG: "I want to earn 200k" (that's professional > professional_aspirations!)""",

    "personal_goals": """
CRITICAL: NON-CAREER life goals.
✓ CORRECT: "I want to run a marathon", "I want more family time"
✗ WRONG: "I want to become a VP" (that's professional > professional_aspirations!)""",

    "lifestyle_preferences": """
CRITICAL: How you want to LIVE—schedule, balance, flexibility.
✓ CORRECT: "I need flexible hours", "Remote work is non-negotiable"
✗ WRONG: "I prefer collaborative environments" (that's psychological > working_style_preferences!)""",

    "life_constraints": """
CRITICAL: Things that RESTRICT your career options.
✓ CORRECT: "I can't travel", "I can't relocate", "I can't take a pay cut"
✗ WRONG: "My in-laws help with childcare" (that's life_enablers, a positive!)""",

    "life_enablers": """
CRITICAL: Things that HELP and SUPPORT your career.
✓ CORRECT: "My spouse is supportive", "Free childcare from in-laws"
✗ WRONG: "I can't afford risks" (that's life_constraints or financial_situation!)""",
}

MULTI_LABEL_EXAMPLES = [
    {
        "message": "I'm married with a 1.5 year old and another on the way, my in-laws help with childcare 3 days a week which is a lifesaver, but I have about 40k in student debt and I'm the primary earner so I can't afford career risks right now",
        "entities": ["personal_life", "life_enablers", "financial_situation", "life_constraints"],
        "sub_entities": ["relationship_status", "children", "family_support", "enabler_type", "debt", "income_dependency", "constraint_type", "impact_on_career"]
    },
    {
        "message": "I have chronic back pain and anxiety that's managed with medication, I'm also 9 months sober and attend AA three times a week, overall my wellbeing is about a 5 out of 10 because the physical pain really gets me down some days",
        "entities": ["health_and_wellbeing"],
        "sub_entities": ["physical_health", "mental_health", "addictions_or_recovery", "overall_wellbeing"]
    },
    {
        "message": "My biggest goal right now is maintaining sobriety and being more present with my family, I won't work more than 45 hours a week and remote work is non-negotiable because I need to be near my AA meetings and my kids school",
        "entities": ["personal_goals", "lifestyle_preferences", "life_constraints"],
        "sub_entities": ["non_career_goals", "priority", "work_life_balance", "non_negotiables", "constraint_type", "description"]
    },
    {
        "message": "I just got married and we bought our first house in Austin, I'm very happy with my family life at 9 out of 10 but my career satisfaction is only about 4 because I feel stuck, family is my top priority but I need more fulfillment at work",
        "entities": ["major_life_events", "living_situation", "life_satisfaction", "personal_values"],
        "sub_entities": ["event_type", "impact", "location", "housing_type", "overall_satisfaction", "satisfaction_by_area", "areas_to_improve", "life_values"]
    },
    {
        "message": "I built an open source analytics dashboard that has 500 GitHub stars and I spend about 5 hours a week on it, I also have a vegetable garden that keeps me sane, I'm in my early 30s with no kids so I have a lot of flexibility",
        "entities": ["personal_projects", "personal_life"],
        "sub_entities": ["project_name", "project_description", "time_commitment", "life_stage", "children"]
    },
    {
        "message": "I can't relocate because my partner has tenure at the university, but living in a college town gives me access to great networking opportunities and my parents help with our mortgage so expenses are manageable even on my salary",
        "entities": ["life_constraints", "life_enablers", "living_situation", "financial_situation"],
        "sub_entities": ["constraint_type", "constraints", "enabler_type", "benefit_to_career", "location", "stability"]
    },
    {
        "message": "I value health and family above everything else, my goal is to run a marathon next year and I've been training 4 days a week, I'm pretty satisfied with life overall about a 7 out of 10 but I want to improve my career fulfillment",
        "entities": ["personal_values", "personal_goals", "life_satisfaction"],
        "sub_entities": ["life_values", "importance", "non_career_goals", "timeframe", "progress", "overall_satisfaction", "areas_to_improve"]
    },
    {
        "message": "I got laid off 6 months ago and it was devastating, I'm still processing it while dealing with financial stress because I burned through most of my savings, my spouse is supportive but we're now a single income family with two kids",
        "entities": ["major_life_events", "financial_situation", "life_enablers", "personal_life"],
        "sub_entities": ["event_type", "impact", "stress_level", "emergency_fund", "income_dependency", "enabler_type", "children", "dependents"]
    }
]

MESSAGE_GENERATION_SYSTEM_PROMPT = """You are an expert at generating natural personal context messages for career coaching. Always respond with valid JSON."""

SINGLE_LABEL_PROMPT_TEMPLATE = """Generate {batch_size} diverse, natural personal messages for career coaching specifically about {entity_name} > {sub_entity_name}.

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

MULTI_LABEL_PROMPT_TEMPLATE = """Generate {batch_size} natural, compound messages for career coaching that COMBINE multiple personal topics in a single message.

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
