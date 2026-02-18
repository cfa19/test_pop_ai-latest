"""
Social Context Constants for Hierarchical Multi-Label Classification

Taxonomy: social > entity > sub_entity
4 entities: mentors, mentees, professional_network, networking
"""

CONTEXT_NAME = "social"

ENTITIES = {
    "mentors": {
        "name": "Mentors",
        "description": "Current or past mentors, coaches, advisors who guide career development.",
        "sub_entities": {
            "mentors": "Mentor identity, organization, role, meeting frequency, guidance areas, and impact"
        },
        "examples": [
            "I have a mentor who's a VP at Stripe she's amazing",
            "My mentor meets with me monthly to review my career plan",
            "Sarah helps me with leadership skills and executive presence",
            "I've had a great coach who helped me transition into tech",
            "My manager at my last job was an incredible guide and advisor",
            "I don't currently have a mentor but I really need one",
            "My professor in college was instrumental in helping me find my path",
            "I had a senior colleague who took me under their wing and taught me everything",
            "My mentor is a retired CEO he gives me brutally honest feedback",
            "I meet with my career coach biweekly and it's been transformative"
        ]
    },
    "mentees": {
        "name": "Mentees",
        "description": "People they mentor, guide, or support in their career development.",
        "sub_entities": {
            "mentees": "Mentee identity, background, guidance provided, and progress"
        },
        "examples": [
            "I'm mentoring a junior PM who's transitioning from design",
            "I help new PMs with product strategy and stakeholder management",
            "I mentor 3 people informally at my company",
            "I'm mentoring a college student who wants to get into tech",
            "I volunteer as a mentor at a coding bootcamp for career changers",
            "One of my mentees just got promoted which felt amazing",
            "I mentor a young nurse who reminds me of myself when I started",
            "I'm guiding two junior engineers on my team through their first year"
        ]
    },
    "professional_network": {
        "name": "Professional Network",
        "description": "Professional connections, peers, colleagues, communities, and groups.",
        "sub_entities": {
            "connections": "People in their network: peers, colleagues, relationship strength, interaction frequency",
            "communities": "Professional communities, groups, membership status, engagement level"
        },
        "examples": [
            "I have an accountability partner for my career goals we check in weekly",
            "I collaborate with other PMs in my Reforge cohort",
            "I know the Director of Product at Airbnb from a conference",
            "I have about 450 LinkedIn connections mostly in tech",
            "I haven't talked to John in 6 months should really reconnect",
            "I'm in the Reforge community and it's been super valuable",
            "I attend SF Product Managers meetups monthly",
            "I'm active in a Slack group for PMs with about 2000 members",
            "I'm part of a mastermind group with 5 other founders",
            "My professional network is pretty weak honestly I need to build it up"
        ]
    },
    "networking": {
        "name": "Networking",
        "description": "Professional networking activities, goals, and preferences.",
        "sub_entities": {
            "networking_activities": "Events attended, coffee chats, conferences",
            "networking_goals": "People they want to meet, events to attend, strategy",
            "networking_preferences": "Preferred formats, energy impact, networking style"
        },
        "examples": [
            "I attended ProductCon last month and met some great people",
            "I had coffee with a PM at Google last week super helpful",
            "I'm going to Stripe Sessions in March to network",
            "I want to meet more CPOs and VPs of Product",
            "I'm planning to attend 3 conferences this year for networking",
            "I prefer 1-on-1 coffee chats over big networking events",
            "Networking drains me I'm an introvert but I know I need to do it",
            "I've been cold emailing VPs on LinkedIn with a 15 percent response rate",
            "I want to build relationships with CTOs at AI companies",
            "I do my best networking through warm introductions not cold outreach"
        ]
    }
}

ENTITY_GUIDANCE = {
    "mentors": """
CRITICAL: People who GUIDE and ADVISE them. Uses mentors schema with fields: name, role, organization, frequency, guidanceAreas, impact.
✓ CORRECT: "My mentor at Stripe helps me with leadership", "I meet with my coach monthly"
✗ WRONG: "I mentor junior PMs" (that's mentees—THEY are the mentor!)
✗ WRONG: "I know a VP at Stripe" (that's professional_network unless that person mentors them!)""",

    "mentees": """
CRITICAL: People THEY guide and mentor. Uses mentees schema with fields: name, background, guidanceAreas, progress.
✓ CORRECT: "I mentor a junior PM", "I help new engineers on my team"
✗ WRONG: "My mentor helps me" (that's mentors—they are the MENTEE!)""",

    "professional_network": """
CRITICAL: Two sub_entities: connections (people) and communities (groups). NOT mentoring relationships.
✓ CORRECT: "I have 500 LinkedIn connections", "I'm in a Slack group for PMs"
✗ WRONG: "My mentor is a VP" (that's mentors!)
✗ WRONG: "I attended a conference" (that's networking > networking_activities!)""",

    "networking": """
CRITICAL: MULTI-LABEL entity. Activities, goals, preferences for professional networking.
- networking_activities: "Attended ProductCon", "Had coffee with a PM"
- networking_goals: "I want to meet CTOs", "Planning to attend 3 conferences"
- networking_preferences: "I prefer 1-on-1 over groups", "Networking drains me"
A message like "I want to meet CTOs at AI conferences, preferably in small settings" touches all three.""",

}

MULTI_LABEL_EXAMPLES = [
    {
        "message": (
            "I have a mentor who's a VP at Stripe and she meets with me monthly "
            "to work on my executive presence, I also mentor two junior PMs at my "
            "company helping them with product strategy and stakeholder management"
        ),
        "entities": ["mentors", "mentees"],
        "sub_entities": ["mentors", "mentees"]
    },
    {
        "message": (
            "I attended ProductCon last month and met some great people including "
            "a Director at Airbnb, I'm planning to attend 3 more conferences this "
            "year and want to meet more CPOs, I prefer smaller networking events "
            "over huge conferences"
        ),
        "entities": ["networking", "professional_network"],
        "sub_entities": ["networking_activities", "networking_goals", "networking_preferences", "connections"]
    },
    {
        "message": (
            "I'm very active in the Reforge community and a PM Slack group "
            "with 2000 members which has been great for my career"
        ),
        "entities": ["professional_network"],
        "sub_entities": ["communities"]
    },
    {
        "message": (
            "I'm mentoring a career changer from teaching to tech and she's doing "
            "great, my own mentor is a retired CEO who gives me brutally honest "
            "feedback biweekly, and I just joined a founders mastermind group "
            "that meets every Friday"
        ),
        "entities": ["mentees", "mentors", "professional_network"],
        "sub_entities": ["mentees", "mentors", "communities"]
    },
    {
        "message": (
            "I want to meet more VPs at fintech companies so I've been cold "
            "emailing on LinkedIn with about 15 percent response rate, I also "
            "had coffee with a PM at Google last week"
        ),
        "entities": ["networking"],
        "sub_entities": ["networking_goals", "networking_activities"]
    },
    {
        "message": (
            "My professional network is pretty weak honestly I only have about "
            "200 LinkedIn connections, I don't have a mentor yet and networking "
            "events drain me because I'm an introvert but I know I need to put "
            "myself out there more"
        ),
        "entities": ["professional_network", "mentors", "networking"],
        "sub_entities": ["connections", "mentors", "networking_preferences"]
    },
    {
        "message": (
            "I have a fantastic mentor who helped me land my current role, I also "
            "mentor a junior designer who's transitioning to product management "
            "and she's making great progress"
        ),
        "entities": ["mentors", "mentees"],
        "sub_entities": ["mentors", "mentees"]
    },
    {
        "message": (
            "I'm part of a startup founders community of about 50 people and I "
            "meet with my accountability partner weekly, I'm planning to attend "
            "Stripe Sessions and AWS re:Invent to expand my network and meet "
            "potential investors"
        ),
        "entities": ["professional_network", "networking"],
        "sub_entities": ["communities", "connections", "networking_activities", "networking_goals"]
    }
]

MESSAGE_GENERATION_SYSTEM_PROMPT = (
    "You are an expert at generating natural social context messages for career "
    "coaching. Generate messages that sound like real people talking about their "
    "professional relationships, networks, mentors, and communities. "
    "Cover DIVERSE professions and networking styles. "
    "Always respond with valid JSON."
)

SINGLE_LABEL_PROMPT_TEMPLATE = """\
Generate {batch_size} diverse, natural social context messages for career \
coaching specifically about {entity_name} > {sub_entity_name}.

Context: Social
Entity: {entity_name}
Sub-entity: {sub_entity_name} - {sub_entity_description}
{entity_guidance}

Requirements:
1. Each message MUST be specifically about {sub_entity_name}
2. Vary WIDELY in length:
   - 20% very short (5-12 words): "I have a great mentor"
   - 30% short-medium (13-25 words)
   - 30% medium-long (26-50 words)
   - 20% long paragraphs (51-70 words)
3. Cover DIVERSE professions and relationship types
4. Include REALISTIC TYPOS in ~15%: "im", "becuase", "mentro", "refernce"
5. Vary formulation: "I...", "My...", "We...", fragments
6. Be authentic—real networking talk

Example messages for {entity_name}:
{entity_examples}

Generate {batch_size} unique messages as JSON:
{{"messages": ["message1", "message2", ...]}}"""

MULTI_LABEL_PROMPT_TEMPLATE = """\
Generate {batch_size} natural, compound messages for career coaching \
that COMBINE multiple social topics in a single message.

Each message should naturally touch on {num_labels} or more of these sub-entities: {sub_entity_list}

Requirements:
1. Each message MUST mention at least {num_labels} different sub-entities
2. Length: 30-70 words (paragraphs)
3. Natural and conversational
4. Cover DIVERSE professions and networking situations
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
