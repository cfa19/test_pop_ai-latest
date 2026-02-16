"""
Psychological Context Constants for Hierarchical Multi-Label Classification

Taxonomy: psychological > entity > sub_entity
10 entities: personality_profile, values, motivations, working_style_preferences,
             confidence_and_self_perception, career_decision_making_style,
             work_environment_preferences, stress_and_coping, emotional_intelligence, growth_mindset
"""

CONTEXT_NAME = "psychological"

ENTITIES = {
    "personality_profile": {
        "name": "Personality Profile",
        "description": "Personality type, traits, behavioral tendencies, self-description.",
        "sub_entities": {
            "personality_type": "MBTI, Big Five, Enneagram, etc.",
            "traits": "Key personality traits",
            "self_description": "How they describe themselves"
        },
        "examples": [
            "I'm an INTJ and that pretty much sums me up",
            "I'm introverted and analytical",
            "I'm a perfectionist which is both a strength and weakness",
            "I'm naturally curious and love exploring new ideas",
            "I'm an extrovert who thrives on interaction and collaboration",
            "I'm detail-oriented and methodical in my approach to everything",
            "I'd describe myself as a big picture thinker not a detail person",
            "I'm an Enneagram 3 the achiever type super driven",
            "I'm empathetic sometimes too much so it affects my decision making",
            "I'm a mix of creative and analytical I switch between both modes"
        ]
    },
    "values": {
        "name": "Values",
        "description": "Professional values: what matters at work, value priorities.",
        "sub_entities": {
            "professional_values": "What matters to them at work (autonomy, impact, stability, etc.)",
            "priorities": "Value priorities and trade-offs"
        },
        "examples": [
            "I value autonomy and impact above everything at work",
            "Work-life balance is my top professional priority",
            "I care deeply about mission alignment in my work",
            "I need to feel like my work makes a real difference",
            "Stability matters more to me than excitement",
            "I value transparency and honesty in leadership",
            "Career growth opportunity matters more than current salary",
            "I prioritize team culture over individual achievement",
            "I need intellectual challenge in my work or I get bored",
            "Ethics and integrity are non-negotiable for me"
        ]
    },
    "motivations": {
        "name": "Motivations",
        "description": "What motivates and demotivates them: intrinsic, extrinsic, and negative.",
        "sub_entities": {
            "intrinsic_motivations": "Internal motivators (purpose, mastery, autonomy, curiosity)",
            "extrinsic_motivations": "External motivators (money, status, recognition, promotion)",
            "demotivators": "What kills their motivation"
        },
        "examples": [
            "I'm motivated by solving hard problems that's what gets me out of bed",
            "Money isn't my main motivator I'd take less for meaningful work",
            "Micromanagement completely kills my motivation I shut down",
            "Recognition and praise really fuel me I need to feel appreciated",
            "I'm driven by the desire to build something from scratch",
            "Repetitive work drains me I need variety and novelty",
            "I'm motivated by helping other people succeed and grow",
            "Getting promoted motivates me a lot I'm competitive",
            "Bureaucracy and politics are the biggest demotivators for me",
            "I thrive when I have a clear sense of purpose in my work"
        ]
    },
    "working_style_preferences": {
        "name": "Working Style Preferences",
        "description": "How they prefer to work, collaborate, make decisions, and communicate.",
        "sub_entities": {
            "work_style": "How they prefer to work (independent, structured, flexible)",
            "collaboration_style": "How they work with others",
            "decision_making": "How they make decisions (data-driven, intuitive, consensus)",
            "communication_style": "How they communicate (direct, diplomatic, written, verbal)"
        },
        "examples": [
            "I work best independently with clear goals and minimal oversight",
            "I like collaborative brainstorming sessions early in a project",
            "I make decisions quickly based on data not gut feeling",
            "I prefer written communication over meetings honestly",
            "I'm very direct in my communication style sometimes too blunt",
            "I need structure and clear processes to do my best work",
            "I thrive in ambiguity and figure things out as I go",
            "I prefer async communication and hate unnecessary meetings",
            "I'm a consensus builder I like to get everyone aligned before deciding",
            "I work in bursts of intense focus followed by rest periods"
        ]
    },
    "confidence_and_self_perception": {
        "name": "Confidence & Self-Perception",
        "description": "Confidence levels, imposter syndrome, self-doubt, self-talk, validation needs, confidence-building strategies.",
        "sub_entities": {
            "confidence_levels": "Overall and domain-specific confidence levels",
            "imposter_syndrome_and_doubt": "Imposter feelings, self-doubt, comparison patterns, self-efficacy, resilience",
            "self_talk_and_validation": "Inner critic, self-compassion, need for external validation, reaction to criticism/praise",
            "confidence_building_strategies": "What builds confidence, what hurts it, current efforts, coping strategies"
        },
        "examples": [
            "I'm feeling pretty confident lately about 7 out of 10",
            "My confidence dropped after the project failure",
            "I'm very confident technically 8 out of 10 but not socially",
            "I have moderate imposter syndrome especially around senior leaders",
            "I feel like a fraud when presenting to the executive team",
            "My inner critic is very harsh I beat myself up over small mistakes",
            "I rely too much on other peoples approval for my self-worth",
            "Keeping a wins journal has really helped build my confidence",
            "I dismiss compliments and focus on the negative feedback",
            "Working with an executive coach on building confidence has been great",
            "I bounce back quickly from setbacks I'm pretty resilient",
            "I constantly compare myself to peers who seem more successful"
        ]
    },
    "career_decision_making_style": {
        "name": "Career Decision Making Style",
        "description": "How they make career decisions, what influences them, confidence in decisions.",
        "sub_entities": {
            "decision_style": "Analytical, intuitive, or mixed decision-making approach",
            "decision_factors": "What influences career decisions (data, gut, advice, values)",
            "decision_confidence": "How confident they are in career decisions"
        },
        "examples": [
            "I make career decisions based on data and spreadsheets",
            "I trust my gut when choosing jobs I know within 5 minutes",
            "I struggle with big career decisions I overthink everything",
            "I always consult my mentor before making a big career move",
            "I make a pros and cons list for every career decision",
            "I let my values guide me when I'm stuck on a career choice",
            "I'm indecisive about career moves and it's held me back",
            "I'm very confident in my career decisions once I commit I don't look back"
        ]
    },
    "work_environment_preferences": {
        "name": "Work Environment Preferences",
        "description": "What work environment they thrive in, stressors, and energizers at work.",
        "sub_entities": {
            "ideal_environment": "What environment they thrive in",
            "stressors": "What stresses them at work",
            "energizers": "What gives them energy at work"
        },
        "examples": [
            "I thrive in fast-paced high-growth environments",
            "Open offices stress me out I need quiet to focus",
            "I love working on collaborative cross-functional projects",
            "I need a calm structured environment to do my best work",
            "Ambiguity and constant change energize me I love startups",
            "I hate corporate politics it's the worst part of any job",
            "I'm energized by building products from zero to one",
            "I need a small team where everyone knows each other",
            "I thrive when theres a healthy amount of pressure and deadlines",
            "I'm drained by large meetings and unnecessary process"
        ]
    },
    "stress_and_coping": {
        "name": "Stress & Coping",
        "description": "Current stress level, triggers, coping strategies, and their effectiveness.",
        "sub_entities": {
            "stress_level": "Current stress level (scale or description)",
            "stress_triggers": "What causes stress at work",
            "coping_strategies": "How they cope with stress",
            "effectiveness": "What works and what doesn't work"
        },
        "examples": [
            "I'm pretty stressed right now about 7 out of 10",
            "Tight deadlines stress me out more than anything",
            "Exercise helps me manage stress I run 3 times a week",
            "I meditate every morning for 15 minutes it helps a lot",
            "I tend to overwork when stressed which makes it worse",
            "My stress level is low right now about a 3 out of 10",
            "Conflict with colleagues is my biggest stress trigger",
            "I cope with stress by talking to my therapist weekly",
            "I've tried meditation but it doesn't work for me honestly",
            "Taking breaks and going for walks is my most effective stress relief"
        ]
    },
    "emotional_intelligence": {
        "name": "Emotional Intelligence",
        "description": "Self-awareness, empathy, emotional regulation abilities.",
        "sub_entities": {
            "self_awareness": "Understanding own emotions and their impact",
            "empathy": "Understanding and responding to others' emotions",
            "emotional_regulation": "Managing own emotions effectively"
        },
        "examples": [
            "I'm very self-aware about my emotions and triggers",
            "I'm working on being more empathetic with my team",
            "I can read a room well I pick up on unspoken dynamics",
            "I struggle to regulate my emotions when I'm frustrated",
            "I'm good at empathy but it sometimes becomes people-pleasing",
            "I can stay calm under pressure that's one of my strengths",
            "I know when I'm getting stressed before it affects my work",
            "I have a hard time not taking criticism personally"
        ]
    },
    "growth_mindset": {
        "name": "Growth Mindset",
        "description": "Fixed vs growth mindset, beliefs about talent, approach to challenges.",
        "sub_entities": {
            "mindset_level": "Fixed, growth, or mixed mindset",
            "beliefs_about_talent": "Whether abilities are innate or developed",
            "approach_to_challenges": "How they approach challenges and failures"
        },
        "examples": [
            "I believe abilities can be developed with effort and practice",
            "I see failures as learning opportunities not defeats",
            "I love challenging myself with things outside my comfort zone",
            "I sometimes fall into a fixed mindset when things get hard",
            "I believe talent is 20 percent innate and 80 percent hard work",
            "I embrace challenges they make me stronger",
            "I'm working on not giving up when things get difficult",
            "I get frustrated when I don't pick things up quickly"
        ]
    }
}

ENTITY_GUIDANCE = {
    "values": """
CRITICAL: PROFESSIONAL values—what matters at WORK.
✓ CORRECT: "I value autonomy", "Mission alignment matters most"
✗ WRONG: "Family is my top priority" (that's personal > personal_values!)
✗ WRONG: "I'm motivated by money" (that's motivations!)""",

    "motivations": """
CRITICAL: What DRIVES or KILLS motivation.
✓ CORRECT: "I'm motivated by solving problems", "Micromanagement kills my motivation"
✗ WRONG: "I value autonomy" (that's values—stable priorities, not drivers!)
✗ WRONG: "I'm stressed by deadlines" (that's stress_and_coping!)""",

    "working_style_preferences": """
CRITICAL: HOW they work, not WHERE or what environment.
✓ CORRECT: "I prefer async communication", "I make data-driven decisions"
✗ WRONG: "I thrive in fast-paced environments" (that's work_environment_preferences!)
✗ WRONG: "I want remote work" (that's professional > desired_work_environment or personal > lifestyle_preferences!)""",

    "confidence_and_self_perception": """
CRITICAL: MULTI-LABEL entity. Confidence, imposter syndrome, self-talk, validation needs.
- confidence_levels: "I'm 7/10 confident", "My confidence dropped"
- imposter_syndrome_and_doubt: "I feel like a fraud", "I compare myself to peers"
- self_talk_and_validation: "My inner critic is harsh", "I need external approval"
- confidence_building_strategies: "Wins journal helps", "Working with a coach"
A single message can touch all four.""",

    "work_environment_preferences": """
CRITICAL: What ENVIRONMENT they thrive in—not working style.
✓ CORRECT: "I thrive in startups", "Open offices stress me out"
✗ WRONG: "I prefer async work" (that's working_style_preferences!)
✗ WRONG: "I want remote work" (that's professional > desired_work_environment!)""",

    "stress_and_coping": """
CRITICAL: Stress and how they HANDLE it.
✓ CORRECT: "I'm stressed 7/10", "Exercise helps me cope"
✗ WRONG: "I have anxiety" (that's personal > health_and_wellbeing > mental_health!)
✗ WRONG: "I'm burned out" (borderline—burnout from work = stress_and_coping, clinical burnout = personal > health)""",
}

MULTI_LABEL_EXAMPLES = [
    {
        "message": "I'm an INTJ perfectionist who values autonomy above everything, I'm motivated by solving hard problems but micromanagement completely kills my drive, I work best independently with clear goals and minimal oversight",
        "entities": ["personality_profile", "values", "motivations", "working_style_preferences"],
        "sub_entities": ["personality_type", "traits", "professional_values", "intrinsic_motivations", "demotivators", "work_style"]
    },
    {
        "message": "My confidence is about 6 out of 10, I have moderate imposter syndrome especially around senior leaders, my inner critic is very harsh and I rely too much on external validation, working with a coach has been helping though",
        "entities": ["confidence_and_self_perception"],
        "sub_entities": ["confidence_levels", "imposter_syndrome_and_doubt", "self_talk_and_validation", "confidence_building_strategies"]
    },
    {
        "message": "I thrive in fast-paced startup environments where I can build from scratch, tight deadlines stress me but exercise and meditation help me cope, I believe challenges make me stronger and I see failures as learning opportunities",
        "entities": ["work_environment_preferences", "stress_and_coping", "growth_mindset"],
        "sub_entities": ["ideal_environment", "energizers", "stress_triggers", "coping_strategies", "approach_to_challenges", "mindset_level"]
    },
    {
        "message": "I make career decisions based on data and always consult my mentor, I'm very self-aware about my emotions and I can read a room well but I struggle with not taking criticism personally and that affects my confidence",
        "entities": ["career_decision_making_style", "emotional_intelligence", "confidence_and_self_perception"],
        "sub_entities": ["decision_style", "decision_factors", "self_awareness", "empathy", "emotional_regulation", "confidence_levels"]
    },
    {
        "message": "I'm an extrovert who loves collaborative brainstorming but I prefer direct written communication, I'm motivated by recognition and building things from scratch, and I need a small team environment where everyone knows each other",
        "entities": ["personality_profile", "working_style_preferences", "motivations", "work_environment_preferences"],
        "sub_entities": ["traits", "self_description", "collaboration_style", "communication_style", "extrinsic_motivations", "intrinsic_motivations", "ideal_environment"]
    },
    {
        "message": "I'm pretty stressed right now about 7 out of 10, conflict with colleagues triggers it most, I try to cope by exercising and talking to my therapist but sometimes I fall into a fixed mindset and just want to give up",
        "entities": ["stress_and_coping", "growth_mindset"],
        "sub_entities": ["stress_level", "stress_triggers", "coping_strategies", "effectiveness", "mindset_level", "approach_to_challenges"]
    },
    {
        "message": "I value mission alignment and intellectual challenge at work, I trust my gut for career decisions and I'm very confident once I commit, I believe talent is mostly developed through hard work and I embrace challenges",
        "entities": ["values", "career_decision_making_style", "growth_mindset"],
        "sub_entities": ["professional_values", "priorities", "decision_style", "decision_confidence", "beliefs_about_talent", "approach_to_challenges"]
    },
    {
        "message": "I'm empathetic sometimes too much so which leads to people-pleasing, I need external validation more than I'd like to admit, this affects my confidence which dropped after a project failure, I'm trying meditation and coaching to help",
        "entities": ["emotional_intelligence", "confidence_and_self_perception", "stress_and_coping"],
        "sub_entities": ["empathy", "emotional_regulation", "self_talk_and_validation", "confidence_levels", "confidence_building_strategies", "coping_strategies"]
    }
]

MESSAGE_GENERATION_SYSTEM_PROMPT = """You are an expert at generating natural psychological context messages for career coaching. Generate messages that sound like real people talking about their personality, values, motivations, confidence, stress, and mindset. Be authentic and sometimes vulnerable. Always respond with valid JSON."""

SINGLE_LABEL_PROMPT_TEMPLATE = """Generate {batch_size} diverse, natural psychological messages for career coaching specifically about {entity_name} > {sub_entity_name}.

Context: Psychological
Entity: {entity_name}
Sub-entity: {sub_entity_name} - {sub_entity_description}
{entity_guidance}

Requirements:
1. Each message MUST be specifically about {sub_entity_name}
2. Vary WIDELY in length:
   - 20% very short (5-12 words): "I'm a perfectionist"
   - 30% short-medium (13-25 words)
   - 30% medium-long (26-50 words)
   - 20% long paragraphs (51-70 words)
3. Cover DIVERSE personality types and psychological profiles
4. Include REALISTIC TYPOS in ~15%: "im", "becuase", "confindence", "motivaton"
5. Vary formulation: "I...", "My...", "I tend to...", fragments
6. Be authentic and vulnerable—real psychological self-reflection

Example messages for {entity_name}:
{entity_examples}

Generate {batch_size} unique messages as JSON:
{{"messages": ["message1", "message2", ...]}}"""

MULTI_LABEL_PROMPT_TEMPLATE = """Generate {batch_size} natural, compound messages for career coaching that COMBINE multiple psychological topics in a single message.

Each message should naturally touch on {num_labels} or more of these sub-entities: {sub_entity_list}

Requirements:
1. Each message MUST mention at least {num_labels} different sub-entities
2. Length: 30-70 words (paragraphs)
3. Natural and conversational, authentically self-reflective
4. Cover DIVERSE psychological profiles
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
