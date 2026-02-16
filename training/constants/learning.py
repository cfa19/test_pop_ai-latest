"""
Learning Context Constants for Hierarchical Multi-Label Classification

Taxonomy: learning > entity > sub_entity
11 entities: current_skills, languages, education_history, learning_gaps, learning_aspirations,
             certifications, knowledge_areas, learning_preferences, learning_history, publications, academic_awards
"""

CONTEXT_NAME = "learning"

ENTITIES = {
    "current_skills": {
        "name": "Current Skills",
        "description": "Skills they currently have, proficiency level, years of experience, verification.",
        "sub_entities": {
            "skills": "Skills they currently have",
            "proficiency": "How proficient they are (beginner, intermediate, expert)",
            "experience": "Years of experience with the skill",
            "verification": "How the skill is verified (certs, portfolio, peer review)"
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
    "languages": {
        "name": "Languages",
        "description": "Languages spoken, proficiency level, language certifications and test scores.",
        "sub_entities": {
            "language": "Languages spoken",
            "proficiency": "Proficiency level (native, fluent, B1, B2, etc.)",
            "certifications": "Language test scores and certifications (TOEFL, IELTS, DELE)"
        },
        "examples": [
            "I speak Spanish at B1 level",
            "I'm fluent in Mandarin Chinese its my native language",
            "I got a 110 on the TOEFL",
            "English is my second language I'm fluent but not native",
            "I speak French and Italian at a conversational level",
            "I passed the JLPT N2 for Japanese",
            "I'm bilingual English and Portuguese",
            "I'm learning German currently at A2 level",
            "My IELTS score is 8.0 overall",
            "I speak 4 languages English Spanish French and a bit of Arabic"
        ]
    },
    "education_history": {
        "name": "Education History",
        "description": "Degrees earned, schools attended, field of study, GPA, graduation dates.",
        "sub_entities": {
            "degrees": "Degrees earned (BS, MS, MBA, PhD, etc.)",
            "institutions": "Schools and universities attended",
            "field_of_study": "Major, minor, or specialization",
            "gpa": "Academic performance",
            "graduation_date": "When they graduated"
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
            "earned_certs": "Certifications they already have",
            "issue_date": "When they got the certification",
            "expiry_date": "When it expires",
            "status": "Active, expired, or pending renewal"
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
    "knowledge_areas": {
        "name": "Knowledge Areas",
        "description": "Broader domains of expertise and knowledge.",
        "sub_entities": {
            "expertise_domains": "Broader knowledge areas and domains of expertise"
        },
        "examples": [
            "I'm knowledgeable about fintech and payments",
            "I have deep expertise in platform architecture",
            "My knowledge base is mostly in healthcare IT systems",
            "I know a lot about supply chain management and logistics",
            "I have broad knowledge of the SaaS industry",
            "I'm an expert in regulatory compliance for financial services",
            "I have deep domain knowledge in renewable energy",
            "I know the edtech space really well inside and out"
        ]
    },
    "learning_preferences": {
        "name": "Learning Preferences",
        "description": "How they prefer to learn: formats, pace, budget, time available.",
        "sub_entities": {
            "preferred_formats": "How they like to learn (books, videos, hands-on, courses)",
            "pace": "Fast or slow learner, learning speed",
            "budget": "Learning budget available",
            "time_available": "Hours per week available for learning"
        },
        "examples": [
            "I learn best through hands-on projects",
            "I prefer books over video courses",
            "I can dedicate 10 hours per week to learning",
            "I have about 2000 dollars per year for courses and training",
            "I'm a fast learner I pick things up quickly",
            "I like structured online courses with deadlines",
            "I learn best by doing not by watching lectures",
            "I only have about 5 hours a week for studying so I need efficient resources"
        ]
    },
    "learning_history": {
        "name": "Learning History",
        "description": "Past courses taken, books read, bootcamps completed, learning outcomes.",
        "sub_entities": {
            "past_courses": "Courses and bootcamps completed",
            "books": "Books they've read for professional development",
            "outcomes": "What they learned and how it helped"
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
    },
    "publications": {
        "name": "Publications",
        "description": "Articles, papers, blog posts, talks they've written or given.",
        "sub_entities": {
            "publications": "Articles, papers, blog posts, conference talks authored"
        },
        "examples": [
            "I wrote an article about platform products on Medium",
            "I published a paper on machine learning in a peer-reviewed journal",
            "I maintain a technical blog about AI and it gets 5k monthly readers",
            "I gave a talk at PyCon about testing best practices",
            "I write a weekly newsletter about product management",
            "I co-authored a paper on distributed systems",
            "I published a book chapter on healthcare informatics",
            "I have 3 patents in computer vision technology"
        ]
    },
    "academic_awards": {
        "name": "Academic Awards",
        "description": "Academic honors and recognitions from educational institutions.",
        "sub_entities": {
            "academic_awards": "Academic honors, dean's list, scholarships, honors societies"
        },
        "examples": [
            "I made the Dean's List every semester in college",
            "I graduated Summa Cum Laude from my university",
            "I received the Outstanding Student Award in engineering",
            "I was inducted into Phi Beta Kappa",
            "I got a full scholarship to my MBA program",
            "I won the best thesis award in my department",
            "I was valedictorian of my high school class",
            "I received the National Merit Scholarship"
        ]
    }
}

ENTITY_GUIDANCE = {
    "current_skills": """
CRITICAL: Skills they HAVE right now, not skills they want to learn.
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
CRITICAL: Certs they ALREADY HAVE.
✓ CORRECT: "I'm AWS certified", "My PMP expires next year"
✗ WRONG: "I want to get PMP certified" (that's learning_aspirations > certification_aspirations!)""",

    "academic_awards": """
CRITICAL: Awards from EDUCATIONAL institutions.
✓ CORRECT: "Dean's List", "Summa Cum Laude", "Best Thesis Award"
✗ WRONG: "Employee of the Month" (that's professional > awards!)""",
}

MULTI_LABEL_EXAMPLES = [
    {
        "message": "I know Python at an intermediate level and I have 5 years of data analysis experience, but I need to learn machine learning to transition into a data science role, I'm taking Andrew Ng's course on Coursera and I'm about 40 percent done",
        "entities": ["current_skills", "learning_gaps", "learning_aspirations", "learning_history"],
        "sub_entities": ["skills", "proficiency", "experience", "skill_gaps", "skill_aspirations", "past_courses"]
    },
    {
        "message": "I have a BS in Computer Science from Berkeley with a 3.8 GPA, I graduated Summa Cum Laude and made the Dean's List, now I want to get an MBA from Wharton to transition into product management",
        "entities": ["education_history", "academic_awards", "learning_aspirations"],
        "sub_entities": ["degrees", "institutions", "gpa", "academic_awards", "education_aspirations"]
    },
    {
        "message": "I'm AWS Solutions Architect certified and studying for the Google Cloud cert, I have deep expertise in cloud architecture and distributed systems, I learn best through hands-on labs and I dedicate about 10 hours per week to studying",
        "entities": ["certifications", "learning_aspirations", "knowledge_areas", "learning_preferences"],
        "sub_entities": ["earned_certs", "certification_aspirations", "expertise_domains", "preferred_formats", "time_available"]
    },
    {
        "message": "I speak Spanish at B1 level and I'm fluent in English, I scored 110 on the TOEFL, I need better language skills for the international PM role I want so I'm taking an intensive Spanish course 3 evenings a week",
        "entities": ["languages", "learning_gaps", "learning_aspirations"],
        "sub_entities": ["language", "proficiency", "certifications", "skill_gaps", "skill_aspirations"]
    },
    {
        "message": "I published 3 papers on machine learning and I maintain a technical blog with 10k monthly readers, I also have deep expertise in NLP and computer vision, I read about 30 technical books a year to stay current",
        "entities": ["publications", "knowledge_areas", "learning_history"],
        "sub_entities": ["publications", "expertise_domains", "books"]
    },
    {
        "message": "I completed a 12-week bootcamp at General Assembly and got the Google UX Design certificate, now I'm studying for the CSPO certification and want to learn Figma at an advanced level, I have about 2000 a year for training",
        "entities": ["learning_history", "certifications", "learning_aspirations", "learning_preferences"],
        "sub_entities": ["past_courses", "outcomes", "earned_certs", "certification_aspirations", "skill_aspirations", "budget"]
    },
    {
        "message": "I'm a fast learner who picks things up in about half the time of my peers, I prefer hands-on projects over lectures, my biggest knowledge gap right now is cloud architecture which I need for the Solutions Architect role",
        "entities": ["learning_preferences", "learning_gaps"],
        "sub_entities": ["pace", "preferred_formats", "knowledge_gaps", "skill_gaps"]
    },
    {
        "message": "I have expert-level Python and SQL skills verified by my 10 years of experience, I'm knowledgeable about fintech and payments, but I lack experience with blockchain technology which is blocking me from a role at Coinbase",
        "entities": ["current_skills", "knowledge_areas", "learning_gaps"],
        "sub_entities": ["skills", "proficiency", "experience", "expertise_domains", "knowledge_gaps"]
    }
]

MESSAGE_GENERATION_SYSTEM_PROMPT = """You are an expert at generating natural learning context messages for career coaching. Generate messages that sound like real people talking about their skills, education, learning, and knowledge. Cover DIVERSE fields and learning styles. Always respond with valid JSON."""

SINGLE_LABEL_PROMPT_TEMPLATE = """Generate {batch_size} diverse, natural learning messages for career coaching specifically about {entity_name} > {sub_entity_name}.

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

MULTI_LABEL_PROMPT_TEMPLATE = """Generate {batch_size} natural, compound messages for career coaching that COMBINE multiple learning topics in a single message.

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
