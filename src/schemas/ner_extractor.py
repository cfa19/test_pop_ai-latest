"""
Lightweight NER Extractor for Runtime Use

Provides fast entity span extraction for LangGraph workflow integration.
Uses regex patterns and compiled phrase matchers — no spaCy, no LLM dependencies.

Usage:
    from src.schemas.ner_extractor import extract_entity_spans

    spans = extract_entity_spans("I'm a Senior PM at Google for 3 years", "work_history")
    # [{"start": 6, "end": 15, "label": "ROLE"}, ...]
"""

import re

# Mapping from entity types to span labels (inlined from entity_prompts.py)
SPAN_LABELS: dict[str, list[str]] = {
    # Professional context
    "work_history": ["ROLE", "ORG", "DURATION", "ACHIEVEMENT"],
    "professional_achievements": ["ACHIEVEMENT", "ORG", "ROLE"],
    "professional_aspirations": ["ROLE", "ORG", "TRANSITION_INTENT", "TIMELINE"],
    # Learning context
    "knowledge_and_credentials": ["SKILL", "DURATION", "ORG", "DEGREE", "CERTIFICATION"],
    "languages": ["LANGUAGE"],
    "learning_agenda": ["SKILL", "CONSTRAINT_SIGNAL", "TRANSITION_INTENT", "LEARNING_PREFERENCE"],
    "workplace_challenges": ["STRESS_SIGNAL", "BURNOUT_SIGNAL", "EMOTIONAL_SIGNAL", "CONSTRAINT_SIGNAL", "ROLE", "ORG"],
    "job_search_status": ["JOB_SEARCH_STATUS", "ROLE", "ORG", "TIMELINE"],
    # Psychological context
    "mindset_and_values": ["PERSONALITY_TRAIT", "VALUE", "MOTIVATION", "EMOTIONAL_SIGNAL"],
    "working_style_preferences": ["WORK_STYLE"],
    "emotional_state": ["EMOTIONAL_SIGNAL", "CONFIDENCE_SIGNAL", "BURNOUT_SIGNAL", "STRESS_SIGNAL", "ENERGY_SIGNAL", "CONSTRAINT_SIGNAL"],
    # Social context
    "mentorship": ["ROLE", "ORG"],
    "recommendations": ["ROLE", "ORG"],
    "network_and_networking": ["ORG", "ROLE", "TRANSITION_INTENT"],
    # Personal context
    "life_situation": ["LIFE_STAGE", "LOCATION", "CONSTRAINT_SIGNAL", "FINANCIAL_CONSTRAINT", "TIMELINE", "LIFESTYLE_PREFERENCE"],
    "health_and_wellbeing": ["HEALTH_SIGNAL", "BURNOUT_SIGNAL"],
    "personal_projects": ["SKILL", "ORG", "ROLE", "DURATION"],
    "personal_priorities": ["LIFE_GOAL", "TIMELINE", "VALUE", "LIFESTYLE_PREFERENCE"],
}


# =============================================================================
# PHRASE DICTIONARY  (curated for runtime; no ESCO dataset required)
# =============================================================================

PHRASE_DICT: dict[str, list[str]] = {
    "ROLE": [
        # C-suite
        "chief executive officer", "chief technology officer", "chief operating officer",
        "chief product officer", "chief financial officer", "chief marketing officer",
        "chief data officer", "chief information officer",
        "ceo", "cto", "coo", "cpo", "cfo", "cmo", "cdo", "cio",
        # VP / Director
        "vice president of engineering", "vice president of product", "vice president of sales",
        "vice president of marketing", "vice president of data",
        "vp of engineering", "vp of product", "vp of sales", "vp of marketing",
        "director of engineering", "director of product", "director of product management",
        "director of data science", "director of machine learning", "director of analytics",
        # Senior IC
        "principal engineer", "principal software engineer", "senior staff engineer",
        "staff engineer", "staff software engineer",
        "senior software engineer", "senior product manager", "senior data scientist",
        "senior machine learning engineer", "senior data engineer", "senior analyst",
        "senior ux designer", "senior ui designer",
        # Mid-level
        "software engineer", "product manager", "data scientist", "data analyst",
        "machine learning engineer", "data engineer", "frontend engineer",
        "backend engineer", "full-stack engineer", "fullstack engineer",
        "full stack engineer", "ux designer", "ui designer", "ux researcher",
        "product designer", "devops engineer", "platform engineer",
        "site reliability engineer", "sre", "engineering manager",
        "product lead", "tech lead", "technical lead",
        # Abbreviations / short forms
        "pm", "swe", "ml engineer", "sde",
        # Business / finance
        "management consultant", "strategy consultant", "business analyst",
        "financial analyst", "investment banker", "venture capitalist",
        "solutions architect", "cloud architect",
        # Marketing / sales
        "growth marketer", "digital marketer", "content marketer",
        "account executive", "sales manager", "sales engineer",
        # Other
        "startup founder", "co-founder", "entrepreneur", "freelancer",
        "project manager", "program manager", "scrum master", "agile coach",
        "recruiter", "technical recruiter",
    ],
    "SKILL": [
        # Programming languages
        "python", "javascript", "typescript", "java", "golang", "go", "rust",
        "c++", "c#", "ruby", "swift", "kotlin", "scala", "r", "matlab",
        "sql", "nosql", "graphql",
        # Frameworks / libraries
        "react", "vue.js", "angular", "next.js", "node.js", "django", "fastapi",
        "spring boot", "express.js", "pytorch", "tensorflow", "scikit-learn",
        # Cloud / infra
        "aws", "azure", "google cloud", "gcp", "kubernetes", "docker",
        "terraform", "ci/cd", "devops", "mlops", "git",
        # Data / AI
        "machine learning", "deep learning", "nlp",
        "natural language processing", "computer vision",
        "data analysis", "data visualization", "tableau", "power bi",
        "pandas", "spark", "hadoop", "airflow",
        # Product
        "product management", "product strategy", "agile", "scrum", "kanban",
        "user research", "ux design", "ui design", "figma", "prototyping",
        "a/b testing", "roadmapping", "stakeholder management",
        # Soft skills
        "leadership", "public speaking", "communication", "project management",
        "strategic thinking", "problem solving",
    ],
    "DEGREE": [
        "bachelor of science", "bachelor of arts", "bachelor of engineering",
        "master of science", "master of arts", "master of business administration",
        "doctor of philosophy",
        "bachelor's degree", "master's degree", "doctoral degree",
        "bachelor's", "master's", "bachelors", "masters",
        "phd", "ph.d.", "mba", "m.b.a.",
        "b.s.", "m.s.", "b.a.", "m.a.", "msc", "bsc", "bs", "ms", "ba", "ma",
        "associate's degree", "associate's", "associates",
        "high school diploma", "ged", "coding bootcamp", "bootcamp",
    ],
    "CERTIFICATION": [
        "aws certified solutions architect", "aws certified developer",
        "google cloud professional", "azure solutions architect",
        "certified scrum master", "csm", "safe agilist",
        "project management professional", "pmp",
        "certified kubernetes administrator", "cka", "ckad",
        "cissp", "cism", "cisa", "comptia security+", "comptia network+",
        "google analytics certified", "salesforce certified administrator",
        "tensorflow developer certificate",
        "aws certified", "google cloud certified", "azure certified",
        "pmi-acp", "prince2", "cspo",
    ],
    "LANGUAGE": [
        "english", "spanish", "french", "german", "mandarin", "chinese",
        "japanese", "portuguese", "arabic", "russian", "hindi", "italian",
        "dutch", "korean", "swedish", "norwegian", "danish", "polish",
        "turkish", "thai", "vietnamese", "indonesian", "malay", "tagalog",
        "swahili", "hebrew", "persian", "greek", "czech", "hungarian",
        "romanian", "ukrainian", "catalan", "croatian", "finnish",
    ],
    "JOB_SEARCH_STATUS": [
        "actively looking for new roles", "actively searching for jobs",
        "actively job searching", "on the job market", "in job search mode",
        "casually browsing opportunities", "passively looking",
        "open to opportunities", "open to the right opportunity",
        "not actively looking", "not looking right now", "not on the market",
        "currently employed and looking", "exploring new opportunities",
        "in final rounds", "in interviews", "interviewing at",
        "sent out applications", "sent applications", "received an offer",
        "have an offer", "multiple offers in hand",
    ],
    "EMOTIONAL_SIGNAL": [
        "feeling really stressed", "feeling completely overwhelmed", "feeling lost",
        "feeling stuck", "feeling excited about", "feeling nervous about",
        "feeling frustrated", "feeling hopeful", "feeling scared",
        "feeling proud", "feeling motivated", "feeling anxious",
        "feel stressed", "feel overwhelmed", "feel stuck", "feel lost",
        "feel excited", "feel confident", "feel anxious",
        "stressed out", "anxious about", "nervous about",
        "excited about", "worried about", "scared of", "afraid of",
        "struggling with",
    ],
    "CONFIDENCE_SIGNAL": [
        "imposter syndrome", "self-doubt", "doubt my abilities",
        "not confident", "lack confidence", "low self-confidence",
        "don't think i'm good enough", "question my abilities",
        "unsure of myself", "uncertain about my abilities",
        "feel capable", "believe in myself", "trust myself",
        "confident in my skills", "sure i can do it",
    ],
    "BURNOUT_SIGNAL": [
        "completely burned out", "severely burnt out", "total burnout",
        "burned out", "burnt out", "burnout",
        "completely exhausted", "mentally exhausted", "physically exhausted",
        "feel drained", "running on empty", "no energy left",
        "worn out", "depleted", "overworked", "can't take it anymore",
        "need a break from work", "work is draining me",
    ],
    "STRESS_SIGNAL": [
        "high pressure environment", "toxic work environment", "difficult manager",
        "constant deadlines", "overwhelming workload", "too much on my plate",
        "under a lot of pressure", "under pressure",
        "high stress", "extremely stressed",
    ],
    "ENERGY_SIGNAL": [
        "morning person", "night owl", "high energy", "low energy",
        "energized in the morning", "peak productivity", "flow state",
        "best work in the afternoon", "most productive at night",
        "energized", "inspired to work",
    ],
    "WORK_STYLE": [
        "fully remote", "remote work", "work from home", "wfh",
        "hybrid work", "in-office", "on-site",
        "asynchronous work", "async", "flexible hours", "flexible schedule",
        "9-to-5", "deep work", "focus time",
        "fast-paced environment", "startup culture", "corporate culture",
        "collaborative work", "independent work", "structured environment",
    ],
    "PERSONALITY_TRAIT": [
        "introverted", "extroverted", "ambivert",
        "introvert", "extrovert",
        "highly analytical", "detail-oriented", "big-picture thinker",
        "extremely empathetic", "very driven", "highly ambitious",
        "perfectionist", "highly adaptable", "very organized",
        "risk-taker", "risk-averse", "strategic thinker",
    ],
    "VALUE": [
        "work-life balance", "work life balance",
        "making a positive impact", "making a difference", "social impact",
        "financial security", "job security", "career stability",
        "autonomy and independence", "creative freedom", "autonomy",
        "continuous learning", "personal growth", "professional development",
        "diversity and inclusion", "transparency and integrity",
        "family first", "family time",
    ],
    "MOTIVATION": [
        "passionate about building", "passionate about solving",
        "passionate about helping", "passionate about",
        "love working on", "love building", "love solving",
        "driven by impact", "motivated by results",
        "want to make an impact", "want to help people",
        "care deeply about", "inspired by",
        "thrive on challenges", "enjoy solving complex problems",
    ],
    "CONSTRAINT_SIGNAL": [
        "can't relocate", "unable to relocate", "location constraint",
        "family obligations", "family commitments", "caregiver responsibilities",
        "need visa sponsorship", "work authorization", "no work authorization",
        "limited bandwidth", "limited time", "no time to",
        "working full time", "full-time job on the side",
        "health limitations", "disability accommodation",
    ],
    "FINANCIAL_CONSTRAINT": [
        "student loan debt", "student loans", "student debt",
        "can't afford tuition", "financial constraints", "financial pressure",
        "not making enough money", "severely underpaid",
        "tight on money", "broke right now", "limited savings",
        "cost of living is high",
    ],
    "LIFE_STAGE": [
        "recent graduate", "just graduated", "new grad", "fresh out of college",
        "early career professional", "mid-career professional", "career changer",
        "returning to the workforce", "re-entering the workforce",
        "new parent", "new mom", "new dad", "expecting a baby",
        "empty nester", "recently divorced", "recently married",
    ],
    "LEARNING_PREFERENCE": [
        "learn best by doing", "hands-on learner", "visual learner",
        "learn through projects", "self-paced learning", "self-taught",
        "prefer online courses", "in-person training", "prefer bootcamps",
        "learn from mentors", "pair programming", "learn by reading",
        "prefer video tutorials", "like attending workshops",
        "project-based learning", "learn on the job",
    ],
    "LIFESTYLE_PREFERENCE": [
        "digital nomad lifestyle", "want to travel", "travel frequently",
        "want to relocate", "prefer to stay local", "love living in big cities",
        "want to move abroad", "work from anywhere", "location independent",
        "active outdoor lifestyle", "4-day work week", "part-time work",
        "freelance lifestyle", "entrepreneurial lifestyle",
    ],
    "HEALTH_SIGNAL": [
        "managing chronic illness", "dealing with mental health",
        "anxiety disorder", "clinical depression", "adhd diagnosis",
        "repetitive strain injury", "back issues from sitting",
        "need good health insurance", "health and wellness are important",
        "prioritize my mental health", "physical health challenges",
    ],
}


# =============================================================================
# REGEX PATTERNS  (mirrors extract_ner.py for consistency)
# =============================================================================

_REGEX_PATTERNS: dict[str, list[re.Pattern]] = {
    "DURATION": [
        re.compile(r"\b\d+\+?\s*(?:year|month|week|day)s?\b", re.I),
        re.compile(r"\bsince\s+\d{4}\b", re.I),
        re.compile(r"\bfor\s+(?:over\s+)?\d+\s*(?:year|month|week|day)s?\b", re.I),
        re.compile(r"\b(?:two|three|four|five|six|seven|eight|nine|ten)\s+(?:year|month|week|day)s?\b", re.I),
        re.compile(r"\ba\s+(?:couple|few|handful)\s+(?:of\s+)?(?:year|month)s?\b", re.I),
    ],
    "TIMELINE": [
        re.compile(r"\b(?:within|in|by)\s+\d+\s*(?:year|month|week)s?\b", re.I),
        re.compile(r"\bby\s+20\d{2}\b", re.I),
        re.compile(r"\bnext\s+(?:year|month|quarter|decade)\b", re.I),
        re.compile(r"\bin\s+(?:the\s+)?next\s+\d+\s*(?:year|month)s?\b", re.I),
        re.compile(r"\bwithin\s+(?:a|one|two|three)\s+(?:year|month|decade)s?\b", re.I),
        re.compile(r"\b(?:shortly|soon|eventually|someday)\b", re.I),
    ],
    "SALARY_EXPECTATION": [
        re.compile(r"\$\s*\d{1,3}(?:,\d{3})*(?:\s*k)?\b", re.I),
        re.compile(r"\b\d{2,3}\s*k\s*(?:base|salary|comp|package|total)?\b", re.I),
        re.compile(r"\b(?:six|seven)\s+figures?\b", re.I),
        re.compile(r"\bdouble\s+(?:my\s+)?(?:current\s+)?salary\b", re.I),
        re.compile(r"\b\d{3},\d{3}\b"),
    ],
    "TRANSITION_INTENT": [
        re.compile(
            r"\b(?:want|hope|plan(?:ning)?|looking|aiming|aspiring|trying)\s+to"
            r"\s+(?:become|move|transition|switch|break\s+into|get\s+into|land(?:\s+a)?|join|pivot)\b",
            re.I,
        ),
        re.compile(r"\b(?:dream(?:ing)?|aspir(?:ing|e))\s+(?:of|to)\b", re.I),
        re.compile(r"\btransition(?:ing)?\s+(?:to|into|from)\b", re.I),
        re.compile(r"\bswitch(?:ing)?\s+(?:to|into|careers?)?\b", re.I),
        re.compile(r"\bbreak(?:ing)?\s+into\b", re.I),
        re.compile(r"\bpivot(?:ing)?\s+(?:to|into|from)?\b", re.I),
        re.compile(r"\bmove\s+(?:into|to)\b", re.I),
    ],
}


# =============================================================================
# SUBCATEGORY → LABEL → FIELD MAPPING
#
# Maps (subcategory, NER_label) → the schema field name that label can fill.
# Only includes cases where the span text is a direct, usable value for the
# field — no semantic inference required.  Fields that need LLM understanding
# (booleans, severity scores, free-text descriptions, etc.) are intentionally
# absent so they remain in the "remaining fields" set for OpenAI.
# =============================================================================

_SUBCATEGORY_LABEL_TO_FIELDS: dict[str, dict[str, str]] = {
    "work_history": {
        "ROLE": "role",
        "ORG": "company",
        "DURATION": "startDate",
    },
    "professional_aspirations": {
        "ROLE": "dreamRole",
        "ORG": "targetCompany",
        "TIMELINE": "targetTimeframe",
        "SALARY_EXPECTATION": "targetSalary",
        "TRANSITION_INTENT": "careerChangeType",
    },
    "professional_achievements": {
        "ROLE": "title",
        "ORG": "organization",
    },
    "workplace_challenges": {
        "STRESS_SIGNAL": "challengeType",
        "BURNOUT_SIGNAL": "challengeType",
    },
    "job_search_status": {
        "JOB_SEARCH_STATUS": "searchStatus",
        "TIMELINE": "desiredStartDate",
    },
    "knowledge_and_credentials": {
        "SKILL": "name",
        "DEGREE": "name",
        "CERTIFICATION": "name",
        "ORG": "institution",
        "DURATION": "yearsExperience",
    },
    "languages": {
        "LANGUAGE": "language",
    },
    "learning_agenda": {
        "SKILL": "gapOrGoal",
        "TIMELINE": "targetDate",
        "LEARNING_PREFERENCE": "preferredFormat",
    },
    "mentorship": {
        "ROLE": "role",
        "ORG": "organization",
    },
    "recommendations": {
        "ROLE": "authorRole",
        "ORG": "platform",
    },
    "network_and_networking": {
        "ROLE": "role",
        "ORG": "organization",
    },
    "mindset_and_values": {
        "VALUE": "value",
        "PERSONALITY_TRAIT": "value",
        "MOTIVATION": "description",
    },
    "working_style_preferences": {
        "WORK_STYLE": "preference",
    },
    "emotional_state": {
        "EMOTIONAL_SIGNAL": "context",
        "STRESS_SIGNAL": "context",
        "CONFIDENCE_SIGNAL": "dimension",
        "BURNOUT_SIGNAL": "dimension",
        "ENERGY_SIGNAL": "dimension",
    },
    "life_situation": {
        "LIFE_STAGE": "value",
        "LOCATION": "value",
        "CONSTRAINT_SIGNAL": "description",
        "FINANCIAL_CONSTRAINT": "description",
        "TIMELINE": "timeframe",
        "LIFESTYLE_PREFERENCE": "description",
    },
    "health_and_wellbeing": {
        "HEALTH_SIGNAL": "condition",
        "BURNOUT_SIGNAL": "condition",
    },
    "personal_projects": {
        "SKILL": "skills",
        "ROLE": "role",
        "ORG": "name",
        "DURATION": "hoursPerWeek",
    },
    "personal_priorities": {
        "VALUE": "category",
        "LIFESTYLE_PREFERENCE": "description",
        "TIMELINE": "timeframe",
    },
}


def spans_to_fields(spans: list[dict], subcategory: str, message: str) -> dict:
    """
    Convert NER spans to a partial schema-field dict for a given subcategory.

    Each span's text (message[start:end]) is used as the field value.  When
    multiple spans map to the same field the result is a list; otherwise a
    plain string.

    Args:
        spans:       List of {"start", "end", "label"} dicts from extract_entity_spans().
        subcategory: Entity/subcategory name (e.g. "work_history").
        message:     Original message text, used to slice span text.

    Returns:
        Dict of {field_name: str | list[str]} for fields that NER could fill.
        Empty dict if the subcategory has no label mapping or spans is empty.
    """
    label_map = _SUBCATEGORY_LABEL_TO_FIELDS.get(subcategory, {})
    if not spans or not label_map:
        return {}

    field_values: dict[str, list[str]] = {}
    for sp in spans:
        field = label_map.get(sp["label"])
        if field is None:
            continue
        text = message[sp["start"]: sp["end"]]
        field_values.setdefault(field, []).append(text)

    return {
        field: values[0] if len(values) == 1 else values
        for field, values in field_values.items()
    }


# =============================================================================
# OVERLAP RESOLUTION
# =============================================================================

def _remove_overlaps(spans: list[dict]) -> list[dict]:
    """
    Resolve overlapping spans:
    - Same label overlap → keep whichever was added first (longer wins because
      phrases are sorted longest-first before matching).
    - Different label, containment → keep the outer (longer) span.
    - Different label, partial overlap → keep both.
    """
    sorted_spans = sorted(spans, key=lambda s: (s["start"], s["start"] - s["end"]))
    accepted: list[dict] = []
    for sp in sorted_spans:
        dominated = False
        for acc in accepted:
            if sp["start"] >= acc["end"] or sp["end"] <= acc["start"]:
                continue  # no overlap
            if sp["label"] == acc["label"]:
                dominated = True
                break
            if acc["start"] <= sp["start"] and sp["end"] <= acc["end"]:
                dominated = True
                break
        if not dominated:
            accepted.append(sp)
    return accepted


# =============================================================================
# NER EXTRACTOR CLASS
# =============================================================================

class NERExtractor:
    """
    Lightweight NER extractor using regex + compiled phrase patterns.

    No spaCy, no LLM — suitable for use inside the LangGraph workflow where
    low latency and zero extra dependencies matter.

    Phrase patterns are compiled once on first use and reused for all calls.
    """

    def __init__(self) -> None:
        self._phrase_patterns: dict[str, re.Pattern] | None = None

    def _compile_phrase_patterns(self) -> dict[str, re.Pattern]:
        """Compile one regex per label from PHRASE_DICT (longest phrases first)."""
        patterns: dict[str, re.Pattern] = {}
        for label, phrases in PHRASE_DICT.items():
            if not phrases:
                continue
            sorted_phrases = sorted(phrases, key=len, reverse=True)
            escaped = [re.escape(p) for p in sorted_phrases]
            patterns[label] = re.compile(r"(?<!\w)(?:" + "|".join(escaped) + r")(?!\w)", re.I)
        return patterns

    @property
    def phrase_patterns(self) -> dict[str, re.Pattern]:
        if self._phrase_patterns is None:
            self._phrase_patterns = self._compile_phrase_patterns()
        return self._phrase_patterns

    def _extract_regex(self, text: str, relevant_labels: set) -> list[dict]:
        spans = []
        for label, patterns in _REGEX_PATTERNS.items():
            if label not in relevant_labels:
                continue
            for pat in patterns:
                for m in pat.finditer(text):
                    spans.append({"start": m.start(), "end": m.end(), "label": label})
        return spans

    def _extract_phrases(self, text: str, relevant_labels: set) -> list[dict]:
        spans = []
        for label, pattern in self.phrase_patterns.items():
            if label not in relevant_labels:
                continue
            for m in pattern.finditer(text):
                spans.append({"start": m.start(), "end": m.end(), "label": label})
        return spans

    def extract(self, text: str, entity: str) -> list[dict]:
        """
        Extract spans for a given entity/subcategory name.

        Args:
            text:   The message text to extract from.
            entity: Subcategory name (e.g. "work_history", "languages").

        Returns:
            Sorted list of {"start": int, "end": int, "label": str} dicts.
            Returns [] if the entity has no SPAN_LABELS entry.
        """
        relevant_labels = set(SPAN_LABELS.get(entity, []))
        if not relevant_labels:
            return []

        regex_spans = self._extract_regex(text, relevant_labels)
        phrase_spans = self._extract_phrases(text, relevant_labels)
        all_spans = _remove_overlaps(regex_spans + phrase_spans)
        return sorted(all_spans, key=lambda s: s["start"])


# =============================================================================
# MODULE SINGLETON
# =============================================================================

_extractor_instance: NERExtractor | None = None


def get_ner_extractor() -> NERExtractor:
    """Get (or lazily create) the module-level NER extractor singleton."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = NERExtractor()
    return _extractor_instance


def extract_entity_spans(text: str, entity: str) -> list[dict]:
    """
    Convenience function: extract NER spans for a given entity/subcategory.

    Args:
        text:   Message text.
        entity: Entity/subcategory name (e.g. "work_history", "languages").

    Returns:
        List of {"start": int, "end": int, "label": str} dicts, sorted by position.

    Example:
        >>> extract_entity_spans("I'm a Senior PM at Google for 3 years", "work_history")
        [{"start": 6, "end": 15, "label": "ROLE"},
         {"start": 19, "end": 25, "label": "ORG"},
         {"start": 26, "end": 34, "label": "DURATION"}]
    """
    return get_ner_extractor().extract(text, entity)
