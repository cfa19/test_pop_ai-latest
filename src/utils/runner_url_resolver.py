"""
Runner Tag Resolver

Resolves runner slugs (e.g. [[module-c1-mur-invisible]]) in LLM responses
to their canonical context.subcontext tags.

This avoids LLM hallucination and provides consistent context labels.
"""

import re

from supabase import Client

# Static mapping: runner-slug → context.subcontext
RUNNER_CONTEXT_MAP: dict[str, str] = {
    # Onboarding
    "talent-market-fit-onboarding": "professional.experience",
    # Self-Assessment & Orientation
    "a0-vision-reussite": "psychological.motivations",
    "a1-reussites-rapides": "psychological.confidenceAndSelfPerception",
    "a2-bilan-360-complet": "professional.experience",
    "a3-mini-bilan-express": "professional.experience",
    "a10-validation-adequation": "professional.aspirations",
    "talent-market-fit-positioning-test": "professional.aspirations",
    # CV Analysis
    "cv-analyzer-v3.1": "professional.experience",
    "cv-basic-analyzer": "professional.experience",
    # Personality & Motivation Tests
    "riasec-horizon-pro": "psychological.personalityProfile",
    "big-five-empreinte-pro": "psychological.personalityProfile",
    "mwms-pvqrr-moteur-pro": "psychological.motivations",
    # Career Planning & Life Timeline
    "ligne-de-vie-interactive": "personal.personalLife",
    "grille-faisabilite-projet": "professional.aspirations",
    "plan-montee-competences-24m": "learning.learningAspirations",
    "plan-croissance-progressive-24m": "professional.aspirations",
    "trajectoire-montee-valeur": "professional.aspirations",
    # Strengths, Skills & Portfolio
    "ms1-portrait": "learning.currentSkills",
    "ms2-connexions": "learning.currentSkills",
    "ms3-patterns-competences": "learning.currentSkills",
    "module-b1-atelier-star": "professional.experience",
    "module-b2-portfolio-validation": "professional.experience",
    # Communication & Pitch
    "talent-market-fit-pitch-builder": "professional.aspirations",
    "module-d2-pitch-audio": "professional.aspirations",
    "matrice-adaptation-contextuelle": "professional.experience",
    "mes-5-phrases-essentielles": "psychological.confidenceAndSelfPerception",
    # Objection Handling & Negotiation
    "gestion-objections-freins": "psychological.confidenceAndSelfPerception",
    "methode-arp-objection": "psychological.confidenceAndSelfPerception",
    "ia-copilote-argumentation": "professional.experience",
    "module-c3-kit-reassurance": "psychological.confidenceAndSelfPerception",
    "ma-ligne-rouge-negociation": "professional.aspirations",
    # Interview Simulation
    "module-d1-simulation-entretien": "professional.experience",
    "lecture-signaux-relationnels": "social.professionalNetwork",
    # Psychological Barriers & Mindset
    "module-c1-mur-invisible": "psychological.confidenceAndSelfPerception",
    "ms5-engagement-personnel": "psychological.motivations",
    # Market Analysis & Positioning
    "analyse-marche-comparative": "professional.aspirations",
    "radar-market-fit-5-axes": "professional.aspirations",
    "enquete-metier-guidee": "professional.aspirations",
    # Visibility & Personal Branding
    "audit-outils-visibilite": "social.networking",
    "cartographie-visibilite-multicanale": "social.networking",
    "maintenance-personal-branding": "social.networking",
    # Networking & Relationship Management
    "cartographie-opportunites-5-cercles": "social.professionalNetwork",
    "plan-action-reseau-3-cercles": "social.networking",
    "strategie-fidelisation-professionnelle": "social.professionalNetwork",
    # Action Planning & Follow-Up
    "module-4-1-activation-plan-action": "professional.aspirations",
    "module-4-2-cartographie-accelerateurs": "social.professionalNetwork",
    "tableau-bord-plan-b": "professional.aspirations",
    # Long-Term Sustainability
    "strategie-perennisation-3-axes": "professional.aspirations",
    "radar-veille-adaptation-3x3": "professional.aspirations",
    # Certification (RS6900)
    "cert-1-scenario-generator": "learning.certifications",
    "cert-2-preparation-20min": "learning.certifications",
    "cert-3-simulation-30min": "learning.certifications",
    "cert-4-jury-assessment": "learning.certifications",
    "ia-copilote-closing-suivi": "learning.certifications",
    # --- Runners not yet in chunks (for future use) ---
    "stress-detective": "psychological.stressAndCoping",
    "energy-checkin": "personal.healthAndWellbeing",
    "values-alignment": "psychological.values",
    "belief-buster": "psychological.confidenceAndSelfPerception",
    "isolation-check": "social.professionalNetwork",
    "support-mapper": "social.mentors",
    "meaning-scanner": "psychological.motivations",
    "ikigai-explorer": "psychological.values",
    "dream-role-swipe": "professional.aspirations",
    "origin-story": "professional.experience",
    "learning-style": "learning.learningPreferences",
    "profil-competences": "learning.currentSkills",
    "skills-spotter": "learning.currentSkills",
    "gap-prioritizer": "learning.learningGaps",
    "experience-translator": "professional.experience",
    "ma-direction-v3": "psychological.motivations",
    "network-compass": "social.networking",
    "personal-boardroom": "social.mentors",
    "diagnostic-entreprise": "professional.experience",
    "jury-consultant-ia": "learning.certifications",
    "vae-match-v2.0.0-prod": "learning.certifications",
    "native": "learning.certifications",
}


def resolve_runner_tags(text: str, supabase: Client) -> str:
    """
    Strip [[runner-slug]] markers from text, leaving only the runner name.

    Example:
        Input:  "I recommend **Mur Invisible** [[module-c1-mur-invisible]] to work on..."
        Output: "I recommend **Mur Invisible** to work on..."
    """
    pattern = r"\s*\[\[([a-zA-Z0-9._-]+)\]\]"
    return re.sub(pattern, "", text)
