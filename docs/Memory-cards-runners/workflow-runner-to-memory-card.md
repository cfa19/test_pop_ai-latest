# From Activity to Memory Card: How Harmonia Learns About You

**Audience**: Product team, stakeholders, designers, non-technical contributors
**Date**: February 2026

---

## What This Document Explains

When a user completes an activity on Harmonia (like analyzing their CV, preparing for a job interview, or building a career plan), the platform can **automatically extract key insights** about them — their skills, experiences, aspirations, personality traits — and save them as **memory cards**.

These memory cards build a living portrait of the user over time, helping the AI coach give increasingly personalized advice.

This document explains the full journey: from the moment a user opens an activity to the moment a memory card appears in their profile.

---

## The Big Picture (30-Second Version)

```
User completes an activity
        |
        v
Results are saved to the database
        |
        v
A "to-do" item is automatically created for the AI
        |
        v
The AI checks: "Did this user agree to memory extraction?"
        |
        v
If yes: the AI reads the results and extracts insights
        |
        v
Each insight becomes a memory card — automatically added to the user's profile
```

---

## Step-by-Step Walkthrough

### Step 1: The User Starts an Activity

A user (beneficiary) opens an activity from their dashboard. Activities are powered by **runners** — interactive modules that guide the user through a structured exercise.

Examples of activities:
- **CV Analyzer**: Upload your CV, get an AI-powered analysis of strengths and gaps
- **STAR Workshop**: Build structured evidence stories (Situation, Task, Action, Result)
- **Career Clarity**: Define your career goals and deal-breakers
- **Invisible Wall**: Identify and overcome limiting beliefs
- **360 Assessment**: Complete a comprehensive professional skills evaluation

Each activity has a specific purpose and produces structured results.

### Step 2: The User Completes the Activity

When the user finishes all the steps and clicks "Complete" (or the activity auto-completes), the runner sends its results to the platform. These results include:

- **The main data**: scores, answers, analysis results — everything the activity produced
- **Pre-categorized insights** (when available): the runner developer has already labeled certain pieces of data as "this is a skill", "this is a career goal", "this is a personality trait", etc.
- **Competency codes** (when applicable): standardized codes from the RNCP framework (the French national competency registry) indicating which professional skills the user demonstrated

The platform automatically adds a small label to the results — the activity's name, how long it took, and when it was completed — so the AI knows where the data came from.

### Step 3: The Results Are Saved

The results are stored in the database in the `activity_completions` table. This is the single source of truth for everything a user has done on the platform.

At this point, the user's work is safely saved. The next steps happen **in the background** — the user doesn't need to wait.

### Step 4: A Processing Request Is Automatically Created

The moment a completion is saved, a database mechanism (a trigger) automatically creates an entry in a **processing queue**. Think of it like placing an order at a restaurant: the order slip goes to the kitchen, and the kitchen works through orders one by one.

This queue ensures:
- **Nothing gets lost**: even if the AI service is temporarily offline, the request waits in the queue
- **No duplicates**: the same activity completion is never processed twice
- **Automatic retry**: if something goes wrong, it tries again (up to 3 times)

### Step 5: The AI Service Picks Up the Request

A separate service (the AI backend, built in Python) continuously monitors the queue. Every 5 seconds, it checks: "Are there new requests to process?"

When it finds one, it picks it up and starts working on it. The request is "locked" so no other process can work on the same item simultaneously.

### Step 6: Privacy Check — "Did This User Consent?"

**Before reading any data**, the AI service checks whether the user has given their consent for AI-powered memory extraction. This is a strict, non-negotiable check.

#### How consent is collected

The very first time a user starts any activity on the platform, a consent modal appears:

```
Personalize your AI coaching experience

Harmonia can automatically learn from your activities to give you
better, more personal advice over time.

The AI will identify and save insights about:

  • Your skills and competencies
  • Your professional experiences
  • Your career goals and aspirations
  • Your work preferences
  • Your personality traits and strengths
  • Your confidence level and emotional state
  • Your professional connections

You can view, edit or delete any insight from your profile at any
time. You can also turn this off entirely from your settings.

  [Enable AI personalization]    [Not now]
```

Key points:
- This modal appears **only once** — during the first activity. It never interrupts subsequent activities.
- The user can choose **"Not now"** and the activity starts normally. The platform works either way.
- The consent explicitly lists **all 7 types of insights**, including personality traits and emotional state, which is required by European privacy law (GDPR Art. 9) for psychological data.
- **One consent covers all activity types** — professional, psychological, motivational — because the purpose is the same (personalized AI coaching) and all data types are clearly listed.
- The user can change their choice at any time from their profile settings (one-click toggle).

#### How consent is checked

For every completed activity, the AI service checks:
- If the user said **yes**: processing continues
- If the user said **no** (or never responded): the request is marked as "skipped" and **no data is read or processed**

This check happens at **three independent levels** for maximum safety:
1. The AI service checks before starting
2. The code validates before writing
3. The database itself verifies before accepting the data

Even if one check had a bug, the other two would catch it. The user's choice is always respected.

#### What about consent versioning?

If Harmonia ever adds new types of insights in the future (for example, financial data), the consent text would be updated and **all users would see the modal again** to agree to the expanded scope. This is tracked via a consent version number.

### Step 7: The AI Extracts Insights

This is where the intelligence happens. The AI reads the activity results and identifies meaningful insights about the user. It uses **three different methods**, from most reliable to most creative:

#### Tier 1: Competency Codes (Automatic, Highest Confidence)

Some activities validate specific professional competencies with standardized codes (like "C3.1 — Project Management, Level 3"). These are **already categorized** by the activity itself.

The AI simply converts them into memory cards directly. No interpretation needed.

- Used by: ~40% of activities
- Confidence level: Very High (95%)
- Example: *"Competency C3.1 validated at level 3"* becomes a **competence** card

#### Tier 2: Pre-Categorized Insights (Automatic, High Confidence)

Many activities include a section called `journey_contributions` — data that the activity developer has already organized into meaningful categories like "career goals", "personality strengths", "perceived obstacles", etc.

The AI reads these categories and maps them to memory card types using a predefined set of rules. No AI creativity is involved — it's a direct translation.

- Used by: ~48% of activities
- Confidence level: High (80-90%)
- Example: A "career goals" entry saying *"Become a UX designer within 2 years"* becomes an **aspiration** card

#### Tier 3: AI Interpretation (Creative, Moderate Confidence)

For the ~20% of activities that don't include pre-categorized data, the AI uses a language model (similar to ChatGPT) to read the raw results and identify what's meaningful.

This is the most flexible method but also the least predictable. It's only used when the first two methods don't apply.

- Used by: ~20% of activities (only when Tiers 1 and 2 don't cover the data)
- Confidence level: Moderate (70-80%)
- Example: From a free-form career analysis, the AI might extract *"Strong leadership skills demonstrated through team management experience"* as a **competence** card

### Step 8: Memory Cards Are Created Automatically

Each extracted insight becomes a **memory card** that is **automatically added to the user's profile**. No manual review required — the user already gave their consent during their first activity, so the system works silently in the background.

Each card contains:

- **Content**: A human-readable summary (e.g., *"5 years of project management at Airbus"*)
- **Type**: One of 7 categories:

| Type | What it captures | Example |
|------|-----------------|---------|
| **Competence** | "What can I do?" | Project management, Python programming, public speaking |
| **Experience** | "What have I done?" | 5 years at Airbus, MBA from HEC, PRINCE2 certification |
| **Preference** | "What do I like?" | Remote work, creative environments, structured processes |
| **Aspiration** | "Where am I going?" | Become a UX designer, earn 60K+, move to Bordeaux |
| **Trait** | "Who am I?" | Analytical thinker, strong communicator, risk-averse |
| **Emotion** | "How am I feeling?" | High confidence in skills, anxious about interviews |
| **Connection** | "Who do I know?" | 3 mentors, 12 professional contacts in tech sector |

- **Confidence score**: How sure the AI is about this card (0% to 100%)
- **Source**: Which activity produced this card, and when

The user can always view, edit, or delete their memory cards from their profile if they want to — but they don't have to. The system is designed to work without requiring constant attention.

### Step 9: Memory Cards Shape the AI Coach

As memory cards accumulate, they become the user's **living professional profile**. The AI coach (Harmonia) uses them to:

- Give personalized advice based on the user's actual skills and experience
- Suggest relevant activities based on identified gaps
- Track progress over time (e.g., confidence level increasing)
- Avoid asking questions the platform already knows the answer to

The more activities a user completes, the richer their profile becomes, and the more helpful the AI coach gets.

---

## What Happens If Something Goes Wrong?

The system is designed to be **resilient**:

| Scenario | What happens |
|----------|-------------|
| AI service is down | Requests wait in the queue. When the service restarts, it processes them. |
| Extraction fails | The request is retried automatically (up to 3 times). After 3 failures, it's flagged for manual review. |
| AI service crashes mid-processing | After 10 minutes, the system detects the stuck request and reassigns it. |
| User revokes consent | All existing memory cards are **permanently deleted**. Pending requests in the queue are skipped. Future activities will no longer generate cards. The user can re-enable at any time from settings. |
| Activity has no extractable data | The request is marked as completed with zero cards. No error. |

---

## Privacy by Design

| Principle | How it's implemented |
|-----------|---------------------|
| **Explicit consent** | User must opt in during their first activity via a clear modal. No default "yes". |
| **Specific and informed** | Consent modal explicitly lists all 7 insight types, including personality traits and emotional state. |
| **Freely given** | User can say "Not now" and the activity works normally. The platform is fully functional without AI personalization. |
| **One consent, all activities** | A single consent covers all runner types (professional, psychological, motivational) because the purpose is one and all data types are listed. |
| **Fail-closed** | If consent status is unknown or ambiguous, nothing is processed. |
| **Three-layer verification** | Consent is checked by the AI service, by the code, AND by the database. |
| **User control** | Users can view, edit, or delete any memory card at any time from their profile. |
| **Right to revoke** | User can withdraw consent with one click from settings. This **permanently deletes all memory cards** and stops future processing. A confirmation dialog warns the action cannot be undone. As easy to withdraw as to give (Art. 7(3)). |
| **Consent versioning** | If new data categories are added, all users see the consent modal again for the expanded scope. |
| **Transparency** | Every card shows its source (which activity, when, confidence level). |

---

## Timeline: How Long Does It Take?

| Step | Duration |
|------|----------|
| User completes activity | Depends on the activity (5 min to 1 hour) |
| Results saved to database | Instant (< 1 second) |
| Queue entry created | Instant (automatic trigger) |
| AI picks up the request | Within 5 seconds (polling interval) |
| Consent check | < 100 milliseconds |
| Extraction (Tiers 1 & 2) | < 1 second |
| Extraction (Tier 3, AI) | 2-5 seconds |
| Memory cards added to profile | **Under 10 seconds** after activity completion |

---

## Glossary

| Term | Meaning |
|------|---------|
| **Runner** | An interactive module that powers an activity (e.g., CV Analyzer, STAR Workshop) |
| **Activity completion** | The record of a user finishing an activity, including all results |
| **Memory card** | A single piece of knowledge about the user (a skill, experience, goal, etc.) |
| **Extraction** | The process of identifying meaningful insights from activity results |
| **Queue** | A waiting list that ensures every completion is processed reliably |
| **Consent** | The user's explicit permission for AI to analyze their activity data |
| **Profile** | The collection of all memory cards that the AI coach uses to personalize guidance |
| **RNCP** | Repertoire National des Certifications Professionnelles — France's national skills framework |
| **Journey contributions** | Pre-categorized insights embedded in activity results by the developer |

---

## Visual Summary

```
                    THE USER'S JOURNEY
                    ==================

    [User opens activity]
            |
            v
    [Works through steps: answers questions, uploads CV, ...]
            |
            v
    [Clicks "Complete"]
            |
            |                           BEHIND THE SCENES
            |                           =================
            v
    [Results saved] ---------> [Queue entry created automatically]
            |                           |
            v                           v
    [User sees completion      [AI service picks up request]
     screen / summary]                  |
                                        v
                               [Consent check: did user agree?]
                                   /           \
                                  /             \
                              NO /               \ YES
                                /                 \
                               v                   v
                        [Skip - done]      [Extract insights]
                                                   |
                                          +--------+--------+
                                          |        |        |
                                          v        v        v
                                       Tier     Tier     Tier
                                         1        2        3
                                       (codes) (categories) (AI)
                                          |        |        |
                                          +--------+--------+
                                                   |
                                                   v
                                   [Create memory cards]
                                   [added to user's profile]
                                                   |
                                                   v
                                   [AI coach uses cards for
                                    personalized guidance]
                                                   |
                                                   v
                                   [User can view/edit/delete
                                    cards anytime from profile]
```

---

## Concrete Example: Marie's Bilan 360

Let's say **Marie** just completed the **"Bilan 360 Complet"** runner. She spent 20 minutes answering questions about her career, skills, and goals. When she clicks "Finish", the runner sends this data:

```json
{
  "score_global": 78,
  "forces": ["leadership", "gestion de projet"],
  "axes_amelioration": ["délégation", "prise de parole"],

  "covered_skills": [
    { "code": "C2.1", "level": 3 },
    { "code": "C3.2", "level": 2 }
  ],

  "journey_contributions": {
    "professional": {
      "career_goals": ["Devenir directrice R&D", "Manager une équipe de 15+"],
      "profile": {
        "experiences": [{ "title": "Chef de projet", "company": "Airbus", "years": 5 }],
        "skills": ["Python", "Agile", "Leadership"],
        "years_experience": 8
      },
      "positioning_assessment": {
        "score": 78,
        "strengths": ["leadership", "gestion de projet"],
        "gaps": ["délégation"]
      }
    },
    "personal": {
      "motivations": ["Impact social", "Innovation"],
      "engagement_baseline": {
        "confiance_niveau": 7,
        "date": "2026-02-13",
        "formule": "confiante"
      }
    }
  },

  "dashboard_data": {
    "tiles": [{ "key": "score_global", "value": 78, "unit": "%" }]
  }
}
```

The API adds `_runner_metadata` automatically, then it goes to the database. A trigger puts it in the queue. The AI backend picks it up and starts extracting.

---

### Tier 1 — The easy wins (competency codes)

**Think of it as**: A checklist. Marie validated specific competency codes. No interpretation needed — just read the codes and create cards.

**What it reads**: `covered_skills`

**What it does**:

| Skill | Card created |
|---|---|
| `C2.1` level 3 | "Compétence C2.1 validée au niveau 3" |
| `C3.2` level 2 | "Compétence C3.2 validée au niveau 2" |

**Result**: 2 memory cards, type `competence`, confidence **0.95**

**Why 0.95?** The runner explicitly says "Marie validated C2.1 at level 3". There's nothing to interpret. It's a fact.

**Cost**: Zero. No AI involved. Just string formatting.

---

### Tier 2 — The structured data (journey_contributions)

**Think of it as**: A filing cabinet. Marie's runner already organized her data into labeled folders (`career_goals`, `profile`, `motivations`...). The system just reads the label and files each folder into the right memory card type.

**What it reads**: `journey_contributions`

**How it works**: There's a dictionary of 44 rules. Each rule says: "If you find data at path X, it's card type Y with confidence Z."

```
professional.career_goals           → aspiration (0.90)
professional.profile                → experience (0.90)
professional.positioning_assessment → trait (0.80)
personal.motivations                → trait (0.85)
personal.engagement_baseline        → emotion (0.80)
```

The system walks through all 44 rules. For each path that has data, it creates a card:

| Path found | Data | Card created | Type |
|---|---|---|---|
| `professional.career_goals` | `["Devenir directrice R&D", "Manager une équipe de 15+"]` | "Objectifs de carrière définis (2 éléments)" | **aspiration** |
| `professional.profile` | `{ experiences: [...], skills: [...], years: 8 }` | "Profil professionnel identifié: Chef de projet, Airbus, 5" | **experience** |
| `professional.positioning_assessment` | `{ score: 78, strengths: [...], gaps: [...] }` | "Évaluation de positionnement complétée: 78, leadership, gestion de projet" | **trait** |
| `personal.motivations` | `["Impact social", "Innovation"]` | "Motivations identifiées (2 éléments)" | **trait** |
| `personal.engagement_baseline` | `{ confiance_niveau: 7, formule: "confiante" }` | "Baseline d'engagement mesurée: 7, confiante" | **emotion** |

**Result**: 5 memory cards, confidence **0.80-0.90**

**Why lower confidence than Tier 1?** The data is well-organized, but the mapping is based on the path name, not an explicit code. `career_goals` *should* be an aspiration, but there's a small chance it contains something else.

**Cost**: Zero. No AI involved. Just dictionary lookup + string formatting.

**What about `dashboard_data`?** Completely ignored. It's display config for the consultant dashboard, not memory about Marie.

---

### Tier 3 — The leftovers (LLM extraction)

**Think of it as**: A detective. After Tiers 1 and 2 have taken everything they understand, there's still some loose data lying around. An AI model reads it and tries to figure out what it means.

**What's left after Tiers 1 and 2?**

After removing everything already processed or irrelevant (`_runner_metadata`, `covered_skills`, `journey_contributions`, `dashboard_data`), only this remains:

```json
{
  "score_global": 78,
  "forces": ["leadership", "gestion de projet"],
  "axes_amelioration": ["délégation", "prise de parole"]
}
```

**Decision point**: Should Tier 3 run?

- Did Tier 2 find data? **Yes** (5 cards from `journey_contributions`)
- Is there remaining data? **Yes** (`score_global`, `forces`, `axes_amelioration`)
- But Tier 2 already covered this! (`positioning_assessment` already captured the same score + strengths + gaps)

**In Marie's case, Tier 3 does NOT run** because `journey_contributions` already covered the meaningful data. The flat fields are redundant.

**But for a runner WITHOUT `journey_contributions`**, say a simple quiz runner that only outputs:

```json
{
  "score": 85,
  "forces": ["leadership", "communication"],
  "axes_amelioration": ["délégation"],
  "recommandations": ["Suivre formation management"]
}
```

Then Tier 3 kicks in. It sends this to OpenAI with a prompt saying "categorize this into memory cards." The AI might return:

| Extracted | Card type | Confidence |
|---|---|---|
| "Forces: leadership, communication" | competence | 0.75 |
| "Axe d'amélioration: délégation" | trait | 0.70 |
| "Recommandation: formation management" | aspiration | 0.75 |

**Why lowest confidence?** The AI is guessing. "forces" *probably* means competence, but it could also be a trait. The LLM does its best, but it's interpretation, not certainty.

**Cost**: ~$0.01-0.03 per completion (one OpenAI API call).

---

### Marie's final memory cards

Back to Marie's bilan. Here's everything the pipeline produced:

| # | Content | Type | Confidence | Tier |
|---|---|---|---|---|
| 1 | Compétence C2.1 validée au niveau 3 | `competence` | 0.95 | Tier 1 |
| 2 | Compétence C3.2 validée au niveau 2 | `competence` | 0.95 | Tier 1 |
| 3 | Objectifs de carrière définis (2 éléments) | `aspiration` | 0.90 | Tier 2 |
| 4 | Profil professionnel identifié | `experience` | 0.90 | Tier 2 |
| 5 | Évaluation de positionnement complétée | `trait` | 0.80 | Tier 2 |
| 6 | Motivations identifiées (2 éléments) | `trait` | 0.85 | Tier 2 |
| 7 | Baseline d'engagement mesurée | `emotion` | 0.80 | Tier 2 |

**7 memory cards from one activity completion. Zero LLM calls. Total cost: $0.00.**

Next time Marie uses the AI coach, it knows she wants to become R&D director, has 8 years of experience at Airbus, is strong in leadership, needs to work on delegation, and is generally confident. All extracted automatically, all from structured data the runner developer organized in advance.

---

### The big picture

```
               Tier 1              Tier 2              Tier 3
            ─────────────      ─────────────      ─────────────
Input       covered_skills     journey_contrib.   leftover flat data
Method      Loop + format      Dict lookup         OpenAI LLM call
AI needed?  No                 No                  Yes
Confidence  0.95               0.80-0.90           0.70-0.80
Cost        $0                 $0                  ~$0.01-0.03
Runners     23/58 (40%)        28/58 (48%)         ~12/58 (20%)
Card types  competence only    all 7 types         all 7 types
```

Note: Tiers 1 and 2 often overlap — a single completion can have both `covered_skills` and `journey_contributions`, which is why the percentages add up to more than 100%. Tier 3 only runs when the other tiers don't cover the data.

**The design goal**: the better runners structure their output (Tiers 1-2), the less we rely on the LLM (Tier 3) — saving money, time, and getting more accurate cards.
