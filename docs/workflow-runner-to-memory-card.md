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

The consent works like this:
- During the user's **first activity** (regardless of which activity it is — onboarding, CV analysis, or any other), the user is asked: *"Do you agree to let the AI analyze your activity results to build your professional profile?"*
- This prompt only appears once — on the very first activity the user starts. After responding, it never appears again.
- If they said **yes**: processing continues
- If they said **no** (or never answered): the request is marked as "skipped" and **no data is read or processed**. The AI moves on to the next request.

This consent check happens at **three independent levels** for maximum safety:
1. The AI service checks before starting
2. The code validates before writing
3. The database itself verifies before accepting the data

Even if one check had a bug, the other two would catch it. The user's choice is always respected.

### Step 7: The AI Extracts Insights

This is where the intelligence happens. The AI reads the activity results and identifies meaningful insights about the user. It uses **three different methods**, from most reliable to most creative:

#### Method 1: Competency Codes (Automatic, Highest Confidence)

Some activities validate specific professional competencies with standardized codes (like "C3.1 — Project Management, Level 3"). These are **already categorized** by the activity itself.

The AI simply converts them into memory cards directly. No interpretation needed.

- Used by: ~40% of activities
- Confidence level: Very High (95%)
- Example: *"Competency C3.1 validated at level 3"* becomes a **competence** card

#### Method 2: Pre-Categorized Insights (Automatic, High Confidence)

Many activities include a section called `journey_contributions` — data that the activity developer has already organized into meaningful categories like "career goals", "personality strengths", "perceived obstacles", etc.

The AI reads these categories and maps them to memory card types using a predefined set of rules. No AI creativity is involved — it's a direct translation.

- Used by: ~48% of activities
- Confidence level: High (80-90%)
- Example: A "career goals" entry saying *"Become a UX designer within 2 years"* becomes an **aspiration** card

#### Method 3: AI Interpretation (Creative, Moderate Confidence)

For the ~20% of activities that don't include pre-categorized data, the AI uses a language model (similar to ChatGPT) to read the raw results and identify what's meaningful.

This is the most flexible method but also the least predictable. It's only used when the first two methods don't apply.

- Used by: ~20% of activities (only when Methods 1 and 2 don't cover the data)
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
| User revokes consent | Future activities are skipped. Existing cards can be deleted by the user. |
| Activity has no extractable data | The request is marked as completed with zero cards. No error. |

---

## Privacy by Design

| Principle | How it's implemented |
|-----------|---------------------|
| **Explicit consent** | User must opt in during their first activity. No default "yes". |
| **Fail-closed** | If consent status is unknown or ambiguous, nothing is processed. |
| **Three-layer verification** | Consent is checked by the AI service, by the code, AND by the database. |
| **User control** | Users can view, edit, or delete any memory card at any time from their profile. |
| **Right to refuse** | User can say no at onboarding. The AI will never process their data. |
| **Right to revoke** | User can withdraw consent at any time. Future activities will no longer be processed. |
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
| Extraction (Methods 1 & 2) | < 1 second |
| Extraction (Method 3, AI) | 2-5 seconds |
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
                                       Method   Method   Method
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
