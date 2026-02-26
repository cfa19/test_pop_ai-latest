# Memory Card v2 — Python Backend Integration Guide

> **For**: Python backend team (coach memory extraction)
> **Related tickets**: POP-507
> **Date**: 2026-02-17

---

## What changed

The frontend has migrated memory cards from v1 to v2. The key difference: **cards now carry structured data (`rawData`) that feeds an AI pipeline mapping observations to the canonical profile**. Without `rawData`, validated cards are a dead end — the AI pipeline has nothing to work with.

Additionally, the canonical profile went from 6 contexts to 5. The old `emotional` and `aspirational` contexts no longer exist.

---

## Current card vs expected card

### What the coach currently produces

```json
{
  "content": "\"1 million dollar per month\"",
  "type": "aspiration",
  "confidence": 0.9,
  "source": {
    "type": "coach",
    "sourceId": "89e05c7b-f98b-47b0-af9d-be6169955eb0",
    "extractedAt": "2026-02-17T12:41:02.483733+00:00"
  },
  "tags": ["professional", "professional_aspirations", "compensation_expectations"],
  "linkedContexts": ["professional"]
}
```

### What the coach should produce

```json
{
  "content": "1 million dollar per month",
  "type": "aspiration",
  "confidence": 0.9,
  "source": {
    "type": "coach",
    "sourceId": "89e05c7b-f98b-47b0-af9d-be6169955eb0",
    "extractedAt": "2026-02-17T12:41:02.483733+00:00"
  },
  "rawData": {
    "target_salary": 1000000,
    "currency": "USD",
    "period": "monthly",
    "flexibility": null,
    "priorities": []
  },
  "tags": ["professional", "professional_aspirations", "compensation_expectations"]
}
```

---

## Changes required

### 1. Remove `linkedContexts` (BREAKING)

**Stop sending `linkedContexts` entirely.** This field is deprecated and will be dropped from the database. The tags already encode the context hierarchy (`tags[0]` = context, `tags[1]` = entity, `tags[2]` = sub-entity), making `linkedContexts` redundant.

```diff
- "linkedContexts": ["professional"]    ← REMOVE
```

### 2. Add `rawData` with structured data (CRITICAL)

This is the most important change. `rawData` is what enables Pipeline 2 (AI canonical mapping) to work. Without it, the AI has only a free-text `content` string to interpret — unreliable and lossy.

`rawData` is a free-form `Record<string, unknown>` (no enforced schema per card type). Populate it with whatever structured fields the coach can extract from the conversation. Use the taxonomy sub-entity fields as guidance for what to extract (see section "rawData examples by type" below).

```diff
+ "rawData": { ... }    ← ADD structured data extracted from the conversation
```

---

## API contract

### Endpoint

```
POST /api/harmonia/journal/memory-cards
```

### Request body schema (`CreateMemoryProposal`)

```typescript
{
  content: string;                    // Human-readable summary (REQUIRED)
  type: MemoryCardType;               // One of the 7 types below (REQUIRED)
  confidence: number;                 // 0.0 to 1.0 (REQUIRED)
  source: {                           // Provenance (REQUIRED)
    type: "coach";
    sourceId: string;                 // Conversation ID or coach identifier
    extractedAt: string;              // ISO-8601 timestamp
    activityId?: string;              // UUID, if linked to an activity
    sessionId?: string;               // UUID, if linked to a session
  };
  rawData?: Record<string, unknown>;  // Structured data from extraction (STRONGLY RECOMMENDED)
  tags?: string[];                    // Taxonomy tags [context, entity, sub_entity]
}
```

### Response (201 Created)

```json
{
  "data": {
    "id": "uuid",
    "userId": "uuid",
    "content": "...",
    "type": "aspiration",
    "confidence": 0.9,
    "source": { ... },
    "status": "proposed",
    "tags": [...],
    "rawData": { ... },
    "canonicalMappings": null,
    "appliedFieldPaths": [],
    "processingStartedAt": null,
    "appliedAt": null,
    "mappingError": null,
    "mappingAttempts": 0,
    "createdAt": "...",
    "validatedAt": null
  },
  "success": true
}
```

Fields set automatically by the API (do NOT send):

- `id`, `userId`, `status`, `createdAt`, `validatedAt`
- All Pipeline 2 fields: `canonicalMappings`, `appliedFieldPaths`, `processingStartedAt`, `appliedAt`, `mappingError`, `mappingAttempts`

---

## The 7 card types

| Type         | Question it answers | When to use                                                                                                    |
| ------------ | ------------------- | -------------------------------------------------------------------------------------------------------------- |
| `competence` | "What can I do?"    | Skills, expertise, knowledge areas, proficiency levels                                                         |
| `experience` | "What have I done?" | Past jobs, roles, achievements, projects, volunteer work                                                       |
| `preference` | "What do I like?"   | Work style preferences, learning format preferences, lifestyle preferences, constraints (inverted preferences) |
| `aspiration` | "Where am I going?" | Career goals, salary targets, dream roles, life goals, learning goals                                          |
| `trait`      | "Who am I?"         | Personality traits, values, motivations, working styles, strengths                                             |
| `emotion`    | "How am I feeling?" | Confidence levels, stress, energy, celebrations, resilience                                                    |
| `connection` | "Who do I know?"    | Mentors, mentees, professional network, references                                                             |

---

## Tags format

Tags follow the hierarchical classification taxonomy (`context > entity > sub_entity`):

```json
"tags": ["professional", "professional_aspirations", "compensation_expectations"]
//        ↑ context       ↑ entity                     ↑ sub_entity
```

Always provide all 3 levels when possible. If sub-entity classification is uncertain, provide at least 2 levels.

The 5 valid contexts: `professional`, `learning`, `social`, `psychological`, `personal`.

> The old contexts `emotional` and `aspirational` no longer exist. Emotional data goes under `psychological`, career aspirations go under `professional`, and life goals go under `personal`.

---

## rawData examples by type

Use the taxonomy sub-entity fields as a guide for what to extract. The key principle: **extract every structured piece of information the coach can infer from the conversation, even if the user didn't state it explicitly.**

### `aspiration` — compensation expectations

User says: _"I want to earn 1 million dollars per month"_

```json
{
  "rawData": {
    "target_salary": 1000000,
    "currency": "USD",
    "period": "monthly",
    "flexibility": null,
    "priorities": []
  }
}
```

### `aspiration` — dream role

User says: _"I want to become a VP of Product at a fintech company in 2 years"_

```json
{
  "rawData": {
    "desired_role": "VP of Product",
    "target_industries": ["fintech"],
    "timeframe": "2 years",
    "priority": "high"
  }
}
```

### `competence` — skill

User says: _"I'm an expert in Python with 8 years of experience"_

```json
{
  "rawData": {
    "skill_name": "Python",
    "proficiency": "expert",
    "years_experience": 8
  }
}
```

### `experience` — past role

User says: _"I was a project manager at Airbus for 5 years"_

```json
{
  "rawData": {
    "title": "Project Manager",
    "company": "Airbus",
    "duration_years": 5,
    "responsibilities": [],
    "achievements": []
  }
}
```

### `trait` — personality / values

User says: _"I value autonomy and impact above everything else"_

```json
{
  "rawData": {
    "values": ["autonomy", "impact"],
    "priority": "highest"
  }
}
```

### `emotion` — confidence

User says: _"I'm feeling pretty confident lately, maybe 7 out of 10"_

```json
{
  "rawData": {
    "confidence_level": 7,
    "scale_max": 10,
    "trend": "stable",
    "domain": "overall"
  }
}
```

### `connection` — mentor

User says: _"I have a mentor who's a VP at Stripe, we meet monthly"_

```json
{
  "rawData": {
    "mentor_name": null,
    "mentor_role": "VP",
    "mentor_company": "Stripe",
    "frequency": "monthly",
    "relationship": "formal"
  }
}
```

### `preference` — work style

User says: _"I need remote work, it's non-negotiable"_

```json
{
  "rawData": {
    "preference_type": "work_mode",
    "value": "remote",
    "importance": "non_negotiable"
  }
}
```

---

## The full taxonomy reference

The tags and rawData fields should align with the hierarchical classification taxonomy defined in `docs/hierarchical_classification_taxonomy.md`. Refer to that document for:

- All 5 contexts with their entities and sub-entities
- Example messages for each classification path
- Sub-entity field names to use as rawData keys

---

## What happens after the coach creates a card

```
Coach creates card (status: 'proposed')
    ↓
User sees the card in the UI
    ↓
User validates → status: 'validated'
    ↓
Pipeline 2 (AI) picks up the card
    ↓
AI reads content + rawData + type + user's current profile
    ↓
AI produces canonicalMappings[] → which profile fields to update
    ↓
Profile updated → status: 'applied'
```

The richer the `rawData`, the better Pipeline 2 can map the card to the canonical profile. Without `rawData`, the AI only has the `content` string — which is ambiguous, may be in different languages, and lacks structure.

---

## Summary of changes

| Field            | Before               | After                 | Action                            |
| ---------------- | -------------------- | --------------------- | --------------------------------- |
| `content`        | Double-quoted string | Clean string          | Fix quoting                       |
| `rawData`        | Not sent (null)      | Structured extraction | **Add** — critical for Pipeline 2 |
| `linkedContexts` | `["professional"]`   | Not sent              | **Remove** — deprecated           |
| `tags`           | Already correct      | Already correct       | No change needed                  |
| `type`           | Already correct      | Already correct       | No change needed                  |
| `source`         | Already correct      | Already correct       | No change needed                  |
| `confidence`     | Already correct      | Already correct       | No change needed                  |
