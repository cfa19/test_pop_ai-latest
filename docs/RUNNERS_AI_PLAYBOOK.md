# Runners Playbook: AI Memory Card Extraction

**Version**: 1.0.0
**Date**: February 2026
**Audience**: Runner developers, AI backend team
**Related plans**: `NEXTJS_PLAN.md` (frontend schema fixes), `AI_BACKEND_PLAN.md` (Python/FastAPI extraction pipeline)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Required Output Structure](#2-required-output-structure)
3. [journey_contributions (Critical for AI)](#3-journey_contributions-critical-for-ai)
4. [covered_skills (RNCP Competencies)](#4-covered_skills-rncp-competencies)
5. [dashboard_data (Display Only)](#5-dashboard_data-display-only)
6. [Schema Best Practices](#6-schema-best-practices)
7. [Runner Config Checklist](#7-runner-config-checklist)
8. [Memory Card Type Mapping Guide](#8-memory-card-type-mapping-guide)
9. [Anti-Patterns](#9-anti-patterns)
10. [Testing](#10-testing)
11. [Gap Analysis: NEXTJS_PLAN vs AI_BACKEND_PLAN](#11-gap-analysis-nextjs_plan-vs-ai_backend_plan)
12. [AI_BACKEND_PLAN Amendments](#12-ai_backend_plan-amendments)

---

## 1. Overview

### What is a runner?

A **runner** is a native React component that implements a pedagogical activity. It receives `ActivityRunnerProps` from the platform and returns structured results via `onComplete()`. See `docs/04-runners/PLAYBOOK_CREATION_RUNNERS.md` for general runner development guidance.

This playbook focuses specifically on **how runner output data feeds the AI memory card extraction pipeline**.

### Data flow: Runner output to memory cards

```
Runner Component
    │  onComplete(result, { endMessage? })
    ▼
API Route (activity-sessions/[id]/complete OR activity-completions)
    │  Injects _runner_metadata into responses
    │  Copies dashboard_data to a separate column (for indexed access)
    ▼
activity_completions table (PostgreSQL)
    │  responses JSONB ← ALL runner output (including journey_contributions + _runner_metadata)
    │  dashboard_data JSONB ← COPY of responses.dashboard_data (separate column, GIN indexed)
    │  NOTE: journey_contributions is NOT a separate column — it lives inside responses
    ▼
PostgreSQL trigger (enqueue_memory_extraction)
    │  Fires on INSERT/UPDATE where status='completed'
    │  Inserts into memory_extraction_queue (idempotent, ON CONFLICT DO NOTHING)
    ▼
memory_extraction_queue (status: 'pending')
    │
    ▼
AI Backend — QueueConsumer (Python/FastAPI, polls every 5s)
    │  SELECT ... FOR UPDATE SKIP LOCKED (exactly-once, crash-safe)
    │
    │  1. Check user consent — Layer 1 (orchestrator, fail-closed)
    │     Query: user_consents WHERE consent_type='ai_training' AND consent_given=true AND revoked_at IS NULL
    │     No consent → queue status: 'skipped'
    │
    │  2. Three-tier extraction:
    │     Tier 1: covered_skills → competence cards (deterministic, no LLM)
    │     Tier 2: journey_contributions → mapped cards (deterministic rules, no LLM)
    │     Tier 3: flat data → LLM extraction (only ~20% of completions need this)
    │
    │  3. Queue status: 'completed' or 'failed' (retry max 3 attempts)
    ▼
create_memory_proposal() RPC — Layer 3 (DB-level consent + type validation)
    │  PostgreSQL SECURITY DEFINER function — unforgeable guardrails
    ▼
memory_cards table (status: 'validated' — automatically active, no manual review)
    │  7 types: competence, experience, preference, aspiration, trait, emotion, connection
    ▼
Automatic pipeline (validated → processing → applied)
```

### Why queue-based, not Realtime?

Supabase Realtime is designed for browser subscriptions. For a server-side critical pipeline:
- **Queue**: exactly-once delivery, crash recovery, retry, backfill, visibility
- **Realtime**: fire-and-forget, events lost if service is down, no replay

### The 7 memory card types (POP-507 v2)

| Type | Question it answers | Primary data sources |
|------|---------------------|---------------------|
| `competence` | "What can I do?" | `covered_skills`, `professional.skills_positioning`, `professional.validations` |
| `experience` | "What have I done?" | `professional.profile.experiences`, `professional.certification` |
| `preference` | "What do I like?" | `professional.career_goals`, work style data |
| `aspiration` | "Where am I going?" | `professional.development_plan`, `personal.vision_6m` |
| `trait` | "Who am I?" | `professional.positioning_assessment`, `personal.motivations` |
| `emotion` | "How am I feeling?" | `personal.engagement_baseline`, confidence data |
| `connection` | "Who do I know?" | `network.*`, social interaction data |

---

## 2. Required Output Structure

When a runner calls `onComplete(result)`, the result object should include up to 4 standard top-level fields:

```typescript
interface RunnerOutput {
  // Required: the main activity result data
  data: {
    // Activity-specific fields (scores, analysis, etc.)
    [key: string]: unknown;
  };

  // Recommended: pre-structured data for AI memory extraction
  journey_contributions?: {
    professional?: { /* see Section 3 */ };
    personal?: { /* see Section 3 */ };
    network?: { /* see Section 3 */ };
    market?: { /* see Section 3 */ };
    // ... other domains
  };

  // Optional: RNCP competency codes validated by this activity
  covered_skills?: Array<{
    code: string;   // e.g. 'C3.1'
    level: number;  // 1-5
  }>;

  // Optional: display configuration for consultant dashboard
  dashboard_data?: {
    tiles?: Array<{ key: string; value: number; unit?: string }>;
    sections?: Array<{ title: string; content: string; icon?: string }>;
  };
}
```

### `_runner_metadata` (injected automatically)

The API route injects `_runner_metadata` into `responses` before saving to the database. **Do not override this key in your output.**

```typescript
// Injected by src/app/api/activity-sessions/[id]/complete/route.ts
_runner_metadata: {
  runner_type: string | null;    // e.g. 'cv-basic-analyzer'
  activity_title: string | null; // e.g. 'Analyse de CV basique'
  duration_ms: number | null;    // Time spent in milliseconds
  completed_at: string;          // ISO-8601 timestamp
}
```

The AI backend uses `_runner_metadata.runner_type` as `source.sourceId` in `CreateMemoryProposal`.

---

## 3. `journey_contributions` (Critical for AI)

### Purpose

`journey_contributions` is a pre-structured field that maps directly to memory card types. When present, the AI backend extracts memory proposals **deterministically** (Tier 2 — no LLM needed) with **high confidence (0.80-0.90)** because the data is already categorized by the runner developer.

**28 of 58 runners** (48%) currently include this field. All new runners should include it.

> **Tier 2 extraction**: The AI backend uses a `JOURNEY_MAPPING` dictionary that maps paths like `professional.career_goals` → `('aspiration', 0.90)`. Content is generated via templates. No LLM call is made. This means `journey_contributions` data is extracted faster, cheaper, and more reliably than flat data (Tier 3 / LLM).

### Canonical paths

Below are all `journey_contributions` paths found across the 28 runners that use them, organized by top-level domain.

#### `professional` (most common — 26/28 runners)

```typescript
professional: {
  // Profile & Experience → memory type: experience
  profile: {
    experiences: Experience[];
    education: Education[];
    skills: string[];
    years_experience: number;
  };

  // Positioning → memory type: trait, competence
  unique_value_proposition: {
    differentiators: string[];
    top_differentiator?: Differentiator;
    elevator_pitch?: string;
  };
  skills_positioning: {
    skills_with_benefits: SkillBenefit[];
    translation_complete: boolean;
    market_ready: boolean;
  };
  positioning_assessment: {
    score?: number;
    gaps?: Gap[];
    recommendations?: FormationReco[];
    strengths?: string[];
    quick_wins?: string[];
    strengths_tags?: string[];
  };

  // Development → memory type: aspiration
  development_plan: Plan6m;
  development_priorities: string[];
  career_goals: string[];
  priorities: string[];

  // Mini Bilan → memory type: trait, competence
  mini_bilan: {
    score: number;
    forces: string[];
    progres: string[];
    reco: string[];
    focus?: string;
  };
  patterns: {
    tags: string[];
    critical_gap: string;
  };

  // Market Awareness → memory type: competence, trait
  market_awareness: {
    score: number;
    perceived_needs: string[];
    identified_trends: string[];
    awareness_level: string;
    last_assessed: string;
  };
  market_positioning: {
    assessment_date: string;
    global_score: number;
    profil_type: string;
    target_role: string;
    strengths: string[];
    gaps: string[];
    chosen_strategy: string;
  };

  // Certification → memory type: experience
  certification: {
    scenario_id?: string;
    secteur?: string;
    simulation?: { exchanges: number; audio: boolean; duration_min: number };
    assessment?: { final_score: number; decision: string };
  };

  // Validations → memory type: competence, experience
  validations: {
    evidences: StarCase[] | Evidence[];
  };

  // Negotiation & Communication → memory type: competence
  negotiation: {
    limits_defined: boolean;
    arguments_prepared: number;
    red_line_documented: boolean;
  };
  objection_handling: {
    objections: number;
    scenarios: number;
    readiness: number;
  };

  // Visibility → memory type: competence, aspiration
  visibility: {
    audit_complete?: boolean;
    supports_audites?: string[];
    score_moyen?: number;
    channels_mapped?: number;
    strategy_defined?: boolean;
    activation_plan_ready?: boolean;
  };
  personal_branding: {
    strategy_defined: boolean;
    top_3_supports: string[];
  };

  // Competency Development → memory type: aspiration, competence
  montee_competences: {
    competences_techniques: number;
    competences_commerciales: number;
    competences_posture: number;
    horizon_mois: number;
  };

  // Job Assessments → memory type: experience
  job_assessments: Array<{
    company: string;
    position: string;
    score: number;
    decision: string;
    date: string;
  }>;
  career_clarity: {
    validated_trajectory?: string;
    key_criteria_identified: string[];
    deal_breakers_clarified: string[];
  };

  // Project Assessment → memory type: experience, aspiration
  project: {
    feasibility_assessed: boolean;
    global_score: number;
    diagnosis: string;
    obstacles_identified: number;
  };

  // Activation readiness → memory type: trait
  activation_readiness: {
    signals_identified?: boolean;
    signals_count?: number;
    formal_commitment?: boolean;
    commitment_quality?: 'high' | 'medium' | 'low';
  };
}
```

#### `personal` (5 runners)

```typescript
personal: {
  // Memory type: aspiration, trait
  vision_6m: string;
  keywords: string[];
  readiness_signals: string[];
  goals: Objective[];
  motivations: string[];

  // Memory type: emotion
  engagement_baseline: {
    confiance_niveau: number;
    date: string;
    formule: string;
  };
  commitment: string;

  // Memory type: trait
  perceived_obstacles: string[];
  action_levers: string[];
}
```

#### `network` (2 runners)

```typescript
network: {
  // Memory type: connection
  circles_mapped: number;
  total_contacts_identified: number;
  priority_contacts_selected: number;
  scripts_generated: number;
  activation_readiness: number;
}
```

#### `market` (3 runners)

```typescript
market: {
  // Memory type: competence, aspiration
  research: {
    interviews_prepared: boolean;
    contacts_identified: number;
    insights_gathered: number;
    positioning_adjusted: boolean;
  };
  positioning: {
    adjustment_made: boolean;
    confidence_level: number;
    differentiation_strategy?: boolean;
  };
}
```

#### Other domains

```typescript
// decision (2 runners) → memory type: aspiration, trait
decision: {
  validation: { radar_5d_complete: boolean; recommendation: string };
  market_validation: { fit_score: number; action_plan_defined: boolean; next_steps: string[] };
}

// skills (2 runners) → memory type: competence
skills: {
  communication: { signals_detected: number; accuracy_score: number };
  interview: { preparation_level: string; scenarios_practiced: number };
}

// autonomie (2 runners) → memory type: competence, trait
autonomie: {
  plan_long_terme_defini: boolean;
  competences_priorisees: number;
  score_autonomie: number;
}

// mindset (1 runner) → memory type: trait, emotion
mindset: {
  confidence: { boundaries_set: boolean; values_clarified: number };
}
```

### Examples from existing runners

**a2-bilan-360-complet** (most complete):
```typescript
journey_contributions: {
  professional: {
    profile: { experiences, education, skills, years_experience },
    unique_value_proposition: { differentiators, elevator_pitch },
    skills_positioning: { skills_with_benefits, market_ready },
    positioning_assessment: { score, gaps, recommendations },
    development_plan: { /* 6-month plan */ },
  }
}
```

**module-c1-mur-invisible** (personal + professional):
```typescript
journey_contributions: {
  personal: {
    perceived_obstacles: ['Peur du jugement', 'Syndrome de l\'imposteur'],
    action_levers: ['Reformulation positive', 'Ancrage factuel'],
  }
}
```

**module-b1-atelier-star** (STAR evidence):
```typescript
journey_contributions: {
  professional: {
    validations: {
      evidences: [
        { id, title, situation, task, action, result, metrics, skills }
      ]
    }
  }
}
```

---

## 4. `covered_skills` (RNCP Competencies)

### Purpose

`covered_skills` lists RNCP competency codes validated by the activity. These map **directly** to `competence` memory cards with **high confidence (0.95)** — no LLM interpretation needed.

**23 of 58 runners** (40%) currently include this field.

### Schema

```typescript
covered_skills: Array<{
  code: string;     // RNCP code, e.g. 'C3.1', 'C2.2'
  level: number;    // 1 = awareness, 2 = application, 3 = mastery (some use up to 5)
  validation_date?: string;  // ISO-8601 (only in cert-4-jury-assessment)
}>
```

### Deterministic extraction path

The AI backend creates `competence` memory cards directly from `covered_skills` as **Tier 1** (deterministic, no LLM):

```python
# Pseudo-code for runner_extraction.py
for skill in completion['responses'].get('covered_skills', []):
    proposals.append({
        'content': f"Compétence {skill['code']} validée au niveau {skill['level']}",
        'type': 'competence',
        'confidence': 0.95,
        'source': { 'type': 'runner', 'sourceId': runner_type, ... },
        'rawData': skill,
        'tags': ['rncp', skill['code']],
    })
```

### Common RNCP codes across runners

| Code | Runners using it | Domain |
|------|-----------------|--------|
| `C2.1`, `C2.2` | cert-*, module-b2, module-c3, ms3 | Professional validation |
| `C3.1`, `C3.2` | module-c1, module-c3, ms3 | Mindset & resilience |
| Open codes | a1, a3, a10, audit-*, radar-* | Activity-specific |

### Level semantics

| Level | Meaning | When to use |
|-------|---------|-------------|
| 1 | Awareness / introduction | User has been exposed to the concept |
| 2 | Application / practice | User has applied the skill in exercises |
| 3 | Mastery / demonstration | User has demonstrated competence |
| 4-5 | Extended range | Some runners use for advanced mastery |

---

## 5. `dashboard_data` (Display Only)

### Purpose

`dashboard_data` is configuration for the **consultant dashboard** — it controls how results are displayed to the supervising consultant. **It is NOT used for memory card extraction.**

The AI backend LLM prompt should explicitly say: *"Ignore `dashboard_data` fields — these are display configuration, not user memories."*

### Storage

`dashboard_data` is **copied** to a separate column in `activity_completions` for indexed access, but it also remains inside the `responses` JSONB. When `completeActivity()` saves:
- `responses` ← the full runner output object (includes `dashboard_data`, `journey_contributions`, everything)
- `dashboard_data` column ← `responses.dashboard_data || null` (GIN indexed copy for consultant dashboard queries)

The AI backend reads `responses` from the database (via the queue consumer). Since `dashboard_data` may appear inside `responses`, all extraction tiers skip it explicitly.

### Two common patterns

**Tiles-based** (module-b1-atelier-star, a2-bilan-360-complet):
```typescript
dashboard_data: {
  tiles: [
    { key: 'score_global', value: 85, unit: '%' },
    { key: 'cas_star', value: 4 },
  ],
  sections: { samples: StarCase[] }
}
```

**Sections-based** (module-c1-mur-invisible):
```typescript
dashboard_data: {
  title: 'Résumé Mur Invisible',
  summary: 'Le bénéficiaire a identifié 4 croyances limitantes...',
  sections: [
    { title: 'Croyances limitantes', content: '...', icon: 'brain' },
  ]
}
```

---

## 6. Schema Best Practices

### File structure

```
src/runners/my-runner/
├── runner.config.ts              # Imports schemas, exports ActivityRunner
├── schemas/
│   ├── output.schema.ts          # Zod schema for onComplete() result
│   └── input.schema.ts           # Zod schema for activity configuration
├── components/                    # React components
│   └── MyRunner.tsx
└── index.ts
```

### Export naming conventions (match existing patterns)

| File | Preferred export name | Also acceptable |
|------|-----------------------|-----------------|
| `schemas/output.schema.ts` | `outputSchema` (named) | `resultSchema`, `OutputSchema`, default export |
| `schemas/input.schema.ts` | `inputSchema` (named) | `configSchema`, `InputSchema`, default export |
| `runner.config.ts` | `runnerConfig` (named + default) | `config` (some older runners) |

### Importing schemas in runner.config.ts

```typescript
import type { ActivityRunner } from '@/lib/activity-system/types';

import { outputSchema } from './schemas/output.schema';
import { inputSchema } from './schemas/input.schema';
import MyRunnerComponent from './components/MyRunner';

export const runnerConfig: ActivityRunner = {
  metadata: { id: 'my-runner', name: 'My Runner', version: '1.0.0', /* ... */ },
  Component: MyRunnerComponent,
  schemas: {
    config: inputSchema,
    result: outputSchema,
  },
  // ...
};

export default runnerConfig;
```

### Schema rules

- **No `z.any()`** — use specific types (`z.string()`, `z.number()`, `z.union([...])`, `z.record(z.string(), z.unknown())`)
- **No `.passthrough()`** on inline schemas in runner.config.ts — use properly typed schemas from schema files
- **No `{} as any`** — use `z.object({})` for empty config schemas, or import the real schema
- Use `z.union([z.string(), z.number(), z.boolean(), z.array(z.string())])` for genuinely dynamic AI output

---

## 7. Runner Config Checklist

```typescript
import type { ActivityRunner } from '@/lib/activity-system/types';

import { outputSchema } from './schemas/output.schema';
import { inputSchema } from './schemas/input.schema';
import MyComponent from './components/MyComponent';

export const runnerConfig: ActivityRunner = {
  // ✅ Required: Metadata
  metadata: {
    id: 'my-runner-id',           // Unique, kebab-case, matches folder name
    name: 'My Runner Name',       // Human-readable
    version: '1.0.0',             // Semver
    description: 'What this runner does',
    author: 'Team Name',
    tags: ['category', 'topic'],
    icon: 'brain',                // Lucide icon name (optional)
  },

  // ✅ Required: React component
  Component: MyComponent,

  // ✅ Required: Typed schemas (NO {} as any, NO z.any())
  schemas: {
    config: inputSchema,          // or z.object({}) if no config needed
    result: outputSchema,         // Must match onComplete() output shape
  },

  // Optional: Dashboard for consultant view
  dashboard: {
    enabled: true,
    pattern: 'evaluation',        // 'evaluation' | 'accompaniment' | 'validation' | 'marketing'
  },

  // Optional: AI capabilities
  capabilities: {
    ai: { enabled: true, models: ['gpt-4o'], maxTokens: 4000 },
    credits: { consumption: 1, validation: true },
    storage: { type: 'local', maxSize: 5 * 1024 * 1024 },
  },
};

export default runnerConfig;
```

---

## 8. Memory Card Type Mapping Guide

### Three extraction tiers

| Tier | Source | Method | LLM needed? | ~% of completions |
|------|--------|--------|-------------|-------------------|
| **Tier 1** | `covered_skills` | Deterministic rules | No | ~40% |
| **Tier 2** | `journey_contributions` | Deterministic mapping (`JOURNEY_MAPPING`) | No | ~48% |
| **Tier 3** | Flat result fields | Full LLM extraction | Yes | ~20% |

Note: Tiers 1 and 2 often overlap (a completion can have both `covered_skills` and `journey_contributions`). Tier 3 is only called when neither field is present, or when flat fields contain data not covered by the other tiers.

### Which runner output fields map to which memory card types

| Runner output field | Memory card type | Confidence | Tier |
|---------------------|------------------|------------|------|
| `covered_skills[*]` | `competence` | 0.95 | Tier 1 (deterministic) |
| `journey_contributions.professional.profile` | `experience` | 0.90 | Tier 2 (deterministic) |
| `journey_contributions.professional.certification` | `experience` | 0.90 | Tier 2 (deterministic) |
| `journey_contributions.professional.job_assessments` | `experience` | 0.85 | Tier 2 (deterministic) |
| `journey_contributions.professional.validations` | `competence` | 0.85 | Tier 2 (deterministic) |
| `journey_contributions.professional.skills_positioning` | `competence` | 0.85 | Tier 2 (deterministic) |
| `journey_contributions.professional.development_plan` | `aspiration` | 0.85 | Tier 2 (deterministic) |
| `journey_contributions.professional.career_goals` | `aspiration` | 0.90 | Tier 2 (deterministic) |
| `journey_contributions.professional.positioning_assessment` | `trait` | 0.80 | Tier 2 (deterministic) |
| `journey_contributions.professional.market_positioning` | `trait` | 0.80 | Tier 2 (deterministic) |
| `journey_contributions.personal.perceived_obstacles` | `trait` | 0.85 | Tier 2 (deterministic) |
| `journey_contributions.personal.action_levers` | `trait` | 0.85 | Tier 2 (deterministic) |
| `journey_contributions.personal.vision_6m` | `aspiration` | 0.85 | Tier 2 (deterministic) |
| `journey_contributions.personal.engagement_baseline` | `emotion` | 0.80 | Tier 2 (deterministic) |
| `journey_contributions.personal.motivations` | `trait` | 0.85 | Tier 2 (deterministic) |
| `journey_contributions.network.*` | `connection` | 0.80 | Tier 2 (deterministic) |
| `journey_contributions.market.research` | `competence` | 0.75 | Tier 2 (deterministic) |
| `journey_contributions.mindset.confidence` | `emotion` | 0.80 | Tier 2 (deterministic) |
| Flat result fields (no journey_contributions) | Varies | 0.70-0.80 | Tier 3 (LLM) |

### Confidence levels explained

| Level | Meaning | Tier |
|-------|---------|------|
| **0.95** | Deterministic extraction (`covered_skills` → `competence`) | Tier 1 |
| **0.85-0.90** | Deterministic mapping from `journey_contributions` | Tier 2 |
| **0.80** | Deterministic mapping, less specific path | Tier 2 |
| **0.70-0.80** | Full LLM extraction from flat/unstructured data | Tier 3 |

### What makes good memory data

- **Specific**: "5 ans de gestion de projet chez Airbus" > "Expérience en gestion"
- **Actionable**: Includes concrete skills, dates, companies, scores
- **Categorizable**: Maps clearly to one of the 7 memory card types
- **User-attributable**: Describes the user, not the activity

### What makes bad memory data

- Activity metadata (runner version, processing time)
- Display configuration (dashboard tiles, section layouts)
- Intermediate states (step progress, draft scores)
- PII that shouldn't be stored as memory (passwords, tokens)

---

## 9. Anti-Patterns

### Schema anti-patterns

```typescript
// ❌ WRONG: {} as any in schemas
schemas: { config: {} as any, result: {} as any }

// ✅ CORRECT: Import typed schemas
import { outputSchema } from './schemas/output.schema';
schemas: { config: z.object({}), result: outputSchema }
```

```typescript
// ❌ WRONG: Inline schema with z.any() and passthrough
const resultSchema = z.object({
  actions: z.array(z.any()).optional(),
  kit: z.any().optional(),
}).passthrough();

// ✅ CORRECT: Import the proper schema file
import { outputSchema } from './schemas/output.schema';
```

```typescript
// ❌ WRONG: z.any() in output schema files
elements: z.record(z.string(), z.any())

// ✅ CORRECT: Use typed union for dynamic values
elements: z.record(z.string(), z.union([z.string(), z.number(), z.boolean(), z.array(z.string())]))
```

### Data anti-patterns

```typescript
// ❌ WRONG: Missing journey_contributions (data still extractable but lower quality)
onComplete({
  score: 85,
  forces: ['leadership', 'communication'],
  axes: ['délégation'],
});

// ✅ CORRECT: Include journey_contributions for high-quality extraction
onComplete({
  score: 85,
  forces: ['leadership', 'communication'],
  axes: ['délégation'],
  journey_contributions: {
    professional: {
      positioning_assessment: {
        score: 85,
        strengths: ['leadership', 'communication'],
        gaps: ['délégation'],
      },
    },
  },
});
```

```typescript
// ❌ WRONG: Overriding _runner_metadata (injected by API route)
onComplete({
  _runner_metadata: { runner_type: 'my-runner' },  // Will be overwritten anyway
  data: { ... },
});

// ✅ CORRECT: Let the API route handle _runner_metadata
onComplete({
  data: { ... },
});
```

```typescript
// ❌ WRONG: Storing PII that shouldn't become memory cards
journey_contributions: {
  personal: {
    email: 'user@example.com',      // PII leak
    phone: '+33 6 12 34 56 78',     // PII leak
  }
}

// ✅ CORRECT: Only store career/development data
journey_contributions: {
  personal: {
    motivations: ['Impact social', 'Créativité'],
    perceived_obstacles: ['Manque de confiance'],
  }
}
```

---

## 10. Testing

### Verify your runner output is AI-friendly

**1. Check schema wiring in runner.config.ts:**
```bash
# Find runners still using {} as any
grep -r "as any" src/runners/*/runner.config.ts

# Find runners with passthrough() in inline schemas
grep -r "passthrough()" src/runners/*/runner.config.ts

# Find remaining z.any() in output schemas
grep -r "z.any()" src/runners/*/schemas/output.schema.ts
```

**2. Check journey_contributions presence:**
```bash
# List runners with journey_contributions in output schema
grep -rl "journey_contributions\|journeyContributions" src/runners/*/schemas/output.schema.ts
```

**3. Check covered_skills presence:**
```bash
# List runners with covered_skills in output schema
grep -rl "covered_skills\|coveredSkill" src/runners/*/schemas/output.schema.ts
```

**4. Manual check — complete the activity, then inspect the database:**
```sql
-- In Supabase SQL Editor
SELECT
  id,
  activity_id,
  responses->'_runner_metadata' as metadata,
  responses->'journey_contributions' as contributions,
  responses->'covered_skills' as skills,
  dashboard_data
FROM activity_completions
WHERE user_id = '<your-user-id>'
ORDER BY completed_at DESC
LIMIT 5;
```

**5. Verify the output schema validates your data:**
```typescript
// In a test file
import { outputSchema } from '../schemas/output.schema';

test('onComplete output matches schema', () => {
  const result = {
    // ... your onComplete data
  };
  expect(() => outputSchema.parse(result)).not.toThrow();
});
```

---

## 11. Gap Analysis: NEXTJS_PLAN vs AI_BACKEND_PLAN

The following 10 gaps were identified between the two plans. They are documented here for the AI backend team to reference during implementation.

### GAP 1 (CRITICAL): `user_consents` table already exists

**AI_BACKEND_PLAN** proposes creating `user_consents` table (sql/05) with schema: `user_id, consent BOOLEAN, consent_version TEXT`.

**Reality**: Table already exists in `supabase/migrations/20260203000000_multi_org_complete.sql` with a DIFFERENT schema:

```sql
CREATE TABLE IF NOT EXISTS public.user_consents (
  id uuid DEFAULT gen_random_uuid() NOT NULL PRIMARY KEY,
  user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  consent_type character varying(50) NOT NULL,     -- 'service' | 'ai_training' | 'data_sharing'
  consent_given boolean NOT NULL,                   -- NOT a single 'consent' boolean
  consent_text text NOT NULL,
  consent_version character varying(10) NOT NULL,
  ip_address inet,
  user_agent text,
  created_at timestamp with time zone DEFAULT now(),
  revoked_at timestamp with time zone,              -- NULL if active, timestamp if revoked
  CONSTRAINT valid_consent_type CHECK (consent_type IN ('service', 'ai_training', 'data_sharing'))
);
```

**Consent flow:**
1. During the user's **first activity** (any runner, not just onboarding), the platform checks if an `ai_training` consent row exists. If not, the consent prompt is displayed and the response is written to `user_consents` with `consent_type='ai_training'`
2. For all subsequent activities, the consent prompt is never shown again — the AI backend **only reads** this table, never writes to it
3. If no consent row exists yet (user hasn't started any activity), the check returns `false` (fail-closed)

**Query:**
```sql
SELECT consent_given FROM user_consents
WHERE user_id = $1
  AND consent_type = 'ai_training'
  AND consent_given = true
  AND revoked_at IS NULL
ORDER BY created_at DESC
LIMIT 1;
```

**Fix**: Remove sql/05 from AI_BACKEND_PLAN. Update `consent_manager.py` to query existing schema.

---

### GAP 2 (CRITICAL): Memory cards API requires user JWT, not service tokens

**AI_BACKEND_PLAN** assumes calling `POST /api/harmonia/journal/memory-cards` with `HARMONIA_SERVICE_TOKEN`.

**Reality**: The endpoint uses `getRouteHandlerSession(request)` — requires authenticated user JWT cookie. It also validates `userId` matches the authenticated user (returns 403 on mismatch).

**Impact**: The queue consumer runs server-side without a user session. It CANNOT call this endpoint as-is.

**Fix**: Use a **`create_memory_proposal` RPC function** (`SECURITY DEFINER`) that enforces consent check + type validation at the DB level. The Python code calls this RPC via the Supabase service-role client. Even a bug in Python cannot bypass DB-level guardrails.

```python
# In harmonia_api.py — calls RPC, not direct table insert
supabase_admin = create_client(SUPABASE_URL, SUPABASE_KEY)  # service role
result = supabase_admin.rpc('create_memory_proposal', {
    'p_user_id': user_id,
    'p_content': proposal['content'],
    'p_type': proposal['type'],
    'p_confidence': proposal['confidence'],
    'p_source': proposal['source'],
    'p_raw_data': proposal.get('rawData'),
    'p_tags': proposal.get('tags', []),
}).execute()
# RPC raises exception if consent is missing or type is invalid
```

---

### GAP 3 (HIGH): `_runner_metadata` contract undocumented in AI backend

**NEXTJS_PLAN** injects `_runner_metadata: { runner_type, activity_title, completed_at, duration_ms }` inside `responses` JSONB.

**AI_BACKEND_PLAN** says "Build context from completion's structured data" but never specifies reading `_runner_metadata`.

**Fix**: Update `runner_extraction.py` to explicitly extract:
```python
metadata = completion['responses'].get('_runner_metadata', {})
source = {
    'type': 'runner',
    'sourceId': metadata.get('runner_type', 'unknown'),
    'activityId': completion.get('activity_id'),
    'extractedAt': datetime.utcnow().isoformat() + 'Z',
}
```

---

### GAP 4 (HIGH): `journey_contributions` — the gold mine ignored

**28 of 58 runners** (48%) include `journey_contributions` with pre-structured memory data.

**AI_BACKEND_PLAN** never mentioned this field initially.

**Fix**: Added **Tier 2 deterministic extraction** in `runner_extraction.py` — a `JOURNEY_MAPPING` rules engine that maps paths like `professional.career_goals` → `('aspiration', 0.90)` without any LLM call. Content is generated via templates. This makes `journey_contributions` extraction faster, cheaper, and more reliable than LLM-based extraction.

---

### GAP 5 (HIGH): `covered_skills` maps directly to competence cards

**23 of 58 runners** (40%) output `covered_skills: [{ code: 'C3.1', level: 3 }]`.

These map directly to `competence` memory cards without LLM interpretation.

**Fix**: Add deterministic extraction path in `runner_extraction.py`:
```python
# Before LLM call, extract covered_skills deterministically
covered_skills = completion['responses'].get('covered_skills', [])
for skill in covered_skills:
    proposals.append(CreateMemoryProposal(
        content=f"Compétence {skill['code']} validée au niveau {skill['level']}",
        type='competence',
        confidence=0.95,
        source=source,
        rawData=skill,
        tags=['rncp', skill['code']],
    ))
```

---

### GAP 6 (MEDIUM): `dashboard_data` is a separate DB column, not inside responses

`activity_completions` has a **separate** `dashboard_data` JSONB column. The mutation splits `onComplete({ data, dashboard_data })` into separate columns.

**AI_BACKEND_PLAN** reads `responses` from the database, which may contain dashboard display data.

**Fix**: All extraction tiers explicitly skip `dashboard_data` and `dashboard_summary` keys. The Tier 3 LLM prompt also includes an explicit ignore instruction.

---

### GAP 7 (MEDIUM): `consent_audit_log` table — not needed

**AI_BACKEND_PLAN** proposes `consent_audit_log` table (sql/06).

The existing `user_consents` table already tracks consent history via `created_at` + `revoked_at` timestamps (each consent action creates a new row). The platform also has an `audit_logs` table for general audit trails.

**Fix**: Remove sql/06 from AI_BACKEND_PLAN. Use existing tables.

---

### GAP 8 (MEDIUM): 3 passthrough runners lack `journey_contributions` — FIXED

`maintenance-personal-branding`, `module-4-1-activation-plan-action`, `strategie-fidelisation-professionnelle` had proper output schemas but NO `journey_contributions` field.

**Fix applied**: Added `journey_contributions` with optional fields to all 3 output schemas:
- **maintenance-personal-branding**: `professional.visibility` + `professional.personal_branding`
- **module-4-1-activation-plan-action**: `professional.activation_readiness` + `professional.development_priorities` + `personal.perceived_obstacles` + `personal.action_levers`
- **strategie-fidelisation-professionnelle**: `network.*` (contacts, circles) + `professional.negotiation` + `professional.career_clarity`

---

### GAP 9 (LOW): ~~Realtime event shape uncertainty~~ — RESOLVED

~~**AI_BACKEND_PLAN** assumes `record['new']['responses']`.~~

**No longer relevant.** The architecture now uses a **queue table + PostgreSQL trigger** instead of Supabase Realtime. The queue consumer fetches completion records directly via `SELECT` — no event shape ambiguity.

---

### GAP 10 (LOW): `endMessage` HTML — not a gap

Some runners pass `onComplete(result, { endMessage: '<div>...</div>' })`. The `endMessage` is stored in `metadata` column, NOT in `responses`.

**Fix**: None needed.

---

### GAP 11 (MEDIUM): `journey_contributions` is NOT a separate column — FIXED in playbook

The playbook data flow diagram originally said "Splits: responses / dashboard_data / journey_contributions". This was misleading.

**Reality** (verified in `src/lib/supabase/mutations/activity-mutations.ts`):
- `responses` JSONB ← ALL runner output (including `journey_contributions`, `dashboard_data`, everything)
- `dashboard_data` column ← a **copy** of `responses.dashboard_data` (for GIN-indexed consultant queries)
- There is **NO** separate `journey_contributions` column

**Impact**: The AI backend reads `responses` from the database (via queue consumer) and finds `journey_contributions` at `responses.journey_contributions`. This works correctly — no code change needed, just documentation clarity.

**Fix applied**: Updated the data flow diagram and Section 5 in this playbook.

---

### GAP 12 (CRITICAL): POP-507 migration not applied — DB rejects new types

**Current `memory_cards` table** (verified 2026-02-13):
- `type` CHECK: `['fact', 'preference', 'decision', 'objective', 'constraint', 'sensitivity']` — **OLD 6 types**
- `status` CHECK: `['proposed', 'validated', 'archived', 'rejected']` — **OLD 4 statuses**
- `raw_data` column: **DOES NOT EXIST**

**AI backend writes**: `type: 'competence'`, `status: 'validated'`, `raw_data: {...}`

**Impact**: The AI backend assumes POP-507 schema. Since we're still in development, the DB will be clean — apply POP-507 migration before first deploy. No backward-compatibility handling needed.

**Fix**: Apply POP-507 migration to the clean dev DB before deploying the AI backend. Updated AI_BACKEND_PLAN.md with current DB state for reference.

---

### GAP 13 (CRITICAL): No code writes `ai_training` consent during any activity

**No runner currently collects AI training consent.** No code anywhere in the platform writes `consent_type='ai_training'` to `user_consents`.

**Current state** (verified 2026-02-13):
- `user_consents` table exists with valid `consent_type` values including `'ai_training'`
- Privacy API exists: `PUT /api/privacy/consents/ai_training` can write consent rows
- No runner includes a consent step or checkbox for AI training
- Users can manually toggle consent in privacy settings, but nobody tells them to

**Impact**: `consent_manager.check_consent(user_id)` returns `False` for **every user** → no memory cards are ever extracted. The fail-closed pipeline works as designed, but the entire feature is dead on arrival.

**Design decision:** The consent prompt should be displayed during the user's **first activity** — regardless of which runner it is (onboarding, CV analyzer, or any other). The platform checks if a `user_consents` row with `consent_type='ai_training'` exists for the user. If not, the consent modal/step is shown before or at the start of the activity.

**Fix required (frontend, before AI backend goes live):**
1. Add a generic AI consent check that runs before any activity starts (e.g., in the activity session initialization or in a shared wrapper component)
2. If no `ai_training` consent row exists for the user, display the consent modal/step
3. Call `PUT /api/privacy/consents/ai_training` with `{ consent_given: true/false }` on user response
4. Must capture consent text, version, IP, user agent (RGPD)
5. Once consent is recorded (yes or no), the prompt never appears again

Updated AI_BACKEND_PLAN.md to list this as a hard blocker alongside POP-507.

---

## 12. AI_BACKEND_PLAN Amendments

Based on the gap analysis, the following amendments should be applied when implementing the AI backend:

| # | Amendment | Impact | Status |
|---|-----------|--------|--------|
| 1 | **Remove sql/05 and sql/06** — Use existing `user_consents` table. Query: `consent_type='ai_training' AND consent_given=true AND revoked_at IS NULL` | Avoids table conflicts | Applied to AI_BACKEND_PLAN |
| 2 | **Use RPC function instead of direct table insert** — `create_memory_proposal()` SECURITY DEFINER function with DB-level consent check + type validation. Called via Supabase service-role client | Unforgeable guardrails at DB level | Applied to AI_BACKEND_PLAN |
| 3 | **Read `_runner_metadata` from `responses`** — Use `responses._runner_metadata.runner_type` as `source.sourceId` | Proper source identification | Applied to AI_BACKEND_PLAN |
| 4 | **Tier 1: Deterministic extraction for `covered_skills`** — Skip LLM, create `competence` cards directly (0.95 confidence) | Better accuracy, lower cost | Applied to AI_BACKEND_PLAN |
| 5 | **Tier 2: Deterministic extraction for `journey_contributions`** — `JOURNEY_MAPPING` rules engine, no LLM needed. Reduces LLM calls by ~48% | Faster, cheaper, more reliable | Applied to AI_BACKEND_PLAN |
| 6 | **Ignore `dashboard_data`** — Display config, not memory data. Excluded from all extraction tiers | Avoid false positives | Applied to AI_BACKEND_PLAN |
| 7 | **Remove `consent_audit_log` table** — Existing `user_consents` + `audit_logs` tables suffice | Simpler architecture | Applied to AI_BACKEND_PLAN |
| 8 | **BLOCKER: Apply POP-507 migration** — Current DB has old type/status constraints that reject new values. `raw_data` column missing | Pipeline non-functional without it | Documented as hard blocker |
| 9 | **BLOCKER: Add AI consent to first activity** — No code writes `ai_training` consent. `check_consent()` returns False for all users. Consent prompt must appear on the user's first activity (any runner, not just onboarding) | Zero extraction for all users | Documented as hard blocker |
| 10 | **Replace Realtime with queue table + trigger** — `memory_extraction_queue` with `FOR UPDATE SKIP LOCKED` for exactly-once, crash-safe processing | Reliable delivery, retry, backfill | Applied to AI_BACKEND_PLAN |
| 11 | **Add failure recovery** — Queue tracks status (pending/processing/completed/failed/skipped), max 3 retries, reprocess endpoint | No lost extractions | Applied to AI_BACKEND_PLAN |
| 12 | **Three consent layers** — Layer 1: orchestrator, Layer 2: Python function, Layer 3: PostgreSQL RPC SECURITY DEFINER | Defense in depth, unforgeable | Applied to AI_BACKEND_PLAN |

### CreateMemoryProposal format reminder (POP-507 v2)

```typescript
interface CreateMemoryProposal {
  content: string;                    // Human-readable summary
  type: MemoryCardType;               // 'competence' | 'experience' | 'preference' | 'aspiration' | 'trait' | 'emotion' | 'connection'
  confidence: number;                 // 0-1
  source: {
    type: 'runner' | 'coach' | 'manual';
    sourceId: string;                 // _runner_metadata.runner_type
    activityId?: string;              // Activity UUID
    extractedAt: string;              // ISO-8601
  };
  rawData?: Record<string, unknown>;  // Structured data from runner
  tags?: string[];
}
```

> **Note**: The current `memory_cards` DB table uses OLD types (`fact`, `preference`, `decision`, `objective`, `constraint`, `sensitivity`) and OLD statuses (`proposed`, `validated`, `archived`, `rejected`). POP-507 migration must be applied first to update constraints to the 7 new types and 7 new statuses before the AI backend can write cards. The `create_memory_proposal` RPC function (sql/06) depends on the POP-507 CHECK constraints being in place. Cards are created with `status: 'validated'` (automatically active — user consents once during their first activity).
