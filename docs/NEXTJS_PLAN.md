# Plan: Improve Runner Data Quality for AI Memory Card Extraction

## Context

An external AI backend (Python/FastAPI) consumes `activity_completions` via a queue table (`memory_extraction_queue`, fed by a PostgreSQL trigger on `status='completed'`). It reads the `responses` JSONB column and uses a three-tier extraction pipeline to create `CreateMemoryProposal` objects (POP-507 spec: 7 types - competence, experience, preference, aspiration, trait, emotion, connection). Tiers 1-2 are deterministic (covered_skills + journey_contributions), Tier 3 uses LLM for flat data.

**Problem:** 16 runners have broken/missing schemas (`{} as any` or `.passthrough()`), causing unvalidated, unpredictable data structures. The LLM needs clear, semantic, well-structured data to avoid miscategorization.

**Key finding:** ALL 58 runners already save to `activity_completions.responses` via `onComplete()`. The `storage: local` config flag only controls intermediate persistence, NOT final DB destination. No localStorage migration needed.

**Critical discovery:** Most broken runners already have proper `schemas/output.schema.ts` files sitting right next to their `runner.config.ts` - they're just not wired up. The fix is mostly mechanical imports.

---

## Step 1: Inject `_runner_metadata` in API routes (all 58 runners, 2 files)

### File: `src/app/api/activity-sessions/[id]/complete/route.ts`

The session completion endpoint calls `completeActivity()` with `responses: results || {}`. We enrich `results` with `_runner_metadata` BEFORE passing to `completeActivity()`. The activity data is already available in `popCoachRawActivity` (which includes `runner_type` and `title`).

**Two code paths to modify:**

#### Primary path (~line 289)

Before `completeActivity()` call, add:

```typescript
// Enrich responses with runner metadata for downstream AI extraction
const enrichedResults = {
  ...(results || {}),
  _runner_metadata: {
    runner_type: popCoachRawActivity?.runner_type || null,
    activity_title: popCoachRawActivity?.title || null,
    duration_ms: duration,
    completed_at: new Date().toISOString(),
  },
};
```

Then change `responses: results || {}` to `responses: enrichedResults`.

#### Duplicate detection path (~line 203)

Same enrichment before the `completeActivity()` call in the duplicate branch:

```typescript
const duplicateEnrichedResults = {
  ...(results || {}),
  _runner_metadata: {
    runner_type: popCoachRawActivity?.runner_type || null,
    activity_title: popCoachRawActivity?.title || null,
    duration_ms: duration,
    completed_at: new Date().toISOString(),
  },
};
```

Then change `responses: results || {}` to `responses: duplicateEnrichedResults`.

### File: `src/app/api/activity-completions/route.ts`

Move the activity data fetch (currently at ~line 116) BEFORE the `upsertActivityCompletion()` call (~line 90). Then enrich responses:

```typescript
// Fetch activity data for runner metadata enrichment and Pop Coach notification
const { data: activityData } = await supabase
  .from('activities')
  .select(
    'credit_cost, title, description, short_description, activity_type, runner_type, tags, difficulty_level, estimated_duration, learning_objectives, prerequisites'
  )
  .eq('id', validatedData.activity_id)
  .single();

// Enrich responses with runner metadata for downstream AI extraction
const enrichedResponses = {
  ...(validatedData.responses || {}),
  _runner_metadata: {
    runner_type: activityData?.runner_type || null,
    activity_title: activityData?.title || null,
    duration_ms: validatedData.time_spent ? validatedData.time_spent * 1000 : null,
    completed_at: validatedData.completed_at || new Date().toISOString(),
  },
};
```

Then change `responses: validatedData.responses || {}` to `responses: enrichedResponses` in the `upsertActivityCompletion()` call.

Remove the duplicate activity fetch that was previously at ~line 116 (it's already fetched above).

---

## Step 2: Wire existing schemas for 8 `{} as any` runners

These runners have well-defined output/input schema files that exist but are NOT imported in `runner.config.ts`.

### 2.1 `a2-bilan-360-complet/runner.config.ts`

- **Current:** `export const config: any = { ... schemas: { config: {} as any, result: {} as any } }`
- **Schema files:** `schemas/output.schema.ts` (default export `resultSchema`), `schemas/input.schema.ts` (default export `inputSchema`)
- **Fix:**
  1. Add imports: `import resultSchema from './schemas/output.schema';` and `import inputSchema from './schemas/input.schema';`
  2. Replace `schemas: { config: {} as any, result: {} as any }` with `schemas: { config: inputSchema, result: resultSchema }`
  3. Fix export: Change `export const config: any =` to proper typed export (import `ActivityRunner` type)

### 2.2 `a3-mini-bilan-express/runner.config.ts`

- **Current:** `export const runnerConfig: ActivityRunner = { ... schemas: { config: {} as any, result: {} as any } }`
- **Schema files:** `schemas/output.schema.ts` (named export `OutputSchema`), `schemas/input.schema.ts` (named export `InputSchema`)
- **Fix:**
  1. Add imports: `import { OutputSchema } from './schemas/output.schema';` and `import { InputSchema } from './schemas/input.schema';`
  2. Replace `schemas: { config: {} as any, result: {} as any }` with `schemas: { config: InputSchema, result: OutputSchema }`

### 2.3 `cert-2-preparation-20min/runner.config.ts`

- **Current:** `export const config: any = { ... schemas: { config: {} as any, result: {} as any } }`
- **Schema files:** Root-level `schemas.ts` has named export `OutputSchema`. The `schemas/output.schema.ts` and `schemas/input.schema.ts` files are **empty stubs** (only `import { z } from 'zod'`).
- **Fix:**
  1. Add import: `import { OutputSchema } from './schemas';`
  2. Replace `schemas: { config: {} as any, result: {} as any }` with `schemas: { config: z.object({}), result: OutputSchema }`
  3. Add `import { z } from 'zod';` if not already present
  4. Fix export: Change `export const config: any =` to proper typed export

### 2.4 `module-b1-atelier-star/runner.config.ts`

- **Current:** `export const config: any = { ... schemas: { config: {} as any, result: {} as any } }`
- **Schema files:** `schemas/output.schema.ts` (named export `outputSchema`), `schemas/input.schema.ts` (named export `inputSchema`)
- **Fix:**
  1. Add imports: `import { outputSchema } from './schemas/output.schema';` and `import { inputSchema } from './schemas/input.schema';`
  2. Replace `schemas: { config: {} as any, result: {} as any }` with `schemas: { config: inputSchema, result: outputSchema }`
  3. Fix export: Change `export const config: any =` to proper typed export

### 2.5 `module-b2-portfolio-validation/runner.config.ts`

- **Current:** `export const runnerConfig: ActivityRunner = { ... schemas: { config: {} as any, result: {} as any } }`
- **Schema files:** `schemas/output.schema.ts` (named export `outputSchema`), `schemas/input.schema.ts` (named export `inputSchema`)
- **Fix:**
  1. Add imports: `import { outputSchema } from './schemas/output.schema';` and `import { inputSchema } from './schemas/input.schema';`
  2. Replace `schemas: { config: {} as any, result: {} as any }` with `schemas: { config: inputSchema, result: outputSchema }`

### 2.6 `module-c1-mur-invisible/runner.config.ts`

- **Current:** `export const runnerConfig: ActivityRunner = { ... schemas: { config: {} as any, result: {} as any } }`
- **Schema files:** `schemas/output.schema.ts` (named export `outputSchema`), `schemas/input.schema.ts` (named export `inputSchema`)
- **Fix:**
  1. Add imports: `import { outputSchema } from './schemas/output.schema';` and `import { inputSchema } from './schemas/input.schema';`
  2. Replace `schemas: { config: {} as any, result: {} as any }` with `schemas: { config: inputSchema, result: outputSchema }`

### 2.7 `module-c3-kit-reassurance/runner.config.ts`

- **Current:** `export const runnerConfig: ActivityRunner = { ... schemas: { config: {} as any, result: {} as any } }`
- **Schema files:** `schemas/output.schema.ts` (named export `outputSchema`), `schemas/input.schema.ts` (named export `inputSchema`)
- **Fix:**
  1. Add imports: `import { outputSchema } from './schemas/output.schema';` and `import { inputSchema } from './schemas/input.schema';`
  2. Replace `schemas: { config: {} as any, result: {} as any }` with `schemas: { config: inputSchema, result: outputSchema }`

### 2.8 `module-d1-simulation-entretien/runner.config.ts`

- **Current:** `export const runnerConfig: ActivityRunner = { ... schemas: { config: {} as any, result: {} as any } }`
- **Schema files:** `schemas/output.schema.ts` (named export `outputSchema`), `schemas/input.schema.ts` (named export `inputSchema`)
- **Fix:**
  1. Add imports: `import { outputSchema } from './schemas/output.schema';` and `import { inputSchema } from './schemas/input.schema';`
  2. Replace `schemas: { config: {} as any, result: {} as any }` with `schemas: { config: inputSchema, result: outputSchema }`

### 4 micro-synthesis runners (config only)

ms1-portrait, ms2-connexions, ms3-patterns-competences, ms5-engagement-personnel all have `result` already wired (e.g. `result: ms1ResultSchema`). Only `config: {} as any` needs fixing.

**Fix for each:** Replace `config: {} as any` with `config: z.object({})` and add `import { z } from 'zod';` if not already present.

---

## Step 3: Replace passthrough schemas for 3 runners + fix 1

### 3.1 `module-4-1-activation-plan-action/runner.config.ts`

- **Current inline schema (lines 21-28):**
  ```typescript
  const resultSchema = z.object({
    h1Actions: z.array(z.any()).optional(),
    decompositions: z.array(z.any()).optional(),
    antiBlockageKit: z.any().optional(),
    completedAt: z.string().optional(),
  }).passthrough();
  ```
- **Proper schema:** `schemas/output.schema.ts` exports named `outputSchema` (line 73) - fully typed, no `z.any()`
- **Fix:**
  1. Remove the inline `resultSchema` definition (lines 21-28)
  2. Add import: `import { outputSchema } from './schemas/output.schema';`
  3. Replace `result: resultSchema` with `result: outputSchema` in schemas block

### 3.2 `maintenance-personal-branding/runner.config.ts`

- **Current inline schema (lines 22-52):** 9 instances of `z.any()` + `.passthrough()`
- **Proper schema:** `schemas/output.schema.ts` exports named `outputSchema` (line 93) - fully typed, no `z.any()`
- **Fix:**
  1. Remove the inline `resultSchema` definition (lines 22-52)
  2. Add import: `import { outputSchema } from './schemas/output.schema';`
  3. Replace `result: resultSchema` with `result: outputSchema` in schemas block

### 3.3 `strategie-fidelisation-professionnelle/runner.config.ts`

- **Current inline schema (lines 21-30):** 5 instances of `z.any()` + `.passthrough()`
- **Proper schema:** `schemas/output.schema.ts` exports named `SessionOutputSchema` (line 125) - fully typed, no `z.any()`
- **NOTE:** Export name is `SessionOutputSchema`, not `outputSchema`
- **Fix:**
  1. Remove the inline `resultSchema` definition (lines 21-30)
  2. Add import: `import { SessionOutputSchema } from './schemas/output.schema';`
  3. Replace `result: resultSchema` with `result: SessionOutputSchema` in schemas block

### 3.4 `cv-basic-analyzer` (already wired, Step 4 handles its z.any())

This runner already imports schemas from `./schemas.ts` properly. Its `schemas.ts` has `configSchema` and `resultSchema` correctly imported in `runner.config.ts`. The only issue is a nested `z.any()` addressed in Step 4.

---

## Step 4: Clean up remaining `z.any()` in 3 wired schemas

### 4.1 `cv-basic-analyzer/schemas.ts` - Line 555

**Current:**
```typescript
dashboardData: z.object({
  raw_data: z.record(z.string(), z.any()),  // Line 555
  metrics: z.object({ ... }),
  ai_analysis: z.object({ ... }).optional(),
})
```

**Actual TypeScript type (types.ts:517-530):**
```typescript
raw_data: {
  analysis_result: OutputAnalyseBasique;
  input_metadata: { document_hash: string; processing_time: number; llm_usage: boolean; };
  quality_metrics: { completeness_score: number; confidence_score: number; ambiguity_count: number; invariants_passed: number; };
}
```

**Fix:** Replace `z.record(z.string(), z.any())` with a typed schema matching the TS type:
```typescript
raw_data: z.object({
  analysis_result: z.record(z.string(), z.unknown()).optional(),
  input_metadata: z.object({
    document_hash: z.string(),
    processing_time: z.number(),
    llm_usage: z.boolean(),
  }).optional(),
  quality_metrics: z.object({
    completeness_score: z.number(),
    confidence_score: z.number(),
    ambiguity_count: z.number(),
    invariants_passed: z.number(),
  }).optional(),
}).passthrough()
```

Note: `analysis_result` is `OutputAnalyseBasique` which is a large type - use `z.record(z.string(), z.unknown())` with `.passthrough()` on the parent to avoid over-constraining it. All fields `.optional()` since this is cached/derived data.

### 4.2 `module-c3-kit-reassurance/schemas/output.schema.ts` - Line 32

**Current:**
```typescript
elements: z.record(z.string(), z.any())  // Line 32
```

**Analysis:** This is genuinely dynamic data from AI output (keys like "salaire", "teletravail" with mixed value types). Rendered generically with `String(value)`.

**Fix:** Replace with a union type:
```typescript
elements: z.record(z.string(), z.union([z.string(), z.number(), z.boolean(), z.array(z.string())]))
```

This covers realistic AI output types while being semantic for memory card extraction.

### 4.3 `a2-bilan-360-complet/schemas/output.schema.ts` - Line 78

**Current:**
```typescript
sections: z.object({
  gaps: z.array(gapSchema),
  reco: z.array(formationRecoSchema),
  plan_brief: z.any(),  // Line 78
})
```

**Analysis:** `plan_brief` has zero references in implementation code. The full `plan_6m` field exists separately with proper typing via `plan6mSchema`. This appears to be a dashboard summary.

**Fix:** Replace with:
```typescript
plan_brief: z.string().optional()
```

A brief textual summary of the plan, or `null`/missing if not generated.

---

## Step 5: Fix `export const config: any` typing for 3 runners

Runners `a2-bilan-360-complet`, `cert-2-preparation-20min`, and `module-b1-atelier-star` use `export const config: any =` instead of the proper `export const runnerConfig: ActivityRunner =`.

**For each:**
1. Add import: `import type { ActivityRunner } from '@/lib/activity-system/types';`
2. Change `export const config: any =` to `export const config: ActivityRunner =`
   - Keep the variable name `config` since it's used by `export default config` at the bottom

---

## Verification Strategy

### 1. TypeScript validation
```bash
pnpm typecheck  # 0 errors expected
```

### 2. Lint
```bash
pnpm lint  # No 'any' type violations in runner schemas
```

### 3. Build
```bash
pnpm build  # Schemas used at runtime by runner auto-discovery
```

### 4. Grep audit for remaining issues
```bash
# Find remaining {} as any in runner configs
grep -r "as any" src/runners/*/runner.config.ts

# Find remaining passthrough() in inline schemas (runner.config.ts only)
grep -r "passthrough()" src/runners/*/runner.config.ts

# Find remaining z.any() in output schemas
grep -r "z.any()" src/runners/*/schemas/output.schema.ts
```

---

## Files Modified Summary

| Step | File | Change |
|------|------|--------|
| 1 | `src/app/api/activity-sessions/[id]/complete/route.ts` | Add `_runner_metadata` to responses (2 code paths) |
| 1 | `src/app/api/activity-completions/route.ts` | Move activity fetch up, add `_runner_metadata` to responses |
| 2 | `src/runners/a2-bilan-360-complet/runner.config.ts` | Wire schemas, fix export type |
| 2 | `src/runners/a3-mini-bilan-express/runner.config.ts` | Wire schemas |
| 2 | `src/runners/cert-2-preparation-20min/runner.config.ts` | Wire schema from `./schemas`, fix export type |
| 2 | `src/runners/module-b1-atelier-star/runner.config.ts` | Wire schemas, fix export type |
| 2 | `src/runners/module-b2-portfolio-validation/runner.config.ts` | Wire schemas |
| 2 | `src/runners/module-c1-mur-invisible/runner.config.ts` | Wire schemas |
| 2 | `src/runners/module-c3-kit-reassurance/runner.config.ts` | Wire schemas |
| 2 | `src/runners/module-d1-simulation-entretien/runner.config.ts` | Wire schemas |
| 2 | `src/runners/ms1-portrait/runner.config.ts` | Fix `config: {} as any` -> `config: z.object({})` |
| 2 | `src/runners/ms2-connexions/runner.config.ts` | Fix `config: {} as any` -> `config: z.object({})` |
| 2 | `src/runners/ms3-patterns-competences/runner.config.ts` | Fix `config: {} as any` -> `config: z.object({})` |
| 2 | `src/runners/ms5-engagement-personnel/runner.config.ts` | Fix `config: {} as any` -> `config: z.object({})` |
| 3 | `src/runners/module-4-1-activation-plan-action/runner.config.ts` | Remove inline schema, import proper one |
| 3 | `src/runners/maintenance-personal-branding/runner.config.ts` | Remove inline schema, import proper one |
| 3 | `src/runners/strategie-fidelisation-professionnelle/runner.config.ts` | Remove inline schema, import `SessionOutputSchema` |
| 4 | `src/runners/cv-basic-analyzer/schemas.ts` | Type `raw_data` properly |
| 4 | `src/runners/module-c3-kit-reassurance/schemas/output.schema.ts` | Replace `z.any()` with union type |
| 4 | `src/runners/a2-bilan-360-complet/schemas/output.schema.ts` | Replace `plan_brief: z.any()` with `z.string().optional()` |

**Total: 20 files modified** (2 API routes + 15 runner configs + 3 schema files)

---

## What We're NOT Doing

- No new DB columns or migrations (POP-507 migration is external backend scope)
- No `_memoryHints` annotations (LLM-based extraction handles categorization)
- No localStorage-to-DB migration (all runners already reach DB via `onComplete()`)
- No changes to runner Component code (only `runner.config.ts` schema wiring)
- No new API endpoints
- No changes to the `reports` column
- No Step 5 "metadata audit" from the original plan (deferred - most runners already pass metadata via `onComplete()` and the `_runner_metadata` injection in Step 1 covers the critical fields)

---

## Hard Prerequisites (for AI memory extraction to work end-to-end)

These are NOT in scope of this plan but must happen before the AI backend goes live:

1. **POP-507 DB migration** — Current `memory_cards` table has OLD type/status constraints and no `raw_data` column. Since we're still in development the DB will be clean — apply POP-507 migration before first deploy. Assignee: Roberto Caravaca Herrera.

2. **AI consent on first activity** — No runner currently collects `ai_training` consent. No code writes `consent_type='ai_training'` to `user_consents`. Without this, the AI backend's fail-closed consent check blocks extraction for every user. Needs: a generic consent check that triggers on the user's first activity (any runner, not just onboarding) + call to `PUT /api/privacy/consents/ai_training`.
