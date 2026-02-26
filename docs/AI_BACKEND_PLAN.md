# GDPR-Compliant Memory Card Generation from Runner Completions (Revised)

## Summary of decisions

| Topic | Decision |
|-------|----------|
| Consent revocation | Handled by frontend — we don't implement it |
| Consent collection | Done once during the user's **first activity** (any runner, not just onboarding) — writes to existing `user_consents` table with `consent_type='ai_training'` |
| Consent check | **Three layers**: (1) orchestrator checks before extraction, (2) Python `create_memory_proposal_rpc()` validates type, (3) PostgreSQL RPC function checks consent + type at DB level — fail-closed at all levels |
| Runner mapping | Three-tier: Tier 1 deterministic (covered_skills), Tier 2 deterministic (journey_contributions), Tier 3 LLM (flat data only, ~20% of completions) |
| Memory card format | POP-507 v2 (`CreateMemoryProposal`) with new types and statuses |
| `store_extracted_information()` | Replace entirely with POP-507 v2 format (both chat and runner) |
| Source types | POP-507 compliant: `'runner' \| 'coach' \| 'manual'` (no `'chat'`) |
| Event delivery | Queue table + PostgreSQL trigger (not Realtime — see rationale below) |
| Extraction tiers | Tier 1: `covered_skills` deterministic → Tier 2: `journey_contributions` deterministic → Tier 3: flat data → LLM |
| DB guardrails | `create_memory_proposal()` RPC function enforces consent + type validation at DB level |
| Failure recovery | Queue with status tracking, retry (max 3), reprocess endpoint |
| `activity_completions` schema | In development — design for flexibility |

---

## Prerequisites (HARD BLOCKERS)

### Blocker 1: POP-507 DB migration

**POP-507 DB migration must be applied first.** Our pipeline depends on schema changes defined in POP-507.

**Current DB state** (verified 2026-02-13):
- `memory_cards.type` CHECK constraint: `['fact', 'preference', 'decision', 'objective', 'constraint', 'sensitivity']` — **OLD 6 types**
- `memory_cards.status` CHECK constraint: `['proposed', 'validated', 'archived', 'rejected']` — **OLD 4 statuses**
- `raw_data` column: **DOES NOT EXIST**
- `linked_contexts` column: **EXISTS** (TEXT[], default `'{}'`)

**Required by POP-507:**
- `raw_data` JSONB column — required for `rawData` field in `CreateMemoryProposal`
- `type` constraint updated to 7 new values (`competence`, `experience`, `preference`, `aspiration`, `trait`, `emotion`, `connection`) — inserting `type: 'competence'` will fail with constraint violation until migrated
- `status` constraint updated to 7 new values (`proposed`, `validated`, `processing`, `applied`, `rejected`, `failed`, `archived`)
- `processing_started_at`, `applied_at`, `mapping_error`, `mapping_attempts`, `canonical_mappings`, `applied_field_paths` columns

**Impact:** The AI backend assumes the POP-507 schema is in place. Since we are still in development, the DB will be clean — apply POP-507 migration before first deploy. No backward-compatibility handling needed.

Coordinate with frontend team (POP-507 assignee: Roberto Caravaca Herrera).

### Blocker 2: AI training consent collection on first activity

**No runner currently collects AI training consent.** No code in the platform writes `consent_type='ai_training'` to the `user_consents` table.

**Current state** (verified 2026-02-13):
- `user_consents` table exists with `consent_type IN ('service', 'ai_training', 'data_sharing')`
- Privacy API exists: `PUT /api/privacy/consents/ai_training` can write consent rows
- **But no UI triggers this call during any activity**

**Impact if not fixed:** `consent_manager.check_consent(user_id)` will return `False` for **every user** → no memory cards are ever extracted. The pipeline is fail-closed by design.

**Design decision:** The consent prompt should be displayed during the user's **first activity** — regardless of whether that activity is the onboarding runner or any other runner. The platform checks if `user_consents` already has an `ai_training` row for this user. If not, it shows the consent modal/step before the runner starts (or as the first step within the runner).

**Required work (frontend):**
1. Add a generic AI consent check that runs before any activity starts (e.g., in the activity session initialization or in a shared wrapper component)
2. If no `ai_training` consent row exists for the user, display the consent modal/step
3. On user response, call `PUT /api/privacy/consents/ai_training` with `{ consent_given: true/false }`
4. Must capture: consent text, version, IP address, user agent (RGPD requirement)
5. Once consent is recorded (yes or no), the prompt never appears again in subsequent activities

---

## Architecture

### Why NOT Supabase Realtime

Supabase Realtime is designed for **browser subscriptions**, not server-to-server critical pipelines:
- **No delivery guarantee** — if the Python service is down, events are lost forever (no replay, no dead letter queue)
- **Reconnection gaps** — events fired during WebSocket reconnection are missed
- **In-memory state loss** — service restart loses deduplication state → duplicate cards

Instead we use a **queue table + PostgreSQL trigger**, which provides:
- **Exactly-once processing** via `SELECT ... FOR UPDATE SKIP LOCKED`
- **Crash recovery** — unprocessed rows persist across restarts
- **Automatic retry** — failed rows stay in queue with attempt counter
- **Visibility** — query the queue to see backlog, failure rates, throughput
- **Backfill** — re-insert old completions for reprocessing

### Pipeline

```
activity_completions INSERT/UPDATE (status='completed')
        │ PostgreSQL trigger (enqueue_memory_extraction)
        ▼
memory_extraction_queue (status: 'pending')
        │
        ▼
┌──────────────────────────────────┐
│  QueueConsumer                   │
│  (polls every 5s, FOR UPDATE    │
│   SKIP LOCKED, batch of 10)     │
└──────────┬───────────────────────┘
           ▼
   ConsentManager.check_consent(user_id)  ← Layer 1 (fail-closed)
           │ (if False → queue status: 'skipped')
           │ (if True ↓)
           ▼
   RunnerExtraction.process(completion)
           │
           ├─ Tier 1: covered_skills → competence cards (deterministic, no LLM)
           ├─ Tier 2: journey_contributions → mapped cards (deterministic rules, no LLM)
           └─ Tier 3: remaining flat data → LLM extraction (only ~20% of completions)
           │
           ▼
   create_memory_proposal() RPC function  ← DB-level consent + type validation (Layer 3)
           │
           ▼
   memory_cards table (status: 'validated' — automatically active, no manual review)
   queue status: 'completed' (or 'failed' → retry, max 3 attempts)
```

---

## POP-507 v2 Memory Card Types (7, replacing old 6)

| Type | Question it answers | Primary canonical targets |
|------|---------------------|--------------------------|
| `competence` | "What can I do?" | professional.skillsValidated, professional.industryKnowledge, learning.knowledgeGraph |
| `experience` | "What have I done?" | professional.experiences, professional.certifications, professional.portfolioItems, professional.currentPosition |
| `preference` | "What do I like?" | psychological.workingStyles, learning.preferredFormats, aspirational.industries, aspirational.geographicPreferences |
| `aspiration` | "Where am I going?" | aspirational.dreamRoles, aspirational.lifeGoals, aspirational.salaryExpectations, learning.learningGoals, learning.skillGapsIdentified |
| `trait` | "Who am I?" | psychological.personalityProfile, psychological.valuesHierarchy, psychological.motivationsCore, psychological.strengthsIdentified, psychological.areasForGrowth |
| `emotion` | "How am I feeling?" | emotional.confidenceTrajectory, emotional.energyPatterns, emotional.stressTriggers, emotional.celebrationMoments, emotional.resilienceFactors |
| `connection` | "Who do I know?" | social.mentorsConnected, social.peersJourney, social.testimonialsReceived |

## POP-507 v2 Statuses (7, replacing old 4)

```
proposed → validated → processing → applied
proposed → rejected
processing → failed → validated (retry, max 3)
applied → archived
```

Our pipeline creates cards with `status: 'validated'` — skipping the `proposed` stage. The user consents once during their first activity; after that, all memory cards are created automatically without requiring individual review. Users can still view, edit, or delete cards from their profile at any time.

## POP-507 Source Types

POP-507 defines three valid source types:
- `'runner'` — automated extraction from runner/activity completions
- `'coach'` — data entered by a coach
- `'manual'` — data entered directly by the user

The old `'chat'` source type is **not** in POP-507. The chat flow will use `'manual'` going forward.

## CreateMemoryProposal format (POP-507 v2)

```typescript
interface CreateMemoryProposal {
  content: string;                    // Human-readable summary
  type: MemoryCardType;               // 'competence' | 'experience' | 'preference' | 'aspiration' | 'trait' | 'emotion' | 'connection'
  confidence: number;                 // 0-1
  source: {
    type: 'runner' | 'coach' | 'manual';
    sourceId: string;                 // e.g. "cv-analyzer-v3.1", "chat-session"
    activityId?: string;              // Activity UUID (runner flow)
    sessionId?: string;               // Session UUID (chat flow)
    extractedAt: string;              // ISO-8601
  };
  rawData?: Record<string, unknown>;  // Structured data from runner result or LLM extraction
  tags?: string[];
}
```

### Example: Runner proposal

```json
{
  "userId": "user-uuid",
  "content": "5 years of project management at Airbus",
  "type": "experience",
  "confidence": 0.88,
  "source": {
    "type": "runner",
    "sourceId": "cv-analyzer-v3.1",
    "activityId": "abc-123",
    "extractedAt": "2026-02-12T10:00:00Z"
  },
  "rawData": {
    "title": "Project Manager",
    "company": "Airbus",
    "startDate": "2018-01-01",
    "endDate": "2023-06-30",
    "skills": ["management", "agile"]
  },
  "tags": ["professional", "cv-analysis"],
  "status": "validated"
}
```

### Example: Chat proposal

```json
{
  "userId": "user-uuid",
  "content": "User wants to become a UX designer",
  "type": "aspiration",
  "confidence": 0.85,
  "source": {
    "type": "manual",
    "sourceId": "chat-extraction",
    "sessionId": "session-uuid",
    "extractedAt": "2026-02-12T10:00:00Z"
  },
  "rawData": {
    "title": "UX Designer",
    "appeal": "creative work with user impact"
  },
  "tags": ["aspirational"],
  "status": "validated"
}
```

### Note on `linkedContexts`

The current API supports a `linkedContexts` field (used in the old `store_extracted_information()`). POP-507's `CreateMemoryProposal` does **not** define `linkedContexts` — only `tags`. We drop `linkedContexts` and use `tags` only, per POP-507 spec. If the API still requires `linkedContexts`, we send an empty array.

### Note on `userId` and `status` in request payload

POP-507's `CreateMemoryProposal` interface does **not** include `userId` or `status` as fields the caller sends. The API endpoint may resolve `userId` from the auth token or route parameter. Our pipeline explicitly sets `status: 'validated'` (not the default `'proposed'`) so cards are immediately active without requiring manual user review. **Suggestion**: confirm with the frontend team whether `POST /api/harmonia/journal/memory-cards` expects `userId` and `status` in the body, or if they are handled server-side.

---

## Subcategory → POP-507 type mapping (for chat flow migration)

When updating the chat flow (`information_extraction_node` → `store_information_node`), old extraction subcategories map to new POP-507 types:

| Old subcategory | POP-507 type |
|-----------------|--------------|
| `skills`, `knowledge` | `competence` |
| `learning_velocity` | `competence` (**see note below**) |
| `experiences`, `certifications`, `current_position` | `experience` |
| `preferred_format`, `work_style` | `preference` |
| `dream_roles`, `salary_expectations`, `life_goals`, `impact_legacy`, `skill_expertise` | `aspiration` |
| `personality_profile`, `strengths`, `weaknesses`, `motivations`, `values` | `trait` |
| `confidence`, `stress`, `energy_patterns`, `celebration_moments` | `emotion` |
| `mentors`, `journey_peers`, `people_helped`, `testimonials` | `connection` |

This mapping will be defined as a constant `SUBCATEGORY_TO_CARD_TYPE` in `training/constants/info_extraction.py`.

> **Suggestion**: `learning_velocity` ("how fast I learn") answers "Who am I?" (`trait`) more than "What can I do?" (`competence`). POP-507 has no direct canonical target for learning speed under `competence`. Consider moving `learning_velocity` to `trait` instead.

---

## Files to create

### 1. `src/services/__init__.py`
Empty package init.

### 2. `src/services/consent_manager.py`

**Consent flow:**
1. When a user starts their **first activity** (any runner, not just onboarding), the platform checks if an `ai_training` consent row exists. If not, the consent prompt is displayed before/during the activity.
2. The frontend writes to `user_consents` table: `consent_type='ai_training'`, `consent_given=true/false`
3. For **all subsequent activities**, the consent prompt is never shown again — the AI backend only reads this table, never writes to it

- `check_consent(user_id: str) → bool`
  - Read-only query on **existing** `user_consents` table:
    ```sql
    SELECT consent_given FROM user_consents
    WHERE user_id = $1
      AND consent_type = 'ai_training'
      AND consent_given = true
      AND revoked_at IS NULL
    ORDER BY created_at DESC
    LIMIT 1;
    ```
  - **Fail-closed**: returns `False` on any error, empty result, or if user hasn't completed their first activity yet (no consent row exists)
  - TTL cache (60s) to reduce DB load — most completions from the same user happen in bursts
- `invalidate_consent_cache(user_id: str)` → clears cache entry (called if consent revocation webhook is added later)

### 3. `src/services/runner_extraction.py`

Three-tier pipeline — LLM only for Tier 3 (~20% of completions):

- `process_completion(completion_record: dict) → list[dict]`

  **Step 0: Extract `_runner_metadata` and build source** (GAP 3 fix)
  ```python
  responses = completion_record.get('responses', {})
  metadata = responses.get('_runner_metadata', {})
  source = {
      'type': 'runner',
      'sourceId': metadata.get('runner_type', 'unknown'),
      'activityId': completion_record.get('activity_id'),
      'extractedAt': datetime.utcnow().isoformat() + 'Z',
  }
  ```

  **Step 1 (Tier 1): Deterministic — `covered_skills`** (23/58 runners, ~40%)
  Skip LLM entirely for RNCP competency codes:
  ```python
  covered_skills = responses.get('covered_skills', [])
  for skill in covered_skills:
      proposals.append({
          'content': f"Compétence {skill['code']} validée au niveau {skill['level']}",
          'type': 'competence',
          'confidence': 0.95,
          'source': source,
          'rawData': skill,
          'tags': ['rncp', skill['code']],
      })
  ```

  **Step 2 (Tier 2): Deterministic — `journey_contributions`** (28/58 runners, ~48%)
  No LLM needed — type assignment and content generation via rules engine:
  ```python
  JOURNEY_MAPPING = {
      # path → (card_type, confidence)
      # professional domain
      'professional.profile':              ('experience', 0.90),
      'professional.unique_value_proposition': ('trait', 0.85),
      'professional.skills_positioning':   ('competence', 0.85),
      'professional.positioning_assessment': ('trait', 0.80),
      'professional.development_plan':     ('aspiration', 0.85),
      'professional.development_priorities': ('aspiration', 0.85),
      'professional.career_goals':         ('aspiration', 0.90),
      'professional.mini_bilan':           ('trait', 0.85),
      'professional.market_awareness':     ('competence', 0.80),
      'professional.market_positioning':   ('trait', 0.80),
      'professional.certification':        ('experience', 0.90),
      'professional.validations':          ('competence', 0.85),
      'professional.negotiation':          ('competence', 0.85),
      'professional.objection_handling':   ('competence', 0.85),
      'professional.visibility':           ('competence', 0.80),
      'professional.personal_branding':    ('aspiration', 0.80),
      'professional.montee_competences':   ('competence', 0.85),
      'professional.job_assessments':      ('experience', 0.85),
      'professional.career_clarity':       ('aspiration', 0.85),
      'professional.project':              ('experience', 0.85),
      'professional.activation_readiness': ('trait', 0.80),
      # personal domain
      'personal.vision_6m':                ('aspiration', 0.85),
      'personal.goals':                    ('aspiration', 0.85),
      'personal.motivations':              ('trait', 0.85),
      'personal.engagement_baseline':      ('emotion', 0.80),
      'personal.perceived_obstacles':      ('trait', 0.85),
      'personal.action_levers':            ('trait', 0.85),
      'personal.commitment':               ('trait', 0.80),
      # network domain
      'network':                           ('connection', 0.80),
      # market domain
      'market.research':                   ('competence', 0.75),
      'market.positioning':                ('trait', 0.80),
      # decision domain
      'decision.validation':               ('trait', 0.80),
      'decision.market_validation':        ('aspiration', 0.80),
      # skills domain
      'skills.communication':              ('competence', 0.85),
      'skills.interview':                  ('competence', 0.85),
      # autonomie domain
      'autonomie':                         ('competence', 0.80),
      # mindset domain
      'mindset.confidence':                ('emotion', 0.80),
  }

  def extract_from_journey(journey: dict, source: dict) -> list[dict]:
      """Tier 2: deterministic extraction from journey_contributions."""
      proposals = []
      for path, (card_type, confidence) in JOURNEY_MAPPING.items():
          value = resolve_nested_path(journey, path)
          if value is None or value == {} or value == []:
              continue
          content = generate_content(path, value)
          proposals.append({
              'content': content,
              'type': card_type,
              'confidence': confidence,
              'source': source,
              'rawData': {'path': path, 'value': value},
              'tags': [path.split('.')[0]],
          })
      return proposals

  def resolve_nested_path(data: dict, path: str):
      """Resolve 'a.b.c' into data['a']['b']['c']."""
      keys = path.split('.')
      current = data
      for key in keys:
          if not isinstance(current, dict) or key not in current:
              return None
          current = current[key]
      return current

  def generate_content(path: str, value) -> str:
      """Template-based content generation: French label + structured detail."""
      label = CONTENT_TEMPLATES.get(path, path.replace('.', ' > ').replace('_', ' ').title())

      # Array values: append count
      if isinstance(value, list):
          return f"{label} ({len(value)} éléments)"
      # Dict values: append key scalar summaries
      if isinstance(value, dict):
          parts = []
          for k, v in value.items():
              if isinstance(v, (str, int, float, bool)):
                  parts.append(str(v))
          detail = ', '.join(parts[:3])  # max 3 values for readability
          return f"{label}: {detail}" if detail else label
      # Scalar values
      return f"{label}: {value}"

  CONTENT_TEMPLATES = {
      # professional domain (21 paths)
      'professional.profile':              "Profil professionnel identifié",
      'professional.unique_value_proposition': "Proposition de valeur unique définie",
      'professional.skills_positioning':   "Positionnement compétences réalisé",
      'professional.positioning_assessment': "Évaluation de positionnement complétée",
      'professional.development_plan':     "Plan de développement à 6 mois défini",
      'professional.development_priorities': "Priorités de développement identifiées",
      'professional.career_goals':         "Objectifs de carrière définis",
      'professional.mini_bilan':           "Mini-bilan express réalisé",
      'professional.market_awareness':     "Connaissance du marché évaluée",
      'professional.market_positioning':   "Positionnement marché analysé",
      'professional.certification':        "Certification: scénario complété",
      'professional.validations':          "Évidences professionnelles validées",
      'professional.negotiation':          "Préparation à la négociation réalisée",
      'professional.objection_handling':   "Gestion des objections préparée",
      'professional.visibility':           "Audit de visibilité complété",
      'professional.personal_branding':    "Stratégie de personal branding définie",
      'professional.montee_competences':   "Plan de montée en compétences établi",
      'professional.job_assessments':      "Évaluations de postes réalisées",
      'professional.career_clarity':       "Clarté de trajectoire professionnelle renforcée",
      'professional.project':              "Projet professionnel évalué",
      'professional.activation_readiness': "Préparation à l'activation évaluée",
      # personal domain (7 paths)
      'personal.vision_6m':               "Vision à 6 mois définie",
      'personal.goals':                   "Objectifs personnels identifiés",
      'personal.motivations':             "Motivations identifiées",
      'personal.engagement_baseline':      "Baseline d'engagement mesurée",
      'personal.perceived_obstacles':      "Obstacles perçus identifiés",
      'personal.action_levers':           "Leviers d'action identifiés",
      'personal.commitment':              "Engagement formalisé",
      # network domain (1 path)
      'network':                          "Réseau professionnel cartographié",
      # market domain (2 paths)
      'market.research':                  "Recherche marché réalisée",
      'market.positioning':               "Positionnement marché ajusté",
      # decision domain (2 paths)
      'decision.validation':              "Validation de décision complétée",
      'decision.market_validation':       "Validation marché réalisée",
      # skills domain (2 paths)
      'skills.communication':             "Compétences de communication évaluées",
      'skills.interview':                 "Préparation aux entretiens réalisée",
      # autonomie domain (1 path)
      'autonomie':                        "Niveau d'autonomie évalué",
      # mindset domain (1 path)
      'mindset.confidence':               "Niveau de confiance évalué",
  }
  ```

  **Step 3 (Tier 3): LLM extraction — flat data only** (~20% of completions)
  Only called when completion has **neither** `covered_skills` **nor** `journey_contributions`, or when flat result fields contain data not covered by Tiers 1-2:
  ```python
  # Collect remaining data not already extracted
  already_extracted = {'_runner_metadata', 'covered_skills', 'journey_contributions',
                       'dashboard_data', 'dashboard_summary'}
  remaining = {k: v for k, v in responses.items() if k not in already_extracted}

  if remaining and not journey_had_data:
      # Only call LLM if there's meaningful data left AND journey_contributions didn't cover it
      llm_proposals = call_llm_extraction(remaining, source)
      proposals.extend(llm_proposals)
      # Rate limiting: avoid hitting OpenAI RPM limits during backlog drain
      await asyncio.sleep(0.5)  # ~120 RPM max, well under GPT-4o limits
  ```

  LLM system prompt (Tier 3 only):
  ```
  Extract memory card proposals from this runner completion data.
  Assign each to one of: competence, experience, preference, aspiration, trait, emotion, connection.
  Use confidence 0.70-0.80 for these flat extractions.

  IGNORE:
  - dashboard_data, dashboard_summary (UI layout)
  - _runner_metadata (already processed)
  - covered_skills (already processed)
  - journey_contributions (already processed)
  ```

  **Step 4: Validate each proposal** before sending:
  - `type` must be one of the 7 POP-507 types — discard proposals with invalid types
  - `content` must be a non-empty string
  - `confidence` must be a float between 0 and 1
  - `rawData` must be a dict if present

  **Step 5:** For each valid proposal: call `create_memory_proposal()` RPC via `harmonia_api.py`

- Uses OpenAI client from `src/config.py` (Tier 3 only)
- Allowed types constant: `VALID_CARD_TYPES = {"competence", "experience", "preference", "aspiration", "trait", "emotion", "connection"}`

### 4. `src/services/runner_orchestrator.py`
Queue-based processing loop:
- `process_queue_item(queue_item: dict) → str`
  Returns final status: `'completed'`, `'failed'`, or `'skipped'`

  ```python
  def process_queue_item(queue_item: dict) -> str:
      completion_id = queue_item['completion_id']
      user_id = queue_item['user_id']

      # 1. Fetch full completion record
      completion = supabase_admin.table('activity_completions') \
          .select('*').eq('id', completion_id).single().execute()
      if not completion.data:
          logger.warning(f"Completion {completion_id} not found")
          return 'failed'

      # 2. Check consent (Layer 1 — fail-closed)
      if not consent_manager.check_consent(user_id):
          logger.info(f"No ai_training consent for user {user_id}, skipping")
          return 'skipped'

      # 3. Extract proposals (Tiers 1-3)
      proposals = runner_extraction.process_completion(completion.data)
      logger.info(f"Extracted {len(proposals)} proposals from completion {completion_id}")

      # 4. Write each proposal via RPC (Layer 3 consent check at DB level)
      written = 0
      for proposal in proposals:
          try:
              result = harmonia_api.create_memory_proposal_rpc(user_id, proposal)
              if result:
                  written += 1
          except Exception as e:
              logger.error(f"Failed to write proposal: {e}")
              # Continue with other proposals — don't fail the whole batch

      logger.info(f"Written {written}/{len(proposals)} memory cards for completion {completion_id}")
      return 'completed'
  ```

Deduplication is handled by the queue table: the trigger uses `ON CONFLICT DO NOTHING` on `completion_id`, so duplicate completions are impossible.

### 5. `src/services/queue_consumer.py`
Polling-based queue consumer (replaces `realtime_listener.py`):

```python
class QueueConsumer:
    """Polls memory_extraction_queue for pending items."""

    def __init__(self, poll_interval: float = 5.0, batch_size: int = 10, max_retries: int = 3):
        self.poll_interval = poll_interval
        self.batch_size = batch_size
        self.max_retries = max_retries
        self._running = False

    async def start(self):
        """Start the polling loop. Called from FastAPI lifespan."""
        self._running = True
        logger.info("QueueConsumer started")
        while self._running:
            try:
                processed = await self._poll_batch()
                if processed == 0:
                    await asyncio.sleep(self.poll_interval)
                # If we processed items, immediately check for more (no sleep)
            except Exception as e:
                logger.error(f"Queue poll error: {e}")
                await asyncio.sleep(self.poll_interval * 2)  # back off on error

    async def stop(self):
        """Stop the polling loop gracefully."""
        self._running = False
        logger.info("QueueConsumer stopped")

    async def _poll_batch(self) -> int:
        """Fetch and process a batch of pending items. Returns count processed."""
        # SELECT ... FOR UPDATE SKIP LOCKED prevents concurrent processing
        # and survives crashes (lock released on disconnect)
        items = supabase_admin.rpc('claim_extraction_batch', {
            'batch_size': self.batch_size,
            'max_retries': self.max_retries,
        }).execute()

        if not items.data:
            return 0

        for item in items.data:
            try:
                status = runner_orchestrator.process_queue_item(item)
                supabase_admin.table('memory_extraction_queue').update({
                    'status': status,
                    'completed_at': datetime.utcnow().isoformat(),
                }).eq('id', item['id']).execute()
            except Exception as e:
                logger.error(f"Processing failed for queue item {item['id']}: {e}")
                supabase_admin.table('memory_extraction_queue').update({
                    'status': 'failed',
                    'attempts': item['attempts'] + 1,
                    'last_error': str(e)[:500],
                }).eq('id', item['id']).execute()

        return len(items.data)
```

**Key design choices:**
- **`FOR UPDATE SKIP LOCKED`** — multiple consumers can run without double-processing
- **No sleep after processing** — drains backlog as fast as possible, only sleeps when queue is empty
- **Crash-safe** — if the service dies mid-processing, items stuck in `processing` for >10 minutes are automatically recovered by `claim_extraction_batch` (treated as failed, retried)
- **Batch processing** — fetches 10 items at a time for efficiency
- **Rate-limited LLM** — Tier 3 calls include a 0.5s delay to stay under OpenAI RPM limits
- Started/stopped with FastAPI lifespan
- Uses `SUPABASE_KEY` (service role)

### 6. SQL Migrations

> **NOTE (GAP 1 fix):** `user_consents` table already exists in production
> (`supabase/migrations/20260203000000_multi_org_complete.sql`). Do NOT create sql/05 or sql/06.
> Existing schema uses `consent_type VARCHAR(50)` with CHECK constraint `('service', 'ai_training', 'data_sharing')`,
> `consent_given BOOLEAN`, and `revoked_at TIMESTAMPTZ` (null = active). Each consent action creates a new row,
> providing full audit history — no separate `consent_audit_log` table needed.

#### `sql/05_memory_extraction_queue.sql`
```sql
-- Queue table for reliable extraction processing
CREATE TABLE IF NOT EXISTS memory_extraction_queue (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  completion_id UUID NOT NULL REFERENCES activity_completions(id) ON DELETE CASCADE,
  user_id UUID NOT NULL,
  status VARCHAR(20) NOT NULL DEFAULT 'pending',
  attempts INTEGER NOT NULL DEFAULT 0,
  last_error TEXT,
  created_at TIMESTAMPTZ DEFAULT now(),
  started_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ,
  CONSTRAINT valid_queue_status CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'skipped')),
  CONSTRAINT unique_completion UNIQUE (completion_id)
);

-- Index for queue polling: pending items ordered by creation time
CREATE INDEX idx_extraction_queue_pending
  ON memory_extraction_queue (created_at)
  WHERE status = 'pending';

-- Index for failed items eligible for retry
CREATE INDEX idx_extraction_queue_retry
  ON memory_extraction_queue (created_at)
  WHERE status = 'failed' AND attempts < 3;

-- Trigger: auto-enqueue when a completion reaches status='completed'
CREATE OR REPLACE FUNCTION enqueue_memory_extraction()
RETURNS TRIGGER AS $$
BEGIN
  IF NEW.status = 'completed' AND (TG_OP = 'INSERT' OR OLD.status IS DISTINCT FROM 'completed') THEN
    INSERT INTO memory_extraction_queue (completion_id, user_id)
    VALUES (NEW.id, NEW.user_id)
    ON CONFLICT (completion_id) DO NOTHING;  -- idempotent: no duplicates
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_enqueue_memory_extraction
  AFTER INSERT OR UPDATE ON activity_completions
  FOR EACH ROW EXECUTE FUNCTION enqueue_memory_extraction();

-- RPC: claim a batch of pending items for processing (FOR UPDATE SKIP LOCKED)
-- Also recovers items stuck in 'processing' for >10 minutes (crash recovery)
CREATE OR REPLACE FUNCTION claim_extraction_batch(batch_size INTEGER DEFAULT 10, max_retries INTEGER DEFAULT 3)
RETURNS SETOF memory_extraction_queue AS $$
BEGIN
  RETURN QUERY
  UPDATE memory_extraction_queue
  SET status = 'processing', started_at = now(), attempts = attempts + 1
  WHERE id IN (
    SELECT id FROM memory_extraction_queue
    WHERE status = 'pending'
       OR (status = 'failed' AND attempts < max_retries)
       OR (status = 'processing' AND started_at < now() - interval '10 minutes')  -- stuck recovery
    ORDER BY created_at
    LIMIT batch_size
    FOR UPDATE SKIP LOCKED
  )
  RETURNING *;
END;
$$ LANGUAGE plpgsql;
```

#### `sql/06_create_memory_proposal_rpc.sql`
```sql
-- RPC function: write a memory card with DB-level consent check + type validation
-- This is the THIRD layer of defense (after orchestrator + Python function)
CREATE OR REPLACE FUNCTION create_memory_proposal(
  p_user_id UUID,
  p_content TEXT,
  p_type TEXT,
  p_confidence NUMERIC,
  p_source JSONB,
  p_raw_data JSONB DEFAULT NULL,
  p_tags TEXT[] DEFAULT '{}'
) RETURNS UUID AS $$
DECLARE
  v_consent BOOLEAN;
  v_card_id UUID;
BEGIN
  -- DB-level consent check (Layer 3 — defense in depth)
  SELECT consent_given INTO v_consent
  FROM user_consents
  WHERE user_id = p_user_id
    AND consent_type = 'ai_training'
    AND consent_given = true
    AND revoked_at IS NULL
  ORDER BY created_at DESC
  LIMIT 1;

  IF v_consent IS NOT TRUE THEN
    RAISE EXCEPTION 'No ai_training consent for user %', p_user_id
      USING ERRCODE = 'P0001';
  END IF;

  -- Type validation is enforced by CHECK constraint on memory_cards.type column (POP-507)
  -- Status set to 'validated' — cards are automatically active (user consented during their first activity)
  INSERT INTO memory_cards (user_id, content, type, confidence, source, raw_data, tags, status)
  VALUES (p_user_id, p_content, p_type, p_confidence, p_source, p_raw_data, p_tags, 'validated')
  RETURNING id INTO v_card_id;

  RETURN v_card_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Grant execute to service role only
REVOKE ALL ON FUNCTION create_memory_proposal FROM PUBLIC;
GRANT EXECUTE ON FUNCTION create_memory_proposal TO service_role;
```

---

## Files to modify

### 1. `src/config.py`
Add:
```python
# Runner extraction
RUNNER_EXTRACTION_ENABLED = os.getenv("RUNNER_EXTRACTION_ENABLED", "true").lower() in ("true", "1", "yes")
QUEUE_POLL_INTERVAL = float(os.getenv("QUEUE_POLL_INTERVAL", "5.0"))  # seconds between polls
QUEUE_BATCH_SIZE = int(os.getenv("QUEUE_BATCH_SIZE", "10"))           # items per poll
QUEUE_MAX_RETRIES = int(os.getenv("QUEUE_MAX_RETRIES", "3"))          # max attempts before permanent failure
# HARMONIA_SERVICE_TOKEN removed — using RPC function instead (GAP 2 fix)
```
Add to `Tables` class:
```python
USER_CONSENTS = "user_consents"              # Already exists in production — do NOT create (GAP 1 fix)
MEMORY_EXTRACTION_QUEUE = "memory_extraction_queue"  # New — queue for reliable extraction
# CONSENT_AUDIT_LOG removed — existing user_consents table provides audit history (GAP 7 fix)
```

### 2. `src/utils/harmonia_api.py`

**Replace `store_extracted_information()`** with POP-507 v2 function:
- New function: `create_memory_proposal_rpc(user_id, proposal) → str | None`
  - **Calls the `create_memory_proposal` RPC function** (sql/06) — NOT direct table insert, NOT the HTTP API
  - **Three layers of defense:**
    - Layer 1: Orchestrator checks consent before extraction (early exit)
    - Layer 2: Python function validates `card_type` against allowed set (application-level)
    - Layer 3: RPC function checks consent + type constraint in PostgreSQL (DB-level, unforgeable)
  - Source must conform to: `{ type: 'runner' | 'coach' | 'manual', sourceId, activityId?, sessionId?, extractedAt }`
  - Always sets `status: 'validated'` (cards are automatically active, no manual review)
  - Uses `tags` (not `linkedContexts`)

> **GAP 2 fix:** The HTTP API endpoint `POST /api/harmonia/journal/memory-cards` requires a user JWT
> session. The queue consumer runs server-side without a user session, so it CANNOT call this endpoint.
> Instead, call the `create_memory_proposal` RPC function which runs as `SECURITY DEFINER`, enforcing
> consent check and type validation at the DB level — even a bug in Python code cannot bypass these guardrails.
>
> ```python
> from services.consent_manager import check_consent
> from supabase import create_client
>
> VALID_CARD_TYPES = {'competence', 'experience', 'preference', 'aspiration', 'trait', 'emotion', 'connection'}
>
> def create_memory_proposal_rpc(user_id: str, proposal: dict) -> str | None:
>     """Write a memory card via RPC. Returns card UUID or None."""
>     card_type = proposal.get('type')
>
>     # Layer 2: application-level type validation (fast fail before DB round-trip)
>     if card_type not in VALID_CARD_TYPES:
>         logger.error(f"Invalid card type: {card_type}")
>         return None
>
>     try:
>         supabase_admin = create_client(SUPABASE_URL, SUPABASE_KEY)  # service role
>         result = supabase_admin.rpc('create_memory_proposal', {
>             'p_user_id': user_id,
>             'p_content': proposal['content'],
>             'p_type': card_type,
>             'p_confidence': proposal['confidence'],
>             'p_source': proposal['source'],
>             'p_raw_data': proposal.get('rawData'),
>             'p_tags': proposal.get('tags', []),
>         }).execute()
>         return result.data  # UUID of created card
>     except Exception as e:
>         if 'No ai_training consent' in str(e):
>             logger.warning(f"DB consent check blocked card for user {user_id}")
>         else:
>             logger.error(f"RPC create_memory_proposal failed: {e}")
>         return None
> ```

- **Remove dead code**:
  - `SUBCATEGORY_ENDPOINTS` dict
  - All `transform_*` functions (~30 functions)
  - `check_profile_exists()`, `create_profile()`, `ensure_profile_exists()` — no longer needed since we're creating proposals, not writing to canonical profile directly
  - `HARMONIA_SERVICE_TOKEN` env var — no longer needed (using Supabase service client instead)

### 3. `training/constants/info_extraction.py`

- **Add `SUBCATEGORY_TO_CARD_TYPE` mapping**: maps extraction subcategories to POP-507 types (see mapping table above)
- **Update `EXTRACTION_SCHEMAS`**: change `"type": "fact"` entries to the correct POP-507 type for each subcategory (e.g., `skills` → `"type": "competence"`, `experiences` → `"type": "experience"`, etc.)

### 4. `src/agents/langgraph_workflow.py`

- Update `store_information_node` to call new `create_memory_proposal()` instead of old `store_extracted_information()`
- Use `SUBCATEGORY_TO_CARD_TYPE` to resolve the correct POP-507 type
- Pass extracted JSON as `rawData`
- Build source as `{ type: "manual", sourceId: "chat-extraction", sessionId: <if available>, extractedAt: <now> }`

### 5. `main.py`
- Import `QueueConsumer` and orchestrator
- `startup_event()`: instantiate and start consumer (after model preloading)
  ```python
  consumer = QueueConsumer(
      poll_interval=config.QUEUE_POLL_INTERVAL,
      batch_size=config.QUEUE_BATCH_SIZE,
      max_retries=config.QUEUE_MAX_RETRIES,
  )
  asyncio.create_task(consumer.start())
  ```
- `shutdown_event()`: stop consumer (before queue stop)
  ```python
  await consumer.stop()
  ```

---

## Implementation order

1. **`sql/05_memory_extraction_queue.sql`** — Queue table, trigger, `claim_extraction_batch` RPC (user_consents already exists)
2. **`sql/06_create_memory_proposal_rpc.sql`** — Memory card creation RPC with DB-level consent check
3. **`src/config.py`** — New env vars (queue settings) + table constants
4. **`training/constants/info_extraction.py`** — Add `SUBCATEGORY_TO_CARD_TYPE`, update `EXTRACTION_SCHEMAS` types
5. **`src/utils/harmonia_api.py`** — Replace with POP-507 v2 `create_memory_proposal_rpc()`, remove dead code
6. **`src/agents/langgraph_workflow.py`** — Update chat flow to use new function + types
7. **`src/services/__init__.py`** — Package init
8. **`src/services/consent_manager.py`** — Consent check with TTL cache
9. **`src/services/runner_extraction.py`** — Three-tier pipeline (deterministic + LLM)
10. **`src/services/runner_orchestrator.py`** — Queue item processor
11. **`src/services/queue_consumer.py`** — Polling consumer (replaces realtime_listener.py)
12. **`main.py`** — Integrate queue consumer into FastAPI lifecycle

---

## What we DON'T implement (out of scope)

- Consent revocation / hard-delete (frontend handles it)
- Canonical mappings / `canonicalMappings[]` (happens after card creation, separate pipeline — POP-507 steps 3-6)
- `appliedFieldPaths`, `mappingError`, `mappingAttempts` fields (populated later by AI processing)
- Retry logic for failed canonical mappings (separate system — our queue retry only covers extraction)
- Frontend consent modal
- Database migration for existing memory cards (POP-507 migration SQL is in the issue, frontend team scope)
- `check_profile_exists` / `create_profile` / `ensure_profile_exists` — canonical profile writes happen in the processing→applied pipeline, not at card creation time
- Creating `user_consents` or `consent_audit_log` tables — they already exist in production (GAP 1, GAP 7)
- Supabase Realtime subscription — replaced by queue-based approach for reliability

## Backfill strategy

**Decision: start fresh — do NOT backfill existing completions.**

The trigger only enqueues completions inserted/updated AFTER the migration is applied. Existing `activity_completions` rows are not retroactively enqueued.

Since we are still in development and the DB will be clean, this is the correct approach. There are no historical completions worth processing.

If backfill is ever needed (e.g., after a production incident), use this one-time SQL:
```sql
INSERT INTO memory_extraction_queue (completion_id, user_id)
SELECT id, user_id FROM activity_completions
WHERE status = 'completed'
  AND completed_at > '2026-03-01'  -- adjust date as needed
ON CONFLICT (completion_id) DO NOTHING;
```

---

## Monitoring & observability (recommended)

While not in initial scope, add these as soon as the pipeline is live:

- **Queue dashboard** — `SELECT status, COUNT(*) FROM memory_extraction_queue GROUP BY status` — shows backlog, failure rates
- **Extraction latency** — time between `created_at` and `completed_at` in queue
- **Alert on failure rate** — if `failed / total > 5%` in last hour, alert the team
- **Reprocess endpoint** — `POST /admin/reprocess/{completion_id}` to manually re-enqueue a completion
- **Queue depth alert** — if pending items > 100 and growing, scale up or investigate

## Future consideration: Edge Functions

The Python backend is justified since the chat flow (LangGraph) already runs in Python. However, with the three-tier extraction where Tiers 1-2 are deterministic (no LLM), ~80% of extractions could theoretically run in a **Supabase Edge Function** (Deno) triggered by the database trigger directly — no queue needed for the simple path. Consider this as a simplification if the Python infrastructure becomes a maintenance burden.
