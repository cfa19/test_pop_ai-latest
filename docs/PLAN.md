# GDPR-Compliant Memory Card Generation from Runner Completions (Revised)

## Summary of decisions

| Topic | Decision |
|-------|----------|
| Consent revocation | Handled by frontend — we don't implement it |
| Consent check | Still required, fail-closed before every extraction |
| Runner mapping | LLM-based — runners provide structured data per context, LLM verifies and splits into memory cards |
| Memory card format | POP-507 v2 (`CreateMemoryProposal`) with new types and statuses |
| `store_extracted_information()` | Replace entirely with POP-507 v2 format (both chat and runner) |
| Realtime listener | Only `activity_completions` where `status='completed'` |
| `activity_completions` schema | In development — design for flexibility |

---

## Architecture

```
activity_completions (Supabase)
        │ Realtime INSERT/UPDATE (status='completed')
        ▼
┌──────────────────────────────────┐
│  RealtimeListenerService         │
│  (WebSocket, FastAPI lifespan)   │
└──────────┬───────────────────────┘
           ▼
   on_completion_event()
           │
           ▼
   ConsentManager.check_consent(user_id)  ← fail-closed
           │ (if True)
           ▼
   RunnerExtraction.process(completion)
           │
           ▼
   LLM verifies & splits structured data
           │
           ▼
   Multiple CreateMemoryProposal objects
           │
           ▼
   create_memory_proposal()  ← new harmonia_api.py function (POP-507 v2)
```

---

## POP-507 v2 Memory Card Types (replacing old types)

| Type | Question it answers | Source contexts |
|------|-------------------|----------------|
| `competence` | "What can I do?" | professional skills, knowledge |
| `experience` | "What have I done?" | work history, certifications |
| `preference` | "What do I like?" | work styles, formats, geographic |
| `aspiration` | "Where am I going?" | dream roles, life goals, salary |
| `trait` | "Who am I?" | personality, values, motivations |
| `emotion` | "How am I feeling?" | confidence, stress, energy |
| `connection` | "Who do I know?" | mentors, peers, testimonials |

## POP-507 v2 Statuses

```
proposed → validated → processing → applied
proposed → rejected
processing → failed → validated (retry, max 3)
applied → archived
```

Our pipeline only creates cards with `status: 'proposed'`.

## CreateMemoryProposal format (what we send to API)

```json
{
  "userId": "...",
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
  "status": "proposed"
}
```

---

## Files to create

### 1. `src/services/__init__.py`
Empty package init.

### 2. `src/services/consent_manager.py`
- `check_consent(user_id: str) → bool`
  - Reads `user_consents` table from Supabase
  - **Fail-closed**: returns `False` on any error
  - TTL cache (60s) to reduce DB load
- `invalidate_consent_cache(user_id: str)` → clears cache entry

### 3. `src/services/runner_extraction.py`
Simple pipeline (no LangGraph):
- `process_completion(completion_record: dict) → list[dict]`
  1. Build context from completion's structured data (responses, metadata)
  2. Call LLM to verify data and split into `CreateMemoryProposal` objects
     - LLM receives the structured data + POP-507 type definitions
     - LLM returns array of proposals with correct `type`, `content`, `rawData`, `confidence`
  3. For each proposal: call `create_memory_proposal()` from `harmonia_api.py`
- Reuses: `EXTRACTION_SYSTEM_MESSAGE` from `training/constants/info_extraction.py`
- Uses OpenAI client from `src/config.py`

### 4. `src/services/runner_orchestrator.py`
Event handler glue:
- `on_completion_event(record: dict)`
  1. Extract `user_id` from record
  2. Check consent via `consent_manager.check_consent(user_id)` → skip if False
  3. Call `runner_extraction.process_completion(record)`
  4. Log results

### 5. `src/services/realtime_listener.py`
Supabase Realtime WebSocket service:
- Subscribes to `activity_completions` channel
  - Filter: `status = 'completed'` (INSERT/UPDATE)
- Exponential backoff reconnection (1s → 60s max)
- Started/stopped with FastAPI lifespan
- Uses `SUPABASE_KEY` (service role) for server-side listening

### 6. SQL Migrations

#### `sql/05_create_user_consents_table.sql`
```sql
CREATE TABLE IF NOT EXISTS user_consents (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL UNIQUE,
    consent BOOLEAN NOT NULL DEFAULT FALSE,
    consent_version TEXT NOT NULL DEFAULT '1.0',
    ip_address TEXT,
    user_agent TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
-- RLS + auto-update trigger
```

#### `sql/06_create_consent_audit_log.sql`
```sql
CREATE TABLE IF NOT EXISTS consent_audit_log (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL,
    action TEXT NOT NULL,
    details JSONB DEFAULT '{}'::jsonb,
    ip_address TEXT,
    user_agent TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
-- RLS: service role only (immutable)
```

#### `sql/07_enable_realtime_activity_completions.sql`
```sql
ALTER PUBLICATION supabase_realtime ADD TABLE activity_completions;
```

---

## Files to modify

### 1. `src/config.py`
Add:
```python
# Runner extraction
RUNNER_EXTRACTION_ENABLED = os.getenv("RUNNER_EXTRACTION_ENABLED", "true").lower() in ("true", "1", "yes")
HARMONIA_SERVICE_TOKEN = os.getenv("HARMONIA_SERVICE_TOKEN", "")
```
Add to `Tables` class:
```python
USER_CONSENTS = "user_consents"
CONSENT_AUDIT_LOG = "consent_audit_log"
```

### 2. `src/utils/harmonia_api.py`
**Replace `store_extracted_information()`** with POP-507 v2 format:
- New function: `create_memory_proposal(proposal: dict, user_id: str, user_token: str = None) → dict`
  - Sends `CreateMemoryProposal` to `POST /api/harmonia/journal/memory-cards`
  - Uses POP-507 v2 types (`competence`, `experience`, etc.)
  - Source: `{ type: "runner" | "chat", sourceId, activityId?, extractedAt }`
  - Includes `rawData` field
- **Update `_make_request()`**: fall back to `HARMONIA_SERVICE_TOKEN` when no `user_token` (server-side processing)
- **Update `store_information_node`** caller in `langgraph_workflow.py` to use new function signature

### 3. `main.py`
- Import `RealtimeListenerService` and orchestrator
- `startup_event()`: instantiate and start listener (after model preloading)
- `shutdown_event()`: stop listener (before queue stop)

### 4. `src/agents/langgraph_workflow.py`
- Update `store_information_node` to call new `create_memory_proposal()` instead of old `store_extracted_information()`
- Map old extraction subcategories to new POP-507 types (e.g., `skills` → `competence`, `experiences` → `experience`, etc.)

---

## Implementation order

1. **SQL migrations** — `user_consents`, `consent_audit_log`, enable Realtime
2. **`src/config.py`** — New env vars + table constants
3. **`src/utils/harmonia_api.py`** — Replace with POP-507 v2 `create_memory_proposal()`
4. **`src/agents/langgraph_workflow.py`** — Update chat flow to use new function
5. **`src/services/__init__.py`** — Package init
6. **`src/services/consent_manager.py`** — Consent check with TTL cache
7. **`src/services/runner_extraction.py`** — LLM verify & split pipeline
8. **`src/services/runner_orchestrator.py`** — Event handler glue
9. **`src/services/realtime_listener.py`** — Supabase Realtime WebSocket
10. **`main.py`** — Integrate listener into FastAPI lifecycle

---

## What we DON'T implement (out of scope)

- Consent revocation / hard-delete (frontend handles it)
- Realtime listener on `user_consents` table
- Canonical mappings (happens after user validates, separate pipeline)
- `canonicalMappings`, `appliedFieldPaths` fields (populated later by AI processing)
- Frontend consent modal
- Database migration for existing memory cards (POP-507 migration SQL is in the issue)
