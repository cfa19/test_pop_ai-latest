# Harmonia API v3.1 - Postman Collection

Complete Postman collection for testing the Harmonia V3.1 Activity Management Platform APIs.

## Overview

This collection provides comprehensive API testing for the Harmonia V3.1 platform, featuring:

- **Native React Runners** (V2 plugin architecture removed)
- **17 database tables** (7 Core + 7 Extended + 3 V3)
- **Multi-tenant isolation** with RLS policies
- **JWT-based authentication** with enriched headers
- **Three-Store Architecture**: Canonical (A), Journal (B), RAG Index (C)

## Files

| File                                    | Description                        |
| --------------------------------------- | ---------------------------------- |
| `Harmonia_API.postman_collection.json`  | Main collection with all endpoints |
| `Harmonia_API.postman_environment.json` | Local development environment      |

## Import Instructions

1. Open Postman
2. Click **Import** button
3. Drag and drop the collection JSON file
4. Import the environment file
5. Select the environment from the dropdown
6. Set your `accessToken` and `supabaseAnonKey`

## Collection Structure

```
Harmonia API v3.1
├── Journal Store (Store B) - 17 endpoints
│   ├── Journal Entries (5 endpoints)
│   │   ├── List Entries - GET /api/harmonia/journal/entries
│   │   ├── Create Entry - POST /api/harmonia/journal/entries
│   │   ├── Get Entry by ID - GET /api/harmonia/journal/entries/:id
│   │   ├── Update Entry - PATCH /api/harmonia/journal/entries/:id
│   │   └── Delete Entry - DELETE /api/harmonia/journal/entries/:id
│   │
│   ├── Journal Events (5 endpoints)
│   │   ├── List Events - GET /api/harmonia/journal/events
│   │   ├── Create Event - POST /api/harmonia/journal/events
│   │   ├── Get Event by ID - GET /api/harmonia/journal/events/:id
│   │   ├── Batch Create Events - POST /api/harmonia/journal/events/batch
│   │   └── Retry Failed Events - POST /api/harmonia/journal/events/retry
│   │
│   ├── Memory Cards (7 endpoints)
│   │   ├── List Memory Cards - GET /api/harmonia/journal/memory-cards
│   │   ├── Create Memory Card - POST /api/harmonia/journal/memory-cards
│   │   ├── Get Memory Card by ID - GET /api/harmonia/journal/memory-cards/:id
│   │   ├── Update Memory Card - PATCH /api/harmonia/journal/memory-cards/:id
│   │   ├── Validate Memory Card - POST /api/harmonia/journal/memory-cards/:id/validate
│   │   ├── Batch Validate - POST /api/harmonia/journal/memory-cards/batch-validate
│   │   └── Delete Memory Card - DELETE /api/harmonia/journal/memory-cards/:id
│   │
│   └── Journal Statistics (1 endpoint)
│       └── Get Stats - GET /api/harmonia/journal/stats
│
└── RAG Index Store (Store C) - 10 endpoints
    ├── Retrieval Chunks (7 endpoints)
    │   ├── List Chunks - GET /api/harmonia/rag-index/chunks
    │   ├── Create Chunk - POST /api/harmonia/rag-index/chunks
    │   ├── Get Chunk by ID - GET /api/harmonia/rag-index/chunks/:id
    │   ├── Update Chunk - PATCH /api/harmonia/rag-index/chunks/:id
    │   ├── Delete Chunk - DELETE /api/harmonia/rag-index/chunks/:id
    │   ├── Batch Create - POST /api/harmonia/rag-index/chunks/batch
    │   └── Sync Chunks - POST /api/harmonia/rag-index/chunks/sync
    │
    ├── RAG Search (3 modes, 1 endpoint)
    │   ├── Hybrid Search - POST /api/harmonia/rag-index/search
    │   ├── Semantic Search - POST /api/harmonia/rag-index/search
    │   └── Keyword Search - POST /api/harmonia/rag-index/search
    │
    └── RAG Statistics (1 endpoint)
        └── Get Stats - GET /api/harmonia/rag-index/stats
```

## Environment Variables

| Variable          | Description                | Required |
| ----------------- | -------------------------- | -------- |
| `baseUrl`         | API base URL               | Yes      |
| `supabaseAnonKey` | Supabase anonymous key     | Yes      |
| `accessToken`     | Supabase JWT access token  | Yes      |
| `userId`          | Authenticated user UUID    | Yes      |
| `userRole`        | User role enum             | Yes      |
| `organizationId`  | Organization ID (B2B only) | No       |
| `activityId`      | Activity UUID              | No       |
| `entryId`         | Journal entry UUID         | No       |
| `eventId`         | Journal event UUID         | No       |
| `memoryCardId`    | Memory card UUID           | No       |
| `chunkId`         | RAG chunk UUID             | No       |
| `sourceId`        | Source document UUID       | No       |
| `idempotencyKey`  | Idempotency key for events | No       |
| `correlationId`   | Correlation ID for tracing | No       |

## Request Body Examples

### Journal Entries

**Create Entry** (POST /api/harmonia/journal/entries):

```json
{
  "userId": "{{userId}}",
  "entryType": "milestone",
  "title": "Completed CV Analysis",
  "content": "Successfully completed CV analysis activity.",
  "metadata": {
    "source": "activity",
    "importance": "high",
    "relatedActivityId": "{{activityId}}",
    "generatedInsights": ["Strong technical skills"]
  },
  "linkedContexts": ["cv-analysis", "career-planning"]
}
```

Entry types: `milestone`, `discovery`, `reflection`, `achievement`
Importance levels: `low`, `medium`, `high`

### Journal Events

**Create Event** (POST /api/harmonia/journal/events):

```json
{
  "userId": "{{userId}}",
  "eventType": "activity_completed",
  "context": "cv-analysis",
  "payload": {
    "activityId": "{{activityId}}",
    "score": 85,
    "insights": ["Leadership potential identified"]
  },
  "metadata": {
    "source": "runner",
    "idempotencyKey": "{{idempotencyKey}}",
    "correlationId": "{{correlationId}}"
  }
}
```

Event types: `profile_update`, `activity_completed`, `milestone_reached`, `memory_extracted`, `user_action`, `system_event`

### Memory Cards

**Create Memory Card** (POST /api/harmonia/journal/memory-cards):

```json
{
  "userId": "{{userId}}",
  "type": "fact",
  "content": "User demonstrates strong leadership qualities.",
  "confidence": 0.85,
  "source": {
    "type": "entry",
    "sourceId": "{{entryId}}",
    "extractedAt": "2026-01-21T00:00:00Z"
  },
  "tags": ["leadership", "teamwork"],
  "linkedContexts": ["career-planning"]
}
```

Card types: `fact`, `preference`, `decision`, `objective`, `constraint`, `sensitivity`
Card statuses: `proposed`, `validated`, `archived`, `rejected`

**Validate Memory Card** (POST /api/harmonia/journal/memory-cards/:id/validate):

```json
{
  "action": "validate"
}
```

Actions: `validate`, `reject`, `archive`

### RAG Chunks

**Create Chunk** (POST /api/harmonia/rag-index/chunks):

```json
{
  "userId": "{{userId}}",
  "sourceType": "entry",
  "sourceId": "{{entryId}}",
  "content": "CV analysis completion with 5 key skills identified.",
  "contentSummary": "CV analysis summary",
  "chunkIndex": 0,
  "metadata": {
    "language": "en",
    "tokenCount": 25,
    "contextPath": "entries.cv-analysis",
    "tags": ["cv-analysis", "skills"]
  }
}
```

Source types: `profile`, `entry`, `chat`, `memory`

### RAG Search

**Hybrid Search** (POST /api/harmonia/rag-index/search):

```json
{
  "query": "career development and leadership skills",
  "userId": "{{userId}}",
  "mode": "hybrid",
  "limit": 10,
  "minConfidence": 0.7,
  "sourceTypes": ["entry", "memory"],
  "semanticWeight": 0.7,
  "keywordWeight": 0.3
}
```

Search modes: `semantic`, `keyword`, `hybrid`

## Authentication

All endpoints require Supabase JWT authentication:

### Required Headers

```
apikey: {{supabaseAnonKey}}
Authorization: Bearer {{accessToken}}
x-user-role: {{userRole}}
x-organization-id: {{organizationId}}  (optional, for B2B)
Content-Type: application/json
```

### User Roles

- **SUPER_ADMIN**: Full system access
- **ORG_ADMIN**: Organization-level management
- **CONSULTANT**: View assigned users and analytics
- **BENEFICIARY**: Access assigned activities
- **INDIVIDUAL_USER**: Personal account access

### Getting Your Access Token

1. Log into the Harmonia application
2. Open browser DevTools (F12)
3. Go to Network tab
4. Find any API request
5. Copy the `Authorization` header value (without "Bearer ")
6. Set as `accessToken` in Postman environment

## API Workflows

### 1. Activity Completion Flow

```
1. POST /journal/events (activity_completed)
2. POST /journal/entries (achievement)
3. POST /journal/memory-cards (extract insights)
4. POST /rag-index/chunks (index for retrieval)
5. PATCH /journal/events/:id (mark completed)
```

### 2. Memory Validation Flow (GDPR)

```
1. GET /journal/memory-cards?status=proposed
2. GET /journal/memory-cards/:id (review)
3. POST /journal/memory-cards/:id/validate
   OR
   POST /journal/memory-cards/batch-validate
```

### 3. RAG Context Retrieval

```
1. POST /rag-index/search (hybrid mode)
2. GET /journal/memory-cards?status=validated
3. GET /journal/entries?importance=high
```

### 4. Event Processing & Retry

```
1. GET /journal/events?processingStatus=failed
2. POST /journal/events/retry
3. GET /journal/stats
```

## Running with Newman

```bash
# Install Newman
pnpm install -g newman

# Run all tests
newman run Harmonia_API.postman_collection.json \
  -e Harmonia_API.postman_environment.json

# Run Journal Store only
newman run Harmonia_API.postman_collection.json \
  -e Harmonia_API.postman_environment.json \
  --folder "Journal Store (Store B)"

# Run RAG Index only
newman run Harmonia_API.postman_collection.json \
  -e Harmonia_API.postman_environment.json \
  --folder "RAG Index Store (Store C)"
```

## Rate Limits

| Endpoint Type      | Rate Limit  |
| ------------------ | ----------- |
| Standard endpoints | 100 req/min |
| Batch endpoints    | 20 req/min  |
| Search endpoint    | 30 req/min  |

## Troubleshooting

### Common Errors

| Error                      | Cause                     | Solution                                           |
| -------------------------- | ------------------------- | -------------------------------------------------- |
| 401 Unauthorized           | Invalid/expired token     | Refresh your accessToken                           |
| 403 Forbidden              | Wrong userId in body      | Ensure body userId matches your authenticated user |
| 404 User profile not found | No journey_profile record | Create an entry first (auto-creates profile)       |
| 400 Request body required  | Empty POST body           | Add JSON body to request                           |
| 422 Validation failed      | Invalid field values      | Check enum values and required fields              |

### Checking Your Setup

1. Verify `accessToken` is set and not expired
2. Verify `supabaseAnonKey` matches your Supabase project
3. Verify `userId` matches your authenticated user
4. For B2B users, ensure `organizationId` is set

## Version

- **Collection Version**: 3.1.0
- **API Version**: 3.1.0
- **Last Updated**: January 2026
- **Architecture**: Harmonia V3.1 - Native React Runners
- **Coverage**: Journal Store (Store B) + RAG Index Store (Store C)
