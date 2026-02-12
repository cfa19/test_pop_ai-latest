# Shared Schema Components

This directory contains reusable schema definitions used across multiple memory card types.

## Purpose

Shared schemas enable:
- **Consistency**: Same data structure across different contexts
- **Reusability**: Define once, reference everywhere
- **Maintainability**: Update in one place, propagate everywhere
- **Type Safety**: Ensure compatible data structures

## Available Schemas

### 1. Role Schema (`role_schema.json`)

**Used by:**
- Professional Experience (past roles)
- Dreamed Roles (aspirational future roles)
- Current Position tracking

**Key Fields:**
- `title`: Job title
- `level`: Career level/seniority
- `function`: Department/function
- `employment_type`: Full-time, contract, etc.
- `work_arrangement`: Remote, hybrid, on-site

### 2. Company Schema (`company_schema.json`)

**Used by:**
- Professional Experience (past employers)
- Dreamed Roles (target companies)
- Industry Research

**Key Fields:**
- `name`: Company name
- `industry`: Industry/sector
- `company_type`: Startup, enterprise, etc.
- `company_stage`: Funding stage
- `company_size`: Employee count

### 3. Skills Schema (`skills_schema.json`)

**Used by:**
- Professional Experience (skills used/developed)
- Dreamed Roles (required skills, skill gaps)
- Professional Context (current skills)
- Learning Context (skills to develop)

**Key Definitions:**
- `skill`: Individual skill with proficiency
- `skill_list`: Array of skills
- `skill_gap`: Gap analysis structure

**Key Fields:**
- `name`: Skill name
- `category`: Technical, leadership, etc.
- `proficiency_level`: Beginner to expert
- `years_of_experience`: Time spent with skill

### 4. Location Schema (`location_schema.json`)

**Used by:**
- Professional Experience (work locations)
- Dreamed Roles (geographic preferences)
- Salary Expectations (cost of living context)

**Key Fields:**
- `city`, `state_province`, `country`
- `country_code`: ISO 3166-1 alpha-2
- `timezone`: IANA timezone
- `cost_of_living_category`: Economic context
- `work_arrangement`: Remote/hybrid/on-site

### 5. Compensation Schema (`compensation_schema.json`)

**Used by:**
- Professional Experience (historical compensation)
- Salary Expectations (target compensation)
- Dreamed Roles (expected salary ranges)

**Key Fields:**
- `currency`: ISO 4217 code
- `base_salary`: Base compensation
- `bonus`: Variable compensation
- `equity`: Stock options, RSUs, etc.
- `benefits`: Health, retirement, perks

### 6. Date Range Schema (`date_range_schema.json`)

**Used by:**
- Professional Experience (employment dates)
- Dreamed Roles (target timelines)
- Education (degree dates)
- Projects (project duration)

**Key Fields:**
- `start_date`, `end_date`
- `is_current`: Ongoing period flag
- `duration`: Calculated time span
- `precision`: Date accuracy level

## Usage Examples

### Referencing in JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ProfessionalExperience",
  "type": "object",
  "properties": {
    "role": {
      "$ref": "https://popskills.ai/schemas/shared/role.json"
    },
    "company": {
      "$ref": "https://popskills.ai/schemas/shared/company.json"
    },
    "location": {
      "$ref": "https://popskills.ai/schemas/shared/location.json"
    },
    "dates": {
      "$ref": "https://popskills.ai/schemas/shared/date_range.json"
    },
    "compensation": {
      "$ref": "https://popskills.ai/schemas/shared/compensation.json"
    },
    "skills_used": {
      "$ref": "https://popskills.ai/schemas/shared/skills.json#/definitions/skill_list"
    }
  }
}
```

### Using in TypeScript

```typescript
import { Role } from './shared/role_schema'
import { Company } from './shared/company_schema'
import { Skill } from './shared/skills_schema'

interface ProfessionalExperience {
  id: string
  user_id: string
  role: Role
  company: Company
  skills_used: Skill[]
  // ... other fields
}
```

### Using in Python (Pydantic)

```python
from pydantic import BaseModel
from typing import List, Optional
from .shared.role_schema import Role
from .shared.company_schema import Company
from .shared.skills_schema import Skill

class ProfessionalExperience(BaseModel):
    id: str
    user_id: str
    role: Role
    company: Company
    skills_used: List[Skill]
    # ... other fields
```

## Schema Design Principles

1. **Self-Contained**: Each schema can stand alone
2. **Optional by Default**: Most fields are optional for flexibility
3. **Extensible**: Use `additionalProperties` for custom fields
4. **Validated**: Include format validators, enums, min/max
5. **Documented**: Every field has a description
6. **Examples**: Provide realistic examples

## Schema Versioning

All schemas include:
- `$id`: Unique schema identifier
- `$schema`: JSON Schema version (draft-07)
- Semantic versioning in `$id` URL when breaking changes occur

## Adding New Shared Schemas

When adding a new shared schema:

1. **Identify Reuse**: Confirm it's used by 2+ memory cards
2. **Design Generically**: Make it flexible for all use cases
3. **Document Thoroughly**: Add to this README
4. **Version Properly**: Use semantic versioning
5. **Test Compatibility**: Ensure it works in all contexts

## Migration Guide

To migrate existing schemas to use shared components:

1. Replace inline definitions with `$ref`
2. Update validation logic to handle references
3. Test data compatibility
4. Update documentation
5. Version bump if breaking changes

## Related Documentation

- [Dreamed Roles Schema](../aspirational/dreamed_role_json_schema.md)
- [Salary Expectations Schema](../aspirational/salary_expectation_schema.json)
- [Professional Values Schema](../aspirational/professional_values_schema.json)
