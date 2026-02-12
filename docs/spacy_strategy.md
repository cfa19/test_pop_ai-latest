Yep — this is a *very* real limitation, and you’re not doing anything wrong. spaCy (and most NER / dependency-based approaches) **cannot infer implied roles**. What you’re running into is the difference between:

* **Explicit mentions** vs
* **Latent / inferred aspirations**

Your example is the canonical one.

> “I wish to lead a global marketing team”

There is **no literal span** in the text that equals
“global marketing team director”.

So spaCy is behaving correctly — it can only extract what’s *there*.

---

## The key insight (this unlocks everything)

You actually have **two different tasks**:

### Task A — Span extraction (what spaCy is good at)

> Extract *explicitly mentioned* entities

### Task B — Role inference / canonicalization (what spaCy will never do)

> Infer a **normalized role** from intent + arguments

Trying to force Task B into Task A will always feel broken.

---

## What you should do instead (clean architecture)

### Step 1: Extract *components*, not the final role

From:

> “I wish to lead a global marketing team”

You want:

| Component  | Extract   |
| ---------- | --------- |
| `ACTION`   | lead      |
| `SCOPE`    | global    |
| `FUNCTION` | marketing |
| `OBJECT`   | team      |

These *are* explicitly present and extractable.

---

### Step 2: Map components → canonical role (separate step)

Only **after extraction**, do inference:

```
lead + marketing + team (+ global)
→ marketing director
→ seniority: director / head
```

This mapping is:

* Domain knowledge
* Business logic
* Not linguistic parsing

---

## Concrete revised label schema (important change)

Add **intent primitives**:

| Label      | Meaning                                               |
| ---------- | ----------------------------------------------------- |
| `ACTION`   | Verbs indicating responsibility (lead, manage, build) |
| `FUNCTION` | Functional domain (marketing, sales, engineering)     |
| `SCOPE`    | Scale / reach (global, regional, large)               |
| `ORG_UNIT` | Team, department, org                                 |

Your previous `ROLE` label becomes **optional / downstream**.

---

## Revised annotation for your example

> “I wish to lead a global marketing team”

```
lead        B-ACTION
global      B-SCOPE
marketing   B-FUNCTION
team        B-ORG_UNIT
```

This is something spaCy *can* do reliably.

---

## Role inference layer (this is the missing piece)

Create a deterministic mapper:

### Rule example

```python
if ACTION == "lead" and FUNCTION == "marketing" and ORG_UNIT == "team":
    role = "marketing director"
    seniority = "director"
```

### Slightly smarter version

* ACTION → seniority band

  * lead/manage → director
  * own/build → manager
  * advise → senior IC
* FUNCTION → role family
* ORG_UNIT size → seniority adjustment
* SCOPE → seniority boost

This logic is:

* Transparent
* Testable
* Business-controlled
* Much safer than letting a model hallucinate

---

## Why *not* let an LLM infer it directly?

You *can*, but:

* It will invent roles inconsistently
* Hard to audit
* Hard to explain to stakeholders
* Expensive at scale

LLMs are great as a **fallback**, not the primary inference engine.

---

## If you really want ML for inference

Use a **classifier**, not NER:

Input:

```
lead global marketing team
```

Output:

```
marketing director
```

This is:

* A multi-class or multi-label classification problem
* Much easier than generation
* Trains well with 1–2k examples

---

## Mental model shift (the big takeaway)

> **NER extracts facts.
> Inference creates meaning.**

spaCy only does the first.

Once you separate them, the system suddenly becomes stable and predictable.

---

If you want next, I can:

* Design a **role-inference ruleset**
* Propose a **hybrid rules + classifier** approach
* Show how to **evaluate inferred roles**
* Help you decide where an LLM actually makes sense here

You’ve hit the *right* problem — this is where good systems get designed.
