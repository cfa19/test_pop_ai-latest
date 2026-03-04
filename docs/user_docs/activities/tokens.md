# Activity Tokens

Tokens allow consultants and admins to share activities with people who do not have a
POPskills account. A token generates a unique secure link that grants one-time access to a
specific activity.

---

## Who can create tokens

| Role | Access |
|---|---|
| **Consultant** | Create tokens for their activities |
| **Org Admin** | Create tokens for organization activities |

---

## Access token management

Navigate to **Dashboard > Management > Tokens**.

The page lists all activities available for token creation, filtered by your role and
organization.

---

## Create a token

1. Navigate to **Dashboard > Management > Tokens**.
2. Select the activity you want to share.
3. Set the **expiration period** in days (default: 30 days, maximum: 365 days).
4. Optionally set a **maximum number of uses** (leave empty for unlimited).
5. Click **Create token**.
6. A unique secure link is generated with an expiration date.
7. Copy the link and share it with the intended recipient (via email, messaging, etc.).

---

## How token access works

When a recipient opens the token link:

1. They are taken directly to the activity — no login or registration required.
2. They complete the activity as usual (all steps, auto-save, etc.).
3. On completion, they see an end-of-activity page with a custom message.
4. Their results are saved and available for the consultant to review.

Token users do not have access to the dashboard, results page, or any other platform feature. They can only complete the specific activity linked to the token.

---

## Token expiration and limits

Tokens have a limited lifespan and optional usage limits:

- **Expiration:** Every token expires after the configured number of days (default 30, maximum 365). After expiration, the link stops working and recipients see an error message.
- **Maximum uses:** If a maximum number of uses is set, the token becomes invalid once that limit is reached. If no limit is set, the token can be used an unlimited number of times until it expires.
- **Revocation:** Tokens can be revoked at any time. A revoked token immediately stops working.

If a recipient opens an expired, revoked, or fully used token link, they see an invalid token page and cannot access the activity.

---

## Use cases

- Sharing an assessment with a **prospect** before they create an account
- Sending a quick activity to someone **outside your organization**
- Running an activity during a **live session** where participants do not need accounts
