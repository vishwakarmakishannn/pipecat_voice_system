# Mswipe production knowledge system

This subsystem is the reviewed factual layer for the Mswipe voice agent. It is
independent from the existing user-upload RAG. The old tables and endpoints are
left intact during rollout.

## Architecture

The control plane runs outside calls:

1. An operator registers an approved source.
2. `knowledge-worker` discovers sitemap and same-domain links, obeys
   `robots.txt`, fetches through pinned public IP addresses, and revalidates
   every redirect.
3. Every response is checksummed and archived privately in S3 or the durable
   local knowledge directory.
4. HTML or PDF becomes canonical Markdown and atomic typed draft units.
   Repeated responsive/hidden HTML blocks and exact cross-page answers are
   removed before review. Framework placeholders such as `[slug]` are never
   fetched.
5. Reviewers approve units, resolve contradiction candidates, build the dense
   index, and run evaluation.
6. A draft release is published atomically. The previous release becomes a
   rollback target.

The serving plane only reads approved units from the one published release. It
normalizes conservative Mswipe/STT aliases, preserves customer and transaction
identifiers, runs PostgreSQL full-text and optional pgvector retrieval, fuses
and deduplicates candidates, applies a confidence threshold, and returns either
evidence or a structured no-answer. The voice processor has its own deadline
and injects query-scoped evidence only for the current turn.

Static knowledge never performs or authorizes live work. Customer-specific
lookups and ticket writes are represented by the typed `MswipeApi` boundary;
its current adapter fails closed until approved endpoint, authentication,
confirmation, and idempotency contracts are supplied.

## Configuration

Add the required values to the root `.env`:

```dotenv
# Keep false until an evaluated release has been published.
MSWIPE_KNOWLEDGE_ENABLED=false
MSWIPE_KNOWLEDGE_ADMIN_USER_IDS=1
MSWIPE_KNOWLEDGE_ALLOWED_DOMAINS=mswipe.com,www.mswipe.com
MSWIPE_KNOWLEDGE_STORAGE_DIR=/var/lib/aura/mswipe-knowledge

# Recommended production mode. `disabled` gives lexical-only retrieval.
MSWIPE_KNOWLEDGE_EMBEDDING_PROVIDER=google
MSWIPE_KNOWLEDGE_EMBEDDING_MODEL=gemini-embedding-001
MSWIPE_KNOWLEDGE_EMBEDDING_DIMENSION=768

MSWIPE_KNOWLEDGE_TOP_K=4
MSWIPE_KNOWLEDGE_MIN_CONFIDENCE=0.42
MSWIPE_KNOWLEDGE_VOICE_TIMEOUT_SECONDS=0.8
MSWIPE_KNOWLEDGE_MAX_CRAWL_PAGES=500
MSWIPE_KNOWLEDGE_MAX_CRAWL_DEPTH=6
MSWIPE_KNOWLEDGE_CRAWL_DELAY_SECONDS=0.25
MSWIPE_KNOWLEDGE_EMBEDDING_BATCH_SIZE=64
MSWIPE_KNOWLEDGE_RESPECT_ROBOTS=true
```

For OpenAI embeddings, set the provider to `openai`, the model to
`text-embedding-3-small`, and configure `OPENAI_API_KEY`. For Google, configure
`GOOGLE_API_KEY`. Provider, model, dimension, and content hash are stored with
every vector. Never change the dimension after applying the migration; use a
new migration and a new release for a dimensional model change.

`S3_BUCKET` plus AWS credentials makes raw snapshots private S3 objects. With
no S3 configuration, the worker uses `MSWIPE_KNOWLEDGE_STORAGE_DIR`, which must
be a durable mounted volume in production.

Admin endpoints deny every user unless their numeric ID is explicitly listed
in `MSWIPE_KNOWLEDGE_ADMIN_USER_IDS`. The operator CLI uses direct database
access and should only be available to trusted deployment operators.

## First-time setup

From the project root:

```bash
docker compose up -d db
docker compose run --rm migrate
docker compose up -d knowledge-worker backend
```

For a local virtual environment, run commands from `backend/`:

```bash
.venv/bin/python -m alembic upgrade head
.venv/bin/python knowledge_worker.py
```

The migration adds only new `knowledge_*` and `ticket_taxonomy_entries` tables.
It does not modify or delete old RAG data.

## Build the first corpus

Register the public website. Sitemap discovery is enabled, and the crawl also
follows same-host links. As verified on 2026-08-27, Mswipe publishes
`https://www.mswipe.com/sitemap.xml`; its `robots.txt` allows public pages and
disallows sign-in, sign-up, and developer-portal paths. The worker checks the
current policy on every crawl.

```bash
cd backend
.venv/bin/python knowledge_cli.py source-add \
  --name "Mswipe public website" \
  --url https://www.mswipe.com/ \
  --authority 3 \
  --max-pages 500 \
  --max-depth 6
```

Copy the returned source UUID into:

```bash
.venv/bin/python knowledge_cli.py aliases-seed
.venv/bin/python knowledge_cli.py crawl SOURCE_UUID
.venv/bin/python knowledge_cli.py jobs
```

The durable worker retries a failed job three times. Page-level errors and the
successful/unchanged/failed counts appear in the job result. Raw snapshot
records and errors are also available through `GET /api/knowledge/admin/snapshots`.
Public pages are still archived even when their units are excluded from the
serving corpus. The default website policy excludes careers, blogs, knowledge
listing/press pages, and about-us content. `robots.txt` exclusions are reported
as policy skips rather than failures.

Import the existing mDesk taxonomy separately:

```bash
.venv/bin/python knowledge_cli.py taxonomy-import \
  ../mswipe-email-agent/customer_complaints_agent/utils/ticket_remarks.csv
```

Only rows whose status is exactly `Active` become active entries. Blocked,
blank, malformed, and exact duplicate rows cannot enter the active taxonomy.
Taxonomy entries are not embedded as general factual answers.
The future ticket tool must call `require_active_ticket_selection` immediately
before any write; `classify_ticket_candidates` can propose up to ten active
values but never authorizes a ticket by itself.

## Review and publish

Audit the corpus first. List and inspect complete draft records, then approve or
retire them individually after checking accuracy, scope, expiry, product,
procedure order, and voice wording:

```bash
.venv/bin/python knowledge_cli.py corpus-audit
.venv/bin/python knowledge_cli.py units --status draft --limit 100
.venv/bin/python knowledge_cli.py units --status draft \
  --source-contains /support --limit 100
.venv/bin/python knowledge_cli.py unit-show UNIT_UUID
.venv/bin/python knowledge_cli.py unit-approve UNIT_UUID
.venv/bin/python knowledge_cli.py unit-retire UNIT_UUID \
  --notes "Not suitable for the customer-support voice corpus"
.venv/bin/python knowledge_cli.py conflicts-detect
.venv/bin/python knowledge_cli.py embed
.venv/bin/python knowledge_cli.py jobs
```

Do not use `release-create --all-approved` until the intended launch scope has
been reviewed. In particular, merchant agreements, privacy policies, grievance
policies, and marketing pages require an explicit product/knowledge-owner
decision before entering a general customer-support release.

The HTTP approval endpoint also records reviewer ID and notes:

```text
POST /api/knowledge/admin/units/{unit_id}/approve
{"review_notes":"Verified against the current product page and support policy."}
```

Open contradiction candidates block publication until an administrator marks
them resolved or ignored with an explanation. When all chosen units are ready:

```bash
.venv/bin/python knowledge_cli.py release-create 2026-08-27.1 \
  --all-approved --description "First reviewed Mswipe website release"
.venv/bin/python knowledge_cli.py release-validate RELEASE_UUID
```

Replace the placeholder stable keys in
`evals/mswipe_knowledge_cases.example.jsonl`, expand it with real English,
Hindi, Hinglish, noisy-STT, no-answer, policy, product, and troubleshooting
cases, then run:

```bash
.venv/bin/python -m scripts.evaluate_mswipe_knowledge \
  evals/mswipe_knowledge_cases.example.jsonl \
  --min-route-accuracy 0.95 --min-recall 0.85
```

The gate reports route accuracy, recall@k, MRR, no-answer accuracy, and
retrieval p50/p95/max latency. Publish only after content review and the release
gate pass:

```bash
.venv/bin/python knowledge_cli.py release-publish RELEASE_UUID
```

Set `MSWIPE_KNOWLEDGE_ENABLED=true` and restart the backend only after the
release is published. `GET /api/knowledge/status` must then report both
`enabled: true` and `serving: true`.

## API surface

Authenticated serving endpoints:

- `GET /api/knowledge/status`
- `POST /api/knowledge/search`
- `POST /api/knowledge/feedback`

Allow-listed admin endpoints manage sources, jobs, snapshots, units, aliases,
conflicts, validation, releases, publication, and rollback under
`/api/knowledge/admin/*`. OpenAPI at `/docs` contains request schemas.

## Rollback and safe shutdown

Rollback is an atomic database transaction:

```bash
.venv/bin/python knowledge_cli.py status
.venv/bin/python knowledge_cli.py release-rollback PREVIOUS_RELEASE_UUID
```

To stop customer-visible knowledge immediately, set
`MSWIPE_KNOWLEDGE_ENABLED=false` and restart the backend. This does not delete a
release or corpus. Do not remove the legacy RAG, its API, UI, or tables until
shadow traffic and the supported-intent pilot pass and a separate destructive
change is explicitly approved.

## What still requires Mswipe input

Before production traffic, Mswipe must provide and approve:

- supported caller groups, languages, intents, and escalation rules;
- authoritative internal manuals/policies and content owners;
- customer verification, live lookup, and ticket API contracts plus UAT access;
- ticket confirmation and idempotency behaviour;
- a cleaned/owned evaluation set with acceptance thresholds;
- legal decisions for recording, PII retention, audit access, and regional use.

Until those inputs exist, the system can serve reviewed public facts but will
fail closed for customer-specific data and real ticket creation.
