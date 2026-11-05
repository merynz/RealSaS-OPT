# BUSTED — Commercial platform foundation v1

Status: **design/preflight only** (2026-11-05). Not a launched checkout, not authorized monetization, and not evidence that public 1–2-view compilation is qualified. BUSTED is the consumer product name; RealSaS remains the underlying engine. Consumer-facing use of "Powered by RealSaS" is intentionally suspended.

## Verified foundation in this repository

- `platform/migrations/sql/00022_public_product_compile_sessions.sql`: explicit PRODUCT release channel; public compile identities; source front/back artifact slots. A public compile is not the same identity as its attempt or ProductRevision.
- `platform/migrations/sql/00026_public_product_session_ownership.sql`: immutable `owner_key_sha256` for public compile sessions. A new customer account cannot be established by rewriting this value.
- `platform/internal/httpapi/auth.go`: private service-to-service bearer token and owner-scoped product route checks; this is **not** a consumer login.
- `deploy/product/.env.example`: `REALSAS_PRODUCT_COMPILE_ENABLED=0` by default and an unbound public source materializer. Controlled exact-8 ingestion is separate from public one/two-view intake.
- `deploy/product/README.md`: web-only loopback edge, private Platform/operator services, pinned production sources and qualification preflight.
- Public shell: fail-closed compiler availability and proof-qualified handoff; static preview does not run a real compile.

## Commercial contract to freeze before writing checkout code

**Customer account and continuity.** Introduce a durable account principal (e.g. `account_id`) and audited account↔browser-session linking. A user's previously owned compile/puppet is claimable only after possession of that session and verified account authentication. Do not mutate the immutable compile owner digest to "transfer" ownership. Define a device-loss recovery policy.

**Persistent character.** Character identity must survive multiple revisions, motion edits, previews, and exports. Proposed entities (not yet migrations): `product_accounts`, `product_characters`, `character_revisions`, `account_character_access`. Bind completed characters to qualified ProductRevisions, not arbitrary uploaded images, predicted assets, or text-only proof labels.

**Usage and charging.** Customer buys an *authorized compile entitlement*, not a fake success event. Proposed append-only `compile_credit_ledger` with `RESERVE`, `CAPTURE`, `RELEASE`, `REFUND`, `ADJUST` movements, each referencing `account_id`, `public_compile_id` (where applicable), `currency`, a stable idempotency key and optional external payment reference. Reserve before genuine execution; capture according to the chosen commercial definition of **successful qualified ProductRevision**, release/credit back on validation failure, cancellation, timeout or unqualified output. Never derive balance by trusting frontend state.

**Payment boundary.** Provider, currencies, local tax/VAT, pricing, refund policy, and jurisdiction are **not selected**. Do not embed card processing into the RealSaS backend. Introduce `checkout_orders` and an append-only `payment_events` inbox only after selecting a compliant hosted provider. Verify webhook signature, event provenance, idempotency, order-amount/currency consistency and replay behavior server-side; client redirect is never payment proof. Payment and compiler roles must not share service tokens.

**Lifecycle.** `draft -> payment_pending -> paid -> credit_granted` are commercial states, **not** compiler quality states. `compile_queued -> running -> qualified_pass/failed` are product states. Order payment cannot bypass source authority, model release admission, Proof gates or per-owner access checks.

**Single vs multiple views.** Single-view reconstruction estimates occluded structure, so public copy must not promise the same source authority as controlled exact-8 input. The 1–2-view path is disabled pending qualified source preparation. Multi-view marketing language such as "We don't generate. We compile." should be used only in a context that accurately describes the selected pipeline.

## Production release gates (all required before a commercial launch)

1. Public intake qualified and independently proven (or the product explicitly limits ordering to an already qualified route). No promise of paid 1-view output until that gate passes.
2. User auth (OIDC/passkeys or comparable), session rotation, recovery and device link proof. Cross-account authorization tests for uploads, compile, product revision, character, motion and export.
3. Entitlement transaction contract implemented and tested for concurrent purchases, retries, duplicate webhooks, failed compiles, delayed success and refunds.
4. Privacy policy, Terms, use of submitted artwork, copyright reporting, retention/deletion, training-use consent policy and support/chargeback route written and reviewed. Never imply user artwork is trained on or deleted without an implemented policy.
5. HTTPS edge with Secure/HttpOnly/SameSite cookies, trusted-forwarded-header policy, CSRF protection where needed, upload size/format verification, per-account/IP abuse limits, CSP, security logging and secret rotation. Keep operator APIs unreachable from the public network.
6. Staging deploy run: pinned canonical source SHA, explicit PRODUCT model releases, DB migration backup/restore rehearsal, gateway secret separation, worker capacity/rate control, observability, alerts and a rollback plan.
7. Full customer path on actual qualified character: account -> payment -> credit -> upload -> proof-qualified compile -> persistent puppet -> Studio -> preview -> authorized native export -> support/refund path.

## Incremental implementation sequence

**A. Now:** Brand hierarchy, truthful copy, visual safety, no payment UI, no fake compiler capabilities. Publish static public preview only as a preview.

**B. Next:** Formal account/character identity and ownership model; account↔session linking; DB migration and integration tests. Keep existing owner hashes immutable.

**C. Then:** Ledger and entitlement reservation/release with idempotency tests and an explicit billing provider adapter interface. Simulated local payment fixtures must be impossible to invoke in production.

**D. After a qualified public compile:** Provider-backed checkout, verified webhook inbox, user-facing purchases and receipts, tax/refund implementation, customer support and production launch verification.

Open decisions requiring product owner sign-off: provider/jurisdiction, offered view counts, price per qualified puppet compile, free retry policy, storage/retention limits, monthly spend ceiling, trial rules and export licensing.

**Non-negotiable:** No paid checkout or "successful compile" language until actual engine source authority, qualified ProductRevision and true backend entitlements agree.
