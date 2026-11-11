# BUSTED V1: one main-owned frontend and product API

Source of truth: `product/busted_site/` in `main`. The historical
`busted-live-preview` branch is now a **generated publication target**, never
the source for product or model semantics.

## Developer/product server

```bash
python -m product.busted_site.server --host 127.0.0.1 --port 8765
```

The server returns the exact BUSTED Awaken and Studio frontend; the existing
`product.living_compile` implementation continues serving proof-qualified
bundles, edits, motion frames and exports. There is no second renderer or
animated rig implementation in the web server.

## External HTTPS frontend (GitHub Pages)

GitHub Pages cannot execute Python or host a model checkpoint. It only hosts
static assets. Set repository variable `BUSTED_PUBLIC_API_ORIGIN` to the
HTTPS origin of the production API (no path, e.g. `https://api.example.org`).
The main-owned publication workflow generates `runtime-config.js` as
ordinary public, nonsecret configuration and publishes static assets from main.

If that variable is empty, API requests remain same-origin. GitHub Pages
returns no product gateway, and Awaken truthfully displays **COMPILER OFFLINE**.

The backend uses a strict Origin allowlist. For the currently published Pages
URL, the HTTP Origin is `https://merynz.github.io` (the repository path is
not part of the Origin). Set:

```bash
export BUSTED_PUBLIC_MODE=1
export BUSTED_PUBLIC_ORIGIN=https://merynz.github.io
export BUSTED_WORKSPACE_ROOT=/srv/busted/workspaces
python -m product.busted_site.server --host 127.0.0.1 --port 8765
```

Place a managed TLS reverse proxy in front of the loopback server. The
`BUSTED_WORKSPACE_ROOT` directory must already exist. Public API operations
accept **opaque workspace IDs**, never a client-supplied filesystem path.
The project service currently advertises `compile_available=false` and
`public_compile_gateway=false`; this is intentional until the real
RSS.vNext inference -> Compiler ProductRevision -> Graphics/Idle -> Studio
publication court can grant product authority. Site readiness must follow
that real executable gate, not the existence of an endpoint or a checkpoint.

## Proof and security limits

- Only two observed FRONT/BACK inputs become model source authority.
- The native C++ Renderer and CanonicalMesh/Rig/Skin are existing Compiler
  consumers, not arbitrary browser animation math.
- User-edited puppet layers stay unqualified until a proof-backed revision.
- Workspaces exposed publicly require authentication/authorization and
  durable storage before commercial launch. Opaque IDs alone are not a
  substitute for account-level authorization.
- This publishing workflow cannot prove real eight-way illustration quality,
  successful unseen learning, or product deployment readiness.

## Regression tests

```bash
PYTHONPATH=. python -m pytest -q --import-mode=importlib \
  tests/product/test_living_compile_v4.py \
  tests/product/test_busted_main_site_gateway_v1.py
node --experimental-default-type=module tests/product/test_busted_api_origin.mjs
```

Tests for source/model integrity, mechanics and C++ raster remain separate
product CI contracts; historical Knight and byte-exact seals remain
manually inspectable without blocking generic RSS.vNext work.
