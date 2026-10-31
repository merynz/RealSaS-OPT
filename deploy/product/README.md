# RealSaS product deployment boundary

This directory packages the current Awaken → qualified ProductRevision → Studio handoff without choosing a domain, hosting provider, or final consumer-facing product name.

## Security and authority boundary

`web` is the only service published to the host and it is bound to `127.0.0.1` by default. Platform API, control worker, operator endpoints and artifact storage remain container-network/private surfaces. A future TLS/authenticated reverse proxy may publish the web service after account/auth/tenant policy and domain naming are frozen.

The browser never receives the Platform gateway token or internal EngineRelease, Attempt, Artifact, ModelRelease or ProductRevision identifiers. The web service holds the product-gateway token server-side and exchanges only opaque public compile ids with the browser. Operator endpoints require the independent operator token and must not be exposed by the public edge.

Production compile is fail-closed. Platform admission requires the exact canonical-main source binding, an explicitly bound PRODUCT release channel and exact PRODUCTION ModelRelease snapshots. Public front/back compile has an additional gate: its source normalization/materializer must be explicitly qualified and `REALSAS_PRODUCT_COMPILE_ENABLED=1`.

The controlled exact-8 lane is intentionally separate. It accepts already observed `V0..V7` source rasters plus their exact controlled camera bundle, materializes `RealSaS.SourceObservation.v1`, and enters the same shipping `product.input → shape → mesh → rig → skin → mechanics → graphics → motion → proof` DAG. It never mirrors, invents or relabels missing views as observed source authority.

## Images

`Dockerfile.platform` builds `realsas-api`, `realsas-control-worker`, `realsas-migrate` and `realsasctl`. The exact source SHA is injected into `realsas-api` with Go ldflags; production builds must pass the reviewed canonical-main 40-character SHA.

`Dockerfile.product` is the unprivileged Awaken/Studio shell plus the light intake/workspace runtime. It includes the deterministic workspace projection, exact-8 operator launcher and deploy-preflight tool. It intentionally does not silently install or invent learned-model checkpoints.

The learned engine is an external capability-specific image selected with `REALSAS_ENGINE_IMAGE`. This is deliberate: MIRA/other learned capabilities may require CUDA/BF16 admission, and the CPU developer Platform is not a substitute for that product execution contract.

## Compose profiles

The default stack starts migration, Platform API, control worker and the Awaken/Studio web shell. It does **not** claim that learned compile is runnable.

The `workspace-worker` has a built-in deterministic materializer:

```text
python -m product.workspace_worker.materialize_workspace_v1
```

so an empty `REALSAS_WORKSPACE_MATERIALIZER_COMMAND` means “use the canonical built-in default,” not “unbound.”

The public `intake-worker` is different: `REALSAS_SOURCE_MATERIALIZER_COMMAND` remains deliberately unbound until the front/back normalization/provenance capability is qualified. Do not point it at the exact-8 launcher and do not duplicate or mirror missing views as source evidence.

Enable `engine` only after `REALSAS_ENGINE_IMAGE` points at the admitted learned engine image whose exact checkpoints are represented by PRODUCTION ModelReleases.

Example staging validation, with public front/back compile still disabled:

```bash
cp deploy/product/.env.example deploy/product/.env
# edit infrastructure endpoints, exact source SHA, engine image and service tokens

docker compose --env-file deploy/product/.env \
  -f deploy/product/compose.yaml config

docker compose --env-file deploy/product/.env \
  -f deploy/product/compose.yaml up --build
```

The local UI is then available at `http://127.0.0.1:8765` unless `REALSAS_WEB_PORT` is changed. No DNS, certificate, external Platform port binding or hosting-provider mutation is performed by this stack.

## One-command deployment preflight

Run preflight from the product image so it can reach the private `platform-api` service without publishing operator routes. The local `.env` is mounted read-only only for the duration of this operator command.

```bash
docker compose --env-file deploy/product/.env \
  -f deploy/product/compose.yaml \
  run --rm --no-deps \
  -v "${PWD}/deploy/product/.env:/run/realsas-product.env:ro" \
  workspace-worker \
  python /app/tools/product_deploy_preflight_v1.py \
    --env-file /run/realsas-product.env \
    --api-root http://platform-api:8080 \
    --lane exact8
```

The Platform-side preflight verifies the active PRODUCT EngineRelease, canonical-main source SHA and the release's own sealed model requirements against current PRODUCTION model channels. Each selected model must match the required architecture, the active EngineRelease, the canonical-main source SHA and the `PRODUCTION` lifecycle state.

The report exposes two independent booleans:

- `exact8_product_ready`: controlled exact-8 product execution may be exercised;
- `public_frontback_ready`: public 1–2 view intake is also qualified and enabled.

It is valid during productization for the first to be true while the second remains false.

## Controlled exact-8 compile

Prepare one directory containing exactly:

```text
V0.png  V1.png  V2.png  V3.png
V4.png  V5.png  V6.png  V7.png
camera_bundle.json
```

The direction contract is `S, SE, E, NE, N, NW, W, SW`; rasters must be 1024×1024 RGBA PNGs with non-empty foreground and the source border-margin gate. `camera_bundle.json` must be the exact `RealSaS.CameraProjectionBundle.v1` emitted by the controlled source renderer. Per-view recentering or pose changes are rejected.

Run the launcher inside the private compose network:

```bash
docker compose --env-file deploy/product/.env \
  -f deploy/product/compose.yaml \
  run --rm --no-deps \
  -v "/absolute/path/to/exact8:/input:ro" \
  workspace-worker \
  python /app/tools/start_exact8_product_compile_v1.py \
    --views-dir /input \
    --camera-bundle /input/camera_bundle.json \
    --api-root http://platform-api:8080 \
    --display-name "Exact 8-View Character"
```

The launcher validates camera geometry and source rasters locally, round-trips the SourceObservation codec, then sends only the qualified SourceObservation bytes to the operator ingress. The resulting opaque `cmp_...` compile follows the normal shipping capability DAG and proof/ProductRevision rules.

## Model checkpoint admission

Checkpoint admission and promotion are intentionally separate. `tools/admit_product_model_release_v1.py` validates each family with its runtime/inference loader and can register a **DRAFT** ModelRelease bound to the exact EngineRelease/source SHA. It never fabricates evaluation PASS records and never promotes a channel. Real evaluation evidence must pass the Platform promotion policy before the model can enter PRODUCTION.

## Public production-enablement checklist

Before changing `REALSAS_PRODUCT_COMPILE_ENABLED` to `1`, all of the following must be true:

- image was built from the exact reviewed canonical-main SHA and the EngineRelease is bound to that SHA;
- database migrations are current;
- production EngineRelease channel is explicitly bound;
- every required learned family has an admitted `PRODUCTION` ModelRelease and exact checkpoint artifact;
- Temporal and the control worker are healthy;
- the capability engine worker is healthy and can read the shared artifact store;
- front/back product intake materializer is explicitly bound and produces qualified `RealSaS.SourceObservation.v1` bytes without fabricating authority;
- post-proof workspace worker uses the deterministic ProductRevision projection and preserves immutable proof/authority;
- product-gateway and operator tokens are distinct, high-entropy server-side secrets;
- end-user authentication, tenant ownership and rate/abuse limits are enforced by the public edge before internet exposure;
- a TLS reverse proxy is configured without exposing Platform operator routes.

Until the front/back-specific gates are satisfied, the correct public state is fail-closed. That does **not** prevent controlled exact-8 product execution from being admitted and tested once the release/model/engine preflight passes.
