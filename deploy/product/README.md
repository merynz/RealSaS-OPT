# RealSaS product deployment boundary

This directory packages the current Awaken → qualified ProductRevision → Studio handoff without choosing a domain, hosting provider, or final consumer-facing product name.

## Security and authority boundary

`web` is the only service published to the host and it is bound to `127.0.0.1` by default. Platform API, control worker and artifact storage remain container-network/private surfaces. A future TLS/authenticated reverse proxy may publish the web service after account/auth/tenant policy and domain naming are frozen.

The browser never receives the Platform gateway token or internal EngineRelease, Attempt, Artifact, ModelRelease or ProductRevision identifiers. The web service holds the product-gateway token server-side and exchanges only opaque public compile ids with the browser.

Production compile is fail-closed. `REALSAS_PRODUCT_COMPILE_ENABLED=1` is not sufficient by itself: the shell also requires a configured gateway/token and writable workspace bundle root. Platform admission separately requires the exact canonical-main source binding, production release channel and exact production ModelRelease snapshots.

## Images

`Dockerfile.platform` builds `realsas-api`, `realsas-control-worker`, `realsas-migrate` and `realsasctl`. The exact source SHA is injected into `realsas-api` with Go ldflags; production builds must pass the reviewed canonical-main 40-character SHA.

`Dockerfile.product` is the unprivileged Awaken/Studio shell plus light intake/workspace worker runtime. It intentionally does not silently install or invent learned-model checkpoints.

The learned engine is an external capability-specific image selected with `REALSAS_ENGINE_IMAGE`. This is deliberate: MIRA/other learned capabilities may require CUDA/BF16 admission, and the CPU developer Platform is not a substitute for that product execution contract.

## Compose profiles

The default stack starts migration, Platform API, control worker and the Awaken/Studio web shell. It does **not** claim that learned compile is runnable.

Enable `workers` only after `REALSAS_SOURCE_MATERIALIZER_COMMAND` and `REALSAS_WORKSPACE_MATERIALIZER_COMMAND` are real executable commands inside the product image. Enable `engine` only after `REALSAS_ENGINE_IMAGE` points at the admitted learned engine image with the production checkpoints available through the model-release/artifact path.

Example staging validation, with compile still disabled:

```bash
cp deploy/product/.env.example deploy/product/.env
# edit infrastructure endpoints and service tokens

docker compose --env-file deploy/product/.env \
  -f deploy/product/compose.yaml config

docker compose --env-file deploy/product/.env \
  -f deploy/product/compose.yaml up --build
```

The local UI is then available at `http://127.0.0.1:8765` unless `REALSAS_WEB_PORT` is changed. No DNS, certificate, external port binding or hosting-provider mutation is performed by this stack.

## Production-enablement checklist

Before changing `REALSAS_PRODUCT_COMPILE_ENABLED` to `1`, all of the following must be true:

- image was built from the exact reviewed canonical-main SHA and the EngineRelease is bound to that SHA;
- database migrations are current;
- production EngineRelease channel is explicitly bound;
- every required learned family has an admitted `PRODUCTION` ModelRelease and exact checkpoint artifact;
- Temporal and the control worker are healthy;
- the capability engine worker is healthy and can read the shared artifact store;
- front/back product intake materializer is bound and produces qualified `RealSaS.SourceObservation.v1` bytes without fabricating authority;
- post-proof workspace materializer is bound and preserves the immutable ProductRevision/proof while deriving the Studio bundle;
- product-gateway and operator tokens are distinct, non-empty secrets;
- end-user authentication, tenant ownership and rate/abuse limits are enforced by the public edge before internet exposure;
- a TLS reverse proxy is configured without exposing Platform operator routes.

Until those gates are satisfied, the correct product state is the current one: Awaken/Studio can be previewed and existing proof-qualified bundles can be edited, while new production compile remains closed.
