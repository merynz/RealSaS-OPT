# FIT8 Morphology Scale Pilot V0 — Exploratory Only

**Date:** 2026-10-25  
**Status:** `EXPLORATORY__NOT_PREREGISTERED__NO_RESOLUTION_SELECTION_AUTHORITY`

This pilot was performed only to design the next preregistered real-scale extraction court. Thresholds and treatment choices must not be claimed as prospectively selected from this file.

## Source scope

Exact selected source geometry was directly readable for:

- Bat — selected glTF;
- Horse — selected glTF;
- GreenBlob — selected glTF;
- George — selected glTF.

The selected sources for Snake, Spider, T-Rex and Shark are FBX and were not directly parseable in the audit runtime. For this exploratory pilot only, the corresponding official Quaternius pack/model OBJ rest-shape geometry was used as a **morphology proxy**:

- Snake.obj for selected Snake.fbx;
- Spider.obj for selected Spider.fbx;
- Trex.obj for selected Trex.fbx;
- Shark.obj for selected Shark.fbx.

These OBJ proxies are not asserted byte-, topology-, skin- or transform-equivalent to the selected FBX sources and are not admissible as final FIT8 truth.

## Method

For each rest-shape geometry:

1. concatenate authored mesh-node transforms;
2. center the rest-shape bbox;
3. scale the largest bbox dimension to 1.0 TESSA unit for this descriptive proxy;
4. render 24 deterministic orthographic projection directions on a 512x512 binary silhouette;
5. skeletonize each silhouette;
6. estimate local full silhouette thickness as `2 * EuclideanDistanceToBoundary` at skeleton pixels;
7. compare each observed thickness against the finite-phase cylinder observability distribution from `EXTRACTION_TOPOLOGY_OBSERVABILITY_RESULT_V1.md`.

This is a **projected local-thickness proxy**, not 3-D medial thickness and not the final controlled eight-view observation contract.

## Local silhouette-thickness distribution

TESSA-normalized units:

| Subject | p01 | p05 | p10 | p25 | median |
|---|---:|---:|---:|---:|---:|
| Bat | 0.00861 | 0.01722 | 0.02838 | 0.05682 | 0.09896 |
| Horse | 0.01206 | 0.02583 | 0.03183 | 0.04214 | 0.07912 |
| GreenBlob | 0.08608 | 0.11696 | 0.14194 | 0.22727 | 0.41128 |
| George | 0.00603 | 0.01376 | 0.01892 | 0.03099 | 0.05936 |
| Snake* | 0.00431 | 0.01549 | 0.04044 | 0.10154 | 0.15930 |
| Spider* | 0.00431 | 0.00946 | 0.01376 | 0.02322 | 0.02753 |
| T-Rex* | 0.00431 | 0.01292 | 0.01892 | 0.03875 | 0.06973 |
| Shark* | 0.00946 | 0.01892 | 0.02494 | 0.04219 | 0.08265 |

`*` = same-pack OBJ proxy, not exact selected FBX.

## Phase-observability proxy by extraction resolution

For every skeleton thickness, compute the fraction of cylinder sub-cell phases for which a base-grid sign sample would observe that thickness. Two descriptive summaries are shown:

- `mean`: mean phase-observability probability over skeleton samples;
- `>=95%`: fraction of skeleton samples observable in at least 95% of finite phases.

| Subject | R24 mean / >=95% | R32 mean / >=95% | R48 mean / >=95% | R64 mean / >=95% |
|---|---:|---:|---:|---:|
| Bat | 0.881 / 0.791 | 0.918 / 0.860 | 0.951 / 0.913 | 0.968 / 0.938 |
| Horse | 0.862 / 0.670 | 0.940 / 0.823 | 0.978 / 0.957 | 0.987 / 0.973 |
| GreenBlob | 1.000 / 0.999 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| George | 0.754 / 0.580 | 0.835 / 0.693 | 0.920 / 0.819 | 0.957 / 0.904 |
| Snake* | 0.914 / 0.892 | 0.926 / 0.902 | 0.946 / 0.917 | 0.962 / 0.928 |
| Spider* | 0.454 / 0.224 | 0.614 / 0.265 | 0.858 / 0.604 | 0.924 / 0.837 |
| T-Rex* | 0.798 / 0.640 | 0.861 / 0.763 | 0.923 / 0.848 | 0.953 / 0.905 |
| Shark* | 0.836 / 0.687 | 0.901 / 0.790 | 0.961 / 0.895 | 0.982 / 0.954 |

Macro descriptive means:

| Resolution | mean phase observability | fraction >=95% phase observable |
|---:|---:|---:|
| 24 | 0.813 | 0.686 |
| 32 | 0.874 | 0.762 |
| 48 | 0.942 | 0.869 |
| 64 | 0.967 | 0.930 |

## What this pilot says

The current res24 risk is not merely theoretical. Thin morphology is highly subject-dependent; Spider is a particularly strong stress case in the authored same-pack rest shape. Moving from 24 to 48/64 materially improves the analytic observability proxy.

However **no production resolution is selected here**. This pilot was not preregistered and mixes exact selected geometry with four same-pack OBJ proxies. It is explicitly outcome-informed development evidence.

## Authorized use

Use this pilot only to design a prospective validation court that:

- uses exact selected/controlled geometry or an independently verified equivalent;
- freezes morphology-scale metrics and thresholds before evaluation;
- compares fixed 24/32/48/64 and, if justified, an adaptive/coarse-to-fine policy;
- measures compute/memory alongside observability;
- does not use authored triangle edge length as the target scale.
