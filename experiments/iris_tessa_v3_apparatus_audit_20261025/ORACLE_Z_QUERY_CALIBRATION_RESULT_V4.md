# Oracle-Z Query Calibration Result V4

**Run:** `40318916429`  
**Head SHA:** `e7815e0440bbc64b427f5afd6b915b49cbc397a0`  
**Artifact digest:** `sha256:9745cc4943bc2fd0c56aeae751ddd21abc35fbe54d7ff0d6b0abadbf0eaf78a2`  
**Status:** `QUERY_CLASS_SELECTED_V4`  
**Selected query class:** `Q192_D3_FOURIER4`

## Result

The V3 surface target/data were held fixed while only direct-XYZ query capacity/basis changed.

| query arm | params | best LR | mean nRMSE | worst nRMSE | mean local nRMSE | calibrated |
|---|---:|---:|---:|---:|---:|---|
| Q96_D2_PLAIN | 60,504 | 3e-3 | 0.42203 | 0.56042 | 0.50970 | no |
| Q256_D4_PLAIN | 1,196,568 | 1e-3 | 0.16706 | 0.27404 | 0.20403 | no |
| Q128_D3_FOURIER4 | 222,744 | 3e-3 | 0.10400 | 0.10900 | 0.12930 | no (mean gate narrowly missed) |
| **Q192_D3_FOURIER4** | **481,560** | **3e-3** | **0.09863** | **0.12229** | **0.12284** | **yes** |

The selected arm also passed the random-label firewall: mean random-label nRMSE `1.28831`, worst `1.30462`.

## Decision

Freeze the **query class**, not its per-scene calibration weights:

- width `192`;
- three hidden SiLU layers;
- explicit coordinate basis `[xyz, sin/cos(2^k*pi*xyz), k=0..3]`;
- LR `3e-3` is the calibrated tiny-court starting point, not a production optimizer mandate.

The earlier V3 calibration failure is therefore diagnosed primarily as a query-capacity / coordinate-basis confound, not evidence that the surface-response target or current carrier is inherently invalid.

Next: compare matched triplane and dense scene carriers using a **shared** Q of this selected class. Per-scene direct-XYZ Q remains an upper-bound control only and must not be mistaken for a deployable scene representation.
