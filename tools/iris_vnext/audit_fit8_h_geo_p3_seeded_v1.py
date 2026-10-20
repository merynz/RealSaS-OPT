#!/usr/bin/env python3
from __future__ import annotations

import torch
from tools.iris_vnext.audit_fit8_h_geo_p3_v1 import main

PROBE_INITIALIZATION_SEED = 27099991

if __name__ == "__main__":
    torch.manual_seed(PROBE_INITIALIZATION_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(PROBE_INITIALIZATION_SEED)
    raise SystemExit(main())
