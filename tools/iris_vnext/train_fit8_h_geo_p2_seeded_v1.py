#!/usr/bin/env python3
from __future__ import annotations

"""Deterministic scientific entrypoint for H-GEO-P2 carrier training."""

import torch
from tools.iris_vnext.train_fit8_h_geo_p2_v1 import main

INITIALIZATION_SEED = 27060001

if __name__ == "__main__":
    torch.manual_seed(INITIALIZATION_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(INITIALIZATION_SEED)
    raise SystemExit(main())
