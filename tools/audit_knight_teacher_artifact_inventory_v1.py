from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

def describe(path: Path):
    with np.load(path, allow_pickle=False) as z:
        out={}
        for k in z.files:
            a=np.asarray(z[k])
            row={"shape":list(a.shape),"dtype":str(a.dtype)}
            if a.dtype.kind in "SU" and a.size:
                vals=a.reshape(-1)[:12].tolist()
                row["sample"]=[str(x) for x in vals]
            elif a.dtype.kind in "ifub" and a.size:
                x=a.astype(np.float64,copy=False)
                finite=np.isfinite(x)
                if np.any(finite):
                    row["min"]=float(np.min(x[finite])); row["max"]=float(np.max(x[finite]))
            out[k]=row
        return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--bank",type=Path,required=True)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    payload={
        "schema":"RealSaS.KnightTeacherArtifactInventory.v1",
        "bank":describe(a.bank),
        "source":describe(a.source),
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps(payload,indent=2,sort_keys=True))

if __name__=="__main__":
    main()
