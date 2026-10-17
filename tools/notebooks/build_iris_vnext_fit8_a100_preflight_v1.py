from __future__ import annotations

"""Build an A100 preflight-only Colab notebook for the sealed IRIS vNext Fit8 cohort.

The notebook deliberately stops after engineering preflight.  It never starts the
scientific Fit8 optimizer lineage.  A separate training handoff is required after
its report is sealed PASS.
"""

import argparse
import json
import re
from pathlib import Path

FLEXICUBES_COMMIT = "4cc7d6c3d0cee83c011ce36721b81adff0dd7db6"


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--drive-prepared-relative",
        default="RealSaS_Fit8_IRIS_vNext_20261016/prepared",
    )
    p.add_argument(
        "--drive-report-relative",
        default="RealSaS_Fit8_IRIS_vNext_20261016/A100_PREFLIGHT.json",
    )
    return p.parse_args()


def _code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text,
    }


def _markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def build(*, source_commit: str, prepared_relative: str, report_relative: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("VNEXT_A100_NOTEBOOK_SOURCE_COMMIT_INVALID")
    intro = _markdown(
        "# RealSaS IRIS vNext Fit8 — A100 engineering preflight only\n\n"
        "This notebook **does not start scientific Fit8 training**. It verifies the "
        "sealed prepared cohort, exact source commit, B=8 optimizer semantics, A100 "
        "memory/runtime, and the pinned differentiable FlexiCubes path. Stop after PASS."
    )
    bootstrap = _code(
        f'''import os, sys, json, hashlib, shutil, subprocess\n'''
        f'''from pathlib import Path\n'''
        f'''from google.colab import drive\n'''
        f'''drive.mount("/content/drive")\n'''
        f'''SOURCE_COMMIT={source_commit!r}\n'''
        f'''FLEXICUBES_COMMIT={FLEXICUBES_COMMIT!r}\n'''
        f'''DRIVE_PREPARED=Path("/content/drive/MyDrive")/{prepared_relative!r}\n'''
        f'''DRIVE_REPORT=Path("/content/drive/MyDrive")/{report_relative!r}\n'''
        f'''WORK=Path("/content/realsas_iris_vnext_fit8_preflight")\n'''
        f'''REPO=WORK/"repo"\n'''
        f'''COHORT=WORK/"prepared"\n'''
        f'''FLEXI=WORK/"FlexiCubes"\n'''
        f'''WORK.mkdir(parents=True,exist_ok=True)\n'''
        f'''assert (DRIVE_PREPARED/"PREPARATION_RECEIPT.json").is_file(), DRIVE_PREPARED\n'''
        f'''assert (DRIVE_PREPARED/"COHORT_MANIFEST.json").is_file(), DRIVE_PREPARED\n'''
        f'''print("[BOUND] prepared cohort found", DRIVE_PREPARED)\n'''
    )
    install = _code(
        '''subprocess.run([sys.executable,"-m","pip","install","--quiet",\n'''
        '''    "torch==2.8.0","torchvision==0.23.0",\n'''
        '''    "--index-url","https://download.pytorch.org/whl/cu128"],check=True)\n'''
        '''subprocess.run([sys.executable,"-m","pip","install","--quiet",\n'''
        '''    "numpy==2.1.3","scipy==1.14.1","Pillow==11.3.0"],check=True)\n'''
        '''import torch\n'''
        '''assert torch.cuda.is_available(), "Select an A100 GPU runtime; CUDA is unavailable."\n'''
        '''name=torch.cuda.get_device_name(0)\n'''
        '''props=torch.cuda.get_device_properties(0)\n'''
        '''assert "A100" in name.upper(), f"A100 required; found {name}"\n'''
        '''assert props.total_memory >= 30*2**30, f"A100 memory unexpectedly small: {props.total_memory}"\n'''
        '''assert torch.cuda.is_bf16_supported(), "BF16 support required"\n'''
        '''print(f"[A100] torch={torch.__version__} cuda={torch.version.cuda} gpu={name} GiB={props.total_memory/2**30:.1f}")\n'''
    )
    source = _code(
        '''def checkout_exact(url,commit,dst):\n'''
        '''    dst=Path(dst)\n'''
        '''    if dst.exists(): shutil.rmtree(dst)\n'''
        '''    subprocess.run(["git","init",str(dst)],check=True)\n'''
        '''    subprocess.run(["git","-C",str(dst),"remote","add","origin",url],check=True)\n'''
        '''    subprocess.run(["git","-C",str(dst),"fetch","--depth","1","origin",commit],check=True)\n'''
        '''    subprocess.run(["git","-C",str(dst),"checkout","--detach","FETCH_HEAD"],check=True)\n'''
        '''    head=subprocess.check_output(["git","-C",str(dst),"rev-parse","HEAD"],text=True).strip()\n'''
        '''    assert head==commit,(head,commit)\n'''
        '''    return head\n'''
        '''checkout_exact("https://github.com/merynz/RealSaS-OPT.git",SOURCE_COMMIT,REPO)\n'''
        '''checkout_exact("https://github.com/nv-tlabs/FlexiCubes.git",FLEXICUBES_COMMIT,FLEXI)\n'''
        '''sys.path.insert(0,str(REPO))\n'''
        '''os.environ["PYTHONPATH"]=str(REPO)\n'''
        '''os.environ["REALSAS_FLEXICUBES_ROOT"]=str(FLEXI)\n'''
        '''if COHORT.exists(): shutil.rmtree(COHORT)\n'''
        '''shutil.copytree(DRIVE_PREPARED,COHORT)\n'''
        '''def sha(path):\n'''
        '''    h=hashlib.sha256()\n'''
        '''    with Path(path).open("rb") as f:\n'''
        '''        for block in iter(lambda:f.read(4*1024*1024),b""): h.update(block)\n'''
        '''    return h.hexdigest()\n'''
        '''prep=json.loads((COHORT/"PREPARATION_RECEIPT.json").read_text())\n'''
        '''manifest=json.loads((COHORT/"COHORT_MANIFEST.json").read_text())\n'''
        '''builder=REPO/"tools/iris_vnext/build_fit8_vnext_shards_v2.py"\n'''
        '''assert prep["builder_sha256"]==sha(builder), "Prepared cohort builder/source commit mismatch"\n'''
        '''assert prep["cohort_manifest_sha256"]==sha(COHORT/"COHORT_MANIFEST.json")\n'''
        '''assert manifest["schema"]=="RealSaS.IRISVNextFit8Manifest.v4"\n'''
        '''assert prep["scientific_fit_started"] is False\n'''
        '''assert prep["scientific_fit_pass_claimed"] is False\n'''
        '''print("[SOURCE] exact commit",SOURCE_COMMIT)\n'''
        '''print("[COHORT] manifest",manifest["manifest_value_sha256"])\n'''
    )
    preflight = _code(
        '''DRIVE_REPORT.parent.mkdir(parents=True,exist_ok=True)\n'''
        '''cmd=[sys.executable,str(REPO/"tools/iris_vnext/a100_preflight_v1.py"),\n'''
        '''     "--cohort-dir",str(COHORT),\n'''
        '''     "--flexicubes-root",str(FLEXI),\n'''
        '''     "--output",str(DRIVE_REPORT),\n'''
        '''     "--geometry-steps","3"]\n'''
        '''subprocess.run(cmd,check=True,cwd=REPO)\n'''
        '''report=json.loads(DRIVE_REPORT.read_text())\n'''
        '''assert report["status"]=="PASS"\n'''
        '''assert report["scientific_fit_started"] is False\n'''
        '''assert report["preflight_checkpoint_reusable"] is False\n'''
        '''print("\\n[A100 PREFLIGHT PASS]")\n'''
        '''print(json.dumps({\n'''
        '''    "device":report["device"],\n'''
        '''    "model":report["model"],\n'''
        '''    "cohort":report["cohort"],\n'''
        '''    "geometry_optimizer_probe":report["geometry_optimizer_probe"],\n'''
        '''    "flexicubes_analytic_probe":report["flexicubes_analytic_probe"],\n'''
        '''},indent=2))\n'''
        '''print("\\nSTOP: scientific Fit8 training has NOT started. Return this PASS report to Alfred for the training handoff.")\n'''
    )
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "A100", "provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": [intro, bootstrap, install, source, preflight],
    }


def main():
    args = _args()
    value = build(
        source_commit=args.source_commit,
        prepared_relative=args.drive_prepared_relative,
        report_relative=args.drive_report_relative,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")
    print("VNEXT_FIT8_A100_PREFLIGHT_NOTEBOOK_BUILD_PASS", args.output)


if __name__ == "__main__":
    main()
