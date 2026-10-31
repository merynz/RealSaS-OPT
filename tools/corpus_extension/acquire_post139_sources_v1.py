import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import requests

SCHEMA = "realsas.post139_public_acquisition.v1"
UA = {"User-Agent": "RealSaS corpus provenance audit/1.0"}

QUATERNIUS = [
    ("ultimate-animated-character", "https://quaternius.com/packs/ultimateanimatedcharacter.html"),
    ("cube-world", "https://quaternius.com/packs/cubeworldkit.html"),
    ("sushi-restaurant", "https://quaternius.com/packs/sushirestaurantkit.html"),
    ("pirate-kit", "https://quaternius.com/packs/piratekit.html"),
    ("cyberpunk-kit", "https://quaternius.com/packs/cyberpunkgamekit.html"),
    ("scifi-essentials", "https://quaternius.com/packs/scifiessentialskit.html"),
    ("zombie-apocalypse", "https://quaternius.com/packs/zombieapocalypsekit.html"),
    ("universal-base-characters", "https://quaternius.com/packs/universalbasecharacters.html"),
    ("animated-dinosaurs", "https://quaternius.com/packs/animateddinosaurs.html"),
    ("cute-fish", "https://quaternius.com/packs/cutefish.html"),
    ("easy-enemy", "https://quaternius.com/packs/easyenemy.html"),
    ("animated-fish-legacy", "https://quaternius.com/packs/animatedfish.html"),
    ("animated-monster-legacy", "https://quaternius.com/packs/animatedmonster.html"),
    ("animated-robot-legacy", "https://quaternius.com/packs/animatedrobot.html"),
    ("animated-alien-legacy", "https://quaternius.com/packs/animatedalien.html"),
    ("animated-men-legacy", "https://quaternius.com/packs/animatedmen.html"),
    ("animated-man-legacy", "https://quaternius.com/packs/animatedman.html"),
]

KENNEY = [
    ("mini-characters", "https://kenney.nl/media/pages/assets/mini-characters/bfc7e272b4-1774770718/kenney_mini-characters.zip"),
    ("blocky-characters", "https://kenney.nl/media/pages/assets/blocky-characters/8369c0cf30-1749547469/kenney_blocky-characters_20.zip"),
    ("cube-pets", "https://kenney.nl/media/pages/assets/cube-pets/44e58e945f-1774520254/kenney_cube-pets_1.0.zip"),
    ("mini-arena", "https://kenney.nl/media/pages/assets/mini-arena/88f977a0cb-1709220730/kenney_mini-arena.zip"),
    ("mini-dungeon", "https://kenney.nl/media/pages/assets/mini-dungeon/6cd72dc849-1785314274/kenney_mini-dungeon.zip"),
    ("prototype-kit", "https://kenney.nl/media/pages/assets/prototype-kit/4d3b7073ed-1724832076/kenney_prototype-kit.zip"),
    ("animated-protagonists", "https://kenney.nl/media/pages/assets/animated-characters-protagonists/608191acc4-1774773108/kenney_animated-characters-protagonists.zip"),
    ("animated-survivors", "https://kenney.nl/media/pages/assets/animated-characters-survivors/27b16052a7-1774772958/kenney_animated-characters-survivors.zip"),
    ("animated-retro", "https://kenney.nl/media/pages/assets/animated-characters-retro/93305a3c49-1774772819/kenney_animated-characters-retro.zip"),
]

KAYKIT = [
    ("adventurers", "https://github.com/KayKit-Game-Assets/KayKit-Character-Pack-Adventures-1.0/archive/refs/heads/main.zip"),
    ("skeletons", "https://github.com/KayKit-Game-Assets/KayKit-Character-Pack-Skeletons-1.0/archive/refs/heads/main.zip"),
]


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, headers=UA, stream=True, timeout=120, allow_redirects=True) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    return {"url": url, "path": str(dst), "bytes": dst.stat().st_size, "sha256": sha_file(dst)}


def unpack_zip(path, dst):
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as z:
        z.extractall(dst)


def drive_folder_url(page_url):
    r = requests.get(page_url, headers=UA, timeout=60)
    r.raise_for_status()
    html = r.text
    m = re.search(r"https://drive\.google\.com/drive/folders/([A-Za-z0-9_-]+)", html)
    if not m:
        return None
    return "https://drive.google.com/drive/folders/" + m.group(1)


def download_drive_folder(url, dst):
    dst.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "gdown", "--folder", url, "-O", str(dst), "--remaining-ok"]
    p = subprocess.run(cmd, text=True, capture_output=True)
    return {"command": cmd, "returncode": p.returncode, "stdout_tail": p.stdout[-4000:], "stderr_tail": p.stderr[-4000:]}


def main():
    root = Path(os.environ.get("REALSAS_ACQ_ROOT", "post139_sources")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    receipts = []

    qroot = root / "quaternius"
    for pack_id, page in QUATERNIUS:
        rec = {"provider": "quaternius", "pack_id": pack_id, "page": page}
        try:
            folder = drive_folder_url(page)
            rec["drive_folder"] = folder
            if folder:
                rec["download"] = download_drive_folder(folder, qroot / pack_id)
                rec["status"] = "DOWNLOADED" if rec["download"]["returncode"] == 0 else "DOWNLOAD_ERROR"
            else:
                rec["status"] = "NO_PUBLIC_DRIVE_FOLDER"
        except Exception as e:
            rec["status"] = "ERROR"
            rec["error"] = repr(e)
        receipts.append(rec)

    for provider, items in [("kenney", KENNEY), ("kaykit", KAYKIT)]:
        proot = root / provider
        for pack_id, url in items:
            rec = {"provider": provider, "pack_id": pack_id, "url": url}
            try:
                z = proot / (pack_id + ".zip")
                rec["archive"] = download(url, z)
                unpack_zip(z, proot / pack_id)
                rec["status"] = "DOWNLOADED"
            except Exception as e:
                rec["status"] = "ERROR"
                rec["error"] = repr(e)
            receipts.append(rec)

    files = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            files.append({"path": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": sha_file(p)})
    (root / "ACQUISITION_RECEIPT.json").write_text(json.dumps({"schema_version": SCHEMA, "root": str(root), "receipts": receipts, "files": files}, indent=2), encoding="utf-8")
    errors = [r for r in receipts if r.get("status") not in {"DOWNLOADED", "NO_PUBLIC_DRIVE_FOLDER"}]
    print(json.dumps({"packs": len(receipts), "errors": len(errors), "files": len(files)}, indent=2))
    if errors:
        print("Acquisition had non-fatal pack errors; audit will continue on acquired sources.")


if __name__ == "__main__":
    main()
