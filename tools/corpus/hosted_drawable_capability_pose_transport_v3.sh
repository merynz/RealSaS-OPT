#!/usr/bin/env bash
# Original Cat two authored poses × 4 views × 2 Cycles samples × 4 source UV edits.
# Source teacher arrays and original art remain private. Original V3 FAIL also archived.
set -euo pipefail
umask 077
D="$RUNNER_TEMP/drawable-pose-v3"
ROOT="gdrive:corpus370_preparation_v1"
mkdir -p "$D/blender" "$D/src/Textures" "$D/atlas" "$D/out"
export RCLONE_CONFIG="$D/private-rclone.conf"
test -n "$DRIVE_CONFIG_B64"
printf '%s' "$DRIVE_CONFIG_B64" | base64 -d > "$RCLONE_CONFIG"
test -s "$RCLONE_CONFIG"
unset DRIVE_CONFIG_B64

save_private_receipts() {
  REMOTE="$ROOT/hosted_drawable_capability_pose_transport_v3/CAT/$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT"
  for F in "$D/render/"*.json "$D/render/"*.npz "$D/out/"*.json "$D/blender-render.log"; do
    if [ -s "$F" ]; then
      NAME="$(basename "$F")"
      rclone copyto "$F" "$REMOTE/$NAME" --checksum --retries 3
      rclone cat "$REMOTE/$NAME" > "$D/readback-$NAME"
      cmp "$F" "$D/readback-$NAME"
      sha256sum "$F"
    fi
  done
}
trap 'rc=$?; save_private_receipts || true; exit "$rc"' EXIT

sudo apt-get -o Acquire::Retries=6 update -qq
sudo apt-get -o Acquire::Retries=6 install -y --no-install-recommends libxi6 libxrender1 libxfixes3 libxkbcommon0 libsm6 libgl1
python -m pip install 'numpy>=1.26,<3' 'pillow>=10,<13'
curl -fL --retry 4 --max-time 300 https://download.blender.org/release/Blender4.5/blender-4.5.14-linux-x64.tar.xz -o "$D/blender/source.tar.xz"
curl -fL --retry 4 --max-time 120 https://download.blender.org/release/Blender4.5/blender-4.5.14.sha256 -o "$D/blender/checksums"
D="$D" python - <<'PY'
import hashlib,os,re
from pathlib import Path
d=Path(os.environ["D"])/"blender"
rows=[z for z in (d/"checksums").read_text().splitlines()
      if "blender-4.5.14-linux-x64.tar.xz" in z]
assert len(rows)==1
matches=re.findall(r"[a-fA-F0-9]{64}",rows[0])
assert len(matches)==1
with (d/"source.tar.xz").open("rb") as f:
    assert hashlib.file_digest(f,"sha256").hexdigest()==matches[0].lower()
PY
tar -xJf "$D/blender/source.tar.xz" -C "$D/blender"
BLENDER="$D/blender/blender-4.5.14-linux-x64/blender"
test "$(sha256sum "$BLENDER" | cut -d' ' -f1)" = 050c02562f81fe80ba616a80198fa02d381e60f8b61b8d39add881f4bca0d7d8
R="$RUNNER_TEMP/rclone-v1.75.2-linux-amd64.zip"
curl -fLsS --retry 3 --max-time 180 https://github.com/rclone/rclone/releases/download/v1.75.2/rclone-v1.75.2-linux-amd64.zip -o "$R"
echo "349ac8fba6ff65d6247043f1750cdcb518ec5d500ef91463a10d37c0ccdf3702  $R" | sha256sum --check
unzip -q "$R" -d "$RUNNER_TEMP"
export PATH="$RUNNER_TEMP/rclone-v1.75.2-linux-amd64:$PATH"

rclone copyto "$ROOT/hosted_operational_drawable_inputs_v1/42220981274-1/ORIGINAL_TWO_SOURCE_INPUTS.zip" "$D/inputs.zip" --checksum --retries 3
echo "e3c758b0a4dc9bd3e7f889e94366fa7ff40517deb15e68a202d525b62de13948  $D/inputs.zip" | sha256sum --check
NATIVE="$ROOT/hosted_native_temporal_pilot_v1/41915393652-1/subjects/corpus_000ca1ae51f9582fde9e"
rclone copyto "$NATIVE/teacher.npz" "$D/teacher.npz" --checksum --retries 3
rclone copyto "$NATIVE/teacher.npz.json" "$D/teacher.json" --checksum --retries 3
rclone copyto "$ROOT/hosted_operational_cat_source_uv_plan_v2/CAT/42260446130-1/PRIVATE_CAT_ORIGINAL_SOURCE_UV_EDIT_PLAN_V2.json" "$D/plan.json" --checksum --retries 3
EDIT="$ROOT/hosted_operational_cat_uv_texture_cycles_v2/CAT/42263195370-1"
rclone copyto "$EDIT/PRIVATE_SOURCE_UV_ATLAS_EDIT_IMAGES_V1.json" "$D/atlas-assets.json" --checksum --retries 3
# Verify ALL eight immutable V2 atlas arms even though V3 renders only the first four.
for ARM in BASE SINGLE_0 SINGLE_1 SINGLE_2 PAIR_01 PAIR_02 PAIR_12 JOINT_012; do
  rclone copyto "$EDIT/atlas/$ARM.png" "$D/atlas/$ARM.png" --checksum --retries 3
done
echo "c2792c8efb36c4ccdf9178c6b933351b50364bbd28707e9365a3c115db00d603  $D/teacher.npz" | sha256sum --check
echo "a0f224019deca20bf5d807ab4d27a7c341a65bc21930fc99ad8e88ca5b4bc023  $D/teacher.json" | sha256sum --check
echo "8be8fccde3d9cc97f7ba452fd50c1a9f4ebea2e34da590a22d30872b2b0f497b  $D/plan.json" | sha256sum --check
echo "0e9a2fcf6a2c048eb6e99048793290b8865d802689f59b7c8c20bc6390cf479a  $D/atlas-assets.json" | sha256sum --check

D="$D" python - <<'PY'
import hashlib,json,os,zipfile
from pathlib import Path
import numpy as np
from tools.corpus.court_corpus370_original_cat_uv_texture_cycles_v2 import verify_original_atlas_variants
d=Path(os.environ["D"])
with zipfile.ZipFile(d/"inputs.zip") as z:
    manifest=json.loads(z.read("PRIVATE_TRANSPORT_MANIFEST.json"))
    assert manifest["parent_manifest_sha256"]=="cb9ffea96b22bb4500b367afe9965d98fa52b99f51dbe4faf5fac8aabc1ae172"
    rows=manifest["files"]
    wanted=[
      ("/Models/GLB format/animal-cat.glb","src/animal-cat.glb","1e73aaf7d497dd7d2946badf92d189d2333f163e66dced91feb446bc9bdee850"),
      ("/Models/GLB format/Textures/colormap.png","src/Textures/colormap.png","2f6d0032e51c2d49b13b8e689d59eb39aa3ee6c32f2a1851213e11fc7593177d"),
      ("/corpus_000ca1ae51f9582fde9e/SOURCE_REST_RENDER_CANDIDATE.json","policy.json","c8fe4e307ddbbaf5cc0892cffc4267bd08e394f2299f4e70155ffc6207190a4b"),
      ("/corpus_000ca1ae51f9582fde9e/camera_bundle.json","camera.json","300b935a47379f6c44d4b4356554d1c32a86ca3b732a62405c73da9e0ec6adf0")
    ]
    for suffix,relative,expected in wanted:
        matches=[f for f in rows if f["relative_path"].endswith(suffix)]
        assert len(matches)==1,(suffix,len(matches))
        f=matches[0];bytes_=z.read("objects/"+f["sha256"])
        assert len(bytes_)==f["size_bytes"] and hashlib.sha256(bytes_).hexdigest()==f["sha256"]==expected
        dst=d/relative;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(bytes_)
plan=json.loads((d/"plan.json").read_text())
assets=json.loads((d/"atlas-assets.json").read_text())
verify_original_atlas_variants(d/"src/Textures/colormap.png",plan,assets,d/"atlas")
with np.load(d/"teacher.npz",allow_pickle=False) as t:
    posed=np.asarray(t["posed_vertices_source_world"],dtype=np.float64)
    rest=np.asarray(t["rest_vertices_source_world"],dtype=np.float64)
    assert posed.shape==(32,1204,3)
    assert np.max(np.linalg.norm(posed[14]-posed[0],axis=1))>1e-6*max(1.,np.linalg.norm(np.ptp(rest,axis=0)))
print("ORIGINAL_CAT_NATIVE_TEACHER_POSE_AND_GLTF_TOPLEFT_UV_SOURCE_ADDRESSES_SHA_VERIFIED",flush=True)
PY

"$BLENDER" --background --factory-startup --threads 2 --python-exit-code 1 \
    --python tools/truth_foundry/blender_corpus370_cat_authored_pose_uv_edits_v3.py -- \
    --source "$D/src/animal-cat.glb" \
    --original-atlas "$D/src/Textures/colormap.png" \
    --teacher-npz "$D/teacher.npz" --teacher-receipt "$D/teacher.json" \
    --camera-bundle "$D/camera.json" --source-policy "$D/policy.json" \
    --uv-plan "$D/plan.json" --atlas-assets "$D/atlas-assets.json" \
    --atlas-dir "$D/atlas" --output-dir "$D/render" > "$D/blender-render.log" 2>&1 || {
      tail -n 100 "$D/blender-render.log"; exit 1; }
tail -n 12 "$D/blender-render.log"

python -m tools.corpus.court_drawable_capability_pose_transport_v3 \
    --input-root "$D" --output "$D/out/PRIVATE_INDEPENDENT_CAT_POSE_UV_TRANSPORT_V3.json"
D="$D" python - <<'PY'
import json,os
from pathlib import Path
d=Path(os.environ["D"])
r=json.loads((d/"out/PRIVATE_INDEPENDENT_CAT_POSE_UV_TRANSPORT_V3.json").read_text())
assert r["actual_source_original_full_physical_renders"]==64
assert r["source_missing_evidence_obligations"]==[]
assert r["source_uv_surface_attached_edit_native_source_status"]=="SATISFIED_PHYSICAL_ORIGINAL_SOURCE_WITNESS_ONLY"
assert r["same_drawable_pairs"]==r["different_drawable_pairs"]==[]
assert r["training_authorized"] is False
print("ORIGINAL_CAT_NATIVE_POSE_GRAPHICS_SOURCE_CONSEQUENCE_COURTED__NO_ARTIST_OWNER_AUTHORITY",flush=True)
PY
