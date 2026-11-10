#!/usr/bin/env bash
set -euo pipefail
umask 077
test -n "$SUBJECT_ORDINAL"
test -n "$DRIVE_CONFIG_B64"
D=/home/monster/realsas_authority
sudo install -d -o "$USER" -g "$(id -gn)" -m 0700 "$D"
R="$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT"
OUT="$D/corpus370_native_rgba_v14/$R/$SUBJECT_ORDINAL"
REMOTE="gdrive:corpus370_preparation_v1/hosted_native_mechanical_rgba_private_v14/$R/$SUBJECT_ORDINAL"
export RCLONE_CONFIG="$RUNNER_TEMP/realsas-v14-rclone.conf"
printf %s "$DRIVE_CONFIG_B64" | base64 -d > "$RCLONE_CONFIG"
test -s "$RCLONE_CONFIG"
unset DRIVE_CONFIG_B64
mkdir -p "$OUT"
upload() { rclone copy "$OUT" "$REMOTE" --checksum --transfers 2 --checkers 2 --retries 3 || true; }
trap upload EXIT
V5="$D/corpus370_v5_delta_repair_pilot/CORPUS370_PREPARED_OBSERVATION_PACKETS_V5.json"
ROSTER="$D/corpus370_preparation_v1/CORPUS370_EXACT_SOURCE_ROSTER_V1.json"
RAYS="$D/corpus370_hosted_ray_outputs/41883539256-1"
JOIN="$D/corpus370_hosted_native_visibility_pairs/41916724594-1"
GROUP="$D/corpus370_allview_groups_v3/41986038051-1"
NATIVE="$D/corpus370_hosted_native_teacher_pilot/41915393652-1"
V10="$D/corpus370_v14_parent_v10"
V11="$D/corpus370_v14_parent_v11"
V13="$D/corpus370_v14_parent_v13b"
VM=gdrive:corpus370_preparation_v1
rm -f "$RUNNER_TEMP/v14-hosted-process.log"
pip install --disable-pip-version-check numpy pytest >/dev/null
python3 -m py_compile tools/corpus/corpus370_native_raster_join_plan_v14.py tools/corpus/court_corpus370_mechanical_raster_join_v14.py tools/corpus/run_corpus370_native_mechanical_raster_private_v14.py tools/truth_foundry/blender_corpus370_native_mechanical_raster_join_v14.py
PYTHONPATH=. pytest -q tests/corpus/test_corpus370_native_mechanical_raster_join_v14.py
mkdir -p "$RUNNER_TEMP/priv" "$RUNNER_TEMP/cache" "$RAYS" "$JOIN" "$GROUP" "$NATIVE" "$V10/$SUBJECT_ORDINAL" "$V11/$SUBJECT_ORDINAL" "$V13/$SUBJECT_ORDINAL"
rclone copy "$VM/hosted_portability_v1/private_v5_shards" "$RUNNER_TEMP/priv" --checksum --transfers 2 --checkers 2 --retries 3
python3 -u tools/corpus/hosted_v5_private_shards_v1.py hydrate --manifest "$RUNNER_TEMP/priv/PRIVATE_V5_SHARD_MANIFEST.json" --expected-manifest-sha256 cb9ffea96b22bb4500b367afe9965d98fa52b99f51dbe4faf5fac8aabc1ae172 --private-dir "$RUNNER_TEMP/priv" --object-cache "$RUNNER_TEMP/cache" >"$OUT/HYDRATE_V14_LOG.txt" 2>&1
test "$(sha256sum "$V5"|cut -d' ' -f1)" = 8ab2ebf526bbf91d78117580b68ed3784e53bcbafdbd326f012e143c924c0ac4
test "$(sha256sum "$ROSTER"|cut -d' ' -f1)" = 925dac90ecbf949375aa3ef5d2ed2f40788ae7dcf5669a02b708ada8d9435355
rclone copy "$VM/hosted_native_visibility_pair_join_v1/41916724594-1" "$JOIN" --checksum --retries 3
rclone copy "$VM/hosted_allview_geometry_groups_v3/41986038051-1" "$GROUP" --checksum --retries 3
rclone copy "$VM/hosted_native_temporal_pilot_v1/41915393652-1" "$NATIVE" --checksum --retries 3
rclone copyto "$VM/hosted_source_locus_rays_all352/41883539256-1/SOURCE_LOCUS_GEOMETRIC_RAYS_PILOT_INDEX_V1.json" "$RAYS/SOURCE_LOCUS_GEOMETRIC_RAYS_PILOT_INDEX_V1.json" --checksum --retries 3
SID="$(SUBJECT_ORDINAL="$SUBJECT_ORDINAL" JOIN_LOCAL="$JOIN" python3 - <<'PY'
import os,json
from pathlib import Path
d=json.loads((Path(os.environ["JOIN_LOCAL"])/"CORPUS370_NATIVE_VISIBILITY_PAIR_JOIN_V1.json").read_text())
print(sorted(d["subjects"])[int(os.environ["SUBJECT_ORDINAL"])])
PY
)"
test -n "$SID"
mkdir -p "$RAYS/subjects/$SID"
rclone copyto "$VM/hosted_source_locus_rays_all352/41883539256-1/subjects/$SID/SOURCE_LOCUS_GEOMETRIC_RAYS_V1.json" "$RAYS/subjects/$SID/SOURCE_LOCUS_GEOMETRIC_RAYS_V1.json" --checksum --retries 3
rclone copyto "$VM/hosted_heldout_source_branch_probe_private_v11/42024346219-1/$SUBJECT_ORDINAL/PRIVATE_HELDOUT_BRANCH_PROBE_PLAN_V11.json" "$V11/$SUBJECT_ORDINAL/PRIVATE_HELDOUT_BRANCH_PROBE_PLAN_V11.json" --checksum --retries 3
rclone copyto "$VM/hosted_native_edit_equivalence_private_v10/42017257514-1/$SUBJECT_ORDINAL/PRIVATE_NATIVE_EDIT_EQUIVALENCE_COURT_V10.json" "$V10/$SUBJECT_ORDINAL/PRIVATE_NATIVE_EDIT_EQUIVALENCE_COURT_V10.json" --checksum --retries 3
for name in PRIVATE_HELDOUT_NATIVE_EDIT_BATCH_V13B.json PRIVATE_ORIGINAL_ANON_ACTUATOR_MANIFEST_V13B.json PRIVATE_ORIGINAL_ANON_ACTUATOR_MANIFEST_V13B.npz; do
  rclone copyto "$VM/hosted_native_carrier_private_v13b/42040199925-1/$SUBJECT_ORDINAL/$name" "$V13/$SUBJECT_ORDINAL/$name" --checksum --retries 3
done
mkdir -p "$RUNNER_TEMP/blender45"
B=https://download.blender.org/release/Blender4.5
curl -fL --retry 4 --max-time 300 "$B/blender-4.5.14-linux-x64.tar.xz" -o "$RUNNER_TEMP/blender45/package.tar.xz"
curl -fL --retry 4 --max-time 120 "$B/blender-4.5.14.sha256" -o "$RUNNER_TEMP/blender45/checksums"
P="$RUNNER_TEMP/blender45/package.tar.xz"
P="$P" CHECKS="$RUNNER_TEMP/blender45/checksums" python3 - <<'PY'
import os,hashlib,re
from pathlib import Path
lines=[x for x in Path(os.environ["CHECKS"]).read_text().splitlines() if "blender-4.5.14-linux-x64.tar.xz" in x]
assert len(lines)==1
h=re.findall(r"[0-9a-fA-F]{64}",lines[0]);assert len(h)==1
with open(os.environ["P"],"rb") as f:assert hashlib.file_digest(f,"sha256").hexdigest()==h[0].lower()
PY
tar -xJf "$P" -C "$RUNNER_TEMP/blender45"
BLENDER="$RUNNER_TEMP/blender45/blender-4.5.14-linux-x64/blender"
test "$(sha256sum "$BLENDER"|cut -d' ' -f1)" = 050c02562f81fe80ba616a80198fa02d381e60f8b61b8d39add881f4bca0d7d8
python3 -u tools/corpus/run_corpus370_native_mechanical_raster_private_v14.py \
 --extracted-source-root "$D/corpus370_preparation_v1/source_bundle/extracted/5c92e77ae07b215fa4234d221faf27659bc0c47a54d2711d6e49db3d820b8fa9" \
 --blender "$BLENDER" --v5-index "$V5" --exact-roster "$ROSTER" \
 --geometric-ray-index "$RAYS/SOURCE_LOCUS_GEOMETRIC_RAYS_PILOT_INDEX_V1.json" \
 --original-join-root "$JOIN" --allview-groups-root "$GROUP" \
 --native-teacher-root "$NATIVE" --v11-private-root "$V11" --v10-private-root "$V10" --v13b-private-root "$V13" \
 --public-v9 canonical/CORPUS370_NATIVE_ANIMATED_SOURCE_RASTER_REAL_RESULT_V9.json \
 --public-v10 canonical/CORPUS370_NATIVE_EDIT_RESPONSE_EQUIVALENCE_REAL_RESULT_V10.json \
 --public-v11 canonical/CORPUS370_HELDOUT_GEOMETRIC_SOURCE_PROBE_PLAN_REAL_RESULT_V11.json \
 --public-v12 canonical/CORPUS370_HELDOUT_NATIVE_GEOMETRIC_LAYER_REAL_RESULT_V12.json \
 --public-v13b canonical/CORPUS370_NATIVE_MIXED_CARRIER_MECHANICAL_REAL_RESULT_V13B.json \
 --public-v9-sha256 decae1f3b38574bf9d69d72665d6f6cc3ec85c252688176a63617e70f7e227b0 \
 --public-v10-sha256 "$(sha256sum canonical/CORPUS370_NATIVE_EDIT_RESPONSE_EQUIVALENCE_REAL_RESULT_V10.json|cut -d' ' -f1)" \
 --public-v11-sha256 "$(sha256sum canonical/CORPUS370_HELDOUT_GEOMETRIC_SOURCE_PROBE_PLAN_REAL_RESULT_V11.json|cut -d' ' -f1)" \
 --public-v12-sha256 "$(sha256sum canonical/CORPUS370_HELDOUT_NATIVE_GEOMETRIC_LAYER_REAL_RESULT_V12.json|cut -d' ' -f1)" \
 --public-v13b-sha256 "$(sha256sum canonical/CORPUS370_NATIVE_MIXED_CARRIER_MECHANICAL_REAL_RESULT_V13B.json|cut -d' ' -f1)" \
 --output "$OUT" --subject-ordinal "$SUBJECT_ORDINAL" >"$OUT/V14_RENDER_LOG.txt" 2>&1
OUT="$OUT" python3 - <<'PY'
import os,json
from pathlib import Path
d=json.loads((Path(os.environ["OUT"])/"PRIVATE_MECHANICAL_RGBA_JOIN_BATCH_V14.json").read_text())
assert d["schema"]=="RealSaS.Corpus370MechanicalRasterJointPrivateBatch.v14"
assert d["status"]=="MEASURED_REAL_SAME_CAUSAL_NATIVE_CONTROL_RGBA_V14__OWNER_UNKNOWN"
assert d["training_authorized"] is False and d["graphics_owner_authorized"] is False
print("V14_SOURCE_CAUSAL_RGBA_RESEARCH_DONE_NO_OWNERSHIP",d["real_rgba_blender_render_arms"],d["independent_court_status"])
PY
