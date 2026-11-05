#!/usr/bin/env bash
set -euo pipefail
umask 077
test -n "$SUBJECT"
test -n "$SUBJECT_ORDINAL"
test -n "$DRIVE_CONFIG_B64"
D=/home/monster/realsas_authority
sudo install -d -o "$USER" -g "$(id -gn)" -m 0700 "$D"
R="$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT"
OUT="$D/corpus370_original_subpixel_attribution_v18j/$R/$SUBJECT_ORDINAL"
REMOTE="gdrive:corpus370_preparation_v1/hosted_source_subpixel_attribution_v18j/$SUBJECT/$R"
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
rm -f "$RUNNER_TEMP/v17b-hosted-process.log"
pip install --disable-pip-version-check numpy pytest pillow >/dev/null
python3 -m py_compile tools/truth_foundry/blender_corpus370_v18b_reactive_stain_heldout_edits.py tools/corpus/court_corpus370_v18b_reactive_stain_editor_edits.py
PYTHONPATH=. pytest -q tests/corpus/test_corpus370_v18b_reactive_stain_editor_edits.py
mkdir -p "$RUNNER_TEMP/priv" "$RUNNER_TEMP/cache" "$RAYS" "$JOIN" "$GROUP" "$NATIVE" "$V10/$SUBJECT_ORDINAL" "$V11/$SUBJECT_ORDINAL" "$V13/$SUBJECT_ORDINAL"
rclone copy "$VM/hosted_portability_v1/private_v5_shards" "$RUNNER_TEMP/priv" --checksum --transfers 2 --checkers 2 --retries 3
python3 -u tools/corpus/hosted_v5_private_shards_v1.py hydrate --manifest "$RUNNER_TEMP/priv/PRIVATE_V5_SHARD_MANIFEST.json" --expected-manifest-sha256 cb9ffea96b22bb4500b367afe9965d98fa52b99f51dbe4faf5fac8aabc1ae172 --private-dir "$RUNNER_TEMP/priv" --object-cache "$RUNNER_TEMP/cache" >"$OUT/HYDRATE_V18B_LOG.txt" 2>&1
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
# Frozen original V14 SOURCE job, 48 signed single-control/source-soup effects.
V14_RUN=42075029264-1
OLD="$D/corpus370_native_rgba_v14/$V14_RUN/$SUBJECT_ORDINAL"
mkdir -p "$OLD"
for name in \
  PRIVATE_NATIVE_MECHANICAL_RGBA_JOIN_JOB_V14.json \
  PRIVATE_GEOMETRY_ONLY_NATIVE_CONTROL_SELECTION_V14.json \
  PRIVATE_REAL_NATIVE_MECHANICAL_RGBA_MANIFEST_V14.json \
  PRIVATE_REAL_NATIVE_MECHANICAL_RGBA_MANIFEST_V14.npz \
  PRIVATE_INDEPENDENT_GEOMETRIC_MECHANICAL_RGBA_COURT_V14.json \
  PRIVATE_MECHANICAL_RGBA_JOIN_BATCH_V14.json
do
  rclone copyto "$VM/hosted_native_mechanical_rgba_private_v14/$V14_RUN/$SUBJECT_ORDINAL/$name" "$OLD/$name" --checksum --retries 3
  test -s "$OLD/$name"
done

# Independent remote courts for all three V18a original sources were SHA-readback
# checked in run 42097722762. These pins prevent an old/partial court from
# silently qualifying the V18b Blender group edit experiment.
case "$SUBJECT" in
  CAT) EXPECTED_V18A_COURT_SHA=2801bc94c58349f7561071f2c36d9a30b8d13b3b476aa4430e164f2e5de02c7b ;;
  TRICERATOPS) EXPECTED_V18A_COURT_SHA=bd7ac5e561af6d4a4bbab8aac555951e5c7e6b97f589a700ef3691edc89ce5c0 ;;
  COWFISH) EXPECTED_V18A_COURT_SHA=16e20721019911d5f52181f8d427794aff352ebf5b187568c7b57058850b221e ;;
  *) echo "Unexpected original source $SUBJECT" >&2; exit 2 ;;
esac
V18A_RUN=42097722762-1
STAIN="$D/corpus370_reactive_stain_v18a_parent/$V18A_RUN/$SUBJECT"
mkdir -p "$STAIN"
for name in \
  PRIVATE_REACTIVE_STAIN_REPORT_V18A.json \
  PRIVATE_REACTIVE_STAIN_CANDIDATE_V18A.npz \
  PRIVATE_INDEPENDENT_REACTIVE_STAIN_COURT_V18A.json
do
  rclone copyto "$VM/hosted_reactive_stain_v18a/$SUBJECT/$V18A_RUN/$name" "$STAIN/$name" --checksum --retries 3
  test -s "$STAIN/$name"
done
test "$(sha256sum "$STAIN/PRIVATE_INDEPENDENT_REACTIVE_STAIN_COURT_V18A.json" | cut -d' ' -f1)" = "$EXPECTED_V18A_COURT_SHA"
echo "V18B_PARENT_V18A_ORIGINAL_SOURCE_COURT_SHA_MATCHED=$SUBJECT/$EXPECTED_V18A_COURT_SHA"

# Only real independently sealed V18i court and V18h original RGBA
# qualify a retrospective V18j source-ray diagnosis.
case "$SUBJECT" in
  CAT) EXPECTED_V18I_SHA=763c4bfebe78cc04dc37cd48aa1258b2b2ddffd0b222a158fc8fedf7d0d575d0 ;;
  TRICERATOPS) EXPECTED_V18I_SHA=36b1bb856c24fa204dbb1cd3cef0a0cbbdfb2cc7fd5b8b31f59cd22286883bd2 ;;
  *) echo "V18j original source is not in the fixed Cat/Triceratops court" >&2; exit 2 ;;
esac

H="$OUT/v18h_source_original"
mkdir -p "$H"
for ext in json npz; do
  rclone copyto "$VM/hosted_emissive_compositing_v18h/$SUBJECT/42120479356-1/PRIVATE_ORIGINAL_V18H_EMISSIVE_COMPLEMENT_RENDER.$ext" "$H/source.$ext" --checksum --retries 3
done
rclone copyto "$VM/hosted_emissive_alpha_coverage_v18i/$SUBJECT/42123720590-1/PRIVATE_V18I_SOURCE_ALPHA_COVERAGE_VS_ORDER.json" "$H/v18i-source-court.json" --checksum --retries 3
test "$(sha256sum "$H/v18i-source-court.json"|cut -d' ' -f1)" = "$EXPECTED_V18I_SHA"

OLD="$D/corpus370_native_rgba_v14/42075029264-1/$SUBJECT_ORDINAL"
V14_JOB="$OLD/PRIVATE_NATIVE_MECHANICAL_RGBA_JOIN_JOB_V14.json"
test -s "$V14_JOB"
eval "$(V14_JOB="$V14_JOB" python3 - <<'PY'
import json,os,shlex
p=json.load(open(os.environ["V14_JOB"]))
for key,var in (("native_teacher","TEACHER"),("camera","CAMERA"),("rest_geometry","REST")):
  print(var+"="+shlex.quote(p[key]["path"]))
PY
)"
test -s "$TEACHER" && test -s "$CAMERA" && test -s "$REST"
V14_JOB="$V14_JOB" TEACHER="$TEACHER" CAMERA="$CAMERA" REST="$REST" python3 - <<'PY'
import json,os,hashlib
p=json.load(open(os.environ["V14_JOB"]))
for key in ("native_teacher","camera","rest_geometry"):
  filename=os.environ[{"native_teacher":"TEACHER","camera":"CAMERA","rest_geometry":"REST"}[key]]
  with open(filename,"rb") as f:
    assert hashlib.file_digest(f,"sha256").hexdigest()==p[key]["sha256"],key
assert p["training_authorized"] is False
print("V18J_ORIGINAL_V14_CAMERA_TEACHER_REST_BYTE_SHA_VERIFIED",flush=True)
PY

STAIN="$D/corpus370_reactive_stain_v18a_parent/42097722762-1/$SUBJECT"
STAIN_NPZ="$STAIN/PRIVATE_REACTIVE_STAIN_CANDIDATE_V18A.npz"
test -s "$STAIN_NPZ"
PYTHONPATH=. python3 -m py_compile \
  tools/corpus/measure_corpus370_v18j_source_subpixel_order_attribution.py \
  tools/corpus/court_corpus370_v18j_source_subpixel_order_attribution.py
PYTHONPATH=. python3 -m pytest -q tests/corpus/test_corpus370_v18j_source_subpixel_order_attribution.py

PROPOSAL="$OUT/PRIVATE_V18J_SOURCE_SUBPIXEL_RAY_ATTRIBUTION.json"
COURT="$OUT/PRIVATE_INDEPENDENT_V18J_SOURCE_SUBPIXEL_COURT.json"
python3 -u tools/corpus/measure_corpus370_v18j_source_subpixel_order_attribution.py \
  --v14-job "$V14_JOB" --teacher "$TEACHER" --camera "$CAMERA" --rest "$REST" \
  --stain-npz "$STAIN_NPZ" \
  --v18h-manifest "$H/source.json" --v18h-rgba "$H/source.npz" \
  --v18i-court "$H/v18i-source-court.json" \
  --expected-v18i-sha "$EXPECTED_V18I_SHA" --output "$PROPOSAL"

python3 -u tools/corpus/court_corpus370_v18j_source_subpixel_order_attribution.py \
  --proposal "$PROPOSAL" --v14-job "$V14_JOB" \
  --teacher "$TEACHER" --camera "$CAMERA" --rest "$REST" \
  --stain-npz "$STAIN_NPZ" \
  --v18h-manifest "$H/source.json" --v18h-rgba "$H/source.npz" \
  --v18i-court "$H/v18i-source-court.json" \
  --expected-v18i-sha "$EXPECTED_V18I_SHA" --output "$COURT"

COURT="$COURT" python3 - <<'PY'
import json,os
from pathlib import Path
d=json.loads(Path(os.environ["COURT"]).read_text())
assert d["status"]=="V18J_ORIGINAL_4X4_SOURCE_RAY_STACKS_INDEPENDENTLY_REMEASURED__OWNER_UNKNOWN"
assert d["complete_private_source_ray_reread"] is True
assert d["image_error_pixel_selection_is_retrospective_not_heldout"] is True
assert d["unknown_is_not_a_negative_edge"] is True
assert d["source_artist_same_different_part_labels"]=="UNKNOWN"
assert all(d[k] is False for k in (
  "graphics_owner_authorized","iris_training_authorized","tessa_training_authorized",
  "materialization_authorized","training_authorized"))
print("V18J_ORIGINAL_SUBPIXEL_SOURCE_RAYS_INDEPENDENTLY_REMEASURED_NO_AUTHORITY",flush=True)
PY

# Upload the smallest private evidence set, retaining raw source rays privately.
rclone copyto "$PROPOSAL" "$REMOTE/PRIVATE_V18J_SOURCE_SUBPIXEL_RAY_ATTRIBUTION.json" --checksum --retries 3
rclone copyto "$COURT" "$REMOTE/PRIVATE_INDEPENDENT_V18J_SOURCE_SUBPIXEL_COURT.json" --checksum --retries 3
rclone cat "$REMOTE/PRIVATE_INDEPENDENT_V18J_SOURCE_SUBPIXEL_COURT.json" > "$RUNNER_TEMP/v18j-court-remote.json"
test "$(sha256sum "$COURT"|cut -d' ' -f1)" = "$(sha256sum "$RUNNER_TEMP/v18j-court-remote.json"|cut -d' ' -f1)"
echo "V18J_SOURCE_SUBPIXEL_INDEPENDENT_COURT_REMOTE_SHA=$(sha256sum "$COURT"|cut -d' ' -f1)"
