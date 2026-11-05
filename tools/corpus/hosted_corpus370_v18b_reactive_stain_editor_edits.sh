#!/usr/bin/env bash
set -euo pipefail
umask 077
test -n "$SUBJECT"
test -n "$SUBJECT_ORDINAL"
test -n "$DRIVE_CONFIG_B64"
D=/home/monster/realsas_authority
sudo install -d -o "$USER" -g "$(id -gn)" -m 0700 "$D"
R="$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT"
OUT="$D/corpus370_reactive_edit_v18b/$R/$SUBJECT_ORDINAL"
REMOTE="gdrive:corpus370_preparation_v1/hosted_reactive_edit_v18b/$SUBJECT/$R"
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
  PRIVATE_REAL_NATIVE_MECHANICAL_RGBA_MANIFEST_V14.json \
  PRIVATE_REAL_NATIVE_MECHANICAL_RGBA_MANIFEST_V14.npz \
  PRIVATE_INDEPENDENT_GEOMETRIC_MECHANICAL_RGBA_COURT_V14.json \
  PRIVATE_MECHANICAL_RGBA_JOIN_BATCH_V14.json
do
  rclone copyto "$VM/hosted_native_mechanical_rgba_private_v14/$V14_RUN/$SUBJECT_ORDINAL/$name" "$OLD/$name" --checksum --retries 3
  test -s "$OLD/$name"
done

# Frozen V16 actual simultaneous source-native AB poses and complete original
# source alpha/RGBA frames. These are ONLY copied, never rerendered or guessed.
V16_RUN=42083242419-1
DUAL="$D/corpus370_v16_pose_ray_parent/$V16_RUN/$SUBJECT"
mkdir -p "$DUAL"
for name in \
  PRIVATE_REAL_DUAL_SOURCE_CONTROL_RENDER_V16.json \
  PRIVATE_REAL_DUAL_SOURCE_CONTROL_RENDER_V16.npz \
  PRIVATE_INDEPENDENT_DUAL_NATIVE_CONTROL_RGBA_COURT_V16.json
do
  rclone copyto "$VM/hosted_native_dual_control_v16/$SUBJECT/$V16_RUN/$name" "$DUAL/$name" --checksum --retries 3
  test -s "$DUAL/$name"
done


# Original V17 privately sealed pose-aware ray stacks and independently
# re-evaluated court. No ray/redraw cache is edited or substituted.
V17_RUN=42088058238-1
PRIOR="$D/corpus370_v17b_original_pose_ray_parent/$V17_RUN/$SUBJECT"
mkdir -p "$PRIOR"
for name in \
  PRIVATE_V17_SOURCE_POSED_RAY_CAUSAL_RGBA_PROPOSAL.json \
  PRIVATE_V17_INDEPENDENT_POSED_RAY_RGBA_COURT.json
do
  rclone copyto "$VM/hosted_posed_source_ray_rgba_join_v17/$SUBJECT/$V17_RUN/$name" "$PRIOR/$name" --checksum --retries 3
  test -s "$PRIOR/$name"
done



# Use a specific already SHA-courted original source V18a stain job. This is
# NOT a general latest or dynamic-retrained layer search.
V18A_RUN=42095639620-1
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

# Exact original renderer binary and downloaded SHA-file. No new source
# shapes are generated or inferred. The original authored character is used.
mkdir -p "$RUNNER_TEMP/v18b-blender"
BLENDER_URL=https://download.blender.org/release/Blender4.5
curl -fL --retry 4 --max-time 300 "$BLENDER_URL/blender-4.5.14-linux-x64.tar.xz" \
  -o "$RUNNER_TEMP/v18b-blender/package.tar.xz"
curl -fL --retry 4 --max-time 120 "$BLENDER_URL/blender-4.5.14.sha256" \
  -o "$RUNNER_TEMP/v18b-blender/checksums"
P="$RUNNER_TEMP/v18b-blender/package.tar.xz" CHECKS="$RUNNER_TEMP/v18b-blender/checksums" python3 - <<'PY'
import hashlib,os,re
from pathlib import Path
rows=[x for x in Path(os.environ["CHECKS"]).read_text().splitlines()
      if "blender-4.5.14-linux-x64.tar.xz" in x]
assert len(rows)==1
hexes=re.findall(r"[0-9a-fA-F]{64}",rows[0])
assert len(hexes)==1
with open(os.environ["P"],"rb") as f:
    assert hashlib.file_digest(f,"sha256").hexdigest()==hexes[0].lower()
PY
tar -xJf "$RUNNER_TEMP/v18b-blender/package.tar.xz" -C "$RUNNER_TEMP/v18b-blender"
BLENDER="$RUNNER_TEMP/v18b-blender/blender-4.5.14-linux-x64/blender"
test "$(sha256sum "$BLENDER" | cut -d' ' -f1)" = 050c02562f81fe80ba616a80198fa02d381e60f8b61b8d39add881f4bca0d7d8

"$BLENDER" --background --factory-startup --python-exit-code 5 \
  --python tools/truth_foundry/blender_corpus370_v18b_reactive_stain_heldout_edits.py \
  -- \
  --job "$OLD/PRIVATE_NATIVE_MECHANICAL_RGBA_JOIN_JOB_V14.json" \
  --stain-report "$STAIN/PRIVATE_REACTIVE_STAIN_REPORT_V18A.json" \
  --stain-npz "$STAIN/PRIVATE_REACTIVE_STAIN_CANDIDATE_V18A.npz" \
  --stain-court "$STAIN/PRIVATE_INDEPENDENT_REACTIVE_STAIN_COURT_V18A.json" \
  --v14-original-pixels "$OLD/PRIVATE_REAL_NATIVE_MECHANICAL_RGBA_MANIFEST_V14.npz" \
  --output "$OUT/PRIVATE_ORIGINAL_V18B_HELDOUT_STAIN_EDITOR_RENDER.json" \
  > "$OUT/PRIVATE_V18B_REAL_BLENDER_EDIT_LOG.txt" 2>&1

python3 -u tools/corpus/court_corpus370_v18b_reactive_stain_editor_edits.py \
  --render-manifest "$OUT/PRIVATE_ORIGINAL_V18B_HELDOUT_STAIN_EDITOR_RENDER.json" \
  --edited-npz "$OUT/PRIVATE_ORIGINAL_V18B_HELDOUT_STAIN_EDITOR_RENDER.npz" \
  --stain-report "$STAIN/PRIVATE_REACTIVE_STAIN_REPORT_V18A.json" \
  --stain-npz "$STAIN/PRIVATE_REACTIVE_STAIN_CANDIDATE_V18A.npz" \
  --stain-court "$STAIN/PRIVATE_INDEPENDENT_REACTIVE_STAIN_COURT_V18A.json" \
  --v14-job "$OLD/PRIVATE_NATIVE_MECHANICAL_RGBA_JOIN_JOB_V14.json" \
  --v14-original-pixels "$OLD/PRIVATE_REAL_NATIVE_MECHANICAL_RGBA_MANIFEST_V14.npz" \
  --output "$OUT/PRIVATE_INDEPENDENT_ORIGINAL_STAIN_EDITOR_COURT_V18B.json"

OUT="$OUT" python3 - <<'PY'
import json,os
from pathlib import Path
d=json.loads((Path(os.environ["OUT"])/"PRIVATE_INDEPENDENT_ORIGINAL_STAIN_EDITOR_COURT_V18B.json").read_text())
assert d["status"]=="REAL_ORIGINAL_HELDOUT_POSE_STAIN_GROUP_EDITS_MEASURED__AUTHOR_DRAWABLE_UNKNOWN"
assert d["heldout_original_source_authored_pose_ordinal"]==2
assert len(d["original_camera_views"])==4
assert d["precommitted_heldout_source_target_count"]==6
assert d["frozen_candidate_similarity_thresholds"]==[.045,.1,.18]
assert d["actual_real_blender_source_render_arms_per_camera"]==52
assert d["actual_heldout_source_edit_cases"]==72
assert d["genuine_independent_edit_collateral_measured"] is True
assert d["single_face_proxy_is_not_author_layer"] is True
assert d["artist_same_drawable_positive_pairs"]==d["artist_different_drawable_negative_pairs"]==0
assert all(d[k] is False for k in (
    "graphics_owner_authorized","full_alpha_caa_qualified",
    "iris_training_authorized","tessa_training_authorized",
    "materialization_authorized","training_authorized"))
print("V18B_REAL_ORIGINAL_STAIN_HIDDEN_POSE_EDIT_REPLAY_COURT_PASS__ARTIST_OWNER_UNKNOWN",
      d["original_source_id"],d["per_candidate_heldout_edit_collateral"],flush=True)
PY

# All source art/render/source-face identity remains in private Drive. Verify
# uploaded receipt byte-for-byte; success without independent proof is invalid.
rclone copy "$OUT" "$REMOTE" --checksum --transfers 2 --checkers 2 --retries 3
LOCAL_SHA="$(sha256sum "$OUT/PRIVATE_INDEPENDENT_ORIGINAL_STAIN_EDITOR_COURT_V18B.json" | cut -d' ' -f1)"
verified=0
for attempt in 1 2 3 4 5 6 7 8; do
  if rclone cat "$REMOTE/PRIVATE_INDEPENDENT_ORIGINAL_STAIN_EDITOR_COURT_V18B.json" > "$RUNNER_TEMP/v18b-court-reopened.json" 2>/dev/null; then
    if test "$LOCAL_SHA" = "$(sha256sum "$RUNNER_TEMP/v18b-court-reopened.json" | cut -d' ' -f1)"; then
      verified=1
      break
    fi
  fi
  sleep 5
done
test "$verified" -eq 1
echo "V18B_REAL_ORIGINAL_STAIN_HELDOUT_EDITOR_COURT_REMOTE_SHA=$LOCAL_SHA"
