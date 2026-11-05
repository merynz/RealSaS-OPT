#!/usr/bin/env bash
set -euo pipefail
umask 077
test -n "$SUBJECT"
test -n "$SUBJECT_ORDINAL"
test -n "$DRIVE_CONFIG_B64"
D=/home/monster/realsas_authority
sudo install -d -o "$USER" -g "$(id -gn)" -m 0700 "$D"
R="$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT"
OUT="$D/corpus370_posed_ray_v17/$R/$SUBJECT_ORDINAL"
REMOTE="gdrive:corpus370_preparation_v1/hosted_posed_source_ray_rgba_join_v17/$SUBJECT/$R"
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
rm -f "$RUNNER_TEMP/v17-hosted-process.log"
pip install --disable-pip-version-check numpy pytest >/dev/null
python3 -m py_compile tools/corpus/measure_corpus370_v17_posed_source_ray_to_causal_rgba_v1.py tools/corpus/court_corpus370_v17_posed_source_ray_to_rgba_join_v1.py
PYTHONPATH=. pytest -q tests/corpus/test_corpus370_v17_posed_source_ray_to_rgba_join_v1.py
mkdir -p "$RUNNER_TEMP/priv" "$RUNNER_TEMP/cache" "$RAYS" "$JOIN" "$GROUP" "$NATIVE" "$V10/$SUBJECT_ORDINAL" "$V11/$SUBJECT_ORDINAL" "$V13/$SUBJECT_ORDINAL"
rclone copy "$VM/hosted_portability_v1/private_v5_shards" "$RUNNER_TEMP/priv" --checksum --transfers 2 --checkers 2 --retries 3
python3 -u tools/corpus/hosted_v5_private_shards_v1.py hydrate --manifest "$RUNNER_TEMP/priv/PRIVATE_V5_SHARD_MANIFEST.json" --expected-manifest-sha256 cb9ffea96b22bb4500b367afe9965d98fa52b99f51dbe4faf5fac8aabc1ae172 --private-dir "$RUNNER_TEMP/priv" --object-cache "$RUNNER_TEMP/cache" >"$OUT/HYDRATE_V17_LOG.txt" 2>&1
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

python3 -u tools/corpus/measure_corpus370_v17_posed_source_ray_to_causal_rgba_v1.py \
  --source "$SUBJECT" \
  --v14-public canonical/CORPUS370_V14_REAL_NATIVE_CONTROL_RGBA_CAUSAL_REPAIR_RESULT_V1.json \
  --v16-public canonical/CORPUS370_V16_REAL_DUAL_NATIVE_CONTROL_INTERACTION_RESULT_V1.json \
  --v14-dir "$OLD" \
  --v14-job "$OLD/PRIVATE_NATIVE_MECHANICAL_RGBA_JOIN_JOB_V14.json" \
  --v16-manifest "$DUAL/PRIVATE_REAL_DUAL_SOURCE_CONTROL_RENDER_V16.json" \
  --v16-court "$DUAL/PRIVATE_INDEPENDENT_DUAL_NATIVE_CONTROL_RGBA_COURT_V16.json" \
  --v16-pixels "$DUAL/PRIVATE_REAL_DUAL_SOURCE_CONTROL_RENDER_V16.npz" \
  --output "$OUT/PRIVATE_V17_SOURCE_POSED_RAY_CAUSAL_RGBA_PROPOSAL.json"

python3 -u tools/corpus/court_corpus370_v17_posed_source_ray_to_rgba_join_v1.py \
  --source "$SUBJECT" \
  --proposal "$OUT/PRIVATE_V17_SOURCE_POSED_RAY_CAUSAL_RGBA_PROPOSAL.json" \
  --v14-public canonical/CORPUS370_V14_REAL_NATIVE_CONTROL_RGBA_CAUSAL_REPAIR_RESULT_V1.json \
  --v16-public canonical/CORPUS370_V16_REAL_DUAL_NATIVE_CONTROL_INTERACTION_RESULT_V1.json \
  --v14-dir "$OLD" \
  --v14-job "$OLD/PRIVATE_NATIVE_MECHANICAL_RGBA_JOIN_JOB_V14.json" \
  --v16-manifest "$DUAL/PRIVATE_REAL_DUAL_SOURCE_CONTROL_RENDER_V16.json" \
  --v16-court "$DUAL/PRIVATE_INDEPENDENT_DUAL_NATIVE_CONTROL_RGBA_COURT_V16.json" \
  --v16-pixels "$DUAL/PRIVATE_REAL_DUAL_SOURCE_CONTROL_RENDER_V16.npz" \
  --output "$OUT/PRIVATE_V17_INDEPENDENT_POSED_RAY_RGBA_COURT.json"

OUT="$OUT" python3 - <<'PY'
import os,json
from pathlib import Path
d=json.loads((Path(os.environ["OUT"])/"PRIVATE_V17_INDEPENDENT_POSED_RAY_RGBA_COURT.json").read_text())
assert d["status"]=="V17_REAL_POSED_SOURCE_GEOMETRIC_RAY_AND_RGBA_LINK_RECALCULATED__ARTIST_OWNER_UNKNOWN"
q=d["recalculated_stats"]
assert q["original_source_pose_view_pair_cases"]==48
assert q["geometry_selected_rays"]>0 and q["complete_untied_nonempty_rays"]>0
assert q["geometry_selected_rays"]==q["complete_untied_nonempty_rays"]+q["nohit_or_incomplete_or_tied_rays"]
assert not d["actual_alpha_ordered_layer_render_proved"]
assert d["author_same_drawable_positive_pairs"]==d["author_different_drawable_negative_pairs"]==0
assert all(not d[k] for k in ("graphics_owner_authorized","training_authorized","materialization_authorized","iris_training_authorized","tessa_training_authorized"))
print("V17_POSED_SOURCE_REAL_RAY_AND_RGBA_COURTED__OWNER_UNKNOWN",
    d["source_label_private_teacher_only"],q,flush=True)
PY

# Original pixel/face/bary/rig bytes and per-source proposal/court remain in
# private Drive only, with exact remote SHA readback.
rclone copy "$OUT" "$REMOTE" --checksum --transfers 2 --checkers 2 --retries 3
LOCAL_SHA="$(sha256sum "$OUT/PRIVATE_V17_INDEPENDENT_POSED_RAY_RGBA_COURT.json" | cut -d' ' -f1)"
verified=0
for attempt in 1 2 3 4 5 6 7 8; do
  if rclone cat "$REMOTE/PRIVATE_V17_INDEPENDENT_POSED_RAY_RGBA_COURT.json" > "$RUNNER_TEMP/v17-remote-court.json" 2>/dev/null; then
    if test "$LOCAL_SHA" = "$(sha256sum "$RUNNER_TEMP/v17-remote-court.json" | cut -d' ' -f1)"; then
      verified=1
      break
    fi
  fi
  sleep 5
done
test "$verified" -eq 1
echo "V17_REAL_ORIGINAL_SOURCE_POSED_RAY_COURT_REMOTE_SHA=$LOCAL_SHA"
