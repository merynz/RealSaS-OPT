#!/usr/bin/env bash
set -euo pipefail
umask 077
test -n "$SUBJECT"
test -n "$SUBJECT_ORDINAL"
test -n "$DRIVE_CONFIG_B64"
D=/home/monster/realsas_authority
sudo install -d -o "$USER" -g "$(id -gn)" -m 0700 "$D"
R="$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT"
OUT="$D/corpus370_controlled_emission_compositing_v18h/$R/$SUBJECT_ORDINAL"
REMOTE="gdrive:corpus370_preparation_v1/hosted_emissive_compositing_v18h/$SUBJECT/$R"
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

# Cat+Triceratops original Cycles transparent-bounce cause is independently
# measured and SHA-reopened. No moving "latest" reference or relaxed threshold.
case "$SUBJECT" in
  CAT) EXPECTED_BOUNCE_SHA=d31d987a6cf592263503bb96201be95872638822f62b19435d6d42e21125c693 ;;
  TRICERATOPS) EXPECTED_BOUNCE_SHA=d750452cd3db1f6552909aa7ba893533d266babb3d43d1eb9ea8ba6c2f0e2674 ;;
  *) echo "V18E two-source physical canary only" >&2; exit 2 ;;
esac
BOUNCE_RUN=42106976442-1
BOUNCE="$OUT/PRIVATE_INDEPENDENT_V18B_EMPTY_BOUNCE_COURT_V1.json"
rclone copyto "$VM/hosted_v18b_empty_bounce_witness_v1/$SUBJECT/$BOUNCE_RUN/PRIVATE_INDEPENDENT_V18B_EMPTY_BOUNCE_COURT_V1.json" "$BOUNCE" --checksum --retries 3
test "$(sha256sum "$BOUNCE" | cut -d' ' -f1)" = "$EXPECTED_BOUNCE_SHA"
BOUNCE="$BOUNCE" python3 - <<'PY'
import os,json
from pathlib import Path
p=json.loads(Path(os.environ["BOUNCE"]).read_text())
assert p["status"]=="ORIGINAL_EMPTY_SOUP_BOUNCE_WITNESS_INDEPENDENTLY_MEASURED__NO_ARTIST_AUTHORITY"
assert p["transparent_bounce_limit_hypothesis_supported"] is True
assert p["all_original_failed_views_repaired_by_increased_bounces"] is True
assert p["physically_hidden_empty_all_views"] is True
assert p["training_authorized"] is False
PY

# Original V18e physical face canary independent court receipts, both SHA-readback.
# This closes the mechanism-validation dependency before full 52-arm expansion.
case "$SUBJECT" in
  CAT) EXPECTED_V18E_SHA=72c80db1bd73b39f6bdd0cfc51789a7bf76de0b8c8f4c71f236d1d67193ea030 ;;
  TRICERATOPS) EXPECTED_V18E_SHA=fb1fb612c89cf3b79098ab84f800d71dc732087ad2af1558061ab9d03d026d4b ;;
  *) echo "Unexpected V18f source" >&2; exit 2 ;;
esac
V18E_RUN=42111807283-1
V18E_COURT="$OUT/PRIVATE_V18E_SOURCE_PHYSICAL_CANARY_PARENT_COURT.json"
rclone copyto "$VM/hosted_physical_edit_canary_v18e/$SUBJECT/$V18E_RUN/PRIVATE_INDEPENDENT_ORIGINAL_V18E_PHYSICAL_EDIT_CANARY_COURT.json" "$V18E_COURT" --checksum --retries 3
test "$(sha256sum "$V18E_COURT"|cut -d' ' -f1)" = "$EXPECTED_V18E_SHA"
PARENT="$V18E_COURT" python3 - <<'PY'
import os,json
from pathlib import Path
p=json.loads(Path(os.environ["PARENT"]).read_text())
assert p["status"]=="ORIGINAL_SOURCE_PHYSICAL_TRIANGLE_REMOVAL_CANARY_COURTED__OWNER_UNKNOWN"
assert p["source_physical_edit_render_parity_qualified"] is True
assert p["observable_blind_source_face_references"]>0
assert p["training_authorized"] is False
PY

# All V18h sources are descendants of exact immutable independently-courted
# original V18f physical edits and V18g lit-image compositing diagnostics.
case "$SUBJECT" in
  CAT)
    EXPECTED_V18F_SHA=9fba1f6f74ff5fd9a60ffb842f49ff8ba0a4169e2c8cf3ed504bec0374c26276
    EXPECTED_V18G_SHA=89e7fd6f71bee76e32937c5f1693ff0c177bc8eea26550a82156162df5f65f3d
    ;;
  TRICERATOPS)
    EXPECTED_V18F_SHA=6c578d92f8f7bc4d867632afd629d989b79f045f38a8edead4e75ab14b1b5f38
    EXPECTED_V18G_SHA=9f3677de0ff1243982a1cdd15f0f6760ac425872f391884c4ea37d0a86bbfd1d
    ;;
  *) echo "Unknown V18h source" >&2; exit 2 ;;
esac
V18F="$OUT/PRIVATE_V18F_INDEPENDENT_SOURCE_PARENT_COURT.json"
V18G="$OUT/PRIVATE_V18G_INDEPENDENT_COMPOSITING_PARENT_COURT.json"
rclone copyto "$VM/hosted_full_physical_edit_v18f/$SUBJECT/42114401233-1/PRIVATE_INDEPENDENT_ORIGINAL_V18F_FULL_PHYSICAL_EDIT_COURT.json" "$V18F" --checksum --retries 3
rclone copyto "$VM/hosted_original_physical_compositing_v18g/$SUBJECT/42116712393-1/PRIVATE_V18G_ORIGINAL_PHYSICAL_COMPOSITING_DIAGNOSIS.json" "$V18G" --checksum --retries 3
test "$(sha256sum "$V18F"|cut -d' ' -f1)" = "$EXPECTED_V18F_SHA"
test "$(sha256sum "$V18G"|cut -d' ' -f1)" = "$EXPECTED_V18G_SHA"
PYTHONPATH=. python3 -m py_compile \
 tools/truth_foundry/blender_corpus370_v18h_emissive_compositing_witness.py \
 tools/corpus/court_corpus370_v18h_emissive_compositing_witness.py
PYTHONPATH=. python3 -m pytest -q \
 tests/corpus/test_corpus370_v18h_emissive_compositing_witness.py
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
  --python tools/truth_foundry/blender_corpus370_v18h_emissive_compositing_witness.py \
  -- \
  --job "$OLD/PRIVATE_NATIVE_MECHANICAL_RGBA_JOIN_JOB_V14.json" \
  --stain-report "$STAIN/PRIVATE_REACTIVE_STAIN_REPORT_V18A.json" \
  --stain-npz "$STAIN/PRIVATE_REACTIVE_STAIN_CANDIDATE_V18A.npz" \
  --stain-court "$STAIN/PRIVATE_INDEPENDENT_REACTIVE_STAIN_COURT_V18A.json" \
  --v14-original-pixels "$OLD/PRIVATE_REAL_NATIVE_MECHANICAL_RGBA_MANIFEST_V14.npz" \
  --parent-v18f-court "$V18F" \
  --parent-v18g-court "$V18G" \
  --expected-v18f-sha "$EXPECTED_V18F_SHA" \
  --expected-v18g-sha "$EXPECTED_V18G_SHA" \
  --output "$OUT/PRIVATE_ORIGINAL_V18H_EMISSIVE_COMPLEMENT_RENDER.json" \
  > "$OUT/PRIVATE_V18H_EMISSIVE_BLENDER_LOG.txt" 2>&1

python3 -u tools/corpus/court_corpus370_v18h_emissive_compositing_witness.py \
  --render-manifest "$OUT/PRIVATE_ORIGINAL_V18H_EMISSIVE_COMPLEMENT_RENDER.json" \
  --render-npz "$OUT/PRIVATE_ORIGINAL_V18H_EMISSIVE_COMPLEMENT_RENDER.npz" \
  --stain-npz "$STAIN/PRIVATE_REACTIVE_STAIN_CANDIDATE_V18A.npz" \
  --stain-court "$STAIN/PRIVATE_INDEPENDENT_REACTIVE_STAIN_COURT_V18A.json" \
  --parent-v18f-court "$V18F" --parent-v18g-court "$V18G" \
  --expected-v18f-sha "$EXPECTED_V18F_SHA" \
  --expected-v18g-sha "$EXPECTED_V18G_SHA" \
  --output "$OUT/PRIVATE_INDEPENDENT_V18H_EMISSIVE_COMPOSITING_COURT.json"

OUT="$OUT" python3 - <<'PY'
import json,os
from pathlib import Path
d=json.loads((Path(os.environ["OUT"])/"PRIVATE_INDEPENDENT_V18H_EMISSIVE_COMPOSITING_COURT.json").read_text())
assert d["status"]=="ORIGINAL_SOURCE_EMISSIVE_COMPOSITING_CONTRAST_COURTED__NO_DRAWABLE_AUTHORITY"
assert len(d["per_camera_threshold"])==12
assert set(d["per_threshold"])=={"0.045","0.1","0.18"}
assert d["fixed_two_color_emission_only_diagnostic"] is True
assert d["source_graphic_appearance_truth_evaluated"] is False
assert all(d[x] is False for x in ("graphics_owner_authorized","iris_training_authorized",
    "tessa_training_authorized","materialization_authorized","training_authorized"))
print("V18H_ORIGINAL_EMISSIVE_COMPLEMENT_INDEPENDENT_COURT_PASS__NO_LAYER_TRUTH",flush=True)
PY
rclone copy "$OUT" "$REMOTE" --checksum --transfers 2 --checkers 2 --retries 3
LOCAL_SHA="$(sha256sum "$OUT/PRIVATE_INDEPENDENT_V18H_EMISSIVE_COMPOSITING_COURT.json"|cut -d' ' -f1)"
rclone cat "$REMOTE/PRIVATE_INDEPENDENT_V18H_EMISSIVE_COMPOSITING_COURT.json" > "$RUNNER_TEMP/v18h-court-readback.json"
test "$LOCAL_SHA" = "$(sha256sum "$RUNNER_TEMP/v18h-court-readback.json"|cut -d' ' -f1)"
echo "V18H_ORIGINAL_EMISSIVE_COMPOSITING_REMOTE_COURT_SHA=$LOCAL_SHA"
