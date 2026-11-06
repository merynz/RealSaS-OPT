"""V18n original source status migration must be the only allowed mutation."""
import json
from pathlib import Path

import pytest

from tools.corpus.recover_corpus370_v18n_original_material_status_receipt import (
    recover,sha,WRONG,CORRECT,RENDER_SCHEMA,RECEIPT_SCHEMA
)


def raw(tmp_path):
    source=tmp_path/"source.json"
    rgba=tmp_path/"source.npz"
    rgba.write_bytes(b"sealed-source-original-cycles-rgba")
    manifest={
      "schema":RENDER_SCHEMA,"status":WRONG,
      "new_source_actual_edited_rgba_npz_sha256":sha(rgba),
      "source_edit_arms":["SOURCE_NATIVE","SOUP_FULL","SOUP_ISOLATE","SOUP_REMOVE"],
      "source_material_policy":"ORIGINAL_SEALED_SOURCE_MATERIAL_GRAPH_AND_UV_REPLAY",
      "source_appearance_material_policy":"SEALED_SOURCE_APPEARANCE_POLICY_REPLAY",
      "heldout_source_camera_views":[0,2,4,6],
      "original_source_cycles_sample_indices":[0,4,8,12,16,20,24,28],
      "source_original_v18n_parent_v18m_run":"42135271057-1",
      "source_original_v18n_parent_v18l_run":"42131579613-1",
      "full_original_caa_draw_order_qualified":False,
      "source_vs_original_v14_RGBA_reference_source_parity_verified":False,
      "original_source_sha256":"abc123",
      "graphics_owner_authorized":False,"iris_training_authorized":False,
      "tessa_training_authorized":False,"materialization_authorized":False,
      "training_authorized":False,
    }
    source.write_text(json.dumps(manifest,sort_keys=True)+"\n")
    return source,rgba,manifest


def test_only_status_literal_changes_and_original_bytes_preserved(tmp_path):
    source,rgba,original=raw(tmp_path)
    canonical=recover(source,rgba,sha(source))
    assert canonical["status"]==CORRECT
    assert set(canonical)==set(original)
    assert all(canonical[k]==original[k] for k in original if k!="status")
    assert canonical["new_source_actual_edited_rgba_npz_sha256"]==sha(rgba)
    assert not canonical["training_authorized"]
    assert RECEIPT_SCHEMA.endswith(".v18n")


@pytest.mark.parametrize("tamper",[
  "change_source_rgba","change_sha","fake_training_admit",
  "modify_group","altered_source_policy","source_status_already_changed",
  "stolen_new_camera","pretend_original_caa_parity",
])
def test_never_recover_when_other_scientific_claim_changed(tmp_path,tamper):
    source,rgba,original=raw(tmp_path)
    if tamper=="change_source_rgba":rgba.write_bytes(b"changed-pixels")
    elif tamper=="change_sha":original["new_source_actual_edited_rgba_npz_sha256"]="0"*64
    elif tamper=="fake_training_admit":original["training_authorized"]=True
    elif tamper=="modify_group":original["source_edit_arms"]=["SOURCE_NATIVE","SOUP_FULL"]
    elif tamper=="altered_source_policy":original["source_material_policy"]="RGB_EMISSION"
    elif tamper=="source_status_already_changed":original["status"]=CORRECT
    elif tamper=="stolen_new_camera":original["heldout_source_camera_views"]=[0,1,2,3]
    else:original["source_vs_original_v14_RGBA_reference_source_parity_verified"]=True
    source.write_text(json.dumps(original,sort_keys=True)+"\n")
    with pytest.raises(ValueError,match="V18N_RECOVERY_"):
        recover(source,rgba,sha(source))


def test_original_manifest_sha_required(tmp_path):
    source,rgba,_=raw(tmp_path)
    with pytest.raises(ValueError,match="V18N_RECOVERY_SOURCE_ORIGINAL_MANIFEST_SHA_MISMATCH"):
        recover(source,rgba,"f"*64)
