from __future__ import annotations
import argparse,hashlib,inspect,json
from pathlib import Path
import dataset,losses,matcher,train_mini_v2 as train,finalize_mini_v2 as finalize,run_mini_extractability_v1 as runner
EXPECTED_MEMBERSHIP_CANONICAL_SHA256='4e223c799cf479a210716a86701ab96459673fe21852142e7cc7bd3a9d30e055';EXPECTED_PANEL_DIGEST='366b5fffb1ff93c1c7bbad0ac4746c4f2675a633ec01745c026cecb2b7820961'
def canonical_json_sha(path):
 obj=json.load(open(path,encoding='utf-8'));return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest(),obj
def main():
 ap=argparse.ArgumentParser(description='Frozen mini contract semantic preflight');ap.add_argument('--membership',default=str(Path(__file__).with_name('MINI_EXTRACTABILITY_MEMBERSHIP_V1.json')));a=ap.parse_args();msh,mem=canonical_json_sha(a.membership)
 if msh!=EXPECTED_MEMBERSHIP_CANONICAL_SHA256:raise RuntimeError(f'membership canonical SHA drift {msh}')
 assert mem['source_panel_asset_id_digest']==EXPECTED_PANEL_DIGEST and mem['sealed_splits_opened'] is False
 assert {k:len(mem['roles'][k]) for k in ('FIT_TRAIN','FIT_SELECT','TUNE_FINAL')}=={'FIT_TRAIN':128,'FIT_SELECT':32,'TUNE_FINAL':26};sets=[set(mem['roles'][k]) for k in ('FIT_TRAIN','FIT_SELECT','TUNE_FINAL')];assert not(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
 ms=inspect.getsource(matcher.match_query);assert 'outputs["N"]' not in ms and "outputs['N']" not in ms;assert 'outputs["Z_coarse"]' in ms and 'outputs["P"]' in ms and 'outputs["Z_fine"]' in ms
 cs=inspect.getsource(losses.coarse_correspondence_loss);fs=inspect.getsource(losses.fine_local_loss);gs=inspect.getsource(losses.geometry_loss);ds=inspect.getsource(dataset.IRISV2Dataset.__getitem__);assert 'track_n_view' not in cs+fs+ds;assert 'batch["geom_n"]' in gs;assert '_uniform_track_subsample' in fs and '[:max_tracks_per_pair]' not in fs
 base={'P_p95':.004,'Zc_top8':.93,'oracle_Zf_top1_p95_native_px':12.,'family_Zc_top8_p10':.80,'family_P_p95_p90':.005,'min_style_Zc_top8':.90,'max_style_P_p95':.005,'N_p95_deg':10.};score=train.selection_score(base);assert train.selection_score(dict(base,N_p95_deg=170.))==score;assert train.selection_score(dict(base,P_p95=.006))>score
 assert train.EPOCHS==16 and train.CANDIDATE_EPOCHS==(4,8,12,16) and train.WARMUP_EPOCHS==3 and train.GRAD_ACCUM==4 and train.TRACK_SAMPLES_TRAIN==128 and train.TRACK_SAMPLES_EVAL==512
 ts=inspect.getsource(train);tm=inspect.getsource(train.main);assert '--tune-cache' not in ts and "split='TUNE'" not in tm and "build_report(model,ck" not in tm;assert 'CHECKPOINT_SELECTION_FROZEN.json' in ts and "'tune_seen_during_selection':False" in ts and "'tune_stage_cache_exists_during_selection':False" in ts
 rs=inspect.getsource(runner.main);i_fit=rs.index("'--splits','FIT'");i_train=rs.index("'train_mini_v2.py'");i_freeze=rs.index("'CHECKPOINT_SELECTION_FROZEN.json'");i_tune=rs.index("'--splits','TUNE'");i_final=rs.index("'finalize_mini_v2.py'");assert i_fit<i_train<i_freeze<i_tune<i_final;assert "if tune_stage.exists() or tune_cache.exists():raise RuntimeError('TUNE stage/cache exists before checkpoint freeze')" in rs
 fz=inspect.getsource(finalize.main);assert 'CHECKPOINT_SELECTION_FROZEN.json' in fz and '--tune-cache' in fz and 'build_report' in fz
 print(json.dumps({'status':'PASS','membership_frozen':True,'membership_canonical_sha256':msh,'roles':{'FIT_TRAIN':128,'FIT_SELECT':32,'TUNE_FINAL':26},'matcher_N_forbidden':True,'N_local_geometry_only':True,'N_checkpoint_selection_forbidden':True,'fine_track_prefix_bias_removed':True,'checkpoint_selection_frozen':True,'tune_cli_absent_from_training_phase':True,'tune_stage_after_checkpoint_freeze_enforced':True,'sealed_splits_opened':False,'optimizer_steps':0},indent=2))
if __name__=='__main__':main()
