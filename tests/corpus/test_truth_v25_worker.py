import csv, pathlib, tempfile, sqlite3, json, subprocess, sys, zipfile
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2] / 'tools' / 'corpus' / 'truth_foundry' / 'evidence_v25'))
import corpus_court_worker as c

def test_synthetic_scope_guard():
    with tempfile.TemporaryDirectory() as temp:
        root=pathlib.Path(temp)
        f=root/'test.obj';f.write_text('v 0 0 0\nv 0 1 0\nv 1 0 0\nf 1 2 3\n')
        status,ob=c.scene_container_observation(f)
        assert status=='RAW_SOURCE_OBJ_VERTEX_FACE_RECORDS_ONLY'
        assert ob['vertex_records']==3 and ob['polygon_records']==1
        f.write_text('v 0 0 0\nf 1 2 3\n')
        try:c.scene_container_observation(f)
        except ValueError as e:assert 'OBJ_FACE_REFERENCE' in str(e)
        else:raise AssertionError('invalid reference accepted')
        z=root/'corrupt.npz';z.write_bytes(b'not-zip')
        try:c.scene_container_observation(z)
        except zipfile.BadZipFile:pass
        else:raise AssertionError('bad NPZ accepted')
        db=sqlite3.connect(root/'x.db');c.initialize(db)
        try:db.execute("INSERT INTO source_court (source_key,provider,scope,status,training_authorized) VALUES ('x','p','s','ok',1)")
        except sqlite3.IntegrityError:pass
        else:raise AssertionError('training admission without gates')
        db.close()
if __name__=='__main__':
    test_synthetic_scope_guard();print('PASS: synthetic worker guards')