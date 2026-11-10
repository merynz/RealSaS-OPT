#!/usr/bin/env python3
"""No live Drive or private files in CI; fail-closed synthetic API contract tests."""
from __future__ import annotations
import tempfile,pathlib,csv,json,hashlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import drive_bulk_master_v25 as bulk

class DummyReader(bulk.DriveReader):
    def __init__(self):pass
    def list_pages(self,q,*args,**kwargs):
        if 'mimeType' in q:
            return iter([{'name':'objaverse_animated_originals_'+'a'*64,'id':'folder_one'},
                         {'name':'other_random_folder','id':'other'}])
        return iter([{'name':'normalized.npz','id':'npz_one','size':'256','parents':['folder_one']},
                     {'name':'source_structure.json','id':'json_one','size':'80','parents':['folder_one']}])

def test_transport_fails_closed():
    d=DummyReader()
    f=d.folder_index('variants_folder')
    assert len(f)==1 and list(f)==['objaverse_animated_originals_'+'a'*64]
    c=d.child_index(f)
    assert sorted(c['folder_one'])==['normalized.npz','source_structure.json']
    class BadReader(DummyReader):
        def list_pages(self,q,*args,**kwargs):
            return iter([{'name':'normalized.npz','id':'one','parents':['folder_one']},
                         {'name':'normalized.npz','id':'two','parents':['folder_one']}])
    try:BadReader().child_index(f)
    except bulk.CourtBlocker as exc:assert 'DUPLICATE' in str(exc)
    else:raise AssertionError('duplicate accepted')
    with tempfile.TemporaryDirectory() as tmp:
        p=pathlib.Path(tmp)
        db=bulk.database(p/'check.sqlite')
        db.execute("INSERT INTO attempts(source_name,status,updated_at) VALUES('asset','COURT_TRANSPORT_BLOCKED','2026-01-01')")
        db.commit()
        assert db.execute('SELECT status FROM attempts').fetchone()[0]=='COURT_TRANSPORT_BLOCKED'
        db.close()
        with (p/'test.json').open('w') as f:json.dump({'ok':True},f)
        assert len(bulk.sha256(p/'test.json'))==64
        assert not bulk.SOURCE_RE.fullmatch('objaverse_animated_originals_'+'a'*63+'z')
        assert bulk.SOURCE_RE.fullmatch('objaverse_animated_originals_'+'a'*64)
    print('PASS synthetic Drive transport, duplicate guard and resumable DB')

if __name__=='__main__':test_transport_fails_closed()