#!/usr/bin/env python3
"""Authenticated Drive -> SHA-verified staged Master normalized derivatives.

One provider-locked transport stage, NOT original-source parity, character truth,
semantic supervision, rights, training admission, or a substitute for numeric_court.
Requires the user's existing Google application-default credentials or a short-lived
read-only OAuth bearer token in REALSAS_DRIVE_TOKEN; credentials are never logged.

Example (in an authenticated Drive/Colab worker):
  python drive_bulk_master_v25.py --v17 TruthFoundry_V17_Bulk_16397_Evidence.zip \
      --v20 TruthFoundry_V20_190_NPZ_Evidence.zip \
      --v21 TruthFoundry_V21_266_NPZ_Evidence.zip \
      --v22 TruthFoundry_V22_334_NPZ_Evidence.zip \
      --v23 TruthFoundry_V23_366_NPZ_Evidence.zip \
      --v24 TruthFoundry_V24_386_Source_Court_Evidence.zip \
      --v25-receipt NEW_MASTER726_NPZ_COURT_V25.csv.gz \
      --variants-id <AUTHORIZED_VARIANTS_FOLDER_ID> --work /mnt/data/transfer \
      --max-new 80
"""
from __future__ import annotations
import argparse,csv,gzip,hashlib,io,json,os,pathlib,random,re,sqlite3,time,zipfile
from urllib.parse import quote
import requests

DRIVE='https://www.googleapis.com/drive/v3/files'
SOURCE_RE=re.compile(r'^objaverse_animated_originals_[0-9a-f]{64}$')
RETRIES=5

class CourtBlocker(RuntimeError):
    pass

class DailyQuotaBlocker(CourtBlocker):
    pass

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for c in iter(lambda:f.read(4<<20),b''):h.update(c)
    return h.hexdigest()

def readzip(archive,member):
    with zipfile.ZipFile(archive) as z:
        with gzip.open(z.open(member),'rt',encoding='utf-8',newline='') as f:
            yield from csv.DictReader(f)

def readcsv(path):
    with gzip.open(path,'rt',newline='') if str(path).endswith('.gz') else open(path,'rt',newline='') as f:
        yield from csv.DictReader(f)

def select_unprocessed(args):
    allow={r['relative_folder_under_variants']:r for r in readzip(args.v17,'MASTER726_SAFE_RECOVERY_PLAN_V17.csv.gz')}
    if len(allow)!=726 or not all(SOURCE_RE.fullmatch(k) for k in allow):
        raise CourtBlocker('V17_ALLOWLIST_INVALID')
    if any(r['creator_original_byte_parity']!='False' or r['rights_training_admission']!='False' for r in allow.values()):
        raise CourtBlocker('ILLEGAL_ALLOWLIST_AUTHORITY')
    prev=set()
    sources=[
      (args.v20,'MASTER726_NPZ_NUMERIC_COURT_V20.csv.gz',190),
      (args.v21,'NEW_MASTER726_NPZ_COURT_V21.csv.gz',76),
      (args.v22,'NEW_MASTER726_NPZ_COURT_V22.csv.gz',68),
      (args.v23,'NEW_MASTER726_NPZ_COURT_V23.csv.gz',32),
      (args.v24,'NEW_MASTER726_NPZ_COURT_V24.csv.gz',20),
    ]
    for archive,member,count in sources:
        rows=list(readzip(archive,member))
        if len(rows)!=count:raise CourtBlocker('PREDECESSOR_CARDINALITY_DRIFT:'+member)
        for r in rows:
            k=r['folder_name']
            if k not in allow or k in prev or r.get('rights_training_admission',r.get('training_admission'))!='False':
                raise CourtBlocker('PREDECESSOR_CONFLICT_OR_AUTHORITY:'+k)
            prev.add(k)
    rows=list(readcsv(args.v25_receipt))
    if len(rows)!=36:raise CourtBlocker('V25_CARDINALITY_DRIFT')
    for r in rows:
        k=r['folder_name']
        if k not in allow or k in prev or r.get('rights_training_admission')!='False':
            raise CourtBlocker('V25_OVERLAP_OR_AUTHORITY:'+k)
        prev.add(k)
    if len(prev)!=422:raise CourtBlocker('PREVIOUS_SET_SIZE')
    return {k:allow[k] for k in sorted(allow.keys()-prev)}

class DriveReader:
    """Tiny Drive v3 transport; no service-account key contents on disk."""
    def __init__(self, session=None, token_fn=None):
        self.http=session or requests.Session()
        self.credentials=None
        self.env_token=os.environ.get('REALSAS_DRIVE_TOKEN','')
        if token_fn is not None:self.token_fn=token_fn
        elif self.env_token:self.token_fn=lambda:self.env_token
        else:
            try:
                import google.auth
                from google.auth.transport.requests import Request
                cred,_=google.auth.default(scopes=['https://www.googleapis.com/auth/drive.readonly'])
                self.credentials=cred;self.google_request=Request()
                self.token_fn=self._refreshable_token
            except Exception as exc:raise CourtBlocker('DRIVE_OAUTH_REQUIRED: configure ADC or REALSAS_DRIVE_TOKEN') from exc

    def _refreshable_token(self):
        if not self.credentials.valid:
            self.credentials.refresh(self.google_request)
        if not self.credentials.token:raise CourtBlocker('GOOGLE_OAUTH_TOKEN_MISSING')
        return self.credentials.token

    def request(self, method,url,*,params=None,headers=None,stream=False):
        for i in range(RETRIES+1):
            hdr={'Authorization':'Bearer '+self.token_fn()}
            hdr.update(headers or {})
            try:
                r=self.http.request(method,url,params=params,headers=hdr,stream=stream,timeout=(15,150))
            except (requests.ConnectionError,requests.Timeout) as exc:
                if i==RETRIES:raise CourtBlocker('DRIVE_NETWORK_RETRIES_EXHAUSTED') from exc
                time.sleep(min(30,2**i));continue
            if r.status_code in (200,206):return r
            body=r.text[:900] if not stream else r.content[:900].decode('utf8','replace')
            try:
                reasons=[x.get('reason','') for x in r.json().get('error',{}).get('errors',[])]
            except (ValueError,KeyError):reasons=[]
            reason=';'.join(reasons)
            r.close()
            if r.status_code in (401,403) and ('dailyLimitExceeded' in reason or 'downloadQuotaExceeded' in reason or 'storageQuotaExceeded' in reason or 'quotaExceeded' in reason):
                raise DailyQuotaBlocker('HARD_DRIVE_QUOTA:'+reason)
            if r.status_code in (429,500,502,503,504) or (r.status_code==403 and ('rateLimitExceeded' in reason or 'userRateLimitExceeded' in reason)):
                if i==RETRIES:raise CourtBlocker('DRIVE_TRANSIENT_RETRIES_EXHAUSTED:'+reason)
                time.sleep(min(60,2**i+random.uniform(0,0.5)));continue
            raise CourtBlocker(f'DRIVE_HTTP_{r.status_code}:{reason or body[:120]}')
        raise AssertionError('unreachable')

    def list_pages(self,q,fields='nextPageToken,files(id,name,mimeType,size,parents,md5Checksum)',page_size=1000):
        token=None
        while True:
            params={'q':q,'fields':fields,'pageSize':page_size,'supportsAllDrives':'true',
                    'includeItemsFromAllDrives':'true'}
            if token:params['pageToken']=token
            resp=self.request('GET',DRIVE,params=params)
            try:body=resp.json()
            finally:resp.close()
            for f in body.get('files',[]):yield f
            token=body.get('nextPageToken')
            if not token:break

    def folder_index(self,parent):
        arr={}
        for f in self.list_pages(f"'{parent}' in parents and trashed = false and mimeType = 'application/vnd.google-apps.folder'"):
            name=f.get('name','')
            if not SOURCE_RE.fullmatch(name):continue
            if name in arr:raise CourtBlocker('DUPLICATE_PROVIDER_FOLDER_NAME:'+name)
            arr[name]=f['id']
        return arr

    def child_index(self,folders,chunk=12):
        """OR parent queries reduce N+1; isolate a rejected query by falling back to singles."""
        out={p:{} for p in folders.values()}
        ids=list(folders.values())
        for i in range(0,len(ids),chunk):
            chunk_ids=ids[i:i+chunk]
            q='('+' or '.join(f"'{p}' in parents" for p in chunk_ids)+') and trashed = false'
            try:items=list(self.list_pages(q))
            except CourtBlocker:
                if len(chunk_ids)==1:raise
                items=[]
                for p in chunk_ids:items.extend(self.list_pages(f"'{p}' in parents and trashed = false"))
            for item in items:
                if item.get('name') not in ('normalized.npz','source_structure.json'):continue
                for parent in item.get('parents',[]):
                    if parent in out:
                        if item['name'] in out[parent]:raise CourtBlocker('AMBIGUOUS_FILE_DUPLICATE:'+parent)
                        out[parent][item['name']]=item
        return out

    def download(self,file_id,target,expected_size,max_bytes):
        if expected_size<=0 or expected_size>max_bytes:raise CourtBlocker('FILE_SIZE_OUTSIDE_LIMIT')
        target.parent.mkdir(parents=True,exist_ok=True)
        part=target.with_suffix(target.suffix+'.part')
        if target.is_file() and target.stat().st_size==expected_size:
            return sha256(target)
        if target.exists():target.unlink()
        # Never trust a stale partial range after a server-side file change.
        if part.exists():part.unlink()
        url=DRIVE+'/'+quote(file_id,safe='')
        resp=self.request('GET',url,params={'alt':'media'},stream=True)
        n=0;h=hashlib.sha256()
        try:
            with part.open('wb') as f:
                for chunk in resp.iter_content(chunk_size=2<<20):
                    if not chunk:continue
                    n+=len(chunk)
                    if n>expected_size or n>max_bytes:raise CourtBlocker('DRIVE_BODY_EXCEEDS_DECLARED_SIZE')
                    h.update(chunk);f.write(chunk)
                f.flush();os.fsync(f.fileno())
        except Exception:
            if part.exists():part.unlink()
            raise
        finally:resp.close()
        if n!=expected_size:
            part.unlink(missing_ok=True);raise CourtBlocker('DRIVE_BODY_SIZE_MISMATCH')
        part.replace(target)
        return h.hexdigest()

def database(path):
    db=sqlite3.connect(path)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('''CREATE TABLE IF NOT EXISTS attempts (
       source_name TEXT PRIMARY KEY, status TEXT NOT NULL, npz_sha256 TEXT, structure_sha256 TEXT,
       npz_file_id TEXT, structure_file_id TEXT, npz_size INTEGER, structure_size INTEGER,
       error TEXT DEFAULT '', updated_at TEXT NOT NULL)''')
    db.commit();return db

def run(args,drive=None):
    work=pathlib.Path(args.work);work.mkdir(parents=True,exist_ok=True)
    allow=select_unprocessed(args)
    db=database(work/'transport.sqlite')
    drive=drive or DriveReader()
    folder_ids=drive.folder_index(args.variants_id)
    missing=set(allow)-set(folder_ids)
    # Fail closed if the historic allowlist cannot be resolved to actual parent folder IDs.
    folder_selected={k:folder_ids[k] for k in allow if k in folder_ids}
    if not folder_selected:raise CourtBlocker('NO_ELIGIBLE_DRIVE_FOLDERS')
    child=drive.child_index(folder_selected)
    ready=[];blocked=[];hard_stop=None
    for source in allow:
        fid=folder_ids.get(source)
        if not fid:
            blocked.append((source,'FOLDER_NOT_FOUND'));continue
        pair=child[fid];npz=pair.get('normalized.npz');st=pair.get('source_structure.json')
        if not npz or not st:
            blocked.append((source,'MISSING_DERIVATIVE_PAIR'));continue
        if npz.get('mimeType')=='application/vnd.google-apps.folder' or st.get('mimeType')=='application/vnd.google-apps.folder':
            blocked.append((source,'NONFILE_ARTIFACT'));continue
        try:ns,ss=int(npz['size']),int(st['size'])
        except (KeyError,ValueError):blocked.append((source,'NO_DECLARED_FILE_SIZE'));continue
        ready.append((source,npz,st,ns,ss))
    successes=[];seen=set()
    for source,npz,st,ns,ss in ready:
        old=db.execute('SELECT status,npz_sha256,structure_sha256,npz_file_id,structure_file_id FROM attempts WHERE source_name=?',(source,)).fetchone()
        npz_path=work/'sources'/(source+'.npz');struct_path=work/'structures'/(source+'.json')
        if old and old[0]=='TRANSFER_SHA_OBSERVED' and old[3]==npz['id'] and old[4]==st['id'] and npz_path.is_file() and struct_path.is_file() and npz_path.stat().st_size==ns and struct_path.stat().st_size==ss and sha256(npz_path)==old[1] and sha256(struct_path)==old[2]:
            hashes=(old[1],old[2])
        else:
            if args.max_new and len(successes)>=args.max_new:break
            try:
                psha=drive.download(npz['id'],npz_path,ns,args.max_file_bytes)
                ssha=drive.download(st['id'],struct_path,ss,3<<20)
                with zipfile.ZipFile(npz_path) as z:
                    if z.testzip() is not None:raise CourtBlocker('NPZ_ZIP_CRC_FAILURE')
                with struct_path.open('r',encoding='utf8') as f: data=json.load(f)
                if not isinstance(data.get('mesh_objects'),list) or not isinstance(data.get('bone_names'),list):
                    raise CourtBlocker('SOURCE_STRUCTURE_FIELDS_INVALID')
                hashes=(psha,ssha)
                db.execute('INSERT OR REPLACE INTO attempts VALUES (?,?,?,?,?,?,?,?,?,?)',
                    (source,'TRANSFER_SHA_OBSERVED',psha,ssha,npz['id'],st['id'],ns,ss,'',time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
                db.commit()
            except DailyQuotaBlocker as exc:
                hard_stop=str(exc);blocked.append((source,hard_stop));break
            except (CourtBlocker,ValueError,zipfile.BadZipFile,requests.RequestException) as exc:
                why=f'{type(exc).__name__}:{exc}'[:200]
                db.execute('INSERT OR REPLACE INTO attempts VALUES (?,?,?,?,?,?,?,?,?,?)',
                    (source,'COURT_TRANSPORT_BLOCKED','','',npz['id'],st['id'],ns,ss,why,time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
                db.commit();blocked.append((source,why));continue
        successes.append(dict(source_key=source,npz_id=npz['id'],npz_size=ns,structure_id=st['id'],structure_size=ss,npz_sha256=hashes[0],structure_sha256=hashes[1]))
        seen.add(source)
    fieldnames=['source_key','npz_id','npz_size','structure_id','structure_size','npz_sha256','structure_sha256']
    with (work/'transfer_manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames);w.writeheader();w.writerows(successes)
    with (work/'transport_blockers.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['source_key','blocked_reason']);w.writerows(blocked)
    summary={'schema':'RealSaS.V25.Master726BulkTransfer.v1','eligible_remaining':304,
        'folders_resolved':len(folder_selected),'files_received_sha256':len(successes)*2,
        'sources_ready_for_typed_numeric_court':len(successes),'holdout_or_unallowlisted_processed':0,
        'missing_folders':len(missing),'other_blocked':len(blocked)-len(missing),
        'hard_quota_stop':hard_stop,'rights_admissions':0,'training_admissions':0,
        'note':'Transfer and CRC only. Run typed numeric court before updating any facet.'}
    (work/'transport_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2,sort_keys=True))
    return summary

def cli():
    p=argparse.ArgumentParser()
    for ver in (17,20,21,22,23,24):p.add_argument('--v'+str(ver),type=pathlib.Path,required=True)
    p.add_argument('--v25-receipt',type=pathlib.Path,required=True)
    p.add_argument('--variants-id',required=True)
    p.add_argument('--work',type=pathlib.Path,required=True)
    p.add_argument('--max-new',type=int,default=80)
    p.add_argument('--max-file-bytes',type=int,default=400<<20)
    a=p.parse_args()
    run(a)
if __name__=='__main__':cli()