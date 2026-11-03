"""Republish an existing immutable compute package without repeating measurement."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from models.materialization_contract_v1 import require,sha256
from tools.notebooks.build_corpus370_compute_worker_v1 import build as notebook


def write(path,value):
    staged=path.with_suffix('.partial');staged.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n');staged.replace(path)


def prepare(config,output):
    resume=config['existing_capsule'];source=Path(resume['directory'])
    receipt_path=source/'COMPUTE_CAPSULE_RECEIPT.json';seal_path=source/'COMPUTE_CAPSULE.json'
    receipt=json.loads(receipt_path.read_text());seal=json.loads(seal_path.read_text())
    archive=source/'CORPUS370_COMPUTE_CAPSULE_V1.zip'
    require(receipt['code_sha']==seal['code_sha']==resume['code_sha'],'PUBLISH_CODE_DRIFT')
    require(receipt['archive_sha256']==resume['archive_sha256']==sha256(archive),'PUBLISH_ARCHIVE_DRIFT')
    require(receipt['archive_size_bytes']==resume['archive_size_bytes']==archive.stat().st_size,'PUBLISH_SIZE_DRIFT')
    require(receipt['seal_sha256']==resume['seal_sha256']==sha256(seal_path),'PUBLISH_SEAL_DRIFT')
    require(seal['preparation_sha256']==config['preparation']['sha256'],'PUBLISH_PREPARATION_DRIFT')
    require(seal['candidate_subject_count']==receipt['candidate_subject_count']==370
            and seal['worker_subject_count']==receipt['worker_subject_count']==294
            and seal['excluded_subject_count']==76,'PUBLISH_COHORT_DRIFT')
    require(seal['training_authorized'] is False and seal['qualification_minted'] is False
            and receipt['training_authorized'] is False,'PUBLISH_AUTHORITY_DRIFT')
    folder=config['drive_folder']+'/'+seal['code_sha']
    value=notebook(seal_path,receipt_path,folder)
    output.mkdir(parents=True,exist_ok=True)
    # Keep the old export and its cancelled publication evidence unchanged.
    for name in ('COMPUTE_CAPSULE.json','COMPUTE_CAPSULE_RECEIPT.json','CORPUS370_COMPUTE_CAPSULE_V1.zip'):
        destination=output/name
        if destination.exists():require(sha256(destination)==sha256(source/name),'PUBLISH_STAGING_DRIFT')
        else:os.link(source/name,destination)
    (output/'Corpus370_Compute_Worker_Run_All_v1.ipynb').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    (output/'DRIVE_FOLDER.txt').write_text(folder+'\n')
    write(output/'PUBLISH_PROGRESS.json',{'status':'VERIFIED_EXISTING_PACKAGE','archive_sha256':receipt['archive_sha256'],
        'archive_size_bytes':receipt['archive_size_bytes'],'code_sha':receipt['code_sha'],
        'measurement_repeated':False,'qualification_minted':False,'training_authorized':False})
    return receipt


def command(*args):
    return ['rclone',*args,'--contimeout','15s','--timeout','90s','--retries','2',
            '--low-level-retries','3','--retries-sleep','10s','--stats','15s','--stats-one-line',
            '--stats-log-level','NOTICE','--log-level','INFO','--checkers','1','--transfers','1']


def publish(output):
    folder=(output/'DRIVE_FOLDER.txt').read_text().strip();remote='gdrive:'+folder
    started=time.monotonic();progress=json.loads((output/'PUBLISH_PROGRESS.json').read_text())
    def phase(status,**extra):
        progress.update(status=status,elapsed_seconds=time.monotonic()-started,**extra)
        write(output/'PUBLISH_PROGRESS.json',progress);print(json.dumps(progress),flush=True)
    try:
        phase('CHECKING_DRIVE_ACCESS')
        subprocess.run(command('lsf',remote,'--files-only','--max-depth','1'),check=True,timeout=120)
        # These are inspectable before the large upload; the readiness receipt is last.
        for name in ('COMPUTE_CAPSULE.json','Corpus370_Compute_Worker_Run_All_v1.ipynb'):
            subprocess.run(command('copyto',str(output/name),remote+'/'+name,'--checksum'),check=True,timeout=180)
        phase('UPLOADING_IMMUTABLE_ARCHIVE')
        name='CORPUS370_COMPUTE_CAPSULE_V1.zip'
        subprocess.run(command('copyto',str(output/name),remote+'/'+name,'--checksum','--drive-chunk-size','32M'),
                       check=True,timeout=30*60)
        phase('VERIFYING_REMOTE_ARCHIVE')
        # Drive's native MD5 check is in addition to the pinned SHA256 checked by Colab.
        subprocess.run(command('check',str(output),remote,'--include','/'+name,'--one-way'),check=True,timeout=180)
        name='COMPUTE_CAPSULE_RECEIPT.json'
        subprocess.run(command('copyto',str(output/name),remote+'/'+name,'--checksum'),check=True,timeout=180)
        phase('PUBLISHED_AND_REMOTE_CHECKED')
    except (subprocess.SubprocessError,OSError) as error:
        phase('PUBLICATION_FAILED__PACKAGE_PRESERVED',error_type=type(error).__name__)
        raise
    return progress


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('operation',choices=('prepare','publish'))
    parser.add_argument('--config',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    value=prepare(json.loads(args.config.read_text()),args.output) if args.operation=='prepare' else publish(args.output)
    print(json.dumps(value,indent=2))


if __name__=='__main__':main()
