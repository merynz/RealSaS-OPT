#!/usr/bin/env python3
"""PartVerse VLM-caption candidate triage, never anatomical/semantic truth.

This is a deterministic PRE-DOWNLOAD gate: source SHA and exact PartVerse
fileIdentifier are joined to the archive's UUID folder name; captions only
propose broad class and style, not verified morphology or part meaning.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlparse

HEX64 = re.compile(r'^[0-9a-f]{64}$')
HEX32 = re.compile(r'^[0-9a-f]{32}$')
RULES = {
 'ANIMAL_CREATURE':r'\b(?:dragon|dinosaur|monster|creature|beast|wolf|tiger|lion|horse|pony|fox|bear|bird|eagle|shark|octopus|snake|serpent|lizard|crocodile|turtle|insect|spider|arthropod|animal|quadruped|griffin|wyvern|demon|frog|bat|rabbit|penguin|deer|gorilla|elephant|rhino|hippo|cat|dog)\b',
 'ROBOT_MECHA':r'\b(?:robot|robotic|mecha|mech|android|automaton|cyborg|golem|droid|transformer|mechanical humanoid|steam automaton)\b',
 'HUMANOID_CHARACTER':r'\b(?:humanoid|warrior|knight|mage|wizard|sorcerer|orc|goblin|elf|woman|man|female figure|male figure|ninja|samurai|soldier|person|pirate|zombie|character|superhero|villain|skeleton warrior|jester|princess|prince|mermaid|dwarf|centaur|viking|elfin|avatar|fighter|adventurer|witch|orcish)\b',
 'GAMEPLAY_VEHICLE':r'\b(?:tank|tractor|vehicle|airplane|aircraft|fighter jet|helicopter|spaceship|spacecraft|car|truck|train|motorcycle|scooter|boat|submarine|rover|airship|biplane|ufo|armored personnel carrier|bulldozer|forklift)\b',
 'FURNITURE_ARCHITECTURE':r'\b(?:sofa|couch|chair|table|desk|armchair|stool|bed|wardrobe|cabinet|shelf|bookshelf|lampshade|cupboard|house|building|room|interior|street|door|castle|temple|room furniture|environment|scene|kitchen|living room|office|sconce|bookcase)\b',
 'EQUIPMENT_PROP':r'\b(?:sword|shield|helmet|gun|rifle|weapon|pistol|dagger|staff|armor|tool|guitar|bottle|cup|crate|barrel|vase|phone|telephone|bowl|mug)\b',
}
COMPILED = {k:re.compile(v, re.I) for k,v in RULES.items()}
STYLE = {
 'STYLIZED_CLAIM':re.compile(r'\b(?:stylized|stylised|cartoon|low[ -]?poly|cel[ -]?shad\w*|anime|hand[ -]?paint\w*|toon|chibi|voxel|pixel[ -]?art|game[ -]?ready|simplified|blocky|polyhedral|game[ -]?art)\b',re.I),
 'REALISM_RISK':re.compile(r'\b(?:photorealistic|photo[ -]?realistic|photogrammetry|3d scan|scanned|lifelike|highly realistic|realistic skin|hyperrealistic|hyper[ -]?realistic|lifelike human|realistic human)\b',re.I),
}
# weak text tags only. Object phrases override superficial animal/human terms in signs,
# ornaments, furniture, etc. This must never be presented as measured morphology.
DECORATION = re.compile(r'\b(?:cat[ -]?eared|animal[ -]?themed|animal[ -]?shaped|dragon[ -]?shaped|dragon[ -]?motif|figurine|sculpture|statue|plush|plushie|toy[ -]?shop|sign|logo|painting|poster|ornament|mural|fossil|taxidermy|dollhouse)\b',re.I)


def read_caption(raw):
    if not raw:
        return '', 'MISSING'
    try:
        cap = json.loads(raw)
    except (json.JSONDecodeError,TypeError):
        return '', 'UNPARSEABLE'
    if not isinstance(cap,list):
        return '', 'UNSUPPORTED_TYPE'
    cap=[str(x).strip() for x in cap if isinstance(x,str) and x.strip()]
    if not cap:
        return '', 'EMPTY'
    return cap[0], 'VLM_WEAK_UNREVIEWED'


def classify(caption):
    matches = [k for k,v in COMPILED.items() if v.search(caption)]
    decorator=bool(DECORATION.search(caption))
    # Strongly prioritize obvious furniture/scenery; never treat a named design
    # detail, animal-shaped sign, etc. as a living morphology without review.
    if 'FURNITURE_ARCHITECTURE' in matches:
        kind = 'FURNITURE_ARCHITECTURE'
    elif decorator and 'EQUIPMENT_PROP' in matches:
        kind = 'EQUIPMENT_PROP'
    elif decorator and 'ANIMAL_CREATURE' in matches:
        kind = 'DECORATIVE_ANIMAL_REVIEW'
    elif 'ROBOT_MECHA' in matches:
        kind = 'ROBOT_MECHA'
    elif 'ANIMAL_CREATURE' in matches:
        kind = 'ANIMAL_CREATURE'
    elif 'HUMANOID_CHARACTER' in matches:
        kind = 'HUMANOID_CHARACTER'
    elif 'GAMEPLAY_VEHICLE' in matches:
        kind = 'GAMEPLAY_VEHICLE'
    elif 'EQUIPMENT_PROP' in matches:
        kind = 'EQUIPMENT_PROP'
    else:
        kind='UNKNOWN_REVIEW'
    realism=bool(STYLE['REALISM_RISK'].search(caption))
    stylized=bool(STYLE['STYLIZED_CLAIM'].search(caption))
    style='CONFLICT_REVIEW' if realism and stylized else ('REALISM_RISK' if realism else ('STYLIZED_CLAIM' if stylized else 'STYLE_UNVERIFIED'))
    if kind in ('ANIMAL_CREATURE','ROBOT_MECHA','HUMANOID_CHARACTER'):
        tier='CHARACTER_PRIORITY' if not realism else 'REALISTIC_CHARACTER_DEFER'
    elif kind=='GAMEPLAY_VEHICLE':
        tier='GAMEPLAY_VEHICLE_SECONDARY' if not realism else 'REALISTIC_VEHICLE_DEFER'
    elif kind=='DECORATIVE_ANIMAL_REVIEW':
        tier='MORPHOLOGY_AMBIGUOUS_REVIEW'
    else:
        tier='NONCHARACTER_DEFER_OR_REVIEW'
    return kind,style,tier,matches,decorator


def source_uid(file_identifier):
    url=urlparse(file_identifier)
    if url.hostname not in ('sketchfab.com','www.sketchfab.com'):
        return ''
    parts=url.path.strip('/').split('/')
    if len(parts)<2 or parts[-2]!='3d-models':
        return ''
    uid=parts[-1].lower()
    return uid if HEX32.fullmatch(uid) else ''


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def run(metadata_path, output_dir, source_roster=None, max_selected=2500, provider_train=None, provider_val=None):
    output_dir.mkdir(parents=True,exist_ok=True)
    master_sha=set()
    if source_roster:
        with source_roster.open('r',newline='',encoding='utf-8-sig') as f:
            original=list(csv.DictReader(f))
            if len(original)!=4363:raise ValueError('FROZEN_ROSTER_NOT_4363')
            master_sha={x['source_sha256'].lower() for x in original if HEX64.fullmatch(x.get('source_sha256','').lower())}
    provider_train_uids=set()
    provider_val_uids=set()
    if provider_train and provider_val:
        for source, target in [(provider_train,provider_train_uids),(provider_val,provider_val_uids)]:
            with source.open('r',encoding='utf-8-sig',newline='') as f:
                reader=csv.DictReader(f)
                if 'part_id' not in (reader.fieldnames or []):raise ValueError('PARTVERSE_SPLIT_COLUMN_MISSING')
                for row in reader:
                    uid=row['part_id'].strip().lower()
                    if not re.fullmatch(r'[a-z0-9_-]{20,64}',uid):raise ValueError('PARTVERSE_SPLIT_UID_INVALID')
                    target.add(uid)
        if provider_train_uids & provider_val_uids:raise ValueError('PARTVERSE_SPLIT_OVERLAP')
    elif provider_train or provider_val:
        raise ValueError('BOTH_SPLIT_FILES_REQUIRED')
    items=[];seen_uids=set();invalid=Counter()
    with metadata_path.open('r',newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        assert {'sha256','file_identifier','aesthetic_score','captions'}<=set(reader.fieldnames or [])
        for i,r in enumerate(reader,1):
            orig_sha=r['sha256'].strip().lower();uid=source_uid(r['file_identifier'])
            if not uid or not HEX64.fullmatch(orig_sha):
                invalid['SOURCE_KEY_INVALID']+=1; continue
            if uid in seen_uids:
                invalid['DUPLICATE_PARTVERSE_UID']+=1; continue
            seen_uids.add(uid)
            caption,caption_status=read_caption(r['captions'])
            kind,style,tier,matches,decorator=classify(caption)
            try:aesthetic=float(r['aesthetic_score'])
            except (TypeError,ValueError):aesthetic=0.0
            items.append({
                'provider':'PARTVERSE_XL','partverse_uid':uid,'source_sha256_provider_claim':orig_sha,
                'source_url':r['file_identifier'],'aesthetic_provider_score':round(aesthetic,4),
                'caption_first_vlm':caption[:1024], 'caption_source':caption_status,
                'provider_split':('PROVIDER_VAL_SEALED' if uid in provider_val_uids else ('PROVIDER_TRAIN_CANDIDATE' if uid in provider_train_uids else 'SPLIT_UNVERIFIED')),
                'object_type_candidate':kind,'matched_weak_text_categories':';'.join(matches),
                'style_candidate':style,'triage_tier':tier,
                'scene_source_original_sha_matches_current_roster':orig_sha in master_sha,
                'source_identity_join_claim':'PARTVERSE_UID_FROM_SKETCHFAB_SOURCE_URL__NOT_REHASHED',
                'source_byte_sha_verified':False,'source_original_license_court':'UNKNOWN',
                'morphology_visual_court':'UNKNOWN','semantic_part_concept_court':'UNKNOWN',
                'source_face_part_id_court':'UNKNOWN','drawable_court':'UNKNOWN',
                'head_training_admitted':False,
            })
    # deterministic selection gives creatures/humanoids/mecha priority; longtail
    # vehicle assets and uncertain subjects get review capacity, never PASS.
    quotas={'ANIMAL_CREATURE':850,'HUMANOID_CHARACTER':750,'ROBOT_MECHA':450,'GAMEPLAY_VEHICLE':300,'UNKNOWN_REVIEW':150}
    ratio=max_selected / sum(quotas.values())
    caps={k:int(v*ratio) for k,v in quotas.items()}
    caps['UNKNOWN_REVIEW']+=max_selected-sum(caps.values())
    selected=[]
    for category,cap in caps.items():
        pool=[r for r in items if r['object_type_candidate']==category and r['provider_split']=='PROVIDER_TRAIN_CANDIDATE' and r['triage_tier'] not in ('REALISTIC_CHARACTER_DEFER','REALISTIC_VEHICLE_DEFER')]
        # place explicit stylized claims ahead of unknown claims; aesthetic score
        # is a candidate quality hint, not a validated art label.
        pool.sort(key=lambda x:(x['style_candidate']!='STYLIZED_CLAIM',-x['aesthetic_provider_score'],x['partverse_uid']))
        selected.extend(pool[:cap])
    selected=sorted(selected,key=lambda r:r['partverse_uid'])
    assert len({x['partverse_uid'] for x in selected})==len(selected)
    if any(x['provider_split']!='PROVIDER_TRAIN_CANDIDATE' for x in selected):raise ValueError('LEAKED_UNVERIFIED_SPLIT')
    all_fields=list(items[0])
    for name,contents in [('PARTVERSE_METADATA_ALL_WEAK_TRIAGE_V1.csv',items),('PARTVERSE_CANDIDATE_DOWNLOAD_QUEUE_V1.csv',selected)]:
        with (output_dir/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,all_fields)
            w.writeheader();w.writerows(contents)
    manifest={'schema':'RealSaS.PartVerseMorphologyTriage.v1','metadata_sha256':sha(metadata_path),
              'record_count':len(items),'selected_quarantined_count':len(selected),'invalid_rows':dict(invalid),
              'candidate_type_distribution':dict(Counter(x['object_type_candidate'] for x in items)),
              'provider_split_counts':dict(Counter(x['provider_split'] for x in items)),
              'selected_with_protected_val_leak':sum(x['provider_split']=='PROVIDER_VAL_SEALED' for x in selected),
              'selected_type_distribution':dict(Counter(x['object_type_candidate'] for x in selected)),
              'style_candidates':dict(Counter(x['style_candidate'] for x in items)),
              'source_roster_exact_SHA_candidate_overlap':sum(x['scene_source_original_sha_matches_current_roster'] for x in items),
              'source_rights_pass':0,'visual_morphology_court_pass':0,'semantic_concept_pass':0,'drawable_pass':0,'head_train_pass':0,
              'max_selected_requested':max_selected,
              'selection_note':'Weak VLM first-caption lexical triage; PartVerse UID source link crosswalk; no visual/rights/source-byte or head truth admission',
              'files_sha256':{p.name:sha(p) for p in output_dir.glob('*.csv')}}
    (output_dir/'SUMMARY_V1.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--source-roster',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--max-selected',type=int,default=2500)
    p.add_argument('--provider-train',type=Path,required=True)
    p.add_argument('--provider-val',type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(run(args.metadata,args.out,args.source_roster,args.max_selected,args.provider_train,args.provider_val),indent=2))