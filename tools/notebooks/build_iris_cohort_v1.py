"""Build a self-contained Run All capsule from repo source and verified inputs."""
from __future__ import annotations
import argparse
import ast
import base64
import hashlib
import io
import json
import subprocess
from pathlib import Path
import zipfile
from tools.training.prepare_iris_cohort_v1 import verify_observations

ROOT=Path(__file__).resolve().parents[2]
DINO_SOURCE_REVISION="7764ea0f912e53c92e82eb78a2a1631e92725fc8"

def source_closure(entries):
    pending=list(entries);seen=set()
    while pending:
        relative=pending.pop()
        if relative in seen:
            continue
        seen.add(relative)
        path=ROOT/relative
        if path.suffix==".py":
            tree=ast.parse(path.read_text())
            modules=[]
            for node in ast.walk(tree):
                if isinstance(node,ast.Import):
                    modules.extend(x.name for x in node.names)
                elif isinstance(node,ast.ImportFrom):
                    if node.level:
                        package=relative[:-3].split("/")[:-1]
                        prefix=package[:len(package)-node.level+1]
                        module=".".join([*prefix,*((node.module or "").split(".") if node.module else [])])
                    else:
                        module=node.module or ""
                    if module:
                        modules.append(module)
                        modules.extend(module+"."+alias.name for alias in node.names if alias.name!="*")
            for name in modules:
                for candidate in (name.replace(".","/")+".py",name.replace(".","/")+"/__init__.py"):
                    if (ROOT/candidate).is_file() and candidate not in seen:
                        pending.append(candidate)
            for parent in path.relative_to(ROOT).parents:
                candidate=(parent/"__init__.py").as_posix()
                if (ROOT/candidate).is_file() and candidate not in seen:
                    pending.append(candidate)
    return sorted(seen)

def collect_source_files(entries):
    source={p:(ROOT/p).read_bytes() for p in source_closure(entries)}
    for p in source_closure(["compiler/vendor/realsas_synthesis/canonical_graph_optimizer.py",
                             "compiler/vendor/realsas_synthesis/__init__.py"]):
        source[p]=(ROOT/p).read_bytes()
    vendor=ROOT/"compiler/vendor/realsas_v05_current_execution_closure.b64"
    manifest=json.loads((vendor/"manifest.json").read_text())
    source[(vendor/"manifest.json").relative_to(ROOT).as_posix()]=(vendor/"manifest.json").read_bytes()
    for row in manifest["part_records"]:
        path=vendor/row["name"]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row["sha256"]:
            raise ValueError("NOTEBOOK_COMPILER_VENDOR_PART_DRIFT")
        source[path.relative_to(ROOT).as_posix()]=path.read_bytes()
    return source

def capsule(files):
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,data in sorted(files.items()):
            info=zipfile.ZipInfo(name,(2026,10,13,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o100644 << 16
            z.writestr(info,data)
    data=buffer.getvalue()
    return {"sha256":hashlib.sha256(data).hexdigest(),"base64":base64.b64encode(data).decode(),
            "file_sha256":{n:hashlib.sha256(d).hexdigest() for n,d in sorted(files.items())}}

def code(text):
    ast.parse(text)
    return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":text}

def markdown(text):
    return {"cell_type":"markdown","metadata":{},"source":text}

def build(input_root,dino_source,output,source_revision,drive_input_relative_path=None):
    roster=json.loads((ROOT/"canonical/FIT8_DIVERSITY_SELECTION_V1.json").read_text())
    input_root=Path(input_root).resolve()
    index_path=input_root/"INPUT_PACKETS.json"
    packet_index=None
    if index_path.exists():
        packet_index=json.loads(index_path.read_text())
        if packet_index["schema"]!="RealSaS.PreparedIrisInputPackets.v1":
            raise ValueError("NOTEBOOK_INPUT_PACKET_INDEX_SCHEMA")
        packet_index={r["subject_id"]:r for r in packet_index["subjects"]}
        if set(packet_index)!={r["subject_key"] for r in roster["selected"]}:
            raise ValueError("NOTEBOOK_INPUT_PACKET_COHORT_DRIFT")
    inputs={}
    for row in roster["selected"]:
        packet_root=input_root
        if packet_index:
            packet_path=Path(packet_index[row["subject_key"]]["packet_path"]).resolve()
            if not packet_path.is_relative_to(input_root):
                raise ValueError("NOTEBOOK_INPUT_PACKET_PATH_ESCAPE")
            packet_root=packet_path.parent
        folder,proof,binding=verify_observations(packet_root,row)
        names=["IRIS_OBSERVATION_ADMISSION.json","SOURCE_REST_RENDER_CANDIDATE.json",
               "SOURCE_SCENE_SNAPSHOT.json","camera_bundle.json"]
        names += [f"V{i}{suffix}" for i in range(8) for suffix in (".png",".mask.bin")]
        for name in names:
            inputs[row["subject_key"]+"/"+name]=(folder/name).read_bytes()
    entries=["models/iris/training/cohort_v1.py","tools/training/prepare_iris_cohort_v1.py",
             "tools/notebooks/build_iris_cohort_v1.py",
             "tools/training/iris_cohort_preflight_v1.py","tests/iris/test_shared_cohort_v1.py",
             "tools/training/export_iris_cohort_surfaces_v1.py",
             "tools/corpus/prepare_iris_observations_v1.py",
             "tools/observation/blender_source_parity_export_v1.py"]
    source=collect_source_files(entries)
    for p in ("canonical/FIT8_DIVERSITY_SELECTION_V1.json","canonical/FIT8_IRIS_OBSERVATION_POLICY_V1.json",
              "canonical/FIT8_IRIS_SHARED_TRAINING_POLICY_V1.json",
              "canonical/FIT8_IRIS_KNIGHT_RECIPE_REFERENCE_V1.json",
              "canonical/FIT8_IRIS_REVISION3_AUDIT_20261013.json"):
        source[p]=(ROOT/p).read_bytes()
    dino_source=Path(dino_source).resolve()
    # Validate the local builder authority, but do not embed a multi-megabyte Git checkout
    # into a Colab cell. The notebook fetches this exact sealed revision at runtime.
    dino_head=subprocess.check_output(["git","-C",str(dino_source),"rev-parse","HEAD"],text=True).strip()
    if dino_head!=DINO_SOURCE_REVISION:
        raise ValueError(f"NOTEBOOK_DINO_SOURCE_REVISION_DRIFT:{dino_head}")
    packs={"source":capsule(source),"inputs":capsule(inputs)}
    input_archive=None
    if drive_input_relative_path:
        input_archive=Path(output).parent/"IRIS_FIT8_OBSERVATIONS.zip"
        input_archive.parent.mkdir(parents=True,exist_ok=True)
        input_archive.write_bytes(base64.b64decode(packs["inputs"].pop("base64")))
        packs["inputs"]["drive_relative_path"]=drive_input_relative_path
    identity=hashlib.sha256(json.dumps({
        "packs":{k:v["sha256"] for k,v in packs.items()},
        "dino_source_revision":DINO_SOURCE_REVISION,
        "recipe_revision":3,
    },sort_keys=True).encode()).hexdigest()
    first = '''import os, sys, json, time, hashlib, subprocess
from pathlib import Path
from google.colab import drive
drive.mount("/content/drive")
WORK = Path("/content/realsas_iris_fit8")
WORK.mkdir(parents=True, exist_ok=True)
print("[BEGIN] IRIS Fit8 revision 3 | post-SDF Knight staged C lineage | joint cohort optimizer | A100")
print("[STAGE] Drive mounted; active work/cache=/content")
'''
    install = '''subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
    "torch==2.8.0", "torchvision==0.23.0",
    "--index-url", "https://download.pytorch.org/whl/cu128"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
    "numpy==2.1.3", "scipy==1.14.1", "Pillow==11.3.0"], check=True)
import torch
assert torch.__version__.split("+")[0] == "2.8.0", "Restart with a clean runtime: an older torch module is still loaded."
assert torch.cuda.is_available(), "A100 GPU runtime seçin; CUDA bulunamadı."
GPU = torch.cuda.get_device_properties(0)
assert "A100" in GPU.name, f"A100 gerekli; bulunan GPU: {GPU.name}"
assert GPU.total_memory >= 30 * 2**30, "A100 GPU belleği yetersiz."
assert torch.cuda.is_bf16_supported(), "BF16 desteği gerekli."
print(f"[RUNTIME] torch={torch.__version__} CUDA={torch.version.cuda} GPU={GPU.name} GiB={GPU.total_memory/2**30:.1f}")
'''
    unpack = "CAPSULES = "+repr(packs)+"\nCAPSULE_ID = "+repr(identity)+"\nSOURCE_REVISION = "+repr(source_revision)+"\nDINO_SOURCE_REVISION = "+repr(DINO_SOURCE_REVISION)+"\n"+'''import base64, io, zipfile, shutil, importlib
# Run All can follow another capsule in the same Colab process. Drop only this
# capsule's Python namespaces; source-byte checks must never certify old objects.
for name in tuple(sys.modules):
    if name.split(".")[0] in {"models", "compiler", "tools", "tests", "dinov2"}:
        del sys.modules[name]
importlib.invalidate_caches()
def unpack_verified(pack, destination):
    if "base64" in pack:
        data = base64.b64decode(pack["base64"], validate=True)
    else:
        source_archive = Path("/content/drive/MyDrive") / pack["drive_relative_path"]
        assert source_archive.is_file(), f"Exact input packet missing: {source_archive}"
        data = source_archive.read_bytes()
    assert hashlib.sha256(data).hexdigest() == pack["sha256"], "Embedded capsule hash drift"
    # These are notebook-owned snapshots. Preserve Drive state and local shards.
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert set(archive.namelist()) == set(pack["file_sha256"]), "Capsule inventory drift"
        for name in archive.namelist():
            target = (destination / name).resolve()
            assert target.is_relative_to(destination.resolve()), "Capsule path escape"
            content = archive.read(name)
            assert hashlib.sha256(content).hexdigest() == pack["file_sha256"][name], "Embedded file hash drift"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
REPO = WORK / "source"
INPUTS = WORK / "inputs"
DINO = WORK / "dinov2"
unpack_verified(CAPSULES["source"], REPO)
unpack_verified(CAPSULES["inputs"], INPUTS)

def exact_git_checkout(repo_url, revision, destination):
    destination=Path(destination)
    if destination.exists():
        try:
            head=subprocess.check_output(["git","-C",str(destination),"rev-parse","HEAD"],text=True,stderr=subprocess.STDOUT).strip()
        except Exception:
            head=None
        if head!=revision:
            shutil.rmtree(destination)
    if not destination.exists():
        subprocess.run(["git","init",str(destination)],check=True)
        subprocess.run(["git","-C",str(destination),"remote","add","origin",repo_url],check=True)
        subprocess.run(["git","-C",str(destination),"fetch","--depth","1","origin",revision],check=True)
        subprocess.run(["git","-C",str(destination),"checkout","--detach","FETCH_HEAD"],check=True)
    head=subprocess.check_output(["git","-C",str(destination),"rev-parse","HEAD"],text=True).strip()
    assert head==revision, f"DINO source revision drift expected={revision} actual={head}"
    return head

DINO_HEAD=exact_git_checkout("https://github.com/facebookresearch/dinov2.git",DINO_SOURCE_REVISION,DINO)
sys.path.insert(0, str(REPO))
os.environ["PYTHONPATH"] = str(REPO)
PERSIST = Path("/content/drive/MyDrive/RealSaS/Fit8/IRIS") / CAPSULE_ID[:16]
PERSIST.mkdir(parents=True, exist_ok=True)
print(f"[SNAPSHOT] source_revision={SOURCE_REVISION} capsule={CAPSULE_ID[:16]} verified_files={sum(len(v['file_sha256']) for v in CAPSULES.values())}")
print(f"[DINO] exact_git_head={DINO_HEAD} transport=shallow_checkout")
print(f"[PERSIST] {PERSIST}")
'''
    weight = '''from models.iris.training.cohort_v1 import file_hash, atomic_json, validate_knight_recipe, source_identity
import shutil, urllib.request
WEIGHT_SHA = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
WEIGHTS = WORK / "dinov2_vits14_pretrain.pth"
DRIVE_WEIGHT = PERSIST.parent / "_foundation" / WEIGHTS.name
DRIVE_WEIGHT.parent.mkdir(parents=True, exist_ok=True)
if not WEIGHTS.exists() or file_hash(WEIGHTS) != WEIGHT_SHA:
    if DRIVE_WEIGHT.exists() and file_hash(DRIVE_WEIGHT) == WEIGHT_SHA:
        shutil.copyfile(DRIVE_WEIGHT, WEIGHTS)
    else:
        temporary = WEIGHTS.with_suffix(".staged")
        urllib.request.urlretrieve("https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth", temporary)
        assert temporary.stat().st_size == 88283115 and file_hash(temporary) == WEIGHT_SHA, "DINO weight authority drift"
        os.replace(temporary, WEIGHTS)
        temporary_drive = DRIVE_WEIGHT.with_suffix(".staged")
        shutil.copyfile(WEIGHTS, temporary_drive)
        assert file_hash(temporary_drive) == WEIGHT_SHA, "Drive foundation copy drift"
        os.replace(temporary_drive, DRIVE_WEIGHT)
assert file_hash(WEIGHTS) == WEIGHT_SHA
print("[FOUNDATION] exact DINOv2-S source + weights verified; backbone frozen")
'''
    prepare = '''from tools.training.prepare_iris_cohort_v1 import prepare
POLICY = json.loads((REPO / "canonical/FIT8_IRIS_SHARED_TRAINING_POLICY_V1.json").read_text())
validate_knight_recipe(POLICY, source_identity(REPO))
REFERENCE = json.loads((REPO / POLICY["knight_recipe_reference"]).read_text())
assert POLICY["recipe_revision"] == 3, "Old Fit8 recipe must not be reused."
print("[RECIPE] Knight post-SDF TP64 + DINO + F* + C loss bytes guarded | weight_decay=0")
print("[RECIPE] 19200 GLOBAL updates | each update averages all 8 subjects | 153600 total subject exposures")
print("[RECIPE] stages: 4800 direct-F* -> 4800 C -> 9600 low-LR C; eval=200/200/400; TRAIN100 telemetry")
SHARDS = WORK / "input_shards"
started = time.monotonic()
MANIFEST = prepare(INPUTS, SHARDS, DINO, WEIGHTS, POLICY, device="cuda",
                   persist_root=PERSIST / "input_shards")
atomic_json(PERSIST / "TRAINING_POLICY.json", POLICY)
atomic_json(PERSIST / "CAPSULE_BINDING.json", {
    "source_revision": SOURCE_REVISION, "capsule_id": CAPSULE_ID,
    "capsule_sha256": {k:v["sha256"] for k,v in CAPSULES.items()},
    "dino_source_revision": DINO_SOURCE_REVISION, "recipe_revision": 3})
print(f"[STAGE] 8 scoped IRIS observation sets -> DINO/F* + disjoint query pools + certified balanced source rays; seconds={time.monotonic()-started:.1f}")
'''
    preflight = '''import unittest
import contextlib, io, importlib.util
TEST_FILE=REPO/"tests/iris/test_shared_cohort_v1.py"
assert TEST_FILE.is_file(), f"Missing verified contract test: {TEST_FILE}"
spec=importlib.util.spec_from_file_location("realsas_fit8_shared_cohort_tests",TEST_FILE)
test_module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=test_module
spec.loader.exec_module(test_module)
SharedCohortTests=test_module.SharedCohortTests
suite = unittest.defaultTestLoader.loadTestsFromTestCase(SharedCohortTests)
contract_log = io.StringIO()
with contextlib.redirect_stdout(contract_log), contextlib.redirect_stderr(contract_log):
    tested = unittest.TextTestRunner(stream=contract_log, verbosity=1).run(suite)
if not tested.wasSuccessful():
    print(contract_log.getvalue())
assert tested.wasSuccessful(), "Shared trainer contract tests failed"
print(f"[CONTRACT_TESTS] tests={tested.testsRun} PASS | exact CPU resume + drift/corruption checks")
from tools.training.export_iris_cohort_surfaces_v1 import export as verified_surface_export
print("[IMPORT_SMOKE] actual Compiler/decoder dependency closure imported")
from tools.training.iris_cohort_preflight_v1 import preflight
PREFLIGHT = preflight(SHARDS, POLICY, PERSIST / "runtime_preflight_checkpoints",
                      PERSIST / "runtime_preflight_results", device="cuda")
assert PREFLIGHT["a100_executed"], "A100 execution preflight required"
print("[PREFLIGHT_PASS] actual A100 | full production model | 4 GLOBAL optimizer steps / 32 subject exposures | staged checkpoint restore + continuation")
'''
    training = '''from models.iris.training.cohort_v1 import SharedIrisTrainer, source_identity
torch.set_num_threads(4)
import contextlib
class Tee:
    def __init__(self, console, stream):
        self.console, self.stream = console, stream
    def write(self, text):
        self.console.write(text); self.stream.write(text)
        return len(text)
    def flush(self):
        self.console.flush(); self.stream.flush()
RESULTS = WORK / "results"
with (PERSIST / "TRAINING.log").open("a", buffering=1) as log:
    with contextlib.redirect_stdout(Tee(sys.stdout, log)):
        trainer = SharedIrisTrainer(SHARDS, MANIFEST, POLICY, source_identity(REPO),
            PERSIST / "checkpoints", RESULTS, device="cuda",
            persistent_result_root=PERSIST / "results")
        RESULT = trainer.run()
assert RESULT["actual_optimizer_steps"] == POLICY["steps"] == 19200
assert set(RESULT["subject_exposures"].values()) == {POLICY["steps"]}
assert RESULT["total_subject_exposures"] == POLICY["steps"] * POLICY["subject_count"]
assert len(RESULT["training_stage_receipts"]) == 3
print("[TRAINING_COMPLETE] 19200 GLOBAL optimizer updates; each of 8 subjects contributes to every update; total exposures=153600")
'''
    surfaces = '''from tools.training.export_iris_cohort_surfaces_v1 import export
print("[STAGE] selected shared checkpoint -> blind R512 exterior sign audit -> T512/T1024 exact contained decoder -> every subject/view P999 court")
SURFACE_COURT = export(RESULTS/"IRIS_SHARED_SELECTED.pt", SHARDS, INPUTS,
    RESULTS/"surfaces", PERSIST/"results/surfaces", device="cuda")
print(f"[COURT_SUMMARY] all_subjects_raw_court_pass={SURFACE_COURT['all_subjects_raw_court_pass']} Go qualification remains separate")
'''
    persist = '''from models.iris.training.cohort_v1 import value_hash
IMPORT_PAYLOAD = {
    "schema": "RealSaS.SharedIrisExecutionImportBundle.v1",
    "execution": {"path": str(PERSIST/"results/IRIS_SHARED_FIT_RESULT.json"),
                  "sha256": file_hash(PERSIST/"results/IRIS_SHARED_FIT_RESULT.json")},
    "cohort_manifest": {"path": str(PERSIST/"input_shards/COHORT_MANIFEST.json"),
                        "sha256": file_hash(PERSIST/"input_shards/COHORT_MANIFEST.json")},
    "training_policy": {"path": str(PERSIST/"TRAINING_POLICY.json"),
                        "sha256": file_hash(PERSIST/"TRAINING_POLICY.json")},
    "checkpoint": {"path": str(PERSIST/"results/IRIS_SHARED_SELECTED.pt"),
                   "sha256": file_hash(PERSIST/"results/IRIS_SHARED_SELECTED.pt")},
    "surface_court": {"path": str(PERSIST/"results/surfaces/COHORT_SURFACE_COURT.json"),
                      "sha256": file_hash(PERSIST/"results/surfaces/COHORT_SURFACE_COURT.json")},
    "go_import_required": True, "qualification_minted": False}
atomic_json(PERSIST/"SHARED_COHORT_IMPORT_PAYLOAD.json", IMPORT_PAYLOAD)
archive_local = Path(shutil.make_archive(str(WORK/"IRIS_FIT8_RESULTS"), "zip", RESULTS))
archive_drive = PERSIST/"IRIS_FIT8_RESULTS.zip"
temporary = archive_drive.with_suffix(".staged")
shutil.copyfile(archive_local, temporary)
assert file_hash(temporary) == file_hash(archive_local), "Persistent archive hash drift"
os.replace(temporary, archive_drive)
atomic_json(PERSIST/"RUN_ALL_COMPLETE.json", {
    "schema": "RealSaS.SharedIrisNotebookCompletion.v1", "status": "TRAINING_COMPLETE",
    "capsule_id": CAPSULE_ID, "result_archive_sha256": file_hash(archive_drive),
    "a100_preflight": PREFLIGHT, "actual_optimizer_steps": RESULT["actual_optimizer_steps"],
    "fit8_pass_claimed": False, "compiler_qualification_pending": True})
print(f"[RUN_ALL_COMPLETE] results={archive_drive}")
print("[NEXT] exact shared checkpoint + all subject metrics -> existing Go/Compiler qualification lane")
'''
    notebook={"nbformat":4,"nbformat_minor":5,
        "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                    "colab":{"name":Path(output).name,"provenance":[]},"accelerator":"GPU",
                    "realsas":{"capsule_id":identity,"source_revision":source_revision,
                               "scope":"IRIS_SHARED_FSTAR_RESEARCH_FIT","recipe_revision":3,
                               "a100_execution_preflight":"RUN_ALL_REQUIRED"}},
        "cells":[markdown("# RealSaS · IRIS Fit8 · A100 · Reçete 3\n\nA100 GPU seçin ve **Run All** çalıştırın. Drive mount dışında hücre düzenleme gerektirmez. "
                 "Snake, Bat, T-Rex, Horse, Spider, GreenBlob, Shark ve George; tek IRIS ve tek optimizer. "
                 "19.200 GLOBAL optimizer adımı; her global adımda sekiz subject eşit ağırlıklı gradient katkısı verir. "
                 "Toplam 153.600 subject exposure vardır fakat yalnız 19.200 Adam update yapılır.\n\n"
                 "Knight TP64 model, DINO, F* ve loss dosyaları aynı hash’lerle korunur. "
                 "AdamW weight decay sıfır; query oranları ve C silüet loss’u Knight ile aynı otoritededir. "
                 "Eğitim kronolojisi SDF-head sonrası Knight C lineage’ını korur: 4.800 direct-F*, 4.800 C ve 9.600 low-LR C. "
                 "İlk aşamada hard replay her 200-step eval’da yenilenir; C başlangıcında seçili direct checkpoint’ten bir kez dondurulur. "
                 "Bu sıfırdan ortak Fit8 eğitimidir; Knight checkpoint continuation değildir. "
                 "Yeni reçete ayrı kapsül/checkpoint kimliği kullanır; eski checkpoint bu reçeteye yüklenmez.\n\n"
                 "RealSaS kaynak kapsülü ve sabit girdi kimlikleri notebook içindedir; büyük DINO Git ağacı editöre gömülmez, "
                 "Knight ile aynı exact commit runtime’da shallow checkout edilir. 64 doğrulanmış bind gözlemi, kamera ve kanıtlar "
                 "SHA bağlı Drive paketinden otomatik yüklenir. "
                 "Önce tam mimariyle gerçek A100 preflight çalışır. Kesilirse aynı notebook’u Run All ile tekrar açın; "
                 "son atomik checkpoint’ten devam eder. Drive çıktıları aynı kapsül kimliğinde kalır.\n\n"
                 "Her 100 global adımda ortak training telemetry; direct/C eval’larında sırasıyla 200/200/400 adım ritmiyle "
                 "sekiz subject’in heldout F* metrikleri ve worst-subject ortak özeti basılır. Eğitim sonunda seçilen ortak checkpoint, "
                 "mevcut contained T512/T1024 decoder’a verilir; SignedZeroSurface, R512 ve her view P999 metrikleri ayrı saklanır.\n\n"
                 "Sıkı IRIS kabulü: eğitim öncesi kaynak R512 dış bölge sayımı, seçim sonrası aynı sayımla "
                 "sıfır/negatif dış bölge hatası bulunmaması, exact containment ve her subject’in bütün view’larında "
                 "değişmemiş P999 eşikleri. Eğitim loss’u veya cohort ortalaması PASS kanıtı değildir.\n\n"
                 "Kapsam: görüntüden türetilmiş F* ortak eğitim deneyi. Native mekanik truth, IRIS Stage12/13, "
                 "Fit8 ve ürün PASS ayrı doğrulama kapılarıdır; bu notebook eğitim kaybından PASS üretmez."),
                 code(first),code(install),code(unpack),code(weight),code(prepare),
                 code(preflight),code(training),code(surfaces),code(persist)]}
    for i,cell in enumerate(notebook["cells"]):
        cell["id"]=f"iris-fit8-{i:02d}"
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+"\n")
    return {"path":str(output),"sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
            "capsule_id":identity,"capsule_sha256":{k:v["sha256"] for k,v in packs.items()},
            "source_file_count":len(source),"input_file_count":len(inputs),
            "input_archive":str(input_archive) if input_archive else None}

def main():
    parser=argparse.ArgumentParser()
    for flag in ("input-root","dino-source","output"):
        parser.add_argument("--"+flag,type=Path,required=True)
    parser.add_argument("--source-revision",required=True)
    parser.add_argument("--drive-input-relative-path")
    args=parser.parse_args()
    print(json.dumps(build(args.input_root,args.dino_source,args.output,args.source_revision,args.drive_input_relative_path),indent=2))
if __name__=="__main__":
    main()
