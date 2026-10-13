import os
from pathlib import Path
import subprocess
import sys
from tools.notebooks.build_iris_cohort_v1 import collect_source_files, ROOT

def test_cold_capsule_import_contains_ast_and_dynamic_compiler_dependencies(tmp_path):
    files=collect_source_files(["tools/training/export_iris_cohort_surfaces_v1.py",
                                "tools/training/iris_cohort_preflight_v1.py"])
    for name,content in files.items():
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
    # Exclude the checkout so namespace fallback cannot hide a missing snapshot.
    dependencies=[p for p in sys.path if p and not Path(p).resolve().is_relative_to(ROOT)]
    env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(tmp_path),*dependencies]))
    result=subprocess.run([sys.executable,"-c",
        "from tools.training.export_iris_cohort_surfaces_v1 import export; "
        "from tools.training.iris_cohort_preflight_v1 import preflight; "
        "print('COLD_CAPSULE_IMPORT_PASS')"],cwd=tmp_path,env=env,text=True,capture_output=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert "COLD_CAPSULE_IMPORT_PASS" in result.stdout
