"""Small IO boundary for unqualified measurement stages, independent of teachers."""
import json
from pathlib import Path
from compiler.realsas_compiler_core.types import QualificationError
from .adapter_io import sha256_file


def require(condition,code):
    if not condition:raise QualificationError(code)


def ref(path,schema=None):
    p=Path(path).resolve();result={'path':str(p),'sha256':sha256_file(p)}
    if schema:result['schema']=schema
    return result


def artifact(path,schema):
    return {**ref(path,schema),'authority_class':'UNQUALIFIED_DERIVED_MEASUREMENTS'}


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.partial')
    tmp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n');tmp.replace(path)
    return path
