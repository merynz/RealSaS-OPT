"""Whitelisted evaluator provenance; never imports a subject or mutates Platform."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import struct
import sys


def fingerprint():
    import bpy
    import numpy as np
    libraries = {}
    for line in Path('/proc/self/maps').read_text().splitlines():
        path = Path(line.split()[-1])
        if path.name.startswith(('libm.so', 'libc.so', 'ld-linux')) and path.is_file():
            libraries[path.name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    cpu = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                if line.startswith('model name')), 'UNKNOWN')
    return {'schema': 'RealSaS.EvaluatorRuntimeFingerprint.v1',
        'blender_version': bpy.app.version_string, 'binary_sha256': hashlib.sha256(Path(bpy.app.binary_path).read_bytes()).hexdigest(),
        'libc': list(platform.libc_ver()), 'libraries': libraries, 'cpu_model': cpu,
        'python_version': platform.python_version(), 'numpy_version': np.__version__,
        'math_probe_binary64': {name: struct.pack('>d', fn(value)).hex()
            for name, fn, value in [('sin_0.1', math.sin, .1), ('cos_0.1', math.cos, .1), ('sin_pi', math.sin, math.pi)]},
        'subject_imported': False, 'platform_mutation_performed': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    Path(args.output).write_text(json.dumps(fingerprint(), sort_keys=True, indent=2) + '\n')
