from pathlib import Path
from unittest.mock import patch

import pytest

from tools.observation import pinned_evaluator_runtime_v1 as runtime


def test_launch_pins_working_directory_and_library_bytes_before_execution(tmp_path):
    executable = tmp_path / 'blender-glibc239'
    executable.write_bytes(b'synthetic evaluator')
    libraries = tmp_path / 'lib'
    libraries.mkdir(parents=True)
    files = {'loader': tmp_path / 'ld-linux-x86-64.so.2',
             'libc': libraries / 'libc.so.6', 'libm': libraries / 'libm.so.6'}
    for key, path in files.items():
        path.write_bytes(key.encode())
    hashes = {key: runtime.sha(path) for key, path in files.items()}
    with patch.object(runtime, 'PATCHED_BINARY_SHA', runtime.sha(executable)), \
         patch.dict(runtime.LIBRARY_SHA, hashes), patch.dict('os.environ', {'LD_LIBRARY_PATH': '/unrelated', 'LD_PRELOAD': '/unrelated'}):
        manifest = runtime.runtime_manifest(executable)
        options = runtime.launch_options(executable, manifest)
        assert options['cwd'] == executable.parent
        assert 'LD_LIBRARY_PATH' not in options['env']
        assert options['env']['LC_ALL'] == 'C' and 'LD_PRELOAD' not in options['env']
        with pytest.raises(ValueError, match='MANIFEST_OR_BINARY_DRIFT'):
            runtime.launch_options(executable, {**manifest, 'package_sha256': '0' * 64})
        files['libm'].write_bytes(b'changed math library')
        with pytest.raises(ValueError, match='LIBRARY_BYTES_DRIFT'):
            runtime.launch_options(executable, manifest)


def test_unknown_executable_cannot_be_patched_and_historical_launch_is_unchanged():
    with pytest.raises(ValueError, match='ORIGINAL_BINARY_DRIFT'):
        runtime.patch_interpreter(b'unsealed executable')
    assert runtime.launch_options(Path('/unused'), None) == {}
