"""Install a separate pinned evaluator; never modify system or original Blender."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request

from tools.observation.pinned_evaluator_runtime_v1 import (
    LIBRARY_SHA, ORIGINAL_BINARY_SHA, PACKAGE_SHA, PACKAGE_URL, PATCHED_BINARY_SHA,
    launch_options, patch_interpreter, runtime_manifest, sha,
)


def download(url, path, expected):
    if not path.is_file():
        temporary = path.with_suffix(path.suffix + '.partial')
        with urllib.request.urlopen(url, timeout=60) as source, temporary.open('wb') as output:
            while block := source.read(1024 * 1024):
                output.write(block)
        if sha(temporary) != expected:
            raise ValueError('PINNED_RUNTIME_ARCHIVE_BYTES_DRIFT')
        temporary.replace(path)
    if sha(path) != expected:
        raise ValueError('PINNED_RUNTIME_ARCHIVE_BYTES_DRIFT')


def install(executable):
    executable = Path(executable).resolve()
    if executable.name != 'blender-glibc239' or executable.parent.name != 'numeric-runtime-239':
        raise ValueError('PINNED_RUNTIME_SEPARATE_EXECUTABLE_REQUIRED')
    root = executable.parent
    original_root = root.parent
    original_root.parent.mkdir(parents=True, exist_ok=True)
    original = original_root / 'blender'
    if not original.is_file():
        archive = original_root.parent / 'blender-4.0.2-linux-x64.tar.xz'
        download('https://download.blender.org/release/Blender4.0/blender-4.0.2-linux-x64.tar.xz', archive,
                 '5583a5588736da8858c522ef17fff5d73be59c47a6fe91ad29c6f3263e22086a')
        with tarfile.open(archive) as source:
            if not all(Path(row.name).parts[0] == original_root.name for row in source.getmembers()):
                raise ValueError('PINNED_RUNTIME_ARCHIVE_ROOT_DRIFT')
            source.extractall(original_root.parent, filter='data')
    if sha(original) != ORIGINAL_BINARY_SHA:
        raise ValueError('PINNED_RUNTIME_ORIGINAL_BINARY_DRIFT')
    root.mkdir(parents=True, exist_ok=True)
    resources = root / '4.0'
    if not resources.exists():
        resources.symlink_to(original_root / '4.0', target_is_directory=True)
    libraries = root / 'lib'; libraries.mkdir(exist_ok=True)
    for source in (original_root / 'lib').iterdir():
        destination = libraries / source.name
        if not destination.exists():
            destination.symlink_to(source, target_is_directory=source.is_dir())
    package = root / 'libc6_2.39-0ubuntu8.6_amd64.deb'
    download(PACKAGE_URL, package, PACKAGE_SHA)
    manifest = runtime_manifest(executable)
    package_libraries = root / 'glibc-2.39/usr/lib/x86_64-linux-gnu'
    if not package_libraries.is_dir():
        subprocess.run(['dpkg-deb', '-x', str(package), str(root / 'glibc-2.39')], check=True)
    for source in package_libraries.iterdir():
        if not source.is_file():
            continue
        destination = libraries / source.name
        if destination.is_symlink():
            destination.unlink()
        if not destination.exists():
            shutil.copyfile(source, destination)
        if sha(destination) != sha(source):
            raise ValueError('PINNED_RUNTIME_PRIVATE_LIBRARY_DRIFT')
    loader = Path(manifest['files']['loader']['path'])
    if not loader.exists():
        shutil.copyfile(package_libraries / 'ld-linux-x86-64.so.2', loader)
        loader.chmod(0o755)
    if sha(loader) != LIBRARY_SHA['loader']:
        raise ValueError('PINNED_RUNTIME_LOADER_BYTES_DRIFT')
    if not executable.exists():
        temporary = executable.with_suffix('.partial')
        temporary.write_bytes(patch_interpreter(original.read_bytes()))
        temporary.chmod(0o755)
        temporary.replace(executable)
    launch_options(executable, manifest)
    if sha(executable) != PATCHED_BINARY_SHA:
        raise ValueError('PINNED_RUNTIME_DERIVATIVE_BINARY_DRIFT')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text())
    result = install(request['evaluator_executable'])
    if result != request['evaluator_runtime'] or request['expected_evaluator_binary_sha256'] != PATCHED_BINARY_SHA:
        raise ValueError('PINNED_RUNTIME_REQUEST_DRIFT')
    print(json.dumps({'status': 'PINNED_NUMERIC_RUNTIME_INSTALLED', 'binary_sha256': PATCHED_BINARY_SHA,
                      'package_sha256': PACKAGE_SHA}))
