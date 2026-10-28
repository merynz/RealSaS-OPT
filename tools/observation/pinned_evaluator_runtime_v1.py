"""Pinned numeric runtime for the existing admitted Blender observation imports."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import struct

ORIGINAL_BINARY_SHA = '6fe78c9e0a1cca51a598eb37c3288b975357efbe86dcee18d5e6fdecaaf1847e'
PATCHED_BINARY_SHA = 'c47c0b83d061621014c8312abd99db5c1ef057bcf5516faf4bb8d92b3ef6de60'
PACKAGE_SHA = 'cb11743530ba3ee0f943f4124b13dac50363a36842d682be65cd2beb5073c33e'
PACKAGE_URL = 'https://snapshot.ubuntu.com/ubuntu/20251001T000000Z/pool/main/g/glibc/libc6_2.39-0ubuntu8.6_amd64.deb'
LIBRARY_SHA = {'loader': '6222a16be7f2d458d6870efe6e715fc0c8d45766fb79cf7dcc3125538d703e28',
    'libc': '511f825ee075610ac9c0f7f91e2c13de2000d0f7b859f6461137e809a0a009d0',
    'libm': 'f06f2ce1f1833df5f41cf13b6447ff07bea993ad9b27297d3428c2f70ab3f0e7'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def patch_interpreter(raw):
    if hashlib.sha256(raw).hexdigest() != ORIGINAL_BINARY_SHA:
        raise ValueError('PINNED_RUNTIME_ORIGINAL_BINARY_DRIFT')
    result = bytearray(raw)
    if result[:6] != b'\x7fELF\x02\x01':
        raise ValueError('PINNED_RUNTIME_ELF_FORMAT')
    phoff = struct.unpack_from('<Q', result, 32)[0]
    size, count = struct.unpack_from('<HH', result, 54)
    for index in range(count):
        offset = phoff + index * size
        if struct.unpack_from('<I', result, offset)[0] == 3:
            start = struct.unpack_from('<Q', result, offset + 8)[0]
            length = struct.unpack_from('<Q', result, offset + 32)[0]
            if bytes(result[start:start + length]) != b'/lib64/ld-linux-x86-64.so.2\0':
                raise ValueError('PINNED_RUNTIME_INTERPRETER_DRIFT')
            target = b'./ld-linux-x86-64.so.2\0'
            result[start:start + length] = target + b'\0' * (length - len(target))
            if hashlib.sha256(result).hexdigest() != PATCHED_BINARY_SHA:
                raise ValueError('PINNED_RUNTIME_DERIVATIVE_BINARY_DRIFT')
            return bytes(result)
    raise ValueError('PINNED_RUNTIME_INTERPRETER_MISSING')


def runtime_manifest(executable):
    root = Path(executable).resolve().parent
    libraries = root / 'lib'
    paths = {'loader': root / 'ld-linux-x86-64.so.2', 'libc': libraries / 'libc.so.6', 'libm': libraries / 'libm.so.6'}
    return {'schema': 'RealSaS.PinnedObservationNumericRuntime.v1', 'package_sha256': PACKAGE_SHA,
        'original_blender_binary_sha256': ORIGINAL_BINARY_SHA,
        'library_directory': str(libraries),
        'files': {key: {'path': str(path), 'sha256': LIBRARY_SHA[key]} for key, path in paths.items()}}


def launch_options(executable, runtime=None):
    if runtime is None:
        return {}
    executable = Path(executable).resolve()
    if runtime != runtime_manifest(executable) or sha(executable) != PATCHED_BINARY_SHA:
        raise ValueError('PINNED_RUNTIME_MANIFEST_OR_BINARY_DRIFT')
    for ref in runtime['files'].values():
        if sha(ref['path']) != ref['sha256']:
            raise ValueError('PINNED_RUNTIME_LIBRARY_BYTES_DRIFT')
    env = dict(os.environ)
    for key in ('LD_PRELOAD', 'LD_AUDIT', 'LD_LIBRARY_PATH'):
        env.pop(key, None)
    # Blender's sealed $ORIGIN/lib RPATH resolves these private libraries.
    # Do not propagate an old libc through LD_LIBRARY_PATH to system children.
    env['LC_ALL'] = 'C'
    # The relative PT_INTERP is part of the derivative's sealed bytes. Linux
    # resolves it from cwd; all source/program/output arguments are absolute.
    return {'cwd': executable.parent, 'env': env}
