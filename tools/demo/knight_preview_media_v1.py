"""Checked, atomic encoding of exact source-rendered preview frames."""
from pathlib import Path
import json
import os
import subprocess
import io
import zipfile
import numpy as np
from PIL import Image, ImageSequence


def save_npz_checked(path, **arrays):
    path = Path(path)
    memory = io.BytesIO()
    np.savez_compressed(memory, **arrays)
    payload = memory.getvalue()
    temporary = path.with_name(path.name + '.writing')
    with temporary.open('wb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('NPZ_CRC_FAILURE:' + str(path))
    if temporary.stat().st_size != len(payload):
        raise RuntimeError('NPZ_WRITE_TRUNCATED:' + str(path))
    temporary.replace(path)


def save_png_checked(image, path):
    path = Path(path)
    temporary = path.with_name(path.name + '.writing.png')
    for attempt in range(3):
        with temporary.open('wb') as stream:
            image.save(stream, format='PNG')
            stream.flush()
            os.fsync(stream.fileno())
        try:
            with Image.open(temporary) as decoded:
                decoded.verify()
            temporary.replace(path)
            return
        except (OSError, SyntaxError):
            if attempt == 2:
                raise


def probe_video(path):
    return json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=width,height,r_frame_rate,nb_frames,duration', '-of', 'json',
        str(path)], text=True))['streams'][0]


def encode_media(pattern, destination, fps, count):
    """Fail on a corrupt input, validate decoded output before atomic replace."""
    destination = Path(destination)
    temporary = destination.with_name(destination.stem + '.writing' + destination.suffix)
    base = ['ffmpeg', '-y', '-v', 'error', '-xerror', '-framerate', str(fps), '-i', str(pattern)]
    for attempt in range(3):
        try:
            if destination.suffix == '.gif':
                palette = destination.with_name(destination.stem + '.palette.png')
                subprocess.run(base + ['-vf', 'palettegen=stats_mode=diff', '-frames:v', '1',
                                       '-threads', '1', str(palette)], check=True)
                subprocess.run(base + ['-i', str(palette), '-filter_complex',
                    '[0:v][1:v]paletteuse=dither=bayer:bayer_scale=3', '-loop', '0',
                    '-frames:v', str(count), '-threads', '1', str(temporary)], check=True)
                with Image.open(temporary) as decoded:
                    ms = 0
                    for frame in ImageSequence.Iterator(decoded):
                        frame.load()
                        ms += frame.info.get('duration', 0)
                if abs(ms - count / fps * 1000) > 20:
                    raise RuntimeError('GIF_TIMING_DRIFT:' + destination.name)
                palette.unlink(missing_ok=True)
            else:
                subprocess.run(base + ['-frames:v', str(count), '-c:v', 'libx264', '-crf',
                    '17', '-pix_fmt', 'yuv420p', '-threads', '2', str(temporary)], check=True)
                stream = probe_video(temporary)
                if stream['r_frame_rate'] != f'{fps}/1' or int(stream['nb_frames']) != count:
                    raise RuntimeError('MP4_FRAME_COUNT_DRIFT:' + destination.name)
                subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(temporary),
                                '-f', 'null', '-'], check=True)
            with temporary.open('rb') as stream:
                os.fsync(stream.fileno())
            temporary.replace(destination)
            return
        except (OSError, SyntaxError, RuntimeError, subprocess.CalledProcessError):
            if attempt == 2:
                raise


def package_frames(out, prefix, kind, frame_indices, resolution, fps):
    from tools.demo.render_knight_m_caa_v1 import tile
    out = Path(out)
    for fi in np.unique(frame_indices):
        sheet = Image.new('RGB', (resolution * 4, (resolution + 32) * 2), (23, 28, 37))
        for v in range(8):
            raw = out / f'{prefix}_v{v}_{fi:03d}.png'
            with Image.open(raw) as decoded:
                decoded.verify()
            with Image.open(raw) as decoded:
                image = tile(np.asarray(decoded.convert('RGBA')), f'{kind} V{v}')
            save_png_checked(image, out / f'{prefix}_tile_v{v}_{fi:03d}.png')
            sheet.paste(image, ((v % 4) * resolution, (v // 4) * (resolution + 32)))
        save_png_checked(sheet, out / f'{prefix}_unique_grid_{fi:03d}.png')
    for pi, fi in enumerate(frame_indices):
        pairs = [(out / f'{prefix}_unique_grid_{fi:03d}.png', out / f'{prefix}_grid_{pi:03d}.png')]
        pairs += [(out / f'{prefix}_tile_v{v}_{fi:03d}.png', out / f'{prefix}_preview_v{v}_{pi:03d}.png') for v in range(8)]
        for original, replay in pairs:
            replay.unlink(missing_ok=True)
            os.link(original, replay)
    for suffix in ['V0-V7'] + [f'V{v}' for v in range(8)]:
        pattern = out / (f'{prefix}_grid_%03d.png' if suffix == 'V0-V7'
                         else f'{prefix}_preview_v{suffix[1:]}_%03d.png')
        for extension in (['mp4', 'gif'] if suffix == 'V0-V7' else ['mp4']):
            encode_media(pattern, out / f'Knight_{kind}_{suffix}.{extension}', fps, len(frame_indices))
    return probe_video(out / f'Knight_{kind}_V0-V7.mp4')
