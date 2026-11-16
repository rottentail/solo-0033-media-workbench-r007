"""Decode once and stream a frozen listener trajectory to four-channel float WAV."""
from array import array
import sys
import os
from pathlib import Path
import tempfile
from . import media_sources, planner, spatial_plan, spatial_signal
from .assets import fingerprint
from .tools import run


def write_programme(plan, source, destination, block_frames=1024, start_sample=0, samples=None):
    samples = plan['samples']-start_sample if samples is None else samples
    if type(start_sample) is not int or type(samples) is not int or start_sample < 0 or samples < 1 or start_sample+samples > plan['samples']:
        raise ValueError('audition interval outside programme')
    if type(block_frames) is not int or not 1 <= block_frames <= 65536:
        raise ValueError('invalid spatial block size')
    remaining = samples
    written = 0
    with source.open('rb') as handle, destination.open('wb') as output:
        handle.seek(start_sample*16)
        while remaining:
            count = min(block_frames, remaining)
            encoded = handle.read(count*16)
            if len(encoded) != count*16:
                raise ValueError('incomplete four-channel recording')
            pcm = array('f')
            pcm.frombytes(encoded)
            if sys.byteorder != 'little':
                pcm.byteswap()
            # Keep the global programme clock: every block samples the same
            # frozen trajectory, so output cannot depend on block partition.
            transformed = spatial_signal.transform(plan, pcm, written+start_sample)
            if sys.byteorder != 'little':
                transformed.byteswap()
            output.write(transformed.tobytes())
            written += count
            remaining -= count
    if written != samples:
        raise ValueError('spatial output length mismatch')
    return written


def render(conn, plan_id, destination, at_ms=0, duration_ms=None):
    plan = media_sources.bind(conn, spatial_plan.get(conn, plan_id))
    count = plan['samples']-at_ms*48 if duration_ms is None else duration_ms*48
    if type(at_ms) is not int or at_ms < 0 or duration_ms is not None and type(duration_ms) is not int or count <= 0 or at_ms*48+count > plan['samples']:
        raise ValueError('audition interval outside programme')
    planner.verify_sources(plan)
    destination = Path(destination).expanduser().absolute()
    destination = destination.parent.resolve()/destination.name
    if os.path.lexists(destination) or not destination.parent.is_dir():
        raise ValueError('destination exists or directory is missing')
    clip = plan['clips'][0]
    with tempfile.TemporaryDirectory(prefix='.spatial-', dir=destination.parent) as directory:
        root = Path(directory)
        raw, transformed, output = root/'input.f32', root/'field.f32', root/'field.wav'
        run('ffmpeg', ['-v','error','-y','-ss',str(clip['start_ms']/1000),'-i',clip['path'],
            '-t',str(plan['samples']/48000),'-map','0:a:0','-vn','-ar','48000',
            '-c:a','pcm_f32le','-f','f32le',str(raw)])
        if raw.stat().st_size != plan['samples']*16:
            raise ValueError('decoder did not preserve the complete four-channel trim')
        write_programme(plan, raw, transformed, start_sample=at_ms*48, samples=count)
        run('ffmpeg', ['-v','error','-y','-f','f32le','-ar','48000','-ac','4','-i',str(transformed),
            '-c:a','pcm_f32le',str(output)])
        planner.verify_sources(plan)
        checksum = fingerprint(output)
        with output.open('rb') as handle:
            os.fsync(handle.fileno())
        os.link(output, destination)
    return {'plan_id': plan_id, 'output_path': str(destination), 'sha256': checksum,
            'samples': count, 'channels': 4, 'convention': 'ACN/SN3D'}
