"""Convert first-order conventions and express a world field in listener axes."""
from array import array
import math
import sys
from . import spatial_plan

# Input channel order of the three directional coefficients (x-front, y-left,
# z-up) for each supported first-order convention. AmbiX is ACN/SN3D
# [W,Y,Z,X]; FuMa is maxN [W,X,Y,Z].
_DIRECTION_CHANNELS = {
    'ambix': (3, 1, 2),
    'fuma': (1, 2, 3),
}
# Output is always ACN/SN3D [W,Y,Z,X].
_OUTPUT_ORDER = (0, 2, 3, 1)


def canonical(frame, convention):
    channels = list(frame)
    if len(channels) != 4:
        raise ValueError('soundfield must have four channels')
    try:
        indices = _DIRECTION_CHANNELS[convention]
    except KeyError:
        raise ValueError('unsupported soundfield convention')
    pressure = channels[0]
    if convention == 'fuma':
        # FuMa maxN W carries a sqrt(2) gain over the SN3D omnidirectional term.
        pressure *= math.sqrt(2)
    x, y, z = (channels[index] for index in indices)
    return pressure, x, y, z


def listener(frame, q):
    w, x, y, z = frame
    a, b, c, d = q
    # First-order soundfield rotation: the listener-to-world pose acts on the
    # world directional coefficients through the pose's rotation matrix,
    # expressing world-fixed sources in listener axes (passive field rotation).
    return (w,
        (1-2*(c*c+d*d))*x + 2*(b*c+a*d)*y + 2*(b*d-a*c)*z,
        2*(b*c-a*d)*x + (1-2*(b*b+d*d))*y + 2*(c*d+a*b)*z,
        2*(b*d+a*c)*x + 2*(c*d-a*b)*y + (1-2*(b*b+c*c))*z)


def _pose_at(points, sample, cursor):
    """Evaluate the frozen trajectory at one sample, advancing the segment cursor."""
    while cursor+1 < len(points) and sample > points[cursor+1]['sample']:
        cursor += 1
    left = points[cursor]
    if cursor+1 == len(points) or sample == left['sample']:
        return left['q'], cursor
    right = points[cursor+1]
    phase = (sample-left['sample'])/(right['sample']-left['sample'])
    return spatial_plan.slerp(left['q'], right['q'], phase), cursor


def transform(plan, pcm, start_sample):
    if len(pcm) % 4 or any(not math.isfinite(v) for v in pcm):
        raise ValueError('invalid four-channel samples')
    frames = len(pcm)//4
    if not frames:
        return array('f')
    finish = start_sample+frames
    if start_sample < 0 or finish > plan['samples']:
        raise ValueError('pose outside programme')
    points = plan['poses']
    cursor = spatial_plan.segment_index(points, start_sample)
    result = array('f')
    convention = plan['convention']
    for index in range(frames):
        sample = start_sample+index
        q, cursor = _pose_at(points, sample, cursor)
        source = canonical(pcm[index*4:index*4+4], convention)
        w, x, y, z = listener(source, q)
        field = (w, x, y, z)
        result.extend(field[channel] for channel in _OUTPUT_ORDER)
    return result


def blocks(plan, handle, block_frames=1024, start_sample=0, samples=None):
    if type(block_frames) is not int or not 1 <= block_frames <= 65536:
        raise ValueError('invalid spatial block size')
    cursor = start_sample
    stop = plan['samples'] if samples is None else start_sample+samples
    if not 0 <= start_sample < stop <= plan['samples']:
        raise ValueError('invalid spatial sample interval')
    while cursor < stop:
        count = min(block_frames, stop-cursor)
        raw = handle.read(count*16)
        if len(raw) != count*16:
            raise ValueError('incomplete four-channel recording')
        pcm = array('f')
        pcm.frombytes(raw)
        if sys.byteorder != 'little':
            pcm.byteswap()
        output = transform(plan, pcm, cursor)
        if sys.byteorder != 'little':
            output.byteswap()
        yield output.tobytes()
        cursor += count
