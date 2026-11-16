"""Freeze listener poses for first-order soundfields on a continuous sample clock."""
import hashlib
import json
import math
from . import db, media_sources, timeline

RATE = 48000


def unit(values):
    if len(values) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError('pose must be a finite quaternion [w,x,y,z]')
    scale = max(abs(v) for v in values)
    if scale == 0:
        raise ValueError('pose quaternion has no orientation')
    values = [v/scale for v in values]
    norm = math.hypot(*values)
    if scale*norm < 1e-12:
        raise ValueError('pose quaternion has no orientation')
    values = [v/norm for v in values]
    # Canonical representatives also make exactly half-turn paths independent
    # of the signs used to record equivalent endpoint orientations.
    sign = next(v for v in values if v != 0)
    return values if sign > 0 else [-v for v in values]


def poses(keyframes, duration_ms):
    if not isinstance(keyframes, list) or not 2 <= len(keyframes) <= 64:
        raise ValueError('two to sixty-four listener poses required')
    result = []
    for item in keyframes:
        at = item['at_ms']
        if type(at) is not int or not 0 <= at <= duration_ms or result and at*48 <= result[-1]['sample']:
            raise ValueError('pose times must strictly increase within the programme')
        result.append({'sample': at*48, 'q': unit(item['q'])})
    if result[0]['sample'] != 0 or result[-1]['sample'] != duration_ms*48:
        raise ValueError('poses must cover the trimmed programme')
    return result


def orientation(plan, sample):
    points = plan['poses']
    if not 0 <= sample <= plan['samples']:
        raise ValueError('pose outside programme')
    selected = None
    for left, right in zip(points, points[1:]):
        if sample <= right['sample']:
            selected = left, right
            break
    if selected is None:
        return list(points[-1]['q'])
    left, right = selected
    width = right['sample']-left['sample']
    phase = (sample-left['sample'])/width
    coefficients = []
    first = unit(left['q'])
    last = unit(right['q'])
    for previous, following in zip(first, last):
        delta = following-previous
        coefficients.append(previous+phase*delta)
    magnitude = sum(value*value for value in coefficients)
    if magnitude < 1e-20:
        return list(first)
    scale = 1/math.sqrt(magnitude)
    return [value*scale for value in coefficients]


def angular_distance(a, b):
    if sum(x*y for x, y in zip(a, b)) < 0:
        b = [-value for value in b]
    difference = math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))
    together = math.sqrt(sum((x+y)**2 for x, y in zip(a, b)))
    return 4*math.atan2(difference, together)


def compact(points, limit):
    if type(limit) is not int or not 2 <= limit <= 64:
        raise ValueError('pose capacity must be two to sixty-four')
    if len(points) <= limit:
        return points, 0.0
    selected = [0, len(points)-1]
    while len(selected) < limit:
        best = None
        for first, last in zip(selected, selected[1:]):
            segment = {'poses': [points[first], points[last]], 'samples': points[last]['sample']}
            for index in range(first+1, last):
                estimated = orientation(segment, points[index]['sample'])
                deviation = angular_distance(points[index]['q'], estimated)
                candidate = deviation, -index
                if best is None or candidate > best[0]:
                    best = candidate, index
        if best is None:
            break
        selected.append(best[1])
        selected.sort()
    maximum = 0.0
    for first, last in zip(selected, selected[1:]):
        segment = {'poses': [points[first], points[last]], 'samples': points[last]['sample']}
        for index in range(first+1, last):
            maximum = max(maximum, angular_distance(points[index]['q'], orientation(segment, points[index]['sample'])))
    return [points[index] for index in selected], math.degrees(maximum)


def create(conn, project_id, asset_id, start_ms, end_ms, convention, keyframes, max_poses=64):
    if (type(start_ms) is not int or type(end_ms) is not int or start_ms < 0
            or not 1 <= end_ms-start_ms <= 30000):
        raise ValueError('spatial trim must span one millisecond to thirty seconds')
    if convention not in ('ambix', 'fuma'):
        raise ValueError('input convention must be ambix or fuma')
    asset = media_sources.get(conn, asset_id)
    metadata = json.loads(asset['metadata'])
    stream = next((s for s in metadata['streams'] if s['codec_type'] == 'audio'), {})
    if stream.get('channels') != 4 or float(metadata['format']['duration'])*1000+.001 < end_ms:
        raise ValueError('a four-channel recording covering the trim is required')
    clip = {k: asset[k] for k in ('path', 'fingerprint', 'metadata')}
    clip.update(asset_id=asset_id, start_ms=start_ms, end_ms=end_ms, has_audio=True, gain=1)
    original = poses(keyframes, end_ms-start_ms)
    selected, error = compact(original, max_poses)
    plan = {'version': 1, 'project': timeline.snapshot(conn, project_id)['project'],
            'clips': [clip], 'convention': convention, 'samples': (end_ms-start_ms)*48,
            'poses': selected, 'original_poses': original, 'max_poses': max_poses,
            'max_error_deg': error, 'rate': RATE,
            'output_convention': 'ACN/SN3D', 'listener_axes': 'X-front,Y-left,Z-up'}
    payload = json.dumps(plan, sort_keys=True, separators=(',', ':'), allow_nan=False)
    with db.write(conn):
        cursor = conn.execute('INSERT INTO spatial_plans(project_id,payload,payload_sha256) VALUES (?,?,?)',
                              (project_id, payload, hashlib.sha256(payload.encode()).hexdigest()))
    return get(conn, cursor.lastrowid)


def get(conn, plan_id):
    row = conn.execute('SELECT * FROM spatial_plans WHERE id=?', (plan_id,)).fetchone()
    if row is None or hashlib.sha256(row['payload'].encode()).hexdigest() != row['payload_sha256']:
        raise ValueError('spatial plan is missing or damaged')
    return dict(json.loads(row['payload']), id=plan_id)
