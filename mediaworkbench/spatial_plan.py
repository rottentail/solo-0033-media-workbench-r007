"""Freeze listener poses for first-order soundfields on a continuous sample clock."""
import hashlib
import json
import math
from . import db, media_sources, timeline

RATE = 48000

# Half-turn roundoff tie, in the absolute dot product between unit quaternions.
HALF_TURN_EPS = 1e-15
# Subset choices whose worst error is within this many degrees of the optimum
# are considered tied, so lexicographic order decides between them.
ERROR_TIE_DEG = 1e-9


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


def slerp(first, last, phase):
    """Shortest spherical path at constant angular speed between unit poses."""
    dot = sum(a*b for a, b in zip(first, last))
    if dot < 0:
        # Equivalent orientation, opposite sign: take the short way round.
        last = [-v for v in last]
        dot = -dot
    if dot >= 1.0:
        return list(first)
    if abs(dot) <= HALF_TURN_EPS:
        # Exact half-turn: the canonical endpoint signs fix the rotation plane.
        angle = math.pi/2
        sin_angle = 1.0
    else:
        dot = min(1.0, max(-1.0, dot))
        angle = math.acos(dot)
        sin_angle = math.sin(angle)
    scale_first = math.sin((1-phase)*angle)/sin_angle
    scale_last = math.sin(phase*angle)/sin_angle
    return [a*scale_first + b*scale_last for a, b in zip(first, last)]


def segment_index(points, sample):
    """Index of the left keyframe of the span containing sample on the clock."""
    if not points[0]['sample'] <= sample <= points[-1]['sample']:
        raise ValueError('pose outside programme')
    lo, hi = 0, len(points)-1
    while hi-lo > 1:
        mid = (lo+hi)//2
        if sample < points[mid]['sample']:
            hi = mid
        else:
            lo = mid
    return lo


def segment(points, sample):
    """Return the (left, right) keyframes straddling sample on the global clock."""
    lo = segment_index(points, sample)
    return points[lo], points[min(lo+1, len(points)-1)]


def orientation(plan, sample):
    points = plan['poses']
    if not 0 <= sample <= plan['samples']:
        raise ValueError('pose outside programme')
    if sample == points[-1]['sample']:
        return list(points[-1]['q'])
    left, right = segment(points, sample)
    width = right['sample']-left['sample']
    phase = (sample-left['sample'])/width
    return slerp(unit(left['q']), unit(right['q']), phase)


def angular_distance(a, b):
    """Rotation angle (radians) between two poses, independent of quaternion sign."""
    dot = abs(sum(x*y for x, y in zip(a, b)))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(x*x for x in b))
    cos_half = dot/(norm_a*norm_b)
    if cos_half >= 1.0:
        return 0.0
    # atan2 keeps the result precise for near-identical poses where acos loses
    # all accuracy to roundoff.
    return 2*math.atan2(math.sqrt(max(0.0, 1.0-cos_half*cos_half)), cos_half)


def _segment_cost(points, first, last):
    """Greatest angular error of observations strictly inside one retained span."""
    if last <= first+1:
        return 0.0
    left, right = points[first], points[last]
    width = right['sample']-left['sample']
    worst = 0.0
    for index in range(first+1, last):
        phase = (points[index]['sample']-left['sample'])/width
        estimated = slerp(left['q'], right['q'], phase)
        worst = max(worst, angular_distance(points[index]['q'], estimated))
    return worst


def _cost_table(points):
    size = len(points)
    table = [[0.0]*size for _ in range(size)]
    for first in range(size):
        for last in range(first+1, size):
            table[first][last] = _segment_cost(points, first, last)
    return table


def _feasible(cost, max_spans, tolerance):
    """reachable[k][j]: node j reaches the end using at most k more spans."""
    size = len(cost)
    reachable = [[False]*size for _ in range(max_spans+1)]
    for budget in range(max_spans+1):
        reachable[budget][size-1] = True      # terminal node needs no span
    for budget in range(1, max_spans+1):
        for node in range(size-1):
            for nxt in range(node+1, size):
                if cost[node][nxt] <= tolerance and reachable[budget-1][nxt]:
                    reachable[budget][node] = True
                    break
    return reachable


def _lexicographic_path(cost, max_spans, tolerance):
    """Smallest retained-index tuple whose spans all stay within tolerance."""
    reachable = _feasible(cost, max_spans, tolerance)
    end = len(cost)-1
    path = [0]
    budget = max_spans
    while path[-1] != end:
        node = path[-1]
        chosen = None
        for nxt in range(node+1, end+1):
            if cost[node][nxt] <= tolerance and reachable[budget-1][nxt]:
                chosen = nxt
                break
        if chosen is None:
            raise ValueError('pose compression has no feasible subset')
        path.append(chosen)
        budget -= 1
    return path


def compact(points, limit):
    if type(limit) is not int or not 2 <= limit <= 64:
        raise ValueError('pose capacity must be two to sixty-four')
    if len(points) <= limit:
        return points, 0.0
    # The device accepts `limit` poses, i.e. at most limit-1 interpolated spans.
    max_spans = limit-1
    cost = _cost_table(points)
    size = len(points)
    # Exact minimax dynamic program. best[k][j] is the smallest achievable
    # worst span error when node j may use at most k spans to reach the end.
    best = [[math.inf]*size for _ in range(max_spans+1)]
    best[0][size-1] = 0.0
    for budget in range(1, max_spans+1):
        best[budget][size-1] = 0.0
        for node in range(size-2, -1, -1):
            value = math.inf
            for nxt in range(node+1, size):
                candidate = max(cost[node][nxt], best[budget-1][nxt])
                value = min(value, candidate)
            best[budget][node] = value
    optimum = best[max_spans][0]
    tolerance = optimum + math.radians(ERROR_TIE_DEG)
    indices = _lexicographic_path(cost, max_spans, tolerance)
    maximum = max(cost[a][b] for a, b in zip(indices, indices[1:]))
    return [points[index] for index in indices], math.degrees(maximum)


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
