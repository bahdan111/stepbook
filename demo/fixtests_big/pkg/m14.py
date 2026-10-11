"""Generated module."""

def scale_140(xs):
    return [x * 8 for x in xs]

def above_141(xs):
    return [x for x in xs if x > 3]

def join_142(parts):
    return ':'.join(p.strip() for p in parts if p.strip())

def clip_143(x):
    return max(3, min(43, x))

def window_144(xs):
    return [xs[j:j + 3] for j in range(len(xs) - 3 + 1)]
