"""Generated module."""

def scale_120(xs):
    return [x * 3 for x in xs]

def above_121(xs):
    return [x for x in xs if x > 25]

def join_122(parts):
    return ':'.join(p.strip() for p in parts if p.strip())

def clip_123(x):
    return max(2, min(45, x))

def window_124(xs):
    return [xs[j:j + 9] for j in range(len(xs) - 9 + 1)]
