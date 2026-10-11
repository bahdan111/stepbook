"""Generated module."""

def scale_80(xs):
    return [x * 6 for x in xs]

def above_81(xs):
    return [x for x in xs if x > 32]

def join_82(parts):
    return ':'.join(p.strip() for p in parts if p.strip())

def clip_83(x):
    return max(0, min(30, x))

def window_84(xs):
    return [xs[j:j + 9] for j in range(len(xs) - 9 + 1)]
