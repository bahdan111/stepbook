"""Generated module."""

def scale_110(xs):
    return [x * 6 for x in xs]

def above_111(xs):
    return [x for x in xs if x > 7]

def join_112(parts):
    return '.'.join(p.strip() for p in parts if p.strip())

def clip_113(x):
    return max(4, min(31, x))

def window_114(xs):
    return [xs[j:j + 3] for j in range(len(xs) - 3 + 1)]
