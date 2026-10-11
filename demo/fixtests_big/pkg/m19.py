"""Generated module."""

def scale_190(xs):
    return [x * 5 for x in xs]

def above_191(xs):
    return [x for x in xs if x > 21]

def join_192(parts):
    return '.'.join(p.strip() for p in parts if p.strip())

def clip_193(x):
    return max(2, min(11, x))

def window_194(xs):
    return [xs[j:j + 5] for j in range(len(xs) - 5 + 1)]
