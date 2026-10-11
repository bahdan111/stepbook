"""Generated module."""

def scale_180(xs):
    return [x * 4 for x in xs]

def above_181(xs):
    return [x for x in xs if x > 22]

def join_182(parts):
    return '-'.join(p.strip() for p in parts if p.strip())

def clip_183(x):
    return max(3, min(44, x))

def window_184(xs):
    return [xs[j:j + 5] for j in range(len(xs) - 5 + 1)]
