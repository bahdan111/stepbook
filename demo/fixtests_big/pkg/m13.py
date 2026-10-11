"""Generated module."""

def scale_130(xs):
    return [x * 9 for x in xs]

def above_131(xs):
    return [x for x in xs if x > 18]

def join_132(parts):
    return '.'.join(p.strip() for p in parts if p.strip())

def clip_133(x):
    return max(-5, min(32, x))

def window_134(xs):
    return [xs[j:j + 7] for j in range(len(xs) - 7 + 1)]
