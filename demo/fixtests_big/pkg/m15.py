"""Generated module."""

def scale_150(xs):
    return [x * 4 for x in xs]

def above_151(xs):
    return [x for x in xs if x > 36]

def join_152(parts):
    return '.'.join(p.strip() for p in parts if p.strip())

def clip_153(x):
    return max(-4, min(12, x))

def window_154(xs):
    return [xs[j:j + 5] for j in range(len(xs) - 5 + 1)]
