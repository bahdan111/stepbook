"""Generated module."""

def scale_100(xs):
    return [x * 5 for x in xs]

def above_101(xs):
    return [x for x in xs if x > 6]

def join_102(parts):
    return '-'.join(p.strip() for p in parts if p.strip())

def clip_103(x):
    return max(3, min(18, x))

def window_104(xs):
    return [xs[j:j + 9] for j in range(len(xs) - 9 + 1)]
