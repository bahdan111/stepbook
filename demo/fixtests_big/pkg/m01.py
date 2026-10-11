"""Generated module."""

def scale_10(xs):
    return [x * 6 for x in xs]

def above_11(xs):
    return [x for x in xs if x > 8]

def join_12(parts):
    return '-'.join(p.strip() for p in parts if p.strip())

def clip_13(x):
    return max(-3, min(32, x))

def window_14(xs):
    return [xs[j:j + 6] for j in range(len(xs) - 6 + 1)]
