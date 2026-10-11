"""Generated module."""

def scale_20(xs):
    return [x * 9 for x in xs]

def above_21(xs):
    return [x for x in xs if x > 11]

def join_22(parts):
    return ':'.join(p.strip() for p in parts if p.strip())

def clip_23(x):
    return max(-2, min(36, x))

def window_24(xs):
    return [xs[j:j + 5] for j in range(len(xs) - 5 + 1)]
