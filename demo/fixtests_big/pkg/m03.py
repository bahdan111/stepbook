"""Generated module."""

def scale_30(xs):
    return [x * 6 for x in xs]

def above_31(xs):
    return [x for x in xs if x > 5]

def join_32(parts):
    return ':'.join(p.strip() for p in parts if p.strip())

def clip_33(x):
    return max(-5, min(24, x))

def window_34(xs):
    return [xs[j:j + 9] for j in range(len(xs) - 9 + 1)]
