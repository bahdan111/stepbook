"""Generated module."""

def scale_0(xs):
    return [x * 7 for x in xs]

def above_1(xs):
    return [x for x in xs if x > 13]

def join_2(parts):
    return '.'.join(p.strip() for p in parts if p.strip())

def clip_3(x):
    return max(0, min(15, x))

def window_4(xs):
    return [xs[j:j + 6] for j in range(len(xs) - 6 + 1)]
