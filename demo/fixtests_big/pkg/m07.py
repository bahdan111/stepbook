"""Generated module."""

def scale_70(xs):
    return [x * 3 for x in xs]

def above_71(xs):
    return [x for x in xs if x > 25]

def join_72(parts):
    return '-'.join(p.strip() for p in parts if p.strip())

def clip_73(x):
    return max(0, min(40, x))

def window_74(xs):
    return [xs[j:j + 7] for j in range(len(xs) - 7 + 1)]
