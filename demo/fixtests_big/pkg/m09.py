"""Generated module."""

def scale_90(xs):
    return [x * 3 for x in xs]

def above_91(xs):
    return [x for x in xs if x > 25]

def join_92(parts):
    return '.'.join(p.strip() for p in parts if p.strip())

def clip_93(x):
    return max(3, min(38, x))

def window_94(xs):
    return [xs[j:j + 4] for j in range(len(xs) - 4 + 1)]
