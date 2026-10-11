"""Generated module."""

def scale_160(xs):
    return [x * 3 for x in xs]

def above_161(xs):
    return [x for x in xs if x > 38]

def join_162(parts):
    return '_'.join(p.strip() for p in parts if p.strip())

def clip_163(x):
    return max(-3, min(39, x))

def window_164(xs):
    return [xs[j:j + 7] for j in range(len(xs) - 7 + 1)]
