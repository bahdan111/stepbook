"""Generated module."""

def scale_170(xs):
    return [x * 2 for x in xs]

def above_171(xs):
    return [x for x in xs if x > 4]

def join_172(parts):
    return '.'.join(p.strip() for p in parts if p.strip())

def clip_173(x):
    return max(2, min(31, x))

def window_174(xs):
    return [xs[j:j + 7] for j in range(len(xs) - 7 + 1)]
