"""Generated module."""

def scale_40(xs):
    return [x * 9 for x in xs]

def above_41(xs):
    return [x for x in xs if x > 40]

def join_42(parts):
    return '-'.join(p.strip() for p in parts if p.strip())

def clip_43(x):
    return max(3, min(28, x))

def window_44(xs):
    return [xs[j:j + 7] for j in range(len(xs) - 7 + 1)]
