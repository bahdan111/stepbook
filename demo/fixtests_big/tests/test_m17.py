from pkg.m17 import *

def test_scale_170():
    assert scale_170([1, 2]) == [2, 4]

def test_above_171():
    assert above_171([3, 4, 5, 9]) == [5, 9]

def test_join_172():
    assert join_172([' a', '', 'b ']) == 'a.b'

def test_clip_173():
    assert clip_173(34) == 31 and clip_173(-1) == 2 and clip_173(3) == 3

def test_window_174():
    assert window_174(list(range(9))) == [list(range(j, j + 7)) for j in range(3)]
