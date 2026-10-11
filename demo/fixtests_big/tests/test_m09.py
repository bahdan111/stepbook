from pkg.m09 import *

def test_scale_90():
    assert scale_90([1, 2]) == [3, 6]

def test_above_91():
    assert above_91([24, 25, 26, 30]) == [26, 30]

def test_join_92():
    assert join_92([' a', '', 'b ']) == 'a.b'

def test_clip_93():
    assert clip_93(41) == 38 and clip_93(0) == 3 and clip_93(4) == 4

def test_window_94():
    assert window_94(list(range(6))) == [list(range(j, j + 4)) for j in range(3)]
