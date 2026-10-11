from pkg.m08 import *

def test_scale_80():
    assert scale_80([1, 2]) == [6, 12]

def test_above_81():
    assert above_81([31, 32, 33, 37]) == [33, 37]

def test_join_82():
    assert join_82([' a', '', 'b ']) == 'a:b'

def test_clip_83():
    assert clip_83(33) == 30 and clip_83(-3) == 0 and clip_83(1) == 1

def test_window_84():
    assert window_84(list(range(11))) == [list(range(j, j + 9)) for j in range(3)]
