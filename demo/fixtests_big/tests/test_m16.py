from pkg.m16 import *

def test_scale_160():
    assert scale_160([1, 2]) == [3, 6]

def test_above_161():
    assert above_161([37, 38, 39, 43]) == [39, 43]

def test_join_162():
    assert join_162([' a', '', 'b ']) == 'a_b'

def test_clip_163():
    assert clip_163(42) == 39 and clip_163(-6) == -3 and clip_163(-2) == -2

def test_window_164():
    assert window_164(list(range(9))) == [list(range(j, j + 7)) for j in range(3)]
