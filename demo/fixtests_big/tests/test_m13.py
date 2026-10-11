from pkg.m13 import *

def test_scale_130():
    assert scale_130([1, 2]) == [9, 18]

def test_above_131():
    assert above_131([17, 18, 19, 23]) == [19, 23]

def test_join_132():
    assert join_132([' a', '', 'b ']) == 'a.b'

def test_clip_133():
    assert clip_133(35) == 32 and clip_133(-8) == -5 and clip_133(-4) == -4

def test_window_134():
    assert window_134(list(range(9))) == [list(range(j, j + 7)) for j in range(3)]
