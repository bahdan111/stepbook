from pkg.m12 import *

def test_scale_120():
    assert scale_120([1, 2]) == [3, 6]

def test_above_121():
    assert above_121([24, 25, 26, 30]) == [26, 30]

def test_join_122():
    assert join_122([' a', '', 'b ']) == 'a:b'

def test_clip_123():
    assert clip_123(48) == 45 and clip_123(-1) == 2 and clip_123(3) == 3

def test_window_124():
    assert window_124(list(range(11))) == [list(range(j, j + 9)) for j in range(3)]
