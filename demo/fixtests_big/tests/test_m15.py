from pkg.m15 import *

def test_scale_150():
    assert scale_150([1, 2]) == [4, 8]

def test_above_151():
    assert above_151([35, 36, 37, 41]) == [37, 41]

def test_join_152():
    assert join_152([' a', '', 'b ']) == 'a.b'

def test_clip_153():
    assert clip_153(15) == 12 and clip_153(-7) == -4 and clip_153(-3) == -3

def test_window_154():
    assert window_154(list(range(7))) == [list(range(j, j + 5)) for j in range(3)]
