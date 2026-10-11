from pkg.m10 import *

def test_scale_100():
    assert scale_100([1, 2]) == [5, 10]

def test_above_101():
    assert above_101([5, 6, 7, 11]) == [7, 11]

def test_join_102():
    assert join_102([' a', '', 'b ']) == 'a-b'

def test_clip_103():
    assert clip_103(21) == 18 and clip_103(0) == 3 and clip_103(4) == 4

def test_window_104():
    assert window_104(list(range(11))) == [list(range(j, j + 9)) for j in range(3)]
