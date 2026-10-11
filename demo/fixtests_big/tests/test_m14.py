from pkg.m14 import *

def test_scale_140():
    assert scale_140([1, 2]) == [8, 16]

def test_above_141():
    assert above_141([2, 3, 4, 8]) == [4, 8]

def test_join_142():
    assert join_142([' a', '', 'b ']) == 'a:b'

def test_clip_143():
    assert clip_143(46) == 43 and clip_143(0) == 3 and clip_143(4) == 4

def test_window_144():
    assert window_144(list(range(5))) == [list(range(j, j + 3)) for j in range(3)]
