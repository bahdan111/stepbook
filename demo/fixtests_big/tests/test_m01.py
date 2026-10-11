from pkg.m01 import *

def test_scale_10():
    assert scale_10([1, 2]) == [6, 12]

def test_above_11():
    assert above_11([7, 8, 9, 13]) == [9, 13]

def test_join_12():
    assert join_12([' a', '', 'b ']) == 'a-b'

def test_clip_13():
    assert clip_13(35) == 32 and clip_13(-6) == -3 and clip_13(-2) == -2

def test_window_14():
    assert window_14(list(range(8))) == [list(range(j, j + 6)) for j in range(3)]
