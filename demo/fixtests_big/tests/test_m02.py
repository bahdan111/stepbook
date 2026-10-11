from pkg.m02 import *

def test_scale_20():
    assert scale_20([1, 2]) == [9, 18]

def test_above_21():
    assert above_21([10, 11, 12, 16]) == [12, 16]

def test_join_22():
    assert join_22([' a', '', 'b ']) == 'a:b'

def test_clip_23():
    assert clip_23(39) == 36 and clip_23(-5) == -2 and clip_23(-1) == -1

def test_window_24():
    assert window_24(list(range(7))) == [list(range(j, j + 5)) for j in range(3)]
