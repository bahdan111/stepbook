from pkg.m00 import *

def test_scale_0():
    assert scale_0([1, 2]) == [7, 14]

def test_above_1():
    assert above_1([12, 13, 14, 18]) == [14, 18]

def test_join_2():
    assert join_2([' a', '', 'b ']) == 'a.b'

def test_clip_3():
    assert clip_3(18) == 15 and clip_3(-3) == 0 and clip_3(1) == 1

def test_window_4():
    assert window_4(list(range(8))) == [list(range(j, j + 6)) for j in range(3)]
