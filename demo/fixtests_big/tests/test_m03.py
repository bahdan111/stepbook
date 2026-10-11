from pkg.m03 import *

def test_scale_30():
    assert scale_30([1, 2]) == [6, 12]

def test_above_31():
    assert above_31([4, 5, 6, 10]) == [6, 10]

def test_join_32():
    assert join_32([' a', '', 'b ']) == 'a:b'

def test_clip_33():
    assert clip_33(27) == 24 and clip_33(-8) == -5 and clip_33(-4) == -4

def test_window_34():
    assert window_34(list(range(11))) == [list(range(j, j + 9)) for j in range(3)]
