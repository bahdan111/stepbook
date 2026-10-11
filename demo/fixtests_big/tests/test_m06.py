from pkg.m06 import *

def test_scale_60():
    assert scale_60([1, 2]) == [3, 6]

def test_above_61():
    assert above_61([38, 39, 40, 44]) == [40, 44]

def test_join_62():
    assert join_62([' a', '', 'b ']) == 'a:b'

def test_clip_63():
    assert clip_63(36) == 33 and clip_63(-6) == -3 and clip_63(-2) == -2

def test_window_64():
    assert window_64(list(range(7))) == [list(range(j, j + 5)) for j in range(3)]
