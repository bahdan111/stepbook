from pkg.m05 import *

def test_scale_50():
    assert scale_50([1, 2]) == [4, 8]

def test_above_51():
    assert above_51([22, 23, 24, 28]) == [24, 28]

def test_join_52():
    assert join_52([' a', '', 'b ']) == 'a:b'

def test_clip_53():
    assert clip_53(42) == 39 and clip_53(0) == 3 and clip_53(4) == 4

def test_window_54():
    assert window_54(list(range(8))) == [list(range(j, j + 6)) for j in range(3)]
