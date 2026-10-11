from pkg.m07 import *

def test_scale_70():
    assert scale_70([1, 2]) == [3, 6]

def test_above_71():
    assert above_71([24, 25, 26, 30]) == [26, 30]

def test_join_72():
    assert join_72([' a', '', 'b ']) == 'a-b'

def test_clip_73():
    assert clip_73(43) == 40 and clip_73(-3) == 0 and clip_73(1) == 1

def test_window_74():
    assert window_74(list(range(9))) == [list(range(j, j + 7)) for j in range(3)]
