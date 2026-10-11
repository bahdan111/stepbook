from pkg.m04 import *

def test_scale_40():
    assert scale_40([1, 2]) == [9, 18]

def test_above_41():
    assert above_41([39, 40, 41, 45]) == [41, 45]

def test_join_42():
    assert join_42([' a', '', 'b ']) == 'a-b'

def test_clip_43():
    assert clip_43(31) == 28 and clip_43(0) == 3 and clip_43(4) == 4

def test_window_44():
    assert window_44(list(range(9))) == [list(range(j, j + 7)) for j in range(3)]
