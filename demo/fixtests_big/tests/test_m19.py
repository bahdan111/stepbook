from pkg.m19 import *

def test_scale_190():
    assert scale_190([1, 2]) == [5, 10]

def test_above_191():
    assert above_191([20, 21, 22, 26]) == [22, 26]

def test_join_192():
    assert join_192([' a', '', 'b ']) == 'a.b'

def test_clip_193():
    assert clip_193(14) == 11 and clip_193(-1) == 2 and clip_193(3) == 3

def test_window_194():
    assert window_194(list(range(7))) == [list(range(j, j + 5)) for j in range(3)]
