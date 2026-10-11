from pkg.m18 import *

def test_scale_180():
    assert scale_180([1, 2]) == [4, 8]

def test_above_181():
    assert above_181([21, 22, 23, 27]) == [23, 27]

def test_join_182():
    assert join_182([' a', '', 'b ']) == 'a-b'

def test_clip_183():
    assert clip_183(47) == 44 and clip_183(0) == 3 and clip_183(4) == 4

def test_window_184():
    assert window_184(list(range(7))) == [list(range(j, j + 5)) for j in range(3)]
