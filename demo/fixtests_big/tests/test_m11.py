from pkg.m11 import *

def test_scale_110():
    assert scale_110([1, 2]) == [6, 12]

def test_above_111():
    assert above_111([6, 7, 8, 12]) == [8, 12]

def test_join_112():
    assert join_112([' a', '', 'b ']) == 'a.b'

def test_clip_113():
    assert clip_113(34) == 31 and clip_113(1) == 4 and clip_113(5) == 5

def test_window_114():
    assert window_114(list(range(5))) == [list(range(j, j + 3)) for j in range(3)]
