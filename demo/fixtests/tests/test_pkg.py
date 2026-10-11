import pytest

from pkg import (Account, chunk, clamp, fib, is_palindrome, mean, median, moving_average, parse_kv, percentile,
                 roman, slugify)


def test_clamp_inside(): assert clamp(5, 0, 10) == 5
def test_clamp_low(): assert clamp(-1, 0, 10) == 0
def test_clamp_high(): assert clamp(11, 0, 10) == 10
def test_clamp_bad_bounds():
    with pytest.raises(ValueError): clamp(1, 5, 0)
def test_mean(): assert mean([1, 2, 3, 4]) == 2.5
def test_mean_empty():
    with pytest.raises(ValueError): mean([])
def test_median_odd(): assert median([3, 1, 2]) == 2
def test_median_even(): assert median([4, 1, 3, 2]) == 2.5
def test_moving_average(): assert moving_average([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]
def test_moving_average_k(): 
    with pytest.raises(ValueError): moving_average([1], 0)
def test_parse_kv(): assert parse_kv("a = 1\n# c\nb=x") == {"a": "1", "b": "x"}
def test_parse_kv_blank(): assert parse_kv("\n\n") == {}
def test_slugify(): assert slugify("Hello, World!") == "hello-world"
def test_slugify_collapse(): assert slugify("a  --  b") == "a-b"
def test_chunk(): assert chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
def test_chunk_bad():
    with pytest.raises(ValueError): chunk([1], 0)
def test_percentile_median(): assert percentile([1, 2, 3, 4, 5], 50) == 3
def test_percentile_interp(): assert percentile([1, 2, 3, 4], 50) == 2.5
def test_percentile_range():
    with pytest.raises(ValueError): percentile([1], 101)
def test_account_deposit():
    a = Account(); a.deposit(10); assert a.balance == 10
def test_account_withdraw():
    a = Account(20); a.withdraw(5); assert a.balance == 15
def test_account_insufficient():
    with pytest.raises(ValueError): Account(1).withdraw(5)
def test_account_negative():
    with pytest.raises(ValueError): Account().deposit(-1)
def test_fib(): assert [fib(i) for i in range(8)] == [0, 1, 1, 2, 3, 5, 8, 13]
def test_fib_negative():
    with pytest.raises(ValueError): fib(-1)
def test_palindrome(): assert is_palindrome("A man, a plan, a canal: Panama")
def test_not_palindrome(): assert not is_palindrome("hello")
def test_roman(): assert roman(1994) == "MCMXCIV"
def test_roman_small(): assert roman(4) == "IV"
def test_roman_range():
    with pytest.raises(ValueError): roman(0)
