import unittest

from toy_math import add


class AddTests(unittest.TestCase):
    def test_add_two_positive_numbers(self):
        self.assertEqual(add(2, 3), 5)
