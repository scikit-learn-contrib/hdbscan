import pytest

from hdbscan._prediction_utils import safe_always_positive_division


@pytest.mark.parameterize('numerator, denominator')
def test_safe_always_positive_division(numerator, denominator):
    # Given negative, zero and positive denominator and positive numerator
    value = safe_always_positive_division(1, 0)
    # Make sure safe division is always positive and doesn't raise ZeroDivision error
    assert value >= 0
