from calc.stats import mean


def test_mean_integer_values():
    assert mean([2, 4, 6]) == 4
