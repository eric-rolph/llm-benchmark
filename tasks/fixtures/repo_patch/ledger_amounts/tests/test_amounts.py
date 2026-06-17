from ledger.amounts import parse_amount, total_amounts


def test_parse_simple_currency_amounts():
    assert parse_amount("$1.50") == 1.5
    assert parse_amount("2.25") == 2.25


def test_total_amounts_sums_rows():
    rows = [{"amount": "$1.00"}, {"amount": "$2.50"}]
    assert total_amounts(rows) == 3.5
