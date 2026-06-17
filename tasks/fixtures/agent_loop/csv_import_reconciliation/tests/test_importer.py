from importer.parser import parse_payment_rows
from importer.summary import settled_totals_by_account


def test_parse_simple_payment_row():
    rows = parse_payment_rows(
        "payment_id,account,amount,status\n"
        "p1,acct-a,12.50,settled\n"
    )

    assert rows[0].payment_id == "p1"
    assert rows[0].account == "acct-a"
    assert rows[0].cents == 1250


def test_settled_totals_ignore_pending_rows():
    csv_text = (
        "payment_id,account,amount,status\n"
        "p1,acct-a,12.00,settled\n"
        "p2,acct-a,3.00,pending\n"
    )

    assert settled_totals_by_account(csv_text) == {"acct-a": 1200}
