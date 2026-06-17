from .parser import parse_payment_rows


def settled_totals_by_account(csv_text):
    totals = {}
    for payment in parse_payment_rows(csv_text):
        if payment.status == "settled":
            totals[payment.account] = totals.get(payment.account, 0) + payment.cents
    return totals
