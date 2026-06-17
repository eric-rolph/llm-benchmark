from .models import Payment


def parse_payment_rows(csv_text):
    lines = [line for line in csv_text.strip().splitlines() if line.strip()]
    if not lines:
        return []

    payments = []
    for line in lines[1:]:
        payment_id, account, amount, status = line.split(",")
        cents = int(round(float(amount) * 100))
        payments.append(Payment(payment_id, account, cents, status))
    return payments
