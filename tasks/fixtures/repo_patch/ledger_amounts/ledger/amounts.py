def parse_amount(text):
    return float(text.strip().replace("$", ""))


def total_amounts(rows):
    return sum(parse_amount(row["amount"]) for row in rows)
