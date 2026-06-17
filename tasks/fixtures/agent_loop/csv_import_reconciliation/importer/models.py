from dataclasses import dataclass


@dataclass(frozen=True)
class Payment:
    payment_id: str
    account: str
    cents: int
    status: str
