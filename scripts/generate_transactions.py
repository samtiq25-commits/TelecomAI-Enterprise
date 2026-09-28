import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


random.seed(42)
np.random.seed(42)


def generate_transactions(n=5000):
    countries = [
        "Pakistan",
        "Spain",
        "France",
        "Germany",
        "United Kingdom",
    ]

    payment_types = [
        "Carrier Billing",
        "Credit Card",
        "Mobile Wallet",
    ]

    device_types = [
        "Android",
        "iOS",
        "Web",
    ]

    start_time = datetime.now() - timedelta(days=90)

    rows = []

    for i in range(n):

        timestamp = start_time + timedelta(
            minutes=random.randint(0, 90 * 24 * 60)
        )

        customer_id = f"CUST-{random.randint(10000, 10999)}"

        amount = round(
            random.choice(
                [
                    2.99,
                    4.99,
                    6.99,
                    9.99,
                    14.99,
                    19.99,
                ]
            ),
            2,
        )

        country = random.choice(countries)

        payment_type = random.choice(payment_types)

        device_type = random.choice(device_types)

        previous_transactions = random.randint(0, 30)

        hour = timestamp.hour

        transaction_velocity = random.randint(1, 8)

        is_new_device = random.choice([0, 1])

        # -------------------------
        # Fraud generation
        # -------------------------

        fraud = 0

        risk_signals = 0

        if transaction_velocity >= 8:
            risk_signals += 1

        if is_new_device and transaction_velocity >= 7:
            risk_signals += 1

        if hour <= 3 and transaction_velocity >= 7:
            risk_signals += 1

        if amount >= 19.99 and transaction_velocity >= 7:
            risk_signals += 1

        if previous_transactions <= 2 and transaction_velocity >= 7:
            risk_signals += 1

        # Multiple suspicious signals
        # indicate a stronger fraud pattern.
        if risk_signals >= 2:
            fraud = 1

        # Small background fraud rate.
        if random.random() < 0.01:
            fraud = 1

        # -------------------------
        # Store transaction
        # -------------------------

        rows.append(
            {
                "transaction_id": f"TX-{100000 + i}",
                "timestamp": timestamp,
                "customer_id": customer_id,
                "amount": amount,
                "country": country,
                "payment_type": payment_type,
                "device_type": device_type,
                "previous_transactions": previous_transactions,
                "transaction_velocity": transaction_velocity,
                "is_new_device": is_new_device,
                "hour": hour,
                "fraud": fraud,
            }
        )

    return pd.DataFrame(rows)
if __name__ == "__main__":
    output_path = Path("data/transactions.csv")

    df = generate_transactions()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Generated {len(df)} transactions."
    )

    print(
        f"Saved to: {output_path}"
    )

    print()

    print("Fraud distribution:")

    print(
        df["fraud"].value_counts()
    )    