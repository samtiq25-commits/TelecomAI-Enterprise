from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "customer_profiles.csv"


np.random.seed(42)

NUM_CUSTOMERS = 2000


countries = [
    "Spain",
    "United Kingdom",
    "France",
    "Germany",
    "Italy",
]


def generate_profiles():
    customer_ids = [
        f"CUST-{10000 + i}"
        for i in range(NUM_CUSTOMERS)
    ]

    monthly_spend = np.round(
        np.random.lognormal(
            mean=2.3,
            sigma=0.45,
            size=NUM_CUSTOMERS
        ),
        2
    )

    monthly_spend = np.clip(
        monthly_spend,
        5,
        150
    )

    data_usage_gb = np.round(
        np.random.gamma(
            shape=2.5,
            scale=4.0,
            size=NUM_CUSTOMERS
        ),
        2
    )

    data_usage_gb = np.clip(
        data_usage_gb,
        0.5,
        100
    )

    call_minutes = np.round(
        np.random.gamma(
            shape=2.2,
            scale=90,
            size=NUM_CUSTOMERS
        )
    ).astype(int)

    call_minutes = np.clip(
        call_minutes,
        10,
        1500
    )

    complaints = np.random.poisson(
        lam=1.2,
        size=NUM_CUSTOMERS
    )

    complaints = np.clip(
        complaints,
        0,
        10
    )

    tenure_months = np.random.randint(
        1,
        121,
        size=NUM_CUSTOMERS
    )

    country = np.random.choice(
        countries,
        size=NUM_CUSTOMERS,
        p=[0.35, 0.20, 0.18, 0.15, 0.12]
    )

    df = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "monthly_spend": monthly_spend,
            "data_usage_gb": data_usage_gb,
            "call_minutes": call_minutes,
            "complaints": complaints,
            "tenure_months": tenure_months,
            "country": country,
        }
    )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        DATA_PATH,
        index=False
    )

    print(f"Generated {len(df)} customer profiles.")
    print(f"Saved to: {DATA_PATH}")

    print("\nSample:")
    print(df.head())


if __name__ == "__main__":
    generate_profiles()