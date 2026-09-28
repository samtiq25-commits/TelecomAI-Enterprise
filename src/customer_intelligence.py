from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "customer_profiles.csv"
SEGMENTED_DATA_PATH = BASE_DIR / "data" / "customer_segments.csv"

FEATURES = [
    "monthly_spend",
    "data_usage_gb",
    "call_minutes",
    "complaints",
    "tenure_months",
]


def load_customer_profiles():
    return pd.read_csv(DATA_PATH)


def segment_customers(n_clusters=4):
    df = load_customer_profiles().copy()

    scaler = StandardScaler()

    X = scaler.fit_transform(
        df[FEATURES]
    )

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )

    df["segment"] = model.fit_predict(X)

    return df

def describe_segments(customers):
    summary = (
        customers.groupby("segment")
        .agg(
            customers=("customer_id", "count"),
            avg_monthly_spend=("monthly_spend", "mean"),
            avg_data_usage_gb=("data_usage_gb", "mean"),
            avg_call_minutes=("call_minutes", "mean"),
            avg_complaints=("complaints", "mean"),
            avg_tenure_months=("tenure_months", "mean"),
        )
        .round(2)
        .reset_index()
    )

    labels = {}

    for _, row in summary.iterrows():
        if (
            row["avg_monthly_spend"] >= summary["avg_monthly_spend"].median()
            and row["avg_data_usage_gb"] >= summary["avg_data_usage_gb"].median()
        ):
            label = "High Value"

        elif row["avg_complaints"] >= summary["avg_complaints"].median():
            label = "At Risk"

        elif row["avg_tenure_months"] <= summary["avg_tenure_months"].median():
            label = "New Customer"

        else:
            label = "Regular"

        labels[row["segment"]] = label

    summary["segment_name"] = summary["segment"].map(labels)

    return summary, labels    

def save_customer_segments(customers):
    SEGMENTED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    customers.to_csv(
        SEGMENTED_DATA_PATH,
        index=False
    )

    print(
        f"Saved customer segments to: {SEGMENTED_DATA_PATH}"
    )    
def generate_personalized_recommendation(customer):
    recommendations = []

    segment = customer.get("segment_name", "Regular")

    monthly_spend = float(customer.get("monthly_spend", 0))
    data_usage = float(customer.get("data_usage_gb", 0))
    call_minutes = int(customer.get("call_minutes", 0))
    complaints = int(customer.get("complaints", 0))
    tenure = int(customer.get("tenure_months", 0))

    # Segment-based recommendation
    if segment == "High Value":
        recommendations.append(
            "Offer a premium plan or exclusive service benefits."
        )

    elif segment == "At Risk":
        recommendations.append(
            "Prioritize retention with a personalized support offer."
        )

        if complaints >= 2:
            recommendations.append(
                "Follow up on recent complaints before promoting new services."
            )

    elif segment == "New Customer":
        recommendations.append(
            "Recommend an onboarding offer based on early usage behavior."
        )

    else:
        recommendations.append(
            "Recommend plans or services based on current usage patterns."
        )

    # Usage-based recommendations
    if data_usage >= 15:
        recommendations.append(
            "Consider a higher-data plan because of heavy data usage."
        )

    if call_minutes >= 400:
        recommendations.append(
            "Consider a voice-focused plan because of high call usage."
        )

    # Only recommend affordability for customers who are
    # not already classified as High Value.
    if monthly_spend < 10 and segment != "High Value":
        recommendations.append(
            "Consider an affordable upgrade or value-focused package."
        )

    # Loyalty recommendation
    if tenure >= 60:
        recommendations.append(
            "Consider a loyalty benefit for long-term engagement."
        )

    return recommendations
    recommendations = []

    segment = customer.get("segment_name", "Regular")

    monthly_spend = float(customer.get("monthly_spend", 0))
    data_usage = float(customer.get("data_usage_gb", 0))
    call_minutes = float(customer.get("call_minutes", 0))
    complaints = int(customer.get("complaints", 0))
    tenure = int(customer.get("tenure_months", 0))

    if segment == "High Value":
        recommendations.append(
            "Offer a premium plan or exclusive service benefits."
        )

    elif segment == "At Risk":
        recommendations.append(
            "Prioritize retention with a personalized support offer."
        )

        if complaints >= 2:
            recommendations.append(
                "Follow up on recent complaints before promoting new services."
            )

    elif segment == "New Customer":
        recommendations.append(
            "Recommend an onboarding offer based on early usage behavior."
        )

    else:
        recommendations.append(
            "Recommend plans or services based on current usage patterns."
        )

    if data_usage >= 15:
        recommendations.append(
            "Consider a higher-data plan because of heavy data usage."
        )

    if call_minutes >= 400:
        recommendations.append(
            "Consider a voice-focused plan because of high call usage."
        )

    if monthly_spend < 10:
        recommendations.append(
            "Consider an affordable upgrade or value-focused package."
        )

    if tenure >= 60:
        recommendations.append(
            "Consider a loyalty benefit for long-term engagement."
        )

    return recommendations

if __name__ == "__main__":
    customers = segment_customers()

    summary, labels = describe_segments(customers)

    customers["segment_name"] = customers["segment"].map(labels)

    save_customer_segments(customers)

    print("\nCustomer Segmentation")
    print("=" * 40)

    print(summary.to_string(index=False))

    print("\nSegment Labels:")
    for segment, label in labels.items():
        print(f"Segment {segment}: {label}")

    print("\nPersonalized Recommendations")
    print("=" * 40)

    for _, customer in customers.head(5).iterrows():
        recommendations = generate_personalized_recommendation(
            customer.to_dict()
        )

        print(f"\nCustomer: {customer['customer_id']}")
        print(f"Segment: {customer['segment_name']}")

        for recommendation in recommendations:
            print(f"• {recommendation}")