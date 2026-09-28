from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "customer_segments.csv"


def load_customer_segments():
    return pd.read_csv(DATA_PATH)


def generate_marketing_recommendations(customer):
    recommendations = []

    segment = customer.get("segment_name", "Regular")

    monthly_spend = float(customer.get("monthly_spend", 0))
    data_usage = float(customer.get("data_usage_gb", 0))
    call_minutes = int(customer.get("call_minutes", 0))
    complaints = int(customer.get("complaints", 0))
    tenure = int(customer.get("tenure_months", 0))

    # Retention should take priority over promotional campaigns.
    if segment == "At Risk":
        recommendations.append(
            {
                "campaign": "Retention Offer",
                "reason": "Customer is classified as At Risk.",
                "priority": "HIGH",
            }
        )

        if complaints >= 2:
            recommendations.append(
                {
                    "campaign": "Service Recovery",
                    "reason": "Customer has multiple complaints.",
                    "priority": "CRITICAL",
                }
            )

        return recommendations

    # New customers receive onboarding campaigns.
    if segment == "New Customer":
        recommendations.append(
            {
                "campaign": "Welcome & Onboarding",
                "reason": "Customer is relatively new.",
                "priority": "MEDIUM",
            }
        )

    # High data usage indicates a possible data upgrade opportunity.
    if data_usage >= 15:
        recommendations.append(
            {
                "campaign": "Data Upgrade",
                "reason": "High monthly data usage.",
                "priority": "HIGH",
            }
        )

    # High call usage indicates a possible voice upgrade.
    if call_minutes >= 400:
        recommendations.append(
            {
                "campaign": "Voice Upgrade",
                "reason": "High monthly voice usage.",
                "priority": "HIGH",
            }
        )

    # High-value customers are candidates for premium services.
    if segment == "High Value":
        recommendations.append(
            {
                "campaign": "Premium Plan",
                "reason": "Customer is classified as High Value.",
                "priority": "HIGH",
            }
        )

    # Long-term customers can receive loyalty campaigns.
    if tenure >= 60:
        recommendations.append(
            {
                "campaign": "Loyalty Offer",
                "reason": "Long-term customer relationship.",
                "priority": "MEDIUM",
            }
        )

    # Regular customers can receive general offers.
    if (
        segment == "Regular"
        and not recommendations
        and monthly_spend >= 10
    ):
        recommendations.append(
            {
                "campaign": "Engagement Offer",
                "reason": "Regular customer with established spending behavior.",
                "priority": "MEDIUM",
            }
        )

    if not recommendations:
        recommendations.append(
            {
                "campaign": "No Campaign",
                "reason": "No strong campaign opportunity detected.",
                "priority": "LOW",
            }
        )

    return recommendations


if __name__ == "__main__":
    customers = load_customer_segments()

    print("Marketing Intelligence")
    print("=" * 40)

    for _, customer in customers.head(10).iterrows():
        recommendations = generate_marketing_recommendations(
            customer.to_dict()
        )

        print(f"\nCustomer: {customer['customer_id']}")
        print(f"Segment: {customer['segment_name']}")

        for recommendation in recommendations:
            print(
                f"• {recommendation['campaign']} "
                f"| {recommendation['priority']} "
                f"| {recommendation['reason']}"
            )