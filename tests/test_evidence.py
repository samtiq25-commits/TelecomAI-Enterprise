import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_generator import load_data
from src.ml_models import detect_anomalies
from src.agent import collect_additional_evidence


network, _, _ = load_data()

anomalies = detect_anomalies(network)

recent = (
    anomalies[
        anomalies.tower_id == "TWR-1024"
    ]
    .sort_values("timestamp")
    .tail(1)
)

anomaly = recent.iloc[0].to_dict() if not recent.empty else {}

result = collect_additional_evidence(
    {
        "tower_id": "TWR-1024",
        "anomaly": anomaly,
    }
)

print("\n========== ADDITIONAL EVIDENCE TEST ==========\n")

for evidence in result["additional_evidence"]:
    print(f"• {evidence}")