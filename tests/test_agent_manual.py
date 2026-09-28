import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_generator import load_data
from src.ml_models import detect_anomalies
from src.agent import build_agent


network, _, _ = load_data()

anomalies = detect_anomalies(network)

recent = (
    anomalies[anomalies.tower_id == "TWR-1024"]
    .sort_values("timestamp")
    .tail(1)
)

anomaly = recent.iloc[0].to_dict() if not recent.empty else {}

result = build_agent().invoke(
    {
        "issue": "Internet is very slow and packet loss appears high.",
        "tower_id": "TWR-1024",
        "anomaly": anomaly,
    },
    config={
        "configurable": {
            "thread_id": "test-TWR-1024"
        }
    }
)


print("\n========== AGENT TEST ==========\n")

print("Issue:")
print(result.get("issue"))

print("\nTower:")
print(result.get("tower_id"))

print("\nKPI Status:")
print(result.get("kpi_status"))

print("\nInvestigation:")
print(result.get("investigation"))

print("\nDiagnosis:")
print(result.get("diagnosis"))

print("\nRoot Cause:")
print(result.get("root_cause"))

print("\nPossible Causes:")
for cause in result.get("possible_causes", []):
    print(f"• {cause}")

print("\nRoot Cause Confidence:")
print(result.get("root_cause_confidence"))

print("\nRecommendation:")
print(result.get("recommendation"))

print("\nExplanation:")
print(result.get("explanation"))

print("\nApproval:")
print("HUMAN ENGINEER REVIEW REQUIRED")
print("\nIncident Report:")
print(result.get("incident_report"))
print("\nVerification Status:")
print(result.get("verification_status"))

print("\nVerification Action:")
print(result.get("verification_action"))
print("\nInvestigation Attempts:")
print(result.get("investigation_attempts"))