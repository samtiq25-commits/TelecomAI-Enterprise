import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_generator import load_data
from src.ml_models import detect_anomalies
from src.agent import build_agent


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

if "timestamp" in anomaly:
    anomaly["timestamp"] = str(anomaly["timestamp"])


agent = build_agent()

config = {
    "configurable": {
        "thread_id": "recovery-test-TWR-1024"
    }
}


print("\n========== RUNNING AGENT ==========\n")

result = agent.invoke(
    {
        "issue": "Internet is very slow and packet loss appears high.",
        "tower_id": "TWR-1024",
        "anomaly": anomaly,
    },
    config=config,
)


print("Initial Root Cause:")
print(result.get("root_cause"))

print("\nInitial Confidence:")
print(result.get("root_cause_confidence"))

print("\nInitial Verification:")
print(result.get("verification_status"))


print("\n========== READING CHECKPOINT ==========\n")

checkpoint = agent.get_state(config)

saved_state = checkpoint.values

print("Recovered Tower:")
print(saved_state.get("tower_id"))

print("\nRecovered Investigation Attempts:")
print(saved_state.get("investigation_attempts"))

print("\nRecovered Root Cause:")
print(saved_state.get("root_cause"))

print("\nRecovered Confidence:")
print(saved_state.get("root_cause_confidence"))

print("\nRecovered Verification:")
print(saved_state.get("verification_status"))