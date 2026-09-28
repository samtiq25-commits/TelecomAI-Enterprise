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
        "thread_id": "checkpoint-test-TWR-1024"
    }
}


print("\n========== FIRST INVOCATION ==========\n")

result = agent.invoke(
    {
        "issue": "Internet is very slow and packet loss appears high.",
        "tower_id": "TWR-1024",
        "anomaly": anomaly,
    },
    config=config,
)

print("Investigation Attempts:", result.get("investigation_attempts"))
print("Root Cause:", result.get("root_cause"))
print("Confidence:", result.get("root_cause_confidence"))
print("Verification:", result.get("verification_status"))


print("\n========== CHECKPOINT STATE ==========\n")

checkpoint = agent.get_state(config)

saved_state = checkpoint.values

print("Saved Investigation Attempts:",
      saved_state.get("investigation_attempts"))

print("Saved Root Cause:",
      saved_state.get("root_cause"))

print("Saved Confidence:",
      saved_state.get("root_cause_confidence"))

print("Saved Verification:",
      saved_state.get("verification_status"))