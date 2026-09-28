import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.types import Command

from src.data_generator import load_data
from src.ml_models import detect_anomalies
from src.agent import build_agent
from src.database import audit

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
        "thread_id": "human-review-TWR-1024"
    }
}


print("\n========== STARTING AGENT ==========\n")

result = agent.invoke(
    {
        "issue": "Internet is very slow and packet loss appears high.",
        "tower_id": "TWR-1024",
        "anomaly": anomaly,
    },
    config=config,
)

print("Agent paused for human review.")

print("\nCheckpointed state before approval:")

state = agent.get_state(config)

print("Tower:", state.values.get("tower_id"))
print("Investigation Attempts:",
      state.values.get("investigation_attempts"))
print("Root Cause:",
      state.values.get("root_cause"))
print("Confidence:",
      state.values.get("root_cause_confidence"))
print("Verification:",
      state.values.get("verification_status"))


print("\n========== HUMAN APPROVAL ==========\n")

result = agent.invoke(
    Command(resume="approved"),
    config=config,
)
audit(
    actor="human-engineer",
    role="noc_engineer",
    action="human_review_approved",
    resource_id="TWR-1024",
    details=(
        f"Root cause: "
        f"{result.get('root_cause', '')}; "
        f"Confidence: "
        f"{result.get('root_cause_confidence', 'UNKNOWN')}; "
        f"Verification: "
        f"{result.get('verification_status', 'UNKNOWN')}"
    )
)

print("Agent resumed successfully.")

print("\n========== FINAL RESULT ==========\n")

print("Root Cause:",
      result.get("root_cause"))

print("Confidence:",
      result.get("root_cause_confidence"))

print("Verification:",
      result.get("verification_status"))

print("Recommendation:",
      result.get("recommendation"))

print("\nIncident Report:")
print(result.get("incident_report"))