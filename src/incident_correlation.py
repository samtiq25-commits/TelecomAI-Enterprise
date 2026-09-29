from .database import recent_incidents
import pandas as pd


def correlate_operational_events(
    operational_events,
    incident_limit=100,
    freshness_hours=12,
):
    """
    Correlate recent significant operational events with
    existing active network incidents.

    Historical events are ignored for automatic incident
    creation/update decisions.
    """

    if not operational_events:
        return []

    incidents = recent_incidents(incident_limit)

    active_incidents = {}

    for incident in incidents:

        if incident.incident_type != "NETWORK":
            continue

        if incident.case_status not in {
            "OPEN",
            "IN_PROGRESS",
        }:
            continue

        tower_id = incident.tower_id

        if not tower_id:
            continue

        if tower_id not in active_incidents:
            active_incidents[tower_id] = incident

    # ---------------------------------------------------------
    # Determine simulated operational "now"
    # ---------------------------------------------------------
    event_times = [
        pd.to_datetime(
            event.get("timestamp"),
            errors="coerce",
        )
        for event in operational_events
        if event.get("timestamp") is not None
    ]

    event_times = [
        timestamp
        for timestamp in event_times
        if pd.notna(timestamp)
    ]

    if not event_times:
        return []

    latest_event_time = max(event_times)
    # ---------------------------------------------------------
    # Keep only recent events
    # ---------------------------------------------------------

    latest_events = {}

    for event in operational_events:

        if not event.get("trigger_investigation"):
            continue

        event_time = event.get("timestamp")

        if event_time is None:
            continue

        event_time = pd.to_datetime(
            event_time,
            errors="coerce",
        )

        if pd.isna(event_time):
            continue

        age_hours = (
            latest_event_time - event_time
        ).total_seconds() / 3600

        if age_hours > freshness_hours:
            continue

        tower_id = event.get("tower_id")

        if not tower_id:
            continue

        existing = latest_events.get(tower_id)

        if (
            existing is None
            or pd.to_datetime(event["timestamp"])
            > pd.to_datetime(existing["timestamp"])
        ):
            latest_events[tower_id] = event

    # ---------------------------------------------------------
    # Correlate recent events with active incidents
    # ---------------------------------------------------------

    results = []

    for tower_id, event in latest_events.items():

        existing_incident = active_incidents.get(
            tower_id
        )

        if existing_incident:

            results.append({
                "action": "UPDATE_EXISTING",
                "incident_id": existing_incident.id,
                "tower_id": tower_id,
                "case_status": (
                    existing_incident.case_status
                ),
                "thread_id": existing_incident.thread_id,
                "event_timestamp": event["timestamp"],
                "event_type": event["event_type"],
                "event_name": event["event_name"],
                "operational_severity": (
                    event["operational_severity"]
                ),
                "reason": event["reason"],
            })

        else:

            results.append({
                "action": "CREATE_NEW",
                "incident_id": None,
                "tower_id": tower_id,
                "case_status": "OPEN",
                "event_timestamp": event["timestamp"],
                "event_type": event["event_type"],
                "event_name": event["event_name"],
                "operational_severity": (
                    event["operational_severity"]
                ),
                "reason": event["reason"],
            })

    return sorted(
        results,
        key=lambda x: x["event_timestamp"],
        reverse=True,
    )
def test_invalid_timestamps_are_ignored(monkeypatch):
    monkeypatch.setattr(
        incident_correlation,
        "recent_incidents",
        lambda limit: [],
    )

    events = [
        {
            "trigger_investigation": True,
            "timestamp": "not-a-date",
            "tower_id": "TWR-1001",
        }
    ]

    result = incident_correlation.correlate_operational_events(
        events
    )

    assert result == []
def test_invalid_timestamps_are_ignored(monkeypatch):
    monkeypatch.setattr(
        incident_correlation,
        "recent_incidents",
        lambda limit: [],
    )

    events = [
        {
            "trigger_investigation": True,
            "timestamp": "not-a-date",
            "tower_id": "TWR-1001",
        }
    ]

    result = incident_correlation.correlate_operational_events(
        events
    )

    assert result == []    
def test_invalid_timestamps_are_ignored(monkeypatch):
    monkeypatch.setattr(
        incident_correlation,
        "recent_incidents",
        lambda limit: [],
    )

    events = [
        {
            "trigger_investigation": True,
            "timestamp": "not-a-date",
            "tower_id": "TWR-1001",
        }
    ]

    result = incident_correlation.correlate_operational_events(
        events
    )

    assert result == []    