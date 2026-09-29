import pandas as pd

from src import incident_correlation


def test_all_events_with_missing_timestamps_return_empty(
    monkeypatch,
):
    monkeypatch.setattr(
        incident_correlation,
        "recent_incidents",
        lambda limit: [],
    )

    events = [
        {
            "trigger_investigation": True,
            "timestamp": None,
            "tower_id": "TWR-1001",
        },
        {
            "trigger_investigation": True,
            "timestamp": None,
            "tower_id": "TWR-1002",
        },
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
def test_only_latest_event_per_tower_is_returned(
    monkeypatch,
):
    monkeypatch.setattr(
        incident_correlation,
        "recent_incidents",
        lambda limit: [],
    )

    events = [
        {
            "timestamp": "2026-09-29T10:00:00",
            "tower_id": "TWR-1001",
            "event_type": "NETWORK",
            "event_name": "High Latency",
            "operational_severity": "HIGH",
            "trigger_investigation": True,
            "reason": "Latency increased",
        },
        {
            "timestamp": "2026-09-29T11:00:00",
            "tower_id": "TWR-1001",
            "event_type": "NETWORK",
            "event_name": "Packet Loss",
            "operational_severity": "CRITICAL",
            "trigger_investigation": True,
            "reason": "Packet loss increased",
        },
    ]

    result = (
        incident_correlation
        .correlate_operational_events(events)
    )

    assert len(result) == 1
    assert result[0]["tower_id"] == "TWR-1001"
    assert result[0]["event_name"] == "Packet Loss"
    assert result[0]["action"] == "CREATE_NEW"
def test_active_incident_is_updated_instead_of_creating_new(
    monkeypatch,
):
    class FakeIncident:
        id = 123
        incident_type = "NETWORK"
        case_status = "IN_PROGRESS"
        tower_id = "TWR-1001"
        thread_id = "thread-123"

    monkeypatch.setattr(
        incident_correlation,
        "recent_incidents",
        lambda limit: [FakeIncident()],
    )

    events = [
        {
            "timestamp": "2026-09-29T11:00:00",
            "tower_id": "TWR-1001",
            "event_type": "NETWORK",
            "event_name": "Packet Loss",
            "operational_severity": "CRITICAL",
            "trigger_investigation": True,
            "reason": "Packet loss increased",
        }
    ]

    result = (
        incident_correlation
        .correlate_operational_events(events)
    )

    assert len(result) == 1
    assert result[0]["action"] == "UPDATE_EXISTING"
    assert result[0]["incident_id"] == 123
    assert result[0]["thread_id"] == "thread-123"
    assert result[0]["case_status"] == "IN_PROGRESS"