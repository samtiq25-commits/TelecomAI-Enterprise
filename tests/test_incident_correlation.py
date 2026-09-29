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