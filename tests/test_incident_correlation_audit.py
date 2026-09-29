from unittest.mock import patch

from src import incident_correlation


def test_correlate_and_audit_records_decisions():
    events = [
        {
            "timestamp": "2026-09-29 10:00:00",
            "tower_id": "TWR-TEST-01",
            "trigger_investigation": True,
            "event_type": "KPI",
            "event_name": "High Latency",
            "operational_severity": "HIGH",
            "reason": "Test event",
        }
    ]

    with patch.object(
        incident_correlation,
        "recent_incidents",
        return_value=[],
    ), patch.object(
        incident_correlation,
        "audit",
    ) as mock_audit:

        results = (
            incident_correlation
            .correlate_and_audit_operational_events(
                events
            )
        )

    assert len(results) == 1
    assert results[0]["action"] == "CREATE_NEW"

    mock_audit.assert_called_once()

    audit_call = mock_audit.call_args.kwargs

    assert (
        audit_call["action"]
        == "OPERATIONAL_EVENT_CORRELATED"
    )
    assert audit_call["resource_id"] == "TWR-TEST-01"