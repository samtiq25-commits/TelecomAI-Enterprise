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
def test_existing_incident_audit_uses_incident_id():
    class FakeIncident:
        id = 123
        incident_type = "NETWORK"
        case_status = "IN_PROGRESS"
        tower_id = "TWR-TEST-02"
        thread_id = "thread-123"

    events = [
        {
            "timestamp": "2026-09-29 11:00:00",
            "tower_id": "TWR-TEST-02",
            "trigger_investigation": True,
            "event_type": "KPI",
            "event_name": "High Packet Loss",
            "operational_severity": "CRITICAL",
            "reason": "Test packet loss event",
        }
    ]

    with patch.object(
        incident_correlation,
        "recent_incidents",
        return_value=[FakeIncident()],
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
    assert results[0]["action"] == "UPDATE_EXISTING"
    assert results[0]["incident_id"] == 123

    mock_audit.assert_called_once()

    audit_call = mock_audit.call_args.kwargs

    assert audit_call["resource_id"] == "123"
    assert (
        audit_call["action"]
        == "OPERATIONAL_EVENT_CORRELATED"
    )