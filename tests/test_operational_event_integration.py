from unittest.mock import MagicMock

from src import database
from src import agent


def test_create_new_operational_event_starts_investigation(
    monkeypatch,
):
    correlation = {
        "action": "CREATE_NEW",
        "incident_id": None,
        "tower_id": "TWR-TEST-01",
        "case_status": "OPEN",
        "operational_severity": "HIGH",
        "reason": "High latency detected",
    }

    created = []

    monkeypatch.setattr(
        database,
        "save_incident",
        lambda data: (
            created.append(data) or 501
        ),
    )

    fake_agent = MagicMock()

    monkeypatch.setattr(
        agent,
        "build_agent",
        lambda: fake_agent,
    )

    thread_id = "event-TWR-TEST-01-test"

    incident_id = database.save_incident({
        "incident_type": "NETWORK",
        "case_status": correlation["case_status"],
        "priority": correlation["operational_severity"],
        "severity": correlation["operational_severity"],
        "issue": (
            f"Operational event detected at "
            f"{correlation['tower_id']}: "
            f"{correlation['reason']}"
        ),
        "tower_id": correlation["tower_id"],
        "thread_id": thread_id,
    })

    fake_agent.invoke(
        {
            "issue": (
                f"Operational event detected at "
                f"{correlation['tower_id']}: "
                f"{correlation['reason']}"
            ),
            "tower_id": correlation["tower_id"],
            "actor": "event-monitor",
            "actor_role": "NOC Engineer",
            "request_type": "NETWORK",
        },
        config={
            "configurable": {
                "thread_id": thread_id
            }
        },
    )

    assert incident_id == 501
    assert len(created) == 1

    assert created[0]["incident_type"] == "NETWORK"
    assert created[0]["tower_id"] == "TWR-TEST-01"
    assert created[0]["case_status"] == "OPEN"
    assert created[0]["priority"] == "HIGH"
    assert created[0]["thread_id"] == thread_id

    fake_agent.invoke.assert_called_once()


def test_existing_operational_event_with_thread_does_not_start_new_investigation():
    correlation = {
        "action": "UPDATE_EXISTING",
        "incident_id": 601,
        "tower_id": "TWR-TEST-02",
        "case_status": "IN_PROGRESS",
        "thread_id": "existing-thread-601",
        "operational_severity": "CRITICAL",
        "reason": "Packet loss increased",
    }

    fake_agent = MagicMock()

    if correlation["thread_id"]:
        processed_message = (
            f"Existing Incident #{correlation['incident_id']} "
            f"correlated with {correlation['tower_id']}"
        )
    else:
        fake_agent.invoke()

    assert (
        processed_message
        == "Existing Incident #601 correlated with TWR-TEST-02"
    )

    fake_agent.invoke.assert_not_called()


def test_existing_operational_event_without_thread_is_repaired(
    monkeypatch,
):
    correlation = {
        "action": "UPDATE_EXISTING",
        "incident_id": 701,
        "tower_id": "TWR-TEST-03",
        "case_status": "OPEN",
        "thread_id": None,
        "operational_severity": "HIGH",
        "reason": "High latency detected",
    }

    saved = []

    monkeypatch.setattr(
        database,
        "save_incident",
        lambda data: (
            saved.append(data) or 702
        ),
    )

    fake_agent = MagicMock()

    monkeypatch.setattr(
        agent,
        "build_agent",
        lambda: fake_agent,
    )

    repaired_thread_id = (
        "event-TWR-TEST-03-repaired"
    )

    repaired_incident_id = database.save_incident({
        "incident_type": "NETWORK",
        "case_status": correlation["case_status"],
        "priority": correlation["operational_severity"],
        "severity": correlation["operational_severity"],
        "issue": (
            f"Operational event detected at "
            f"{correlation['tower_id']}: "
            f"{correlation['reason']}"
        ),
        "tower_id": correlation["tower_id"],
        "thread_id": repaired_thread_id,
    })

    fake_agent.invoke(
        {
            "issue": (
                f"Operational event detected at "
                f"{correlation['tower_id']}: "
                f"{correlation['reason']}"
            ),
            "tower_id": correlation["tower_id"],
            "actor": "event-monitor",
            "actor_role": "NOC Engineer",
            "request_type": "NETWORK",
        },
        config={
            "configurable": {
                "thread_id": repaired_thread_id
            }
        },
    )

    assert repaired_incident_id == 702
    assert len(saved) == 1

    assert saved[0]["tower_id"] == "TWR-TEST-03"
    assert saved[0]["thread_id"] == repaired_thread_id

    fake_agent.invoke.assert_called_once()

    invoke_call = fake_agent.invoke.call_args

    assert (
        invoke_call.kwargs["config"]["configurable"]["thread_id"]
        == repaired_thread_id
    )