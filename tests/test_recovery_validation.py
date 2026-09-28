import src.voice_assistant as voice
import src.agent as agent
from langgraph.checkpoint.memory import MemorySaver
from types import SimpleNamespace
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command


def mock_tower_kpi(monkeypatch, kpi):
    monkeypatch.setattr(
        agent,
        "get_tower_kpi",
        SimpleNamespace(
            invoke=lambda args: {
                "found": True,
                "kpi": kpi,
            }
        ),
    )


def test_missing_kpi_does_not_confirm_recovery(monkeypatch):
    mock_tower_kpi(
        monkeypatch,
        {
            "latency": 10,
            "packet_loss": 0.5,
        },
    )

    result = agent.verify_service_recovery(
        {"tower_id": "TWR-1024"}
    )

    assert result["service_recovery_status"] == "NOT_RECOVERED"
    assert "call_drop_rate" in result["service_recovery_reason"]


def test_invalid_kpi_does_not_confirm_recovery(monkeypatch):
    mock_tower_kpi(
        monkeypatch,
        {
            "latency": "invalid",
            "packet_loss": 0.5,
            "call_drop_rate": 0.5,
        },
    )

    result = agent.verify_service_recovery(
        {"tower_id": "TWR-1024"}
    )

    assert result["service_recovery_status"] == "NOT_RECOVERED"


def test_valid_kpi_can_confirm_recovery(monkeypatch):
    mock_tower_kpi(
        monkeypatch,
        {
            "latency": 20,
            "packet_loss": 1,
            "call_drop_rate": 1,
        },
    )

    result = agent.verify_service_recovery(
        {"tower_id": "TWR-1024"}
    )

    assert result["service_recovery_status"] == "RECOVERED"


def test_high_latency_does_not_confirm_recovery(monkeypatch):
    mock_tower_kpi(
        monkeypatch,
        {
            "latency": 45,
            "packet_loss": 1,
            "call_drop_rate": 1,
        },
    )

    result = agent.verify_service_recovery(
        {
            "tower_id": "TWR-1024",
            "recovery_baseline": {
                "avg_latency": 20,
                "avg_packet_loss": 1,
            },
        }
    )

    assert result["service_recovery_status"] == "NOT_RECOVERED"
    assert "latency" in result["service_recovery_reason"]


def test_recovered_incident_becomes_resolved(monkeypatch):
    status_updates = []
    metrics = []

    monkeypatch.setattr(
        agent,
        "update_case_status",
        lambda *args, **kwargs: (
            status_updates.append((args, kwargs)) or True
        ),
    )

    monkeypatch.setattr(
        agent,
        "save_investigation_metric",
        lambda **kwargs: metrics.append(kwargs),
    )

    result = agent.evaluate_incident_resolution(
        {
            "incident_id": 999999,
            "case_status": "IN_PROGRESS",
            "service_recovery_status": "RECOVERED",
            "actor": "test_user",
            "actor_role": "Network Engineer",
            "thread_id": "test-thread",
            "tower_id": "TWR-1024",
        }
    )

    assert result["case_status"] == "RESOLVED"
    assert status_updates[0][0][1] == "RESOLVED"
    assert metrics[0]["outcome"] == "RESOLVED"


def test_unrecovered_incident_stays_in_progress(monkeypatch):
    status_updates = []
    metrics = []

    monkeypatch.setattr(
        agent,
        "update_case_status",
        lambda *args, **kwargs: (
            status_updates.append((args, kwargs)) or True
        ),
    )

    monkeypatch.setattr(
        agent,
        "save_investigation_metric",
        lambda **kwargs: metrics.append(kwargs),
    )

    result = agent.evaluate_incident_resolution(
        {
            "incident_id": 999999,
            "case_status": "IN_PROGRESS",
            "service_recovery_status": "NOT_RECOVERED",
            "actor": "test_user",
            "actor_role": "Network Engineer",
            "thread_id": "test-thread",
            "tower_id": "TWR-1024",
        }
    )

    assert result["case_status"] == "IN_PROGRESS"
    assert status_updates == []
    assert metrics[0]["outcome"] == "NOT_RECOVERED"


def test_failed_status_update_does_not_save_metric(monkeypatch):
    metrics = []

    monkeypatch.setattr(
        agent,
        "update_case_status",
        lambda *args, **kwargs: False,
    )

    monkeypatch.setattr(
        agent,
        "save_investigation_metric",
        lambda **kwargs: metrics.append(kwargs),
    )

    result = agent.evaluate_incident_resolution(
        {
            "incident_id": 999999,
            "case_status": "IN_PROGRESS",
            "service_recovery_status": "RECOVERED",
            "actor": "test_user",
            "actor_role": "Network Engineer",
            "thread_id": "test-thread",
            "tower_id": "TWR-1024",
        }
    )

    assert result == {}
    assert metrics == []


def test_recovery_verification_updates_incident_status(monkeypatch):
    status_updates = []
    metrics = []

    mock_tower_kpi(
        monkeypatch,
        {
            "latency": 20,
            "packet_loss": 1,
            "call_drop_rate": 1,
        },
    )

    monkeypatch.setattr(
        agent,
        "update_case_status",
        lambda *args, **kwargs: (
            status_updates.append((args, kwargs)) or True
        ),
    )

    monkeypatch.setattr(
        agent,
        "save_investigation_metric",
        lambda **kwargs: metrics.append(kwargs),
    )

    state = {
        "tower_id": "TWR-1024",
        "incident_id": 999999,
        "case_status": "IN_PROGRESS",
        "actor": "test_user",
        "actor_role": "Network Engineer",
        "thread_id": "isolated-recovery-test",
    }

    recovery_result = agent.verify_service_recovery(state)
    state.update(recovery_result)

    incident_result = agent.evaluate_incident_resolution(state)

    assert recovery_result["service_recovery_status"] == "RECOVERED"
    assert incident_result["case_status"] == "RESOLVED"
    assert status_updates[0][0][1] == "RESOLVED"
    assert metrics[0]["outcome"] == "RESOLVED"


def build_isolated_network_test_graph():
    graph = StateGraph(agent.State)

    graph.add_node(
        "network_engineer_review",
        agent.network_engineer_review,
    )
    graph.add_node(
        "create_network_incident",
        agent.create_network_incident,
    )
    graph.add_node(
        "move_incident_to_in_progress",
        agent.move_incident_to_in_progress,
    )
    graph.add_node(
        "verify_service_recovery",
        agent.verify_service_recovery,
    )
    graph.add_node(
        "evaluate_incident_resolution",
        agent.evaluate_incident_resolution,
    )

    graph.add_edge(START, "network_engineer_review")
    graph.add_edge(
        "network_engineer_review",
        "create_network_incident",
    )
    graph.add_edge(
        "create_network_incident",
        "move_incident_to_in_progress",
    )
    graph.add_edge(
        "move_incident_to_in_progress",
        "verify_service_recovery",
    )
    graph.add_edge(
        "verify_service_recovery",
        "evaluate_incident_resolution",
    )
    graph.add_edge("evaluate_incident_resolution", END)

    return graph.compile(checkpointer=MemorySaver())
def test_complete_voice_approval_and_recovery_isolated(monkeypatch):
    created_incidents = []
    status_updates = []
    metrics = []
    audits = []

    monkeypatch.setattr(
        agent,
        "save_incident",
        lambda data: (
            created_incidents.append(data) or 987654
        ),
    )

    monkeypatch.setattr(
        agent,
        "update_case_status",
        lambda *args, **kwargs: (
            status_updates.append((args, kwargs)) or True
        ),
    )

    monkeypatch.setattr(
        agent,
        "save_investigation_metric",
        lambda **kwargs: metrics.append(kwargs),
    )

    monkeypatch.setattr(
        agent,
        "audit",
        lambda *args, **kwargs: audits.append(
            (args, kwargs)
        ),
    )

    mock_tower_kpi(
        monkeypatch,
        {
            "latency": 20,
            "packet_loss": 1,
            "call_drop_rate": 1,
        },
    )

    test_graph = build_isolated_network_test_graph()

    monkeypatch.setattr(
        voice,
        "build_agent",
        lambda: test_graph,
    )

    monkeypatch.setattr(
        voice,
        "save_investigation_metric",
        lambda **kwargs: None,
    )

    thread_id = "isolated-voice-approval-recovery-test"

    result, response = voice.process_voice_request(
        text="My internet is slow and calls keep dropping.",
        audience="employee",
        tower_id="TWR-1024",
        thread_id=thread_id,
    )

    assert result["voice_request_type"] == "NETWORK"
    assert result["network_decision"] == "HUMAN_REVIEW"
    assert "human network engineer review" in response.lower()
    assert "Human approval is required" in response

    config = {"configurable": {"thread_id": thread_id}}
    snapshot = test_graph.get_state(config)

    assert snapshot.next == ("network_engineer_review",)

    final_state = voice.resume_voice_investigation(
        thread_id=thread_id,
        decision="approved",
        feedback="Approved in isolated voice test",
        reviewer_actor="test_user",
        reviewer_role="Network Engineer",
    )

    assert final_state["incident_id"] == 987654
    assert final_state["case_status"] == "RESOLVED"
    assert final_state["service_recovery_status"] == "RECOVERED"

    assert len(created_incidents) == 1
    assert status_updates[0][0][1] == "IN_PROGRESS"
    assert status_updates[1][0][1] == "RESOLVED"
    assert metrics[-1]["outcome"] == "RESOLVED"
    assert len(audits) >= 1