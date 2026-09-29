import pandas as pd

from src import event_intelligence


def make_network():
    return pd.DataFrame([
        {
            "timestamp": pd.Timestamp("2026-09-28 10:00:00"),
            "tower_id": "TWR-1001",
            "latency": 120.0,
            "packet_loss": 6.0,
            "cpu_usage": 90.0,
            "memory_usage": 70.0,
        }
    ])


def make_anomalies():
    return pd.DataFrame([
        {
            "timestamp": pd.Timestamp("2026-09-28 10:00:00"),
            "tower_id": "TWR-1001",
            "is_anomaly": True,
        }
    ])


def make_event(**overrides):
    event = {
        "timestamp": pd.Timestamp("2026-09-28 10:00:00"),
        "tower_id": "TWR-1001",
        "event_type": "BACKHAUL",
        "severity": "NORMAL",
        "backhaul_utilization": 50.0,
        "event_name": "Test event",
        "description": "Test operational event",
    }

    event.update(overrides)
    return event


def run_detector(monkeypatch, event):
    monkeypatch.setattr(
        event_intelligence,
        "load_operational_events",
        lambda: pd.DataFrame([event]),
    )

    return event_intelligence.detect_operational_events(
        make_network(),
        make_anomalies(),
    )


def test_high_latency_triggers_investigation(monkeypatch):
    results = run_detector(
        monkeypatch,
        make_event(),
    )

    assert len(results) == 1
    assert results[0]["trigger_investigation"] is True
    assert results[0]["kpi_severity"] == "CRITICAL"


def test_event_without_supporting_evidence_is_ignored(
    monkeypatch,
):
    network = make_network()
    network["latency"] = 30.0
    network["packet_loss"] = 0.5
    network["cpu_usage"] = 40.0
    network["memory_usage"] = 50.0

    monkeypatch.setattr(
        event_intelligence,
        "load_operational_events",
        lambda: pd.DataFrame([
            make_event(event_type="CONFIGURATION")
        ]),
    )

    anomalies = make_anomalies()
    anomalies["is_anomaly"] = False

    results = event_intelligence.detect_operational_events(
    network,
    anomalies,
)

    assert results == []


def test_event_with_kpi_more_than_two_hours_away_is_ignored(
    monkeypatch,
):
    network = make_network()
    network["timestamp"] = pd.Timestamp(
        "2026-09-28 14:00:00"
    )

    monkeypatch.setattr(
        event_intelligence,
        "load_operational_events",
        lambda: pd.DataFrame([make_event()]),
    )

    results = event_intelligence.detect_operational_events(
        network,
        make_anomalies(),
    )

    assert results == []
def test_event_does_not_match_another_tower(
    monkeypatch,
):
    event = make_event(
        tower_id="TWR-2000",
        event_type="CONFIGURATION",
    )

    results = run_detector(
        monkeypatch,
        event,
    )

    assert results == []
def test_high_backhaul_utilization_triggers_investigation(
    monkeypatch,
):
    event = make_event(
        backhaul_utilization=90.0,
        event_type="BACKHAUL",
    )

    results = run_detector(monkeypatch, event)

    assert len(results) == 1
    assert results[0]["trigger_investigation"] is True
    assert "High backhaul utilization" in results[0]["reason"]
def test_critical_event_triggers_investigation(
    monkeypatch,
):
    network = make_network()
    network["latency"] = 30.0
    network["packet_loss"] = 0.5
    network["cpu_usage"] = 40.0
    network["memory_usage"] = 50.0

    anomalies = make_anomalies()
    anomalies["is_anomaly"] = False

    event = make_event(
        event_type="CONFIGURATION",
        severity="CRITICAL",
    )

    monkeypatch.setattr(
        event_intelligence,
        "load_operational_events",
        lambda: pd.DataFrame([event]),
    )

    results = event_intelligence.detect_operational_events(
        network,
        anomalies,
    )

    assert len(results) == 1
    assert results[0]["trigger_investigation"] is True
    assert results[0]["operational_severity"] == "CRITICAL"
def test_medium_event_with_abnormal_kpi_escalates(
    monkeypatch,
):
    event = make_event(
        event_type="CONFIGURATION",
        severity="MEDIUM",
    )

    results = run_detector(
        monkeypatch,
        event,
    )

    assert len(results) == 1
    assert results[0]["operational_severity"] == "HIGH"
def test_missing_anomaly_value_does_not_escalate(
    monkeypatch,
):
    event = make_event(
        event_type="CONFIGURATION",
        severity="NORMAL",
    )

    network = make_network()
    network["latency"] = 40
    network["packet_loss"] = 0.5
    network["cpu_usage"] = 40
    network["memory_usage"] = 50

    anomalies = make_anomalies()
    anomalies["is_anomaly"] = float("nan")

    monkeypatch.setattr(
        event_intelligence,
        "load_operational_events",
        lambda: pd.DataFrame([event]),
    )

    results = event_intelligence.detect_operational_events(
        network,
        anomalies,
    )

    assert results == []
def test_missing_latency_does_not_crash(
    monkeypatch,
):
    event = make_event(
        event_type="CONFIGURATION",
        severity="NORMAL",
    )

    network = make_network()
    network["latency"] = float("nan")
    network["packet_loss"] = 0.5
    network["cpu_usage"] = 40
    network["memory_usage"] = 50
    network["packet_loss"] = float("nan")

    anomalies = make_anomalies()
    anomalies["is_anomaly"] = False

    monkeypatch.setattr(
        event_intelligence,
        "load_operational_events",
        lambda: pd.DataFrame([event]),
    )

    results = event_intelligence.detect_operational_events(
        network,
        anomalies,
    )

    assert results == []    
def test_invalid_latency_does_not_crash(
    monkeypatch,
):
    event = make_event(
        event_type="CONFIGURATION",
        severity="NORMAL",
    )

    network = make_network()
    network["latency"] = "invalid"
    network["packet_loss"] = 0.5
    network["cpu_usage"] = 40
    network["memory_usage"] = 50
    network["packet_loss"] = "invalid"

    anomalies = make_anomalies()
    anomalies["is_anomaly"] = False

    monkeypatch.setattr(
        event_intelligence,
        "load_operational_events",
        lambda: pd.DataFrame([event]),
    )

    results = event_intelligence.detect_operational_events(
        network,
        anomalies,
    )

    assert results == []
def test_missing_packet_loss_does_not_crash(
    monkeypatch,
):
    event = make_event(
        event_type="CONFIGURATION",
        severity="NORMAL",
    )

    network = make_network()
    network["latency"] = 40
    network["packet_loss"] = float("nan")
    network["cpu_usage"] = 40
    network["memory_usage"] = 50

    anomalies = make_anomalies()
    anomalies["is_anomaly"] = False

    monkeypatch.setattr(
        event_intelligence,
        "load_operational_events",
        lambda: pd.DataFrame([event]),
    )

    results = event_intelligence.detect_operational_events(
        network,
        anomalies,
    )

    assert results == []    
def test_missing_cpu_does_not_crash(
    monkeypatch,
):
    event = make_event(
        event_type="CONFIGURATION",
        severity="NORMAL",
    )

    network = make_network()
    network["latency"] = 40
    network["packet_loss"] = 0.5
    network["cpu_usage"] = float("nan")
    network["memory_usage"] = 50

    anomalies = make_anomalies()
    anomalies["is_anomaly"] = False

    monkeypatch.setattr(
        event_intelligence,
        "load_operational_events",
        lambda: pd.DataFrame([event]),
    )

    results = event_intelligence.detect_operational_events(
        network,
        anomalies,
    )

    assert results == []    
def test_missing_memory_does_not_crash(
    monkeypatch,
):
    event = make_event(
        event_type="CONFIGURATION",
        severity="NORMAL",
    )

    network = make_network()
    network["latency"] = 40
    network["packet_loss"] = 0.5
    network["cpu_usage"] = 40
    network["memory_usage"] = float("nan")

    anomalies = make_anomalies()
    anomalies["is_anomaly"] = False

    monkeypatch.setattr(
        event_intelligence,
        "load_operational_events",
        lambda: pd.DataFrame([event]),
    )

    results = event_intelligence.detect_operational_events(
        network,
        anomalies,
    )

    assert results == []    
def test_missing_kpis_do_not_create_critical_severity(
    monkeypatch,
):
    event = make_event(
        event_type="BACKHAUL",
        severity="HIGH",
    )

    network = make_network()
    network["latency"] = float("nan")
    network["packet_loss"] = float("nan")
    network["cpu_usage"] = float("nan")
    network["memory_usage"] = float("nan")

    anomalies = make_anomalies()
    anomalies["is_anomaly"] = False

    monkeypatch.setattr(
        event_intelligence,
        "load_operational_events",
        lambda: pd.DataFrame([event]),
    )

    results = event_intelligence.detect_operational_events(
        network,
        anomalies,
    )

    assert len(results) == 1
    assert results[0]["kpi_severity"] == "NORMAL"
    assert results[0]["operational_severity"] == "HIGH"