import pytest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import src.database as database


@pytest.fixture
def isolated_database(tmp_path, monkeypatch):
    """
    Create an isolated SQLite database for each test.

    Never modify the real telecomai.db database.
    """

    test_database_path = tmp_path / "test_incidents.db"

    test_engine = create_engine(
        f"sqlite:///{test_database_path}",
        connect_args={"check_same_thread": False},
    )

    test_session = sessionmaker(
        bind=test_engine,
        autoflush=False,
        autocommit=False,
    )

    monkeypatch.setattr(
        database,
        "engine",
        test_engine,
    )

    monkeypatch.setattr(
        database,
        "SessionLocal",
        test_session,
    )

    # Create the schema in the isolated database.
    database.Base.metadata.create_all(test_engine)

    # The schema already exists, so the production
    # migration function is not needed in this fixture.
    monkeypatch.setattr(
        database,
        "init_db",
        lambda: None,
    )

    yield test_session

    test_engine.dispose()


def test_new_network_incident_is_created(
    isolated_database,
):
    incident_id = database.save_incident({
        "incident_type": "NETWORK",
        "tower_id": "TWR-TEST-1001",
        "thread_id": "test-thread-1",
        "case_status": "OPEN",
        "priority": "HIGH",
        "severity": "HIGH",
        "issue": "High network latency",
    })

    with isolated_database() as session:
        incidents = session.query(
            database.Incident
        ).all()

        assert len(incidents) == 1
        assert incidents[0].id == incident_id
        assert incidents[0].case_status == "OPEN"
        assert incidents[0].thread_id == "test-thread-1"


def test_active_network_incident_is_not_duplicated(
    isolated_database,
):
    first_id = database.save_incident({
        "incident_type": "NETWORK",
        "tower_id": "TWR-TEST-1002",
        "case_status": "OPEN",
        "priority": "HIGH",
        "issue": "Initial network event",
    })

    second_id = database.save_incident({
        "incident_type": "NETWORK",
        "tower_id": "TWR-TEST-1002",
        "case_status": "OPEN",
        "priority": "CRITICAL",
        "severity": "CRITICAL",
        "issue": "Another network event",
    })

    assert first_id == second_id

    with isolated_database() as session:
        incidents = session.query(
            database.Incident
        ).all()

        assert len(incidents) == 1
        assert incidents[0].priority == "CRITICAL"
        assert incidents[0].severity == "CRITICAL"


def test_in_progress_incident_is_not_downgraded(
    isolated_database,
):
    incident_id = database.save_incident({
        "incident_type": "NETWORK",
        "tower_id": "TWR-TEST-1003",
        "case_status": "IN_PROGRESS",
        "priority": "HIGH",
        "issue": "Network investigation underway",
    })

    updated_id = database.save_incident({
        "incident_type": "NETWORK",
        "tower_id": "TWR-TEST-1003",
        "case_status": "OPEN",
        "priority": "CRITICAL",
        "issue": "New operational event",
    })

    assert incident_id == updated_id

    with isolated_database() as session:
        incident = session.get(
            database.Incident,
            incident_id,
        )

        assert incident.case_status == "IN_PROGRESS"
        assert incident.priority == "CRITICAL"


def test_existing_incident_receives_thread_id(
    isolated_database,
):
    incident_id = database.save_incident({
        "incident_type": "NETWORK",
        "tower_id": "TWR-TEST-1004",
        "case_status": "OPEN",
        "issue": "Legacy incident without thread",
    })

    updated_id = database.save_incident({
        "incident_type": "NETWORK",
        "tower_id": "TWR-TEST-1004",
        "case_status": "OPEN",
        "thread_id": "repaired-thread-1004",
        "issue": "Event investigation",
    })

    assert incident_id == updated_id

    with isolated_database() as session:
        incident = session.get(
            database.Incident,
            incident_id,
        )

        assert incident.thread_id == "repaired-thread-1004"

        count = session.query(
            database.Incident
        ).count()

        assert count == 1


def test_resolved_incident_allows_new_incident(
    isolated_database,
):
    first_id = database.save_incident({
        "incident_type": "NETWORK",
        "tower_id": "TWR-TEST-1005",
        "case_status": "RESOLVED",
        "issue": "Previous resolved outage",
    })

    second_id = database.save_incident({
        "incident_type": "NETWORK",
        "tower_id": "TWR-TEST-1005",
        "case_status": "OPEN",
        "issue": "New network outage",
    })

    assert first_id != second_id

    with isolated_database() as session:
        incidents = session.query(
            database.Incident
        ).all()

        assert len(incidents) == 2