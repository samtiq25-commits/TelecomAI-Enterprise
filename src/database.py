import os
from datetime import datetime, timezone
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Float,
)
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "")
if DATABASE_URL:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
else:
    engine = create_engine("sqlite:///telecomai.db",
                           connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class Incident(Base):
    __tablename__ = "incidents"
    id = Column(Integer, primary_key=True)
    thread_id = Column(
    String(128),
    nullable=True,
    index=True
)
    incident_type = Column(String(32), default="NETWORK")
    case_status = Column(
    String(32),
    default="OPEN"
)
    assigned_to = Column(
        String(128),
        nullable=True
    )
    priority = Column(
        String(32),
        default="MEDIUM"
    )
    issue = Column(Text, nullable=False)
    tower_id = Column(String(64))
    diagnosis = Column(Text)
    recommendation = Column(Text)
    severity = Column(String(32))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    root_cause_confidence = Column(String(32))


class AuditEvent(Base):
    __tablename__ = "audit_events"
    id = Column(Integer, primary_key=True)
    actor = Column(String(128))
    role = Column(String(64))
    action = Column(String(128))
    details = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    resource_id = Column(String(128))


class InvestigationMetric(Base):
    __tablename__ = "investigation_metrics"

    id = Column(Integer, primary_key=True)

    thread_id = Column(
        String(128),
        nullable=False,
        index=True,
    )

    incident_id = Column(
        Integer,
        nullable=True,
        index=True,
    )

    tower_id = Column(
        String(64),
        nullable=True,
    )

    status = Column(
        String(32),
        default="STARTED",
    )

    outcome = Column(
        String(64),
        nullable=True,
    )

    execution_time_ms = Column(
        Integer,
        nullable=True,
    )

    human_review = Column(
        String(32),
        nullable=True,
    )

    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
    )


class ModelTelemetry(Base):
    __tablename__ = "model_telemetry"

    id = Column(Integer, primary_key=True)

    thread_id = Column(
        String(128),
        nullable=True,
    )

    incident_id = Column(
        Integer,
        nullable=True,
    )

    provider = Column(
        String(64),
        nullable=True,
    )

    model_name = Column(
        String(128),
        nullable=True,
    )

    operation = Column(
        String(64),
        nullable=True,
    )

    input_tokens = Column(
        Integer,
        nullable=True,
    )

    output_tokens = Column(
        Integer,
        nullable=True,
    )

    total_tokens = Column(
        Integer,
        nullable=True,
    )

    latency_ms = Column(
        Integer,
        nullable=True,
    )

    estimated_cost = Column(
        Float,
        nullable=True,
    )

    status = Column(
        String(32),
        nullable=True,
    )

    error_message = Column(
        Text,
        nullable=True,
    )

    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
    )


def init_db():
    Base.metadata.create_all(bind=engine)

    if engine.url.get_backend_name() == "sqlite":
        with engine.begin() as connection:
            columns = connection.exec_driver_sql(
                "PRAGMA table_info(incidents)"
            ).fetchall()

            column_names = {
                column[1]
                for column in columns
            }

            if "thread_id" not in column_names:
                connection.exec_driver_sql(
                    """
                    ALTER TABLE incidents
                    ADD COLUMN thread_id VARCHAR(128)
                    """
                )

                connection.exec_driver_sql(
                    """
                    CREATE INDEX IF NOT EXISTS
                    ix_incidents_thread_id
                    ON incidents (thread_id)
                    """
                )


def save_incident(result):

    init_db()

    db = SessionLocal()

    try:
        incident_type = result.get(
            "incident_type",
            "NETWORK",
        )

        tower_id = result.get("tower_id")

        requested_status = result.get(
            "case_status",
            "OPEN",
        )

        requested_thread_id = result.get(
            "thread_id"
        )

        # ---------------------------------------------------------
        # Incident Deduplication
        # ---------------------------------------------------------

        if incident_type == "NETWORK" and tower_id:

            existing = (
                db.query(Incident)
                .filter(
                    Incident.incident_type == "NETWORK",
                    Incident.tower_id == tower_id,
                    Incident.case_status.in_(
                        ["OPEN", "IN_PROGRESS"]
                    ),
                )
                .first()
            )

            if existing:

                # Never downgrade an existing
                # IN_PROGRESS incident back to OPEN.
                if existing.case_status != "IN_PROGRESS":
                    existing.case_status = requested_status

                # Keep the latest investigation information.
                existing.priority = result.get(
                    "priority",
                    existing.priority,
                )

                existing.severity = result.get(
                    "severity",
                    existing.severity,
                )
                existing.diagnosis = result.get(
                    "diagnosis",
                    existing.diagnosis,
                )

                existing.recommendation = result.get(
                    "recommendation",
                    existing.recommendation,
                )

                existing.root_cause_confidence = result.get(
                    "root_cause_confidence",
                    existing.root_cause_confidence,
                )

                # Link the incident to the latest LangGraph investigation.
                if requested_thread_id:
                    existing.thread_id = requested_thread_id

                db.commit()

                return existing.id

        # ---------------------------------------------------------
        # Create a new incident
        # ---------------------------------------------------------

        item = Incident(
            incident_type=incident_type,
            thread_id=requested_thread_id,
            case_status=requested_status,
            priority=result.get(
                "priority",
                "MEDIUM",
            ),
            issue=result.get(
                "issue",
                "",
            ),
            assigned_to=result.get(
                "assigned_to"
            ),
            tower_id=tower_id,
            diagnosis=result.get(
                "diagnosis"
            ),
            recommendation=result.get(
                "recommendation"
            ),
            severity=result.get(
            "severity",
            result.get("priority", "MEDIUM")
           ),
            root_cause_confidence=result.get(
                "root_cause_confidence",
                "UNKNOWN",
            ),
        )

        db.add(item)
        db.commit()

        return item.id

    finally:
        db.close()
def update_incident_status(
    incident_id,
    new_status,
    assigned_to=None,
    resolution_notes=None,
):
    """
    Update an incident's lifecycle status.

    Supported lifecycle:
        OPEN -> IN_PROGRESS -> RESOLVED -> CLOSED
    """

    init_db()

    db = SessionLocal()

    try:
        incident = (
            db.query(Incident)
            .filter(Incident.id == incident_id)
            .first()
        )

        if not incident:
            return False

        allowed_transitions = {
            "OPEN": ["IN_PROGRESS"],
            "IN_PROGRESS": ["RESOLVED"],
            "RESOLVED": ["CLOSED"],
            "CLOSED": [],
        }

        current_status = incident.case_status or "OPEN"

        if (
           new_status != current_status
           and new_status not in allowed_transitions.get(
            current_status,
            [],
         )
):
         return False

        incident.case_status = new_status

        if assigned_to is not None:
            incident.assigned_to = assigned_to

        # Store resolution information in the
        # recommendation field for now because the
        # current Incident model does not yet have
        # a dedicated resolution_notes column.
        if resolution_notes:
            incident.recommendation = resolution_notes

        db.commit()

        return True

    finally:
        db.close()
def audit(
    actor,
    role,
    action,
    details="",
    resource_id=None
):
    init_db()
    db = SessionLocal()
    try:
        db.add(
            AuditEvent(
                actor=actor,
                role=role,
                action=action,
                resource_id=resource_id,
                details=details
            )
        )
        db.commit()
    finally:
        db.close()
        


def recent_incidents(limit=25):
    init_db()
    db = SessionLocal()
    try:
        return (
            db.query(Incident)
            .order_by(Incident.created_at.desc())
            .limit(limit)
            .all()
        )
    finally:
        db.close()
def get_incident_by_id(incident_id):

    init_db()

    db = SessionLocal()

    try:
        return (
            db.query(Incident)
            .filter(Incident.id == incident_id)
            .first()
        )

    finally:
        db.close()        
def get_active_incident_by_tower(tower_id):
    init_db()

    db = SessionLocal()

    try:
        return (
            db.query(Incident)
            .filter(
                Incident.incident_type == "NETWORK",
                Incident.tower_id == tower_id,
                Incident.case_status.in_(
                    ["OPEN", "IN_PROGRESS"]
                ),
            )
            .order_by(Incident.id.desc())
            .first()
        )
    finally:
        db.close()        
def find_similar_historical_incidents(
    tower_id=None,
    issue=None,
    limit=5,
):
    """
    Retrieve relevant historical incidents for investigation.

    Historical incidents are used as supporting context only.
    They do not determine the current root cause.
    """
    init_db()
    session = SessionLocal()

    try:
        query = session.query(Incident)

        # Historical incidents only.
        query = query.filter(
            Incident.case_status.in_(
                ["RESOLVED", "CLOSED"]
            )
        )

        if tower_id:
            query = query.filter(
                Incident.tower_id == tower_id
            )

        incidents = (
            query
            .order_by(Incident.created_at.desc())
            .limit(limit)
            .all()
        )

        results = []

        for incident in incidents:
            results.append({
                "incident_id": incident.id,
                "tower_id": incident.tower_id,
                "issue": incident.issue,
                "diagnosis": incident.diagnosis,
                "recommendation": incident.recommendation,
                "root_cause_confidence": (
                    incident.root_cause_confidence
                ),
                "case_status": incident.case_status,
                "created_at": str(incident.created_at),
            })

        return results

    finally:
        session.close()        
def recent_audit_events(limit=50):
    init_db()
    db = SessionLocal()
    try:
        return (
            db.query(AuditEvent)
            .order_by(AuditEvent.created_at.desc())
            .limit(limit)
            .all()
        )
    finally:
        db.close()    
           
                     
def update_case_status(
    incident_id,
    new_status,
    actor="unknown",
    role="unknown",
):
    session = SessionLocal()

    try:
        incident = session.get(Incident, incident_id)

        if not incident:
            return False

        old_status = incident.case_status
        incident.case_status = new_status

        session.commit()

        audit(
            actor,
            role,
            "case_status_updated",
            (
                f"Incident {incident_id} status changed "
                f"from {old_status} to {new_status}."
            ),
            resource_id=str(incident_id),
        )

        return True

    except Exception:
        session.rollback()
        return False

def update_case_assignment(
    incident_id,
    assigned_to,
    actor="unknown",
    role="unknown",
):
    session = SessionLocal()

    try:
        incident = session.get(Incident, incident_id)

        if not incident:
            return False

        old_assignee = incident.assigned_to
        incident.assigned_to = assigned_to

        session.commit()

        audit(
            actor,
            role,
            "case_assignment_updated",
            (
                f"Incident {incident_id} assignment changed "
                f"from {old_assignee} to {assigned_to}."
            ),
            resource_id=str(incident_id),
        )

        return True

    except Exception:
        session.rollback()
        return False

    finally:
        session.close()        
def save_investigation_metric(
    thread_id,
    incident_id=None,
    tower_id=None,
    status="STARTED",
    outcome=None,
    execution_time_ms=None,
    human_review=None,
):
    init_db()
    db = SessionLocal()

    try:

        metric = None

        # When an incident is being attached, first look for
        # the existing timing metric for this thread.
        if incident_id is not None:

            metric = (
                db.query(InvestigationMetric)
                .filter(
                    InvestigationMetric.thread_id == thread_id,
                    InvestigationMetric.incident_id.is_(None),
                    InvestigationMetric.execution_time_ms.isnot(None),
                )
                .order_by(
                    InvestigationMetric.id.desc()
                )
                .first()
            )

        # Otherwise, find the latest metric for this thread.
        if metric is None:

            metric = (
                db.query(InvestigationMetric)
                .filter(
                    InvestigationMetric.thread_id == thread_id
                )
                .order_by(
                    InvestigationMetric.id.desc()
                )
                .first()
            )

        if metric:

            if incident_id is not None:
                metric.incident_id = incident_id

            if tower_id is not None:
                metric.tower_id = tower_id

            if status:
                metric.status = status

            if outcome is not None:
                metric.outcome = outcome

            if execution_time_ms is not None:
                metric.execution_time_ms = execution_time_ms

            if human_review is not None:
                metric.human_review = human_review

            db.commit()

            return metric.id

        metric = InvestigationMetric(
            thread_id=thread_id,
            incident_id=incident_id,
            tower_id=tower_id,
            status=status,
            outcome=outcome,
            execution_time_ms=execution_time_ms,
            human_review=human_review,
        )

        db.add(metric)
        db.commit()

        return metric.id

    finally:
        db.close()

def recent_investigation_metrics(limit=50):
    init_db()
    db = SessionLocal()

    try:
        return (
            db.query(InvestigationMetric)
            .order_by(
                InvestigationMetric.created_at.desc()
            )
            .limit(limit)
            .all()
        )

    finally:
        db.close()
def save_model_telemetry(
    thread_id=None,
    incident_id=None,
    provider=None,
    model_name=None,
    operation=None,
    input_tokens=None,
    output_tokens=None,
    total_tokens=None,
    latency_ms=None,
    estimated_cost=None,
    status="SUCCESS",
    error_message=None,
):
    init_db()

    db = SessionLocal()

    try:

        telemetry = ModelTelemetry(
            thread_id=thread_id,
            incident_id=incident_id,
            provider=provider,
            model_name=model_name,
            operation=operation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            estimated_cost=estimated_cost,
            status=status,
            error_message=error_message,
        )

        db.add(telemetry)

        db.commit()

        return telemetry.id

    finally:
        db.close()


def recent_model_telemetry(limit=100):
    init_db()

    db = SessionLocal()

    try:

        return (
            db.query(ModelTelemetry)
            .order_by(
                ModelTelemetry.created_at.desc()
            )
            .limit(limit)
            .all()
        )

    finally:
        db.close()        
        