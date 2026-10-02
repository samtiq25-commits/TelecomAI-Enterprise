import streamlit as st
import pandas as pd
import plotly.express as px
import uuid
import json
from pathlib import Path
from src.event_intelligence import detect_operational_events
from src.data_generator import load_data
from src.ml_models import detect_anomalies, predict_failures
from src.complaint_analyzer import classify_complaint
from langgraph.types import Command
from src.marketing_intelligence import generate_marketing_recommendations
from src.rag import retrieve
from src.agent import build_agent, list_investigation_threads
from src.database import (
    save_incident, recent_incidents,
    audit,
    recent_audit_events,
    update_case_status,
    update_case_assignment,
    save_investigation_metric,
    recent_investigation_metrics, get_active_incident_by_tower,
    get_incident_by_id, update_incident_status, recent_model_telemetry
)
from src.incident_correlation import (
    correlate_operational_events,
    correlate_and_audit_operational_events,
)
from src.security import verify_password, can_access, roles
from src.voice import transcribe_audio, text_to_speech
from src.voice_assistant import (
    process_voice_request,
    route_voice_intent,
    resume_voice_investigation,
)
from src.monitoring import incident_rows
from src.customer_intelligence import (
    segment_customers,
    describe_segments,
    generate_personalized_recommendation,
)
from src.fraud_detection import (
    predict_fraud,
    explain_fraud_risk,
)
st.set_page_config(page_title="TelecomAI Enterprise",
                   page_icon="📡", layout="wide")

# Demo RBAC gate
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "role" not in st.session_state:
    st.session_state.role = "viewer"

if not st.session_state.authenticated:
    st.title("📡 TelecomAI Enterprise")
    st.caption("Secure telecom intelligence workspace")
    with st.form("login"):
        role = st.selectbox("Role", roles(), index=1)
        password = st.text_input("Demo password", type="password")
        submitted = st.form_submit_button("Sign in")
        if submitted:
            if verify_password(password, role):
                st.session_state.authenticated = True
                st.session_state.role = role
                audit("demo-user", role, "login", "Successful dashboard login")
                st.rerun()
            else:
                st.error("Invalid demo password.")
    st.info("Set TELECOM_ADMIN_PASSWORD in .env before running.")
    st.stop()

network, equipment, complaints = load_data()
anomalies = detect_anomalies(network)
# Assign severity levels to network anomalies


def get_anomaly_severity(score):
    if score <= -0.10:
        return "Critical"
    elif score <= -0.05:
        return "High"
    elif score <= -0.02:
        return "Medium"
    else:
        return "Low"


anomalies["severity"] = anomalies["anomaly_score"].apply(
    get_anomaly_severity
)
failures = predict_failures(equipment)
operational_events = detect_operational_events(
    network,
    anomalies
)
incident_correlations = correlate_operational_events(
    operational_events
)
if "workspace_page" not in st.session_state:
    st.session_state.workspace_page = "Executive Dashboard"
with st.sidebar:
    st.title("TelecomAI")
    st.caption(f"Role: **{st.session_state.role}**")
    pages = [
        ("Executive Dashboard", "dashboard"),
        ("📡 Live NOC", "noc"),
        ("Network Intelligence", "network"),
        ("Predictive Maintenance", "maintenance"),
        ("Customer AI", "customer"),
        ("💳 Billing Fraud Intelligence", "fraud"),
        ("RAG Knowledge", "rag"),
        ("LangGraph Agent", "agent"),
        ("🎙️ Voice Assistant", "voice"),
        ("Incident Center", "incidents"),
        ("🤖 AI Evaluation", "evaluation"),
        ("📊 Observability", "observability"),
    ]
    allowed = [(label, key)
               for label, key in pages if can_access(st.session_state.role, key)]
    page = st.radio(
        "Workspace",
        [x[0] for x in allowed],
        index=(
            [x[0] for x in allowed].index(
                st.session_state.workspace_page
            )
            if st.session_state.workspace_page
            in [x[0] for x in allowed]
            else 0
        )
    )
    st.session_state.workspace_page = page
    if st.button("Sign out"):
        audit("demo-user", st.session_state.role, "logout")
        st.session_state.authenticated = False
        st.rerun()
if page == "Executive Dashboard":
    st.title("📊 Executive Dashboard")
    st.caption(
        "Enterprise-wide network health, operational risk, customer impact, and AI intelligence"
    )

    health = max(
        0,
        100 - anomalies.is_anomaly.mean() * 100
    )

    # ---------------------------------------------------------
    # Executive KPI Cards
    # ---------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Network Health",
        f"{health:.1f}%"
    )
    c1.caption("Overall network condition")

    c2.metric(
        "Active Anomalies",
        int(anomalies.is_anomaly.sum())
    )
    c2.caption("Machine-learning detected")

    c3.metric(
        "High/Critical Equipment",
        int(
            failures.risk_level.isin(
                ["HIGH", "CRITICAL"]
            ).sum()
        )
    )
    c3.caption("Requires attention")

    c4, c5 = st.columns(2)

    c4.metric(
        "Customer Complaints",
        len(complaints)
    )
    c4.caption("Current complaint volume")

    c5.metric(
        "Predicted Failures",
        int(
            (failures.failure_probability >= 0.6).sum()
        )
    )
    c5.caption("AI-predicted risk")

    st.divider()

    # ---------------------------------------------------------
    # Network & Customer Signals
    # ---------------------------------------------------------

    st.subheader("📡 Network & Customer Signals")

    a, b = st.columns(2)

    # ---------------------------------------------------------
    # Network KPI Trend
    # ---------------------------------------------------------

    with a:
        trend = (
            anomalies
            .assign(
                hour=anomalies.timestamp.dt.floor("h")
            )
            .groupby(
                "hour",
                as_index=False
            )
            .agg(
                latency=("latency", "mean"),
                packet_loss=("packet_loss", "mean")
            )
            .tail(100)
        )

        fig = px.line(
            trend,
            x="hour",
            y=["latency", "packet_loss"],
            title="Network KPI Trend"
        )

        fig.update_layout(
            height=300,
            margin=dict(
                l=20,
                r=20,
                t=50,
                b=20
            )
        )

        st.plotly_chart(
            fig,
            width="stretch",
            config={
                "displayModeBar": False
            }
        )

    # ---------------------------------------------------------
    # Customer Complaint Distribution
    # ---------------------------------------------------------

    with b:
        mix = (
            complaints.complaint_category
            .value_counts()
            .reset_index()
        )

        mix.columns = [
            "category",
            "count"
        ]

        fig = px.bar(
            mix,
            x="category",
            y="count",
            title="Customer Complaint Distribution"
        )

        fig.update_layout(
            height=300,
            margin=dict(
                l=20,
                r=20,
                t=50,
                b=20
            )
        )

        st.plotly_chart(
            fig,
            width="stretch",
            config={
                "displayModeBar": False
            }
        )

    # =========================================================
    # ENTERPRISE OPERATIONS OVERVIEW
    # =========================================================

    st.subheader("Enterprise Operations Overview")

    transactions = pd.read_csv(
        "data/transactions.csv"
    )

    total_transactions = len(
        transactions
    )

    flagged_transactions = int(
        transactions["fraud"].sum()
    )

    fraud_rate = (
        flagged_transactions
        / total_transactions
        * 100
        if total_transactions > 0
        else 0
    )

    incident_data = incident_rows(1000)

    total_incidents = len(
        incident_data
    )

    audit_rows = recent_audit_events(1000)

    approved_reviews = sum(
        1
        for row in audit_rows
        if row.action == "human_review_approved"
    )

    rejected_reviews = sum(
        1
        for row in audit_rows
        if row.action == "human_review_rejected"
    )

    total_reviews = (
        approved_reviews
        + rejected_reviews
    )

    approval_rate = (
        approved_reviews
        / total_reviews
        * 100
        if total_reviews > 0
        else 0
    )

    # ---------------------------------------------------------
    # Enterprise Operations KPI Cards
    # ---------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Flagged Transactions",
        flagged_transactions
    )
    c1.caption(
        "Potentially fraudulent activity"
    )

    c2.metric(
        "Fraud Rate",
        f"{fraud_rate:.1f}%"
    )
    c2.caption(
        "Flagged transaction percentage"
    )

    c3.metric(
        "Total Incidents",
        total_incidents
    )
    c3.caption(
        "Recorded operational incidents"
    )

    c4.metric(
        "Human Approval Rate",
        f"{approval_rate:.1f}%"
    )
    c4.caption(
        "Human-reviewed decisions"
    )
if page == "📡 Live NOC":

    st.title("📡 Live Network Operations Center")
    st.caption(
        "Real-time network health, tower status, anomalies, and operational incidents"
    )

    # ---------------------------------------------------------
    # Current Network Snapshot
    # ---------------------------------------------------------

    latest_network = (
        network
        .sort_values("timestamp")
        .groupby("tower_id", as_index=False)
        .tail(1)
        .copy()
    )

    # Get the latest anomaly status for each tower
    latest_anomalies = (
        anomalies
        .sort_values("timestamp")
        .groupby("tower_id", as_index=False)
        .tail(1)
        [["tower_id", "is_anomaly"]]
    )

    # Add anomaly status to the latest network snapshot
    latest_network = latest_network.merge(
        latest_anomalies,
        on="tower_id",
        how="left"
    )

    latest_network["is_anomaly"] = (
        latest_network["is_anomaly"]
        .fillna(False)
        .astype(bool)
    )

    total_towers = len(latest_network)

    active_anomalies = int(latest_network["is_anomaly"].sum())

    # Operational severity based on network KPIs
    def get_operational_status(row):
        if (
            bool(row["is_anomaly"])
            or row["latency"] > 100
            or row["packet_loss"] > 5
        ):
            return "🔴 CRITICAL"

        if row["latency"] > 30 or row["packet_loss"] > 2:
            return "🟡 WARNING"

        return "🟢 HEALTHY"

    latest_network["status"] = latest_network.apply(
        get_operational_status,
        axis=1,
    )

    critical_towers = int(
        (latest_network["status"] == "🔴 CRITICAL").sum()
    )

    warning_towers = int(
        (latest_network["status"] == "🟡 WARNING").sum()
    )

    healthy_towers = int(
        (latest_network["status"] == "🟢 HEALTHY").sum()
    )

    network_health = (
        (
            healthy_towers
            + (warning_towers * 0.75)
            + (critical_towers * 0.25)
        )
        / total_towers
        * 100
        if total_towers > 0
        else 0
    )
    # ---------------------------------------------------------
    # NOC KPI Cards
    # ---------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Network Health",
        f"{network_health:.1f}%"
    )

    c2.metric(
        "Critical Towers",
        critical_towers
    )

    c3.metric(
        "Warning Towers",
        warning_towers
    )

    c4.metric(
        "Healthy Towers",
        healthy_towers
    )

    st.caption(
        f"Monitoring {total_towers} towers • "
        f"{active_anomalies} machine-learning anomalies detected"
    )
    st.divider()

    # ---------------------------------------------------------
    # Tower Operational Status
    # ---------------------------------------------------------

    st.subheader("🗼 Tower Operational Status")

    tower_status = latest_network[
        [
            "tower_id",
            "timestamp",
            "latency",
            "packet_loss",
            "network_traffic",
            "status",
        ]
    ].copy()
    # ---------------------------------------------------------
    # Active Incident Correlation
    # ---------------------------------------------------------

    def get_active_incident_display(tower_id):
        incident = get_active_incident_by_tower(tower_id)
        if incident is None:
            return "None"
        return f"#{incident.id} — {incident.case_status}"

    tower_status["active_incident"] = tower_status["tower_id"].apply(
        get_active_incident_display
    )

    tower_status = tower_status.rename(
        columns={
            "tower_id": "Tower",
            "timestamp": "Last Update",
            "latency": "Latency (ms)",
            "packet_loss": "Packet Loss (%)",
            "network_traffic": "Traffic (%)",
            "status": "Status",
            "active_incident": "Active Incident",
        }
    )

    tower_status = tower_status[
        [
            "Tower",
            "Last Update",
            "Active Incident",
            "Latency (ms)",
            "Packet Loss (%)",
            "Traffic (%)",
            "Status",
        ]
    ]

    st.dataframe(
        tower_status.sort_values(
            "Status",
            ascending=True
        ),
        width="stretch",
        hide_index=True,
    )

    st.divider()

    # ---------------------------------------------------------
    # Network KPI Monitoring
    # ---------------------------------------------------------

    st.subheader("📊 Network KPI Monitoring")

    k1, k2 = st.columns(2)

    with k1:
        latency_chart = network.sort_values("timestamp").tail(300)

        st.plotly_chart(
            px.line(
                latency_chart,
                x="timestamp",
                y="latency",
                color="tower_id",
                title="Network Latency",
            ),
            width="stretch",
        )

    with k2:
        packet_chart = network.sort_values("timestamp").tail(300)

        st.plotly_chart(
            px.line(
                packet_chart,
                x="timestamp",
                y="packet_loss",
                color="tower_id",
                title="Packet Loss",
            ),
            width="stretch",
        )

    st.divider()

    # ---------------------------------------------------------
    # Active Network Anomalies
    # ---------------------------------------------------------

    st.subheader("🚨 Active Network Anomalies")

    active = anomalies[
        anomalies["is_anomaly"]
    ].sort_values(
        "timestamp",
        ascending=False
    )

    if active.empty:

        st.success(
            "No active network anomalies detected."
        )

    else:

        st.dataframe(
            active[
                [
                    "tower_id",
                    "timestamp",
                    "latency",
                    "packet_loss",
                    "network_traffic",
                    "anomaly_score",
                ]
            ].head(50),
            width="stretch",
            hide_index=True,
        )

    st.divider()
    st.subheader("⚡ Operational Event Intelligence")

    if operational_events:

        event_df = pd.DataFrame(
            operational_events
        )

        display_columns = [
            "timestamp",
            "tower_id",
            "event_type",
            "event_name",
            "event_severity",
            "kpi_severity",
            "operational_severity",
            "kpi_status",
            "reason",
        ]

        available_columns = [
            column
            for column in display_columns
            if column in event_df.columns
        ]

        st.dataframe(
            event_df[available_columns],
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.success(
            "No operational events currently require investigation.")

    # ---------------------------------------------------------
    # Recent Operational Incidents
    # ---------------------------------------------------------

    st.subheader("🛡️ Recent Operational Incidents")

    try:

        noc_incidents = incident_rows(20)

        if noc_incidents:

            incidents_df = pd.DataFrame(noc_incidents)

            display_columns = [
                column
                for column in [
                    "id",
                    "incident_type",
                    "severity",
                    "case_status",
                    "assigned_to",
                    "tower_id",
                    "created_at",
                ]
                if column in incidents_df.columns
            ]

            if display_columns:

                st.dataframe(
                    incidents_df[
                        display_columns
                    ],
                    width="stretch",
                    hide_index=True,
                )

            else:

                st.info(
                    "Recent incidents are available, "
                    "but no display fields were found."
                )

        else:

            st.success(
                "No recent operational incidents."
            )
    except Exception as e:

        st.warning(
            f"Unable to load recent incidents: {e}"
        )
    # ---------------------------------------------------------
    # Active Incident Correlation
    # ---------------------------------------------------------

    st.subheader("🔗 Active Incident Correlation")

    try:

        correlation_rows = incident_rows(1000)

        # -----------------------------------------------------
        # Keep only active NETWORK incidents
        # -----------------------------------------------------

        active_network_incidents = [
            incident
            for incident in correlation_rows
            if incident.get("incident_type") == "NETWORK"
            and incident.get("case_status") in (
                "OPEN",
                "IN_PROGRESS",
            )
        ]

        # -----------------------------------------------------
        # Keep only the latest active incident per tower
        #
        # Historical incidents remain in the database.
        # This only cleans the operational view.
        # -----------------------------------------------------

        latest_by_tower = {}

        for incident in active_network_incidents:

            tower_id = incident.get("tower_id")

            if not tower_id:
                continue

            existing = latest_by_tower.get(tower_id)

            if (
                existing is None
                or incident.get("id", 0)
                > existing.get("id", 0)
            ):
                latest_by_tower[tower_id] = incident

        active_network_incidents = list(
            latest_by_tower.values()
        )

        # Newest incidents first
        active_network_incidents.sort(
            key=lambda incident: incident.get("id", 0),
            reverse=True,
        )

        # -----------------------------------------------------
        # No active incidents
        # -----------------------------------------------------

        if not active_network_incidents:

            st.success(
                "No active network incidents require attention."
            )

        else:

            correlation_df = pd.DataFrame(
                active_network_incidents
            )

            display_columns = [
                column
                for column in [
                    "id",
                    "tower_id",
                    "case_status",
                    "priority",
                    "root_cause_confidence",
                    "diagnosis",
                    "thread_id",
                ]
                if column in correlation_df.columns
            ]

            st.dataframe(
                correlation_df[display_columns],
                width="stretch",
                hide_index=True,
            )

            st.caption(
                "Operational view shows the latest active "
                "network incident for each tower. Historical "
                "incident records are preserved."
            )

            st.divider()

            # -------------------------------------------------
            # Select active incident
            # -------------------------------------------------

            selected_correlation_id = st.selectbox(
                "Select an active incident",
                correlation_df["id"].tolist(),
                key="noc_correlation_incident",
            )

            selected_incident = correlation_df[
                correlation_df["id"]
                == selected_correlation_id
            ].iloc[0]

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Incident",
                    f"#{selected_incident['id']}",
                )

                st.write(
                    "**Tower:**",
                    selected_incident.get(
                        "tower_id",
                        "N/A",
                    ),
                )

                st.write(
                    "**Status:**",
                    selected_incident.get(
                        "case_status",
                        "N/A",
                    ),
                )

            with col2:

                st.write(
                    "**Root Cause Confidence:**",
                    selected_incident.get(
                        "root_cause_confidence",
                        "UNKNOWN",
                    ),
                )

                st.write(
                    "**Diagnosis:**",
                    selected_incident.get(
                        "diagnosis",
                        "No diagnosis available.",
                    ),
                )

            with col3:

                st.write(
                    "**Recommendation:**",
                    selected_incident.get(
                        "recommendation",
                        "No recommendation available.",
                    ),
                )

            # -------------------------------------------------
            # LangGraph Thread
            # -------------------------------------------------

            incident_thread_id = selected_incident.get(
                "thread_id"
            )

            if incident_thread_id:

                st.caption(
                    f"LangGraph Thread: "
                    f"{incident_thread_id}"
                )

    except Exception as e:

        st.warning(
            f"Unable to load active incident correlation: {e}"
        )
    # ---------------------------------------------------------
    # Event → Incident Correlation
    # ---------------------------------------------------------

    st.subheader("🔗 Event → Incident Correlation")

    try:

        incident_correlations = correlate_operational_events(
            operational_events
        )

        if incident_correlations:

            correlation_df = pd.DataFrame(
                incident_correlations
            )

            display_columns = [
                "action",
                "tower_id",
                "case_status",
                "event_type",
                "operational_severity",
                "reason",
            ]

            available_columns = [
                column
                for column in display_columns
                if column in correlation_df.columns
            ]

            st.dataframe(
                correlation_df[available_columns],
                width="stretch",
                hide_index=True,
            )

            if st.button(
                "⚙️ Process Incident Correlation",
                type="primary",
                key="process_event_correlation",
            ):

                processed = []

                incident_correlations = correlate_and_audit_operational_events(
                    operational_events
                )

                for correlation in incident_correlations:

                    if correlation["action"] == "CREATE_NEW":

                        severity = correlation[
                            "operational_severity"
                        ]

                        tower_id = correlation["tower_id"]

                        issue = (
                            f"Operational event detected at "
                            f"{tower_id}: "
                            f"{correlation['reason']}"
                        )

                        # ---------------------------------------------------------
                        # Create a dedicated LangGraph thread
                        # ---------------------------------------------------------

                        thread_id = (
                            f"event-{tower_id}-{uuid.uuid4().hex[:12]}"
                        )

                        # ---------------------------------------------------------
                        # Create incident and link it to LangGraph
                        # ---------------------------------------------------------

                        incident_id = save_incident({
                            "incident_type": "NETWORK",
                            "case_status": "OPEN",
                            "priority": severity,
                            "severity": severity,
                            "issue": issue,
                            "tower_id": tower_id,
                            "thread_id": thread_id,
                            "diagnosis": (
                                "Event-driven network condition "
                                "requires investigation."
                            ),
                            "recommendation": (
                                "Initiate network investigation "
                                "and validate affected tower KPIs."
                            ),
                            "root_cause_confidence": "UNKNOWN",
                        })

# ---------------------------------------------------------
# Start LangGraph investigation
# ---------------------------------------------------------

                        agent = build_agent()

                        config = {
                            "configurable": {
                                "thread_id": thread_id
                            }
                        }

                        agent.invoke(
                            {
                                "issue": issue,
                                "tower_id": tower_id,
                                "actor": "event-monitor",
                                "actor_role": "NOC Engineer",
                                "request_type": "NETWORK",
                            },
                            config=config,
                        )

                        processed.append(
                            f"Created Incident #{incident_id} "
                            f"for {tower_id} and started "
                            f"LangGraph investigation."
                        )
                    elif correlation["action"] == "UPDATE_EXISTING":

                        incident_id = correlation["incident_id"]
                        tower_id = correlation["tower_id"]
                        thread_id = correlation.get("thread_id")

                        # Repair older incidents that do not have
                        # a LangGraph investigation thread.
                        if not thread_id:
                            thread_id = (
                                f"event-{tower_id}-{uuid.uuid4().hex[:12]}"
                            )

                            incident_id = save_incident({
                                "incident_type": "NETWORK",
                                "case_status": correlation["case_status"],
                                "priority": correlation["operational_severity"],
                                "severity": correlation["operational_severity"],
                                "issue": (
                                    f"Operational event detected at "
                                    f"{tower_id}: "
                                    f"{correlation['reason']}"
                                ),
                                "tower_id": tower_id,
                                "thread_id": thread_id,
                                "diagnosis": (
                                    "Event-driven network condition "
                                    "requires investigation."
                                ),
                                "recommendation": (
                                    "Initiate network investigation "
                                    "and validate affected tower KPIs."
                                ),
                                "root_cause_confidence": "UNKNOWN",
                            })

                            agent = build_agent()

                            config = {
                                "configurable": {
                                    "thread_id": thread_id
                                }
                            }

                            agent.invoke(
                                {
                                    "issue": (
                                        f"Operational event detected at "
                                        f"{tower_id}: "
                                        f"{correlation['reason']}"
                                    ),
                                    "tower_id": tower_id,
                                    "actor": "event-monitor",
                                    "actor_role": "NOC Engineer",
                                    "request_type": "NETWORK",
                                },
                                config=config,
                            )

                            processed.append(
                                f"Repaired Incident #{incident_id} for "
                                f"{tower_id} and started LangGraph investigation."
                            )

                        else:
                            processed.append(
                                f"Existing Incident #{incident_id} "
                                f"correlated with {tower_id}"
                            )

                for message in processed:
                    st.success(message)

        else:

            st.info(
                "No recent operational events require "
                "incident correlation."
            )

    except Exception as e:

        st.warning(
            f"Unable to process event correlation: {e}"
        )

    # ---------------------------------------------------------
    # NOC Refresh
    # ---------------------------------------------------------

    if st.button(
        "🔄 Refresh Network Data",
        type="primary"
    ):
        st.rerun()
elif page == "Network Intelligence":
    st.title("🚨 Network Intelligence")
    # Network KPI Metrics
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Latency",
        f"{anomalies['latency'].mean():.2f} ms"
    )

    col2.metric(
        "Average Packet Loss",
        f"{anomalies['packet_loss'].mean():.2f}%"
    )

    col3.metric(
        "Average CPU Usage",
        f"{anomalies['cpu_usage'].mean():.2f}%"
    )
    # Network Anomaly Alerts
    st.subheader("🚨 Network Anomaly Alerts")

    anomaly_data = anomalies[anomalies["is_anomaly"]]

    alert_col1, alert_col2, alert_col3 = st.columns(3)

    alert_col1.metric(
        "Anomalous Records",
        len(anomaly_data)
    )

    alert_col2.metric(
        "Affected Towers",
        anomaly_data["tower_id"].nunique()
    )

    alert_col3.metric(
        "Lowest Anomaly Score",
        f"{anomaly_data['anomaly_score'].min():.6f}"
        if not anomaly_data.empty else "N/A"
    )
    # Top 10 Most Affected Towers
    st.subheader("📊 Top 10 Most Affected Towers")

    if not anomaly_data.empty:
        tower_summary = (
            anomaly_data.groupby("tower_id")
            .agg(
                Anomaly_Count=("is_anomaly", "count"),
                Average_Latency=("latency", "mean"),
                Average_Packet_Loss=("packet_loss", "mean")
            )
            .reset_index()
            .sort_values(
                "Anomaly_Count",
                ascending=False
            )
            .head(10)
        )
       # Assign severity based on anomaly count

        def get_severity(count):
            if count >= 13:
                return "High"
            elif count >= 10:
                return "Medium"
            else:
                return "Low"

        tower_summary["Severity"] = (
            tower_summary["Anomaly_Count"]
            .apply(get_severity)
        )

        st.dataframe(
            tower_summary,
            width="stretch",
            hide_index=True
        )
    else:
        st.info("No network anomalies detected.")

# ============================================================
# NETWORK INTELLIGENCE — INVESTIGATION, APPROVAL & VISUALIZATIONS
# ============================================================

    st.divider()
    st.subheader("🔎 Investigate an Affected Tower")

# ------------------------------------------------------------
# 1. TOWER INVESTIGATION
# ------------------------------------------------------------
    anomaly_data = anomalies[
        anomalies["is_anomaly"]
    ].copy()
    if not anomaly_data.empty:

        affected_towers = sorted(
            anomaly_data["tower_id"].dropna().unique()
        )

        selected_tower = st.selectbox(
            "Select tower for investigation",
            affected_towers,
            key="network_investigation_tower"
        )

        st.write("Selected Tower:", selected_tower)

        if st.button(
            "🔍 Investigate Tower",
            key="investigate_selected_tower",
            type="primary"
        ):

            from src.voice_assistant import process_voice_text

            tower_anomalies = anomaly_data[
                anomaly_data["tower_id"] == selected_tower
            ]

            tower_evidence = {
                "anomaly_count": int(len(tower_anomalies)),
                "average_latency": float(
                    tower_anomalies["latency"].mean()
                ),
                "average_packet_loss": float(
                    tower_anomalies["packet_loss"].mean()
                ),
            }

            investigation_result, investigation_response = (
                process_voice_text(
                    text=(
                        "Investigate network anomalies and "
                        "identify the root cause for tower "
                        f"{selected_tower}."
                    ),
                    audience="employee",
                    tower_id=selected_tower,
                    anomaly=tower_evidence,
                )
            )

        # Save investigation in Streamlit session state
            st.session_state[
                "network_investigation_result"
            ] = investigation_result

            st.session_state[
                "network_investigation_response"
            ] = investigation_response

            st.success("Investigation workflow executed.")

            st.write(investigation_response)

            st.json({
                "tower_id": selected_tower,
                "request_type": investigation_result.get(
                    "voice_request_type"
                ),
                "thread_id": investigation_result.get(
                    "thread_id"
                ),
                "network_decision": investigation_result.get(
                    "network_decision"
                ),
            })


# ------------------------------------------------------------
# 2. PERSISTENT NETWORK ENGINEER APPROVAL
# ------------------------------------------------------------

    pending_investigation = st.session_state.get(
        "network_investigation_result"
    )

# Safely handle missing or invalid session state
    if not isinstance(pending_investigation, dict):
        pending_investigation = None

    if (
        pending_investigation is not None
        and pending_investigation.get("network_decision")
        == "HUMAN_REVIEW"
    ):

        st.divider()


# ------------------------------------------------------------
# 3. NETWORK ANOMALY VISUALIZATIONS
# These remain outside the approval condition.
# ------------------------------------------------------------

    st.divider()
    st.subheader("📊 Network Anomaly Analysis")

    if not anomalies.empty:

        available_towers = sorted(
            anomalies["tower_id"].dropna().unique()
        )

        selected = st.multiselect(
            "Towers",
            available_towers,
            default=available_towers[:8],
            key="network_tower_filter"
        )

        if selected:

            df = anomalies[
                anomalies["tower_id"].isin(selected)
            ]

        else:

            df = anomalies

    # Scatter plot
        st.plotly_chart(
            px.scatter(
                df.tail(1500),
                x="latency",
                y="packet_loss",
                size="cpu_usage",
                color="is_anomaly",
                hover_data=[
                    "tower_id",
                    "timestamp"
                ],
            ),
            width="stretch"
        )

        # Anomaly records
        st.subheader("Anomaly Records")
        # Severity summary
        severity_counts = anomalies[
            anomalies["is_anomaly"]
        ]["severity"].value_counts()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Critical",
            severity_counts.get("Critical", 0),
        )

        col2.metric(
            "High",
            severity_counts.get("High", 0),
        )

        col3.metric(
            "Medium",
            severity_counts.get("Medium", 0),
        )

        col4.metric(
            "Low",
            severity_counts.get("Low", 0),
        )

        # Severity filter
        severity_options = [
            "All",
            "Critical",
            "High",
            "Medium",
            "Low",
        ]

        selected_severity = st.selectbox(
            "Filter by Severity",
            severity_options,
            key="network_anomaly_severity_filter",
        )
        # Tower-specific filter
        tower_options = ["All"] + sorted(
            anomalies.loc[
                anomalies["is_anomaly"],
                "tower_id"
            ].dropna().unique().tolist()
        )

        selected_tower = st.selectbox(
            "Filter by Tower",
            tower_options,
            key="network_anomaly_tower_filter",
        )
        # Select detected anomalies
        anomaly_records = anomalies[
            anomalies["is_anomaly"]
        ].copy()

        # Apply severity filter
        if selected_severity != "All":
            anomaly_records = anomaly_records[
                anomaly_records["severity"]
                == selected_severity
            ]

        # Apply tower filter
        if selected_tower != "All":
            anomaly_records = anomaly_records[
                anomaly_records["tower_id"]
                == selected_tower
            ]

        # Sort most anomalous first
        anomaly_records = (
            anomaly_records
            .sort_values(
                "anomaly_score",
                ascending=True
            )
            .head(50)
        )

        # Color severity levels
        def highlight_severity(value):
            colors = {
                "Critical": "background-color: #7f1d1d; color: white",
                "High": "background-color: #c2410c; color: white",
                "Medium": "background-color: #a16207; color: white",
                "Low": "background-color: #166534; color: white",
            }

            return colors.get(value, "")

        styled_anomalies = anomaly_records.style.map(
            highlight_severity,
            subset=["severity"]
        )

        st.dataframe(
            styled_anomalies,
            width="stretch",
            hide_index=True
        )

    else:

        st.info(
            "No network anomaly data is currently available."
        )
    st.divider()

# ------------------------------------------------------------
# SAFE NETWORK ENGINEER APPROVAL
# ------------------------------------------------------------

    pending_investigation = st.session_state.get(
        "network_investigation_result"
    )

    if (
        isinstance(pending_investigation, dict)
        and pending_investigation.get("network_decision")
        == "HUMAN_REVIEW"
    ):

        st.divider()

        st.subheader("Network Engineer Approval")

        st.warning(
            "Engineer approval is required before incident creation."
        )

        st.markdown("### Investigation Summary")

        st.write(
            f"**Tower:** "
            f"{pending_investigation.get('tower_id', 'N/A')}"
        )

        st.write(
            "**Status:** Awaiting Network Engineer Approval"
        )

        engineer_feedback = st.text_area(
            "Engineer review feedback (optional)",
            key="network_engineer_feedback",
            placeholder="Enter findings or approval notes..."
        )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
           "✅ Approve Investigation",
           key="approve_network_investigation"
           ):

            from src.voice_assistant import (
                resume_voice_investigation
            )

            thread_id = pending_investigation.get(
                "thread_id"
            )

            if not thread_id:
                st.error("Missing investigation thread ID.")

            else:
                incident_notice = st.empty()

                def show_created_incident(incident_id):
                    incident_notice.success(
                        f"Incident {incident_id} created. "
                        "Completing recovery verification..."
                    )
                with st.spinner(
                    "Approving and resuming investigation..."
                ):

                    resumed_result = resume_voice_investigation(
                        thread_id=thread_id,
                        decision="approved",
                        feedback=engineer_feedback,
                        reviewer_actor="admin",
                        reviewer_role="Network Engineer",
                        on_incident_created=show_created_incident,
                    )

                st.session_state[
                    "network_investigation_result"
                ] = resumed_result

                st.session_state[
                    "voice_result"
                ] = resumed_result

                st.success(
                    "Investigation approved and workflow resumed."
                )

                st.rerun()

    with col2:

        if st.button(
           "❌ Reject Investigation",
           key="reject_network_investigation"
           ):

            from src.voice_assistant import (
                resume_voice_investigation
            )

            thread_id = pending_investigation.get(
                "thread_id"
            )

            if not thread_id:
                st.error("Missing investigation thread ID.")

            else:
                with st.spinner(
                    "Recording rejection and resuming workflow..."
                ):

                    resumed_result = resume_voice_investigation(
                        thread_id=thread_id,
                        decision="rejected",
                        feedback=engineer_feedback,
                        reviewer_actor="admin",
                        reviewer_role="Network Engineer",
                    )

                st.session_state[
                    "network_investigation_result"
                ] = resumed_result

                st.session_state[
                    "voice_result"
                ] = resumed_result

                st.warning(
                    "Investigation rejected and workflow resumed."
                )


elif page == "Predictive Maintenance":
    st.title("🔧 Predictive Maintenance")
    risk = st.slider("Minimum failure probability", 0.0, 1.0, .60, .05)
    df = failures[failures.failure_probability >= risk].copy()
    df["failure_probability"] = (df.failure_probability*100).round(1)
    st.dataframe(df, width="stretch", hide_index=True)

elif page == "Customer AI":
    st.title("🤖 Customer Complaint Intelligence")

    text = st.text_area("Enter customer complaint")

    if st.button("Analyze", type="primary"):
        a = classify_complaint(text)

        st.write("Category:", a["category"])
        st.write("Sentiment:", a["sentiment"])
        st.write("Priority:", a["priority"])

        audit(
            "demo-user",
            st.session_state.role,
            "complaint_analysis",
            text[:500]
        )

    st.divider()
    st.subheader("👤 Customer Personalization")

    customers = pd.read_csv("data/customer_segments.csv")

    selected_customer = st.selectbox(
        "Select Customer",
        customers["customer_id"].tolist()
    )

    customer = customers[
        customers["customer_id"] == selected_customer
    ].iloc[0]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Customer Segment",
        customer["segment_name"]
    )

    c2.metric(
        "Monthly Spend",
        f"{customer['monthly_spend']:.2f}"
    )

    c3.metric(
        "Tenure",
        f"{int(customer['tenure_months'])} months"
    )

    st.subheader("Customer Behavior")

    b1, b2, b3 = st.columns(3)

    b1.metric(
        "Data Usage",
        f"{customer['data_usage_gb']:.1f} GB"
    )

    b2.metric(
        "Call Minutes",
        int(customer["call_minutes"])
    )

    b3.metric(
        "Complaints",
        int(customer["complaints"])
    )

    st.subheader("🎯 Personalized Recommendations")

    recommendations = generate_personalized_recommendation(
        customer.to_dict()
    )

    for recommendation in recommendations:
        st.write(f"• {recommendation}")

    st.divider()
    st.subheader("📣 Marketing Intelligence")

    marketing_recommendations = generate_marketing_recommendations(
        customer.to_dict()
    )

    for recommendation in marketing_recommendations:
        priority = recommendation["priority"]

        if priority == "CRITICAL":
            st.error(
                f"🔴 {recommendation['campaign']} — {priority}"
            )

        elif priority == "HIGH":
            st.warning(
                f"🟠 {recommendation['campaign']} — {priority}"
            )

        elif priority == "MEDIUM":
            st.info(
                f"🔵 {recommendation['campaign']} — {priority}"
            )

        else:
            st.success(
                f"🟢 {recommendation['campaign']} — {priority}"
            )

        st.write(recommendation["reason"])
elif page == "💳 Billing Fraud Intelligence":
    st.title("💳 Billing Fraud Intelligence")

    transactions = pd.read_csv("data/transactions.csv")
    from src.fraud_detection import predict_risk_levels
    transactions = predict_risk_levels(transactions)
    # Fraud overview
    total_transactions = len(transactions)
    fraud_transactions = int(transactions["fraud"].sum())
    fraud_rate = (
        fraud_transactions / total_transactions * 100
        if total_transactions
        else 0
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total Transactions",
        total_transactions
    )

    c2.metric(
        "Known Fraud Transactions",
        fraud_transactions
    )
    c3.metric(
        "Fraud Rate",
        f"{fraud_rate:.1f}%"
    )
    st.subheader("🤖 Fraud Model Performance")

    p1, p2, p3, p4 = st.columns(4)

    p1.metric("Accuracy", "99%")
    p2.metric("Fraud Precision", "97%")
    p3.metric("Fraud Recall", "93%")
    p4.metric("Fraud F1 Score", "95%")

    st.caption(
        "Model performance is measured on a held-out test set of 1,000 transactions. "
        "The dashboard risk distribution is generated separately across the full transaction dataset. "
        "Fraud metrics refer to the fraud class."
    )

    st.divider()

    st.subheader("Fraud Risk Distribution")

    risk_distribution = (
        transactions["risk_level"]
        .value_counts()
        .reindex(
            ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
            fill_value=0
        )
    )
    risk_counts = risk_distribution

    risk_counts = risk_distribution

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("LOW", int(risk_counts["LOW"]))
    c2.metric("MEDIUM", int(risk_counts["MEDIUM"]))
    c3.metric("HIGH", int(risk_counts["HIGH"]))
    c4.metric("CRITICAL", int(risk_counts["CRITICAL"]))

    elevated_risk = int(
        risk_counts["HIGH"] + risk_counts["CRITICAL"]
    )

    c5.metric("Elevated Risk", elevated_risk)

    st.caption(
        f"Fraud investigation queue: "
        f"{int(risk_counts['CRITICAL'])} critical and "
        f"{int(risk_counts['HIGH'])} high-risk transactions."
    )
    st.plotly_chart(
        px.bar(
            risk_distribution.reset_index(),
            x="risk_level",
            y="count",
            labels={
                "risk_level": "Risk Level",
                "count": "Transactions",
            },
            title="Model-Predicted Risk Distribution",
        ),
        width="stretch",
    )

    st.caption(
        "This chart shows the risk distribution generated by the fraud detection model. "
        "Risk levels are assigned using the predicted fraud probability: "
        "LOW, MEDIUM, HIGH, and CRITICAL."
    )

    st.info(
        f"Out of {total_transactions:,} transactions, "
        f"{fraud_transactions:,} were flagged as potentially fraudulent "
        f"({fraud_rate:.1f}%). The remaining "
        f"{total_transactions - fraud_transactions:,} transactions were classified as LOW risk."
    )

    st.subheader("Fraud Distribution")

    fraud_mix = (
        transactions["fraud"]
        .map({
            0: "Legitimate",
            1: "Fraud"
        })
        .value_counts()
        .reset_index()
    )

    fraud_mix.columns = ["transaction_type", "count"]

    st.plotly_chart(
        px.pie(
            fraud_mix,
            names="transaction_type",
            values="count"
        ),
        width="stretch"
    )

    st.divider()

    # Recent suspicious transactions
    st.subheader("🚨 Recent Suspicious Transactions")

    risk_filter = st.selectbox(
        "Filter by Risk Level",
        ["All", "HIGH", "CRITICAL"],
    )

    if risk_filter == "All":
        suspicious = transactions[
            transactions["risk_level"].isin(["HIGH", "CRITICAL"])
        ]
    else:
        suspicious = transactions[
            transactions["risk_level"] == risk_filter
        ]

    risk_priority = {
        "CRITICAL": 0,
        "HIGH": 1,
    }

    suspicious = suspicious.assign(
        risk_priority=suspicious["risk_level"].map(risk_priority)
    )

    suspicious = suspicious.sort_values(
        ["risk_priority", "timestamp"],
        ascending=[True, False],
    ).drop(
        columns=["risk_priority"]
    )
    display_suspicious = suspicious.head(50).copy()
    display_suspicious["Risk Indicators"] = (
        display_suspicious.apply(
            lambda row: "; ".join(
                explain_fraud_risk(row.to_dict())
            ),
            axis=1,
        )
    )
    display_suspicious["Fraud Probability (%)"] = (
        display_suspicious["fraud_probability"] * 100
    ).round(1)

    display_suspicious = display_suspicious.drop(
        columns=["fraud_probability"]
    )
    st.dataframe(
        display_suspicious,
        width="stretch",
        hide_index=True,
    )
    st.subheader("📝 Fraud Case Review")

    review_transaction = st.selectbox(
        "Select Transaction",
        suspicious["transaction_id"].tolist(),
    )
    review_decision = st.selectbox(
        "Review Decision",
        ["Pending Review", "Approved", "Rejected"],
    )
    if st.button("Submit Review", type="primary"):

        if review_decision == "Pending Review":
            st.warning("Please select Approved or Rejected before submitting.")

        else:
            selected_transaction = suspicious[
                suspicious["transaction_id"] == review_transaction
            ].iloc[0]

            audit(
                "demo-user",
                st.session_state.role,
                f"fraud_review_{review_decision.lower()}",
                (
                    f"Transaction reviewed with model risk level "
                    f"{selected_transaction['risk_level']} and fraud probability "
                    f"{selected_transaction['fraud_probability']:.4f}."
                ),
                resource_id=str(review_transaction),
            )
            if review_decision == "Approved":
                st.success(
                    f"Transaction {review_transaction} approved and logged.")
            elif review_decision == "Rejected":
                st.success(
                    f"Transaction {review_transaction} rejected and logged.")

    # Manual transaction risk analysis
    st.subheader("🔍 Analyze a Transaction")

    col1, col2 = st.columns(2)

    with col1:
        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=4.99,
            step=1.0
        )

        previous_transactions = st.number_input(
            "Previous Transactions",
            min_value=0,
            value=15,
            step=1
        )

        transaction_velocity = st.number_input(
            "Transaction Velocity",
            min_value=0,
            value=1,
            step=1
        )

        is_new_device = st.selectbox(
            "New Device",
            [0, 1],
            index=0,
            format_func=lambda x: "Yes" if x else "No"
        )

    with col2:
        hour = st.slider(
            "Transaction Hour",
            0,
            23,
            14
        )

        country = st.selectbox(
            "Country",
            sorted(transactions["country"].unique())
        )

        payment_type = st.selectbox(
            "Payment Type",
            sorted(transactions["payment_type"].unique())
        )

        device_type = st.selectbox(
            "Device Type",
            sorted(transactions["device_type"].unique())
        )

    if st.button("Analyze Fraud Risk", type="primary"):

        transaction = {
            "amount": amount,
            "previous_transactions": previous_transactions,
            "transaction_velocity": transaction_velocity,
            "is_new_device": is_new_device,
            "hour": hour,
            "country": country,
            "payment_type": payment_type,
            "device_type": device_type,
        }

        result = predict_fraud(transaction)

        st.subheader("Risk Assessment")

        r1, r2 = st.columns(2)

        r1.metric(
            "Fraud Probability",
            f"{result['fraud_probability'] * 100:.1f}%"
        )

        r2.metric(
            "Risk Level",
            result["risk_level"]
        )
        if result["risk_level"] in ["HIGH", "CRITICAL"]:
            st.warning(
                "This transaction should receive additional fraud review "
                "based on the model's predicted risk."
            )
        elif result["risk_level"] == "MEDIUM":
            st.info(
                "This transaction has moderate fraud risk and may require "
                "additional review."
            )
        else:
            st.success(
                "This transaction has low predicted fraud risk."
            )
        if result["risk_level"] == "CRITICAL":
            st.error("🚨 Critical fraud risk detected.")

        elif result["risk_level"] == "HIGH":
            st.warning("⚠️ High fraud risk detected.")

        elif result["risk_level"] == "MEDIUM":
            st.info("⚠️ Medium fraud risk detected.")

        else:
            st.success("✅ Low fraud risk.")

        st.subheader("Risk Indicators")

        for reason in result["reasons"]:
            st.write(f"• {reason}")

        audit(
            "demo-user",
            st.session_state.role,
            "fraud_analysis",
            f"Fraud probability: {result['fraud_probability']}"
        )

elif page == "RAG Knowledge":
    st.title("📚 Grounded Telecom Knowledge")
    q = st.text_area(
        "Question", "What should we check when latency and packet loss increase?")
    if st.button("Retrieve", type="primary"):
        for d in retrieve(q):
            with st.expander(f"{d['source']} · relevance {d['score']:.2f}"):
                st.write(d["content"])

elif page == "LangGraph Agent":

    st.title("🧠 LangGraph Troubleshooting Agent")

    issue = st.text_area(
        "Issue",
        "Internet is very slow and packet loss appears high."
    )

    # Initialize agent session state
    if "agent_thread_id" not in st.session_state:
        st.session_state.agent_thread_id = None

    if "agent" not in st.session_state:
        st.session_state.agent = None

    if "agent_config" not in st.session_state:
        st.session_state.agent_config = None

    if "agent_result" not in st.session_state:
        st.session_state.agent_result = None

    if "agent_issue" not in st.session_state:
        st.session_state.agent_issue = ""
    # ---------------------------------------------------------
    # TOWER SELECTION
    # ---------------------------------------------------------

    tower_options = sorted(
        anomalies.tower_id.dropna().unique().tolist()
    )

    if not tower_options:
        tower_options = ["TWR-1024"]

    default_tower_index = (
        tower_options.index("TWR-1024")
        if "TWR-1024" in tower_options
        else 0
    )

    tower = st.selectbox(
        "Tower ID",
        tower_options,
        index=default_tower_index,
        key="agent_tower_selection"
    )

    # ---------------------------------------------------------
    # INVESTIGATION SELECTION
    # ---------------------------------------------------------

    try:
        saved_threads = list_investigation_threads(
            limit=50
        )

        investigation_options = [
            "New Investigation"
        ] + list(saved_threads.keys())

    except Exception as e:
        saved_threads = {}
        investigation_options = [
            "New Investigation"
        ]

        st.warning(
            f"Unable to load saved investigations: {e}"
        )

    selected_investigation = st.selectbox(
        "Investigation",
        investigation_options,
        key="agent_investigation_selection"
    )

    if selected_investigation == "New Investigation":

        st.info(
            "A new investigation will receive a unique "
            "Thread ID when you run the agent."
        )

    else:

        st.code(
            selected_investigation,
            language="text"
        )

        if st.button(
            "▶️ Resume Investigation",
            key="resume_network_agent"
        ):

            try:
                agent = build_agent()

                config = {
                    "configurable": {
                        "thread_id": selected_investigation
                    }
                }

                state = agent.get_state(config)

                if not state.values:

                    st.warning(
                        "No saved investigation was found "
                        "for this Thread ID."
                    )

                else:

                    st.session_state.agent_thread_id = (
                        selected_investigation
                    )

                    st.session_state.agent = agent
                    st.session_state.agent_config = config
                    st.session_state.agent_result = (
                        state.values
                    )

                    st.session_state.agent_issue = (
                        state.values.get(
                            "issue",
                            ""
                        )
                    )

                    st.success(
                        "Investigation checkpoint restored successfully."
                    )

                    st.warning(
                        "Investigation rejected and workflow resumed."
                    )

                    st.rerun()

            except Exception as e:

                st.error(
                    f"Unable to resume investigation: {e}"
                )

    # ---------------------------------------------------------
    # CURRENT INVESTIGATION THREAD
    # ---------------------------------------------------------

    if st.session_state.agent_thread_id:

        st.info(
            f"Investigation Thread ID: "
            f"{st.session_state.agent_thread_id}"
        )

    # ---------------------------------------------------------
    # RUN AGENT
    # ---------------------------------------------------------

    if st.button("Run Agent", type="primary", key="run_network_agent"):

        # Create a unique LangGraph thread
        st.session_state.agent_thread_id = (
            f"streamlit-{tower}-{uuid.uuid4().hex}"
        )

        config = {
            "configurable": {
                "thread_id": st.session_state.agent_thread_id
            }
        }

        # -----------------------------------------------------
        # GET LATEST ANOMALY FOR THE SELECTED TOWER
        # -----------------------------------------------------

        recent = (
            anomalies[
                anomalies.tower_id == tower
            ]
            .sort_values("timestamp")
            .tail(1)
        )

        anomaly = (
            recent.iloc[0].to_dict()
            if not recent.empty
            else {}
        )

        # Convert timestamp to string if present
        if "timestamp" in anomaly:
            anomaly["timestamp"] = str(
                anomaly["timestamp"]
            )

        # -----------------------------------------------------
        # BUILD LANGGRAPH AGENT
        # -----------------------------------------------------

        agent = build_agent()

        # -----------------------------------------------------
        # INITIAL AGENT INVOCATION
        # -----------------------------------------------------

        result = agent.invoke(
            {
                "issue": issue,
                "tower_id": tower,
                "actor": st.session_state.get("actor", "admin"),
                "actor_role": st.session_state.get(
                    "actor_role",
                    "Network Engineer"
                ),
                "anomaly": anomaly,
            },
            config=config,
        )

        # -----------------------------------------------------
        # SAVE AGENT STATE
        # -----------------------------------------------------

        st.session_state.agent = agent
        st.session_state.agent_config = config
        st.session_state.agent_issue = issue
        st.session_state.agent_result = result

        st.rerun()

    # ---------------------------------------------------------
    # DISPLAY AGENT RESULT
    # ---------------------------------------------------------

    if (
        st.session_state.agent is not None
        and st.session_state.agent_config is not None
        and st.session_state.agent_result is not None
    ):

        agent = st.session_state.agent
        config = st.session_state.agent_config
        result = st.session_state.agent_result

        # -----------------------------------------------------
        # GET CURRENT LANGGRAPH STATE
        # -----------------------------------------------------

        state = agent.get_state(config)

        # -----------------------------------------------------
        # HUMAN ENGINEER REVIEW REQUIRED
        # -----------------------------------------------------

        if state.next:

            st.warning(
                "⏸️ Human engineer review required before "
                "the agent can continue."
            )

            review = state.values

            # -------------------------------------------------
            # REVIEW DETAILS
            # -------------------------------------------------

            st.subheader("🔍 Review Details")

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    f"**Tower ID:** "
                    f"{review.get('tower_id', tower)}"
                )

                st.markdown(
                    f"**Root Cause:** "
                    f"{review.get('root_cause', 'Not available')}"
                )

                st.markdown(
                    f"**Confidence:** "
                    f"{review.get('root_cause_confidence', 'UNKNOWN')}"
                )

            with col2:

                st.markdown(
                    f"**Verification Status:** "
                    f"{review.get('verification_status', 'UNKNOWN')}"
                )

                st.markdown(
                    f"**Verification Action:** "
                    f"{review.get('verification_action', 'Not available')}"
                )

                st.markdown(
                    f"**Network Decision:** "
                    f"{review.get('network_decision', 'Not available')}"
                )

            st.divider()

            # -------------------------------------------------
            # TECHNICAL EVIDENCE
            # -------------------------------------------------

            st.subheader("📊 Technical Evidence")

            anomaly_data = review.get("anomaly", {})

            if anomaly_data:

                evidence_col1, evidence_col2 = st.columns(2)

                with evidence_col1:

                    if "latency_ms" in anomaly_data:
                        st.metric(
                            "Latency",
                            f"{anomaly_data['latency_ms']} ms"
                        )

                    if "packet_loss" in anomaly_data:
                        st.metric(
                            "Packet Loss",
                            f"{anomaly_data['packet_loss']}%"
                        )

                    if "download_speed_mbps" in anomaly_data:
                        st.metric(
                            "Download Speed",
                            f"{anomaly_data['download_speed_mbps']} Mbps"
                        )

                with evidence_col2:

                    if "cpu_usage" in anomaly_data:
                        st.metric(
                            "CPU Usage",
                            f"{anomaly_data['cpu_usage']}%"
                        )

                    if "memory_usage" in anomaly_data:
                        st.metric(
                            "Memory Usage",
                            f"{anomaly_data['memory_usage']}%"
                        )

                    if "signal_strength" in anomaly_data:
                        st.metric(
                            "Signal Strength",
                            f"{anomaly_data['signal_strength']}"
                        )

            else:

                st.info(
                    "No structured anomaly data was available "
                    "for this tower."
                )

            st.divider()

            # -------------------------------------------------
            # ENGINEER FEEDBACK
            # -------------------------------------------------

            st.subheader("👨‍💻 Human Engineer Review")

            human_feedback = st.text_area(
                "Engineer Feedback",
                placeholder=(
                    "Explain why you approve or reject "
                    "the diagnosis..."
                ),
                key="network_engineer_feedback",
            )

            st.caption(
                "The feedback will be stored in the audit trail "
                "together with your review decision."
            )

            # -------------------------------------------------
            # APPROVE / REJECT BUTTONS
            # -------------------------------------------------

            col1, col2 = st.columns(2)

            # =================================================
            # APPROVE
            # =================================================

            with col1:

                if st.button(
                    "✅ Approve",
                    type="primary",
                    key="approve_network_review",
                ):

                    result = agent.invoke(
                        Command(
                            resume={
                                "decision": "approved",
                                "feedback": human_feedback,
                            }
                        ),
                        config=config,
                    )
                    st.write("DEBUG RESULT:", result)
                    # -----------------------------------------
                    # AUDIT HUMAN APPROVAL
                    # -----------------------------------------

                    audit(
                        "human-engineer",
                        st.session_state.role,
                        "human_review_approved",
                        (
                            f"Network engineer approved diagnosis "
                            f"for tower {tower}. "
                            f"Engineer Feedback: "
                            f"{human_feedback or 'No feedback provided'}"
                        ),
                        tower,
                    )

                    # -----------------------------------------
                    # SAVE INCIDENT
                    # -----------------------------------------

                    save_incident(result)

                    # -----------------------------------------
                    # SAVE RESULT
                    # -----------------------------------------

                    st.session_state.agent_result = result

                    st.success(
                        "Human engineer approved the diagnosis."
                    )

                    st.rerun()

            # =================================================
            # REJECT
            # =================================================

            with col2:

                if st.button(
                    "❌ Reject",
                    key="reject_network_review",
                ):

                    result = agent.invoke(
                        Command(
                            resume={
                                "decision": "rejected",
                                "feedback": human_feedback,
                            }
                        ),
                        config=config,
                    )

                    # -----------------------------------------
                    # AUDIT HUMAN REJECTION
                    # -----------------------------------------

                    audit(
                        "human-engineer",
                        st.session_state.role,
                        "human_review_rejected",
                        (
                            f"Network engineer rejected diagnosis "
                            f"for tower {tower}. "
                            f"Engineer Feedback: "
                            f"{human_feedback or 'No feedback provided'}"
                        ),
                        tower,
                    )

                    # -----------------------------------------
                    # SAVE RESULT
                    # -----------------------------------------

                    st.session_state.agent_result = result

                    st.error(
                        "Human engineer rejected the diagnosis."
                    )

                    st.rerun()

        # -----------------------------------------------------
        # AGENT FINISHED
        # -----------------------------------------------------

        else:

            st.success(
                "✅ LangGraph workflow completed."
            )

            # -------------------------------------------------
            # FINAL STATE
            # -------------------------------------------------

            final_state = state.values

            # -------------------------------------------------
            # DIAGNOSIS
            # -------------------------------------------------

            st.subheader("🩺 Diagnosis")

            diagnosis = final_state.get(
                "diagnosis",
                "No diagnosis available."
            )

            st.write(diagnosis)

            # -------------------------------------------------
            # CONFIDENCE
            # -------------------------------------------------

            confidence = final_state.get(
                "root_cause_confidence",
                "UNKNOWN"
            )

            st.info(
                f"Root Cause Confidence: **{confidence}**"
            )

            # -------------------------------------------------
            # VERIFICATION
            # -------------------------------------------------

            st.subheader("🔎 Verification")

            verification_status = final_state.get(
                "verification_status",
                "UNKNOWN"
            )

            verification_action = final_state.get(
                "verification_action",
                "No verification action recorded."
            )

            st.write(
                f"**Verification Status:** "
                f"{verification_status}"
            )

            st.write(
                f"**Verification Action:** "
                f"{verification_action}"
            )

            # -------------------------------------------------
            # HUMAN REVIEW STATUS
            # -------------------------------------------------

            human_review_status = final_state.get(
                "human_review_status",
                "Not required"
            )

            human_review_feedback = final_state.get(
                "human_review_feedback",
                "No feedback provided"
            )

            st.subheader("👨‍💻 Human Engineer Review")

            st.write(
                f"**Review Status:** "
                f"{human_review_status}"
            )

            st.write(
                f"**Engineer Feedback:** "
                f"{human_review_feedback}"
            )

            # -------------------------------------------------
            # NETWORK DECISION
            # -------------------------------------------------

            network_decision = final_state.get(
                "network_decision",
                "UNKNOWN"
            )

            network_decision_reason = final_state.get(
                "network_decision_reason",
                "No decision reason recorded."
            )

            st.subheader("🧠 Agent Decision")

            st.write(
                f"**Decision:** {network_decision}"
            )

            st.write(
                f"**Reason:** {network_decision_reason}"
            )

            # -------------------------------------------------
            # RECOMMENDATION
            # -------------------------------------------------

            recommendation = final_state.get(
                "recommendation"
            )

            if recommendation:

                st.subheader("🛠️ Recommendation")

                st.write(recommendation)

            else:

                st.warning(
                    "No recommendation was generated."
                )

            # -------------------------------------------------
            # INCIDENT
            # -------------------------------------------------

            incident_id = final_state.get(
                "incident_id"
            )

            if incident_id:

                st.subheader("📋 Incident")

                st.write(
                    f"**Incident ID:** {incident_id}"
                )

            # -------------------------------------------------
            # WORKFLOW SUMMARY
            # -------------------------------------------------

            with st.expander("🔬 LangGraph State"):

                st.json(final_state)

            # -------------------------------------------------
            # START NEW INVESTIGATION
            # -------------------------------------------------

            if st.button(
                "🔄 Start New Investigation",
                key="new_network_investigation",
            ):

                st.session_state.agent = None
                st.session_state.agent_config = None
                st.session_state.agent_result = None
                st.session_state.agent_thread_id = None

                st.rerun()
elif page == "🎙️ Voice Assistant":

    st.title("🎙️ TelecomAI Voice Contact Center")

    import uuid

    # -------------------------
    # Voice session state
    # -------------------------

    if "voice_thread_id" not in st.session_state:
        st.session_state.voice_thread_id = (
            f"voice-{uuid.uuid4().hex}"
        )

    if "voice_result" not in st.session_state:
        st.session_state.voice_result = None

    if "voice_response" not in st.session_state:
        st.session_state.voice_response = ""

    if "voice_transaction" not in st.session_state:
        st.session_state.voice_transaction = {}

    st.caption(
        f"Voice session: {st.session_state.voice_thread_id}"
    )

    st.write(
        "The assistant can listen to and respond to both "
        "**customers** and **employees**. Customer responses "
        "are empathetic and non-technical; employee responses "
        "include diagnosis, operational evidence, and "
        "recommended actions."
    )

    if st.session_state.role not in (
        "admin",
        "noc_engineer",
        "support"
    ):
        st.warning(
            "Voice access is restricted to operational/support roles."
        )
        st.stop()

    # -------------------------
    # Voice input
    # -------------------------

    audience = st.radio(
        "Who is speaking?",
        ["Customer", "Employee"],
        horizontal=True
    )

    tower = st.text_input(
        "Tower / Location (optional)",
        "TWR-1024"
    )

    audio = st.audio_input(
        "🎙️ Record the complaint or request"
    )

    st.divider()

    st.subheader("Type the request")

    typed = st.text_area(
        "Complaint / Request",
        "My internet is very slow and calls keep dropping."
    )

    if audio:
        st.audio(audio)

    # -------------------------
    # Process Voice Request
    # -------------------------

    if st.button(
        "Process Voice Request",
        type="primary"
    ):
        try:
            with st.spinner(
                "Understanding the request..."
            ):

                # Prefer typed text when provided.
                # Otherwise, transcribe the uploaded audio.

                if typed and typed.strip():
                    transcript = typed.strip()

                elif audio:
                    transcript = transcribe_audio(
                        audio.getvalue(),
                        audio.name or "voice.webm"
                    )

                else:
                    st.warning("Please type a request or record audio.")
                    st.stop()

                st.subheader("📝 Transcript")
                st.write(transcript)

                recent = (
                    anomalies[
                        anomalies.tower_id == tower
                    ]
                    .sort_values("timestamp")
                    .tail(1)
                )

                anomaly = (
                    recent.iloc[0].to_dict()
                    if not recent.empty
                    else {}
                )
                # Start a fresh workflow context for every new voice request.
                st.session_state.voice_thread_id = f"voice-{uuid.uuid4().hex}"
                st.session_state.voice_transaction = {}
                st.session_state.voice_result = None
                st.session_state.voice_response = ""
                from src.voice_assistant import (
                    process_voice_request,
                    route_voice_intent
                )

                result, response = process_voice_request(
                    transcript,
                    audience=audience.lower(),
                    tower_id=tower,
                    anomaly=anomaly,
                    thread_id=(
                        st.session_state.voice_thread_id
                    ),
                    transaction=(
                        st.session_state.voice_transaction
                    )
                )

                result["thread_id"] = (
                    st.session_state.voice_thread_id
                )

                st.session_state.voice_result = result
                st.session_state.voice_response = response

                # Keep transaction state only for fraud workflows
                if (
                    result.get("voice_request_type")
                    != "BILLING_FRAUD"
                ):
                    st.session_state.voice_transaction = {}

                selected_workflow = route_voice_intent(
                    transcript,
                    audience
                )

                st.subheader("🧭 Agent Decision")

                st.info(
                    f"Selected workflow: {selected_workflow}"
                )

            st.subheader(
                "🤖 Voice Assistant Response"
            )

            st.success(response)

            if audience == "Customer":

                st.info(
                    "Customer mode: the assistant intentionally "
                    "avoids exposing internal network diagnostics."
                )

            else:

                with st.expander(
                    "Employee operational details"
                ):

                    st.write(
                        "**Diagnosis:**",
                        result.get("diagnosis")
                    )

                    st.write(
                        "**Recommended action:**",
                        result.get("recommendation")
                    )

                    st.write(
                        "**Incident report:**"
                    )

                    st.code(
                        result.get(
                            "incident_report",
                            ""
                        )
                    )

        except Exception as exc:

            st.error(
                f"Unable to process the voice request: {exc}"
            )

    # -------------------------
    # Pending Fraud Information
    # -------------------------

    pending_result = (
        st.session_state.get("voice_result")
    )

    if (
        pending_result
        and pending_result.get(
            "voice_request_type"
        ) == "BILLING_FRAUD"
    ):

        missing_fields = (
            pending_result.get(
                "transaction_missing_fields",
                []
            )
        )

        if missing_fields:

            st.divider()

            st.subheader(
                "💳 Missing Transaction Information"
            )

            st.warning(
                "The fraud agent needs the following "
                "information before it can perform the "
                "fraud assessment:"
            )

            for field in missing_fields:

                st.write(
                    f"- {field.replace('_', ' ').title()}"
                )

            st.subheader(
                "💳 Continue Fraud Investigation"
            )

            st.info(
                "Provide the missing transaction information "
                "below so the fraud agent can continue the assessment."
            )

            previous_transactions = st.number_input(
                "Previous Transactions",
                min_value=0,
                value=0,
                step=1,
                key="fraud_previous_transactions"
            )

            transaction_velocity = st.number_input(
                "Transaction Velocity",
                min_value=0,
                value=0,
                step=1,
                key="fraud_transaction_velocity"
            )

            country = st.text_input(
                "Country",
                key="fraud_country"
            )

            payment_type = st.text_input(
                "Payment Type",
                key="fraud_payment_type"
            )

            device_type = st.text_input(
                "Device Type",
                key="fraud_device_type"
            )

            if st.button(
                "🔎 Continue Fraud Assessment",
                type="primary",
                key="continue_fraud_assessment"
            ):

                st.session_state.voice_transaction.update({
                    "previous_transactions":
                        previous_transactions,
                    "transaction_velocity":
                        transaction_velocity,
                    "country":
                        country,
                    "payment_type":
                        payment_type,
                    "device_type":
                        device_type,
                })

                from src.fraud_detection import (
                    predict_fraud
                )

                required_fraud_fields = [
                    "amount",
                    "previous_transactions",
                    "transaction_velocity",
                    "is_new_device",
                    "hour",
                    "country",
                    "payment_type",
                    "device_type",
                ]

                missing_fraud_fields = [
                    field
                    for field in required_fraud_fields
                    if field not in
                    st.session_state.voice_transaction
                ]

                if missing_fraud_fields:

                    st.error(
                        "Fraud assessment cannot continue "
                        "because these transaction fields "
                        "are missing: "
                        f"{', '.join(missing_fraud_fields)}"
                    )

                else:

                    fraud_result = predict_fraud(
                        st.session_state.voice_transaction
                    )

                    st.session_state.voice_result.update({
                        "transaction":
                            st.session_state.voice_transaction,
                        "transaction_missing_fields":
                            [],
                        "fraud_probability":
                            fraud_result[
                                "fraud_probability"
                            ],
                        "fraud_risk_level":
                            fraud_result[
                                "risk_level"
                            ],
                        "fraud_detected":
                            fraud_result[
                                "is_fraud"
                            ],
                        "fraud_reasons":
                            fraud_result[
                                "reasons"
                            ],
                        "fraud_decision":
                            "ASSESSMENT_COMPLETE",
                    })

                    st.session_state.voice_response = (
                        "Billing fraud assessment completed "
                        "successfully. "
                        f"Risk level: "
                        f"{fraud_result['risk_level']}. "
                        f"Fraud probability: "
                        f"{fraud_result['fraud_probability'] * 100:.1f}%."
                    )

                    st.metric(
                        "Fraud Probability",
                        (
                            f"{fraud_result['fraud_probability'] * 100:.1f}%"
                        )
                    )

                    st.metric(
                        "Risk Level",
                        fraud_result["risk_level"]
                    )

                    st.subheader(
                        "Risk Indicators"
                    )

                    for reason in fraud_result["reasons"]:
                        st.write(f"• {reason}")

                    if fraud_result["risk_level"] in (
                        "HIGH",
                        "CRITICAL"
                    ):

                        fraud_incident = {
                            "incident_type": "FRAUD",
                            "thread_id": (
                                st.session_state.voice_thread_id
                            ),
                            "case_status": "IN_PROGRESS",
                            "priority": (
                                fraud_result["risk_level"]
                            ),
                            "issue":
                                "Voice-reported suspicious billing transaction",
                            "assigned_to":
                                "Fraud Analyst",
                            "tower_id":
                                None,
                            "diagnosis":
                                (
                                    f"Fraud probability: "
                                    f"{fraud_result['fraud_probability']:.4f}. "
                                    f"Risk level: "
                                    f"{fraud_result['risk_level']}."
                            ),
                            "recommendation":
                                (
                                    "Human fraud analyst review is "
                                    "required before the case is finalized."
                            ),
                            "complaint_analysis": {
                                "severity":
                                    fraud_result["risk_level"]
                            },
                            "root_cause_confidence":
                                "HIGH",
                        }

                        incident_id = save_incident(
                            fraud_incident
                        )

                        st.session_state.voice_result.update({
                            "fraud_incident_id":
                                incident_id,
                            "fraud_review_status":
                                "PENDING",
                            "fraud_review_feedback":
                                "",
                        })

                        st.warning(
                            f"Fraud Incident {incident_id} created. "
                            "Human fraud analyst review is required."
                        )

    # -------------------------
    # Pending Network Approval
    # -------------------------

    pending_result = (
        st.session_state.get("voice_result")
    )

    current_voice_result = st.session_state.get("voice_result") or {}

    if (
        current_voice_result
        and current_voice_result.get(
            "audience",
            ""
        ).lower() == "employee"
        and current_voice_result.get(
            "network_decision",
            ""
        ) == "HUMAN_REVIEW"
        and current_voice_result.get(
            "human_review_status",
            ""
        ).lower() not in ("approved", "rejected")
    ):
        st.divider()

        st.subheader(
            "👷 Network Engineer Approval"
        )

        st.warning(
            "This network investigation is paused and "
            "requires human network engineer approval "
            "before the workflow can continue."
        )

        feedback = st.text_area(
            "Engineer feedback",
            placeholder=(
                "Example: Approved for network investigation."
            ),
            key="voice_engineer_feedback"
        )

        col1, col2 = st.columns(2)

        with col1:

            if st.button(
                "✅ Approve Investigation",
                type="primary",
                key="voice_approve_investigation"
            ):

                try:

                    from src.voice_assistant import (
                        resume_voice_investigation
                    )

                    with st.spinner(
                        "Resuming network investigation..."
                    ):

                        resumed_result = (
                            resume_voice_investigation(
                                pending_result.get(
                                    "thread_id"
                                ),
                                "approved",
                                feedback
                            )
                        )

                    st.session_state.voice_result = resumed_result

                    st.session_state.voice_response = (
                        resumed_result.get("response")
                        or resumed_result.get("incident_report")
                        or resumed_result.get("recommendation")
                        or "Network investigation was rejected."
                    )

                    st.rerun()

                    st.write(
                        "**Case Status:**",
                        resumed_result.get(
                            "case_status"
                        )
                    )

                    st.write(
                        "**Recommendation:**",
                        resumed_result.get(
                            "recommendation"
                        )
                    )

                except Exception as approval_error:

                    st.error(
                        "Unable to resume the investigation: "
                        f"{approval_error}"
                    )

        with col2:

            if st.button(
                "❌ Reject Investigation",
                key="voice_reject_investigation"
            ):

                try:

                    from src.voice_assistant import (
                        resume_voice_investigation
                    )

                    with st.spinner(
                        "Processing rejection..."
                    ):

                        resumed_result = (
                            resume_voice_investigation(
                                pending_result.get(
                                    "thread_id"
                                ),
                                "rejected",
                                feedback
                            )
                        )

                    st.session_state.voice_result = resumed_result

                    st.session_state.voice_response = (
                        resumed_result.get("response")
                        or resumed_result.get("incident_report")
                        or resumed_result.get("recommendation")
                        or "Network investigation was rejected."
                    )

                    st.rerun()

                except Exception as rejection_error:

                    st.error(
                        "Unable to process the rejection: "
                        f"{rejection_error}"
                    )

    # =========================================================
    # PERSISTENT VOICE RESULT
    # =========================================================

    saved_voice_result = st.session_state.get("voice_result") or {}
    saved_voice_response = st.session_state.get("voice_response", "")

    review_status = saved_voice_result.get(
        "human_review_status",
        ""
    ) if saved_voice_result else ""

    incident_id = saved_voice_result.get(
        "incident_id",
        ""
    ) if saved_voice_result else ""

    case_status = saved_voice_result.get(
        "case_status",
        ""
    ) if saved_voice_result else ""

    review_completed = (
        review_status in ("approved", "rejected")
        or bool(incident_id)
        or bool(case_status)
    )

    if saved_voice_result and review_completed:
        st.divider()

        network_decision = saved_voice_result.get(
            "network_decision",
            ""
        )

        human_review_status = saved_voice_result.get(
            "human_review_status",
            ""
        )

        if network_decision:
            st.write(
                "**Network Decision:**",
                network_decision
            )

        if human_review_status:
            st.write(
                "**Human Review Status:**",
                human_review_status
            )

        if saved_voice_result.get("incident_id"):
            st.write(
                "**Incident ID:**",
                saved_voice_result.get("incident_id")
            )

            if saved_voice_result.get("case_status"):
                st.write(
                    "**Case Status:**",
                    saved_voice_result.get("case_status")
                )

            incident_report = saved_voice_result.get(
                "incident_report",
                ""
            )

            if incident_report:
                st.write("**Incident Report:**")
                st.code(incident_report)
elif page == "Incident Center":

    st.title("🛡️ Incident & Audit Center")

    # -------------------------
    # Saved Investigations
    # -------------------------

    st.subheader("🔄 Saved Investigations")

    try:
        saved_threads = list_investigation_threads(
            limit=50
        )
        saved_metrics = recent_investigation_metrics(100)
        saved_incidents = incident_rows(100)

        metric_lookup = {
            str(metric.thread_id): metric
            for metric in saved_metrics
            if metric.thread_id
        }

        incident_lookup = {
            str(row["id"]): row
            for row in saved_incidents
        }

        if not saved_threads:
            st.info(
                "No saved LangGraph investigations were found."
            )
        else:
            investigation_options = list(saved_threads.keys())

            selected_thread = st.selectbox(
                "Select an investigation to recover",
                investigation_options,
                key="saved_investigation_thread"
            )

            checkpoint = saved_threads.get(selected_thread)
            checkpoint_values = {}

            if checkpoint and checkpoint.checkpoint:
                checkpoint_values = checkpoint.checkpoint.get(
                    "channel_values", {}
                )

            # Initialize before the loop
            investigation_rows = []

            for thread_id, saved_checkpoint in saved_threads.items():
                values = {}

                if saved_checkpoint and saved_checkpoint.checkpoint:
                    values = saved_checkpoint.checkpoint.get(
                        "channel_values", {}
                    )

                investigation_rows.append({
                    "Investigation": thread_id,
                    "Tower": values.get("tower_id", "N/A"),
                    "Case Status": values.get("case_status", "NOT CREATED"),
                    "Root Cause Confidence": values.get(
                        "root_cause_confidence", "UNKNOWN"
                    ),
                    "Verification Status": values.get(
                        "verification_status", "UNKNOWN"
                    ),
                    "Human Review": values.get(
                        "human_review_status", "N/A"
                    ),
                    "Network Decision": values.get(
                        "network_decision", "N/A"
                    ),
                })

            st.dataframe(
                pd.DataFrame(investigation_rows),
                width="stretch",
                hide_index=True
            )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.write(
                "**Investigation Thread ID:**"
            )
            st.code(selected_thread)

        with col2:
            st.write("**Tower:**")
            selected_metric = metric_lookup.get(
                str(selected_thread)
            )

            selected_tower = (
                checkpoint_values.get("tower_id")
                or (
                    selected_metric.tower_id
                    if selected_metric
                    else None
                )
                or "N/A"
            )

            st.write(selected_tower)

            if st.button(
               "▶️ Resume Selected Investigation",
                key="resume_saved_investigation"
               ):
                try:
                    agent = build_agent()

                    config = {
                        "configurable": {
                            "thread_id": selected_thread
                        }
                    }

                    state = agent.get_state(
                        config
                    )

                    if not state.values:
                        st.warning(
                            "No recoverable checkpoint "
                            "was found for this investigation."
                        )
                    else:
                        st.session_state.agent_thread_id = (
                            selected_thread
                        )
                        st.session_state.agent = agent
                        st.session_state.agent_config = config
                        st.session_state.agent_result = (
                            state.values
                        )
                        st.session_state.agent_issue = (
                            state.values.get(
                                "issue",
                                ""
                            )
                        )

                        st.success(
                            "Investigation restored successfully. "
                            "Open LangGraph Agent to continue."
                        )
                except Exception as e:
                    st.error(
                        f"Unable to resume investigation: {e}"
                    )

    except Exception as e:
        st.error(
            f"Unable to load saved investigations: {e}"
        )

    st.divider()
    # -------------------------
    # Investigation Performance
    # -------------------------

    st.subheader("📈 Investigation Performance")

    metrics = recent_investigation_metrics(100)

    if metrics:
        execution_times = [
            metric.execution_time_ms
            for metric in metrics
            if metric.execution_time_ms is not None
        ]

        human_review_count = sum(
            1
            for metric in metrics
            if metric.human_review == "REQUIRED"
        )

        final_outcomes = [
            metric
            for metric in metrics
            if metric.outcome in {"RESOLVED", "NOT_RECOVERED"}
        ]

        resolved_count = sum(
            1
            for metric in final_outcomes
            if metric.outcome == "RESOLVED"
        )

        unresolved_count = sum(
            1
            for metric in final_outcomes
            if metric.outcome == "NOT_RECOVERED"
        )

        average_execution_time = (
            sum(execution_times) / len(execution_times)
            if execution_times
            else 0
        )

    human_review_rate = (
        human_review_count / len(metrics) * 100
    )

    recovery_rate = (
        resolved_count / len(final_outcomes) * 100
        if final_outcomes
        else 0
    )

    unresolved_rate = (
        unresolved_count / len(final_outcomes) * 100
        if final_outcomes
        else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Avg Investigation Time",
        f"{average_execution_time:.0f} ms",
    )

    col2.metric(
        "Human Review Rate",
        f"{human_review_rate:.1f}%",
    )

    col3.metric(
        "Recovery Rate",
        f"{recovery_rate:.1f}%",
    )

    col4.metric(
        "Unresolved Rate",
        f"{unresolved_rate:.1f}%",
    )

    st.divider()
    # -------------------------
    # Investigation Trend
    # -------------------------
    st.subheader("📈 Investigation Performance Trend")

    metrics = recent_investigation_metrics(100)

    trend_metrics = [
        metric
        for metric in metrics
        if metric.execution_time_ms is not None
    ]

    if trend_metrics:
        trend_data = pd.DataFrame(
            [
                {
                    "Created At": metric.created_at,
                    "Execution Time (ms)": metric.execution_time_ms,
                    "Human Review": metric.human_review,
                    "Outcome": metric.outcome,
                }
                for metric in trend_metrics
            ]
        )

        trend_data["Created At"] = pd.to_datetime(
            trend_data["Created At"]
        )

        trend_data = trend_data.sort_values("Created At")

        st.line_chart(
            trend_data.set_index("Created At")[
                ["Execution Time (ms)"]
            ]
        )

    else:
        st.info("No investigation performance data available yet.")

    # -------------------------
    # Investigation History
    # -------------------------

    st.divider()

    st.subheader("🧾 Investigation History")

    try:
        history_metrics = recent_investigation_metrics(100)
        history_threads = list_investigation_threads(limit=100)
        history_incidents = incident_rows(100)

        # Match incidents to their original investigation thread.
        incident_by_thread = {
            str(row["thread_id"]): row
            for row in history_incidents
            if row.get("thread_id")
        }

        history_rows = []

        for metric in history_metrics:

            thread_id = str(metric.thread_id)

            # -------------------------
            # Get checkpoint information
            # -------------------------

            checkpoint = history_threads.get(thread_id)

            checkpoint_values = {}

            if checkpoint and checkpoint.checkpoint:
                checkpoint_values = checkpoint.checkpoint.get(
                    "channel_values",
                    {}
                )

            # -------------------------
            # Get incident information
            # -------------------------
            # Find the incident belonging to this exact investigation.
            incident = incident_by_thread.get(thread_id)

            # -------------------------
            # Resolve values
            # -------------------------

            tower_id = (
                metric.tower_id
                or checkpoint_values.get("tower_id")
                or (
                    incident.get("tower_id")
                    if incident
                    else "N/A"
                )
            )

            case_status = (
                checkpoint_values.get(
                    "case_status",
                    None
                )
                or (
                    incident.get("case_status")
                    if incident
                    and "case_status" in incident
                    else "NOT CREATED"
                )
            )

            network_decision = checkpoint_values.get(
                "network_decision",
                "N/A"
            )

            human_review = (
                metric.human_review
                or checkpoint_values.get(
                    "human_review_status",
                    "N/A"
                )
            )

            history_rows.append(
                {
                    "Investigation": thread_id,
                    "Incident": (
                        incident.get("id")
                        if incident
                        else "N/A"
                    ),
                    "Tower": tower_id,
                    "Investigation Time": (
                        f"{metric.execution_time_ms:.0f} ms"
                        if metric.execution_time_ms is not None
                        else "N/A"
                    ),
                    "Human Review": human_review,
                    "Decision": network_decision,
                    "Outcome": metric.outcome or "N/A",
                    "Case Status": case_status,
                }
            )

        if history_rows:

            history_df = pd.DataFrame(history_rows)

            st.dataframe(
                history_df,
                width="stretch",
                hide_index=True,
            )

            st.caption(
                "Investigation history links LangGraph execution, "
                "human review, network decision, incident creation, "
                "and case lifecycle status."
            )

        else:
            st.info(
                "No investigation history is available yet."
            )

    except Exception as e:
        st.error(
            f"Unable to load investigation history: {e}"
        )
    # -------------------------
    # Investigation Details
    # -------------------------

    st.divider()

    st.subheader("🔎 Investigation Details")
    try:

        detail_metrics = recent_investigation_metrics(100)
        detail_threads = list_investigation_threads(limit=100)
        detail_incidents = recent_incidents(100)

        if detail_metrics:

            detail_options = [
                str(metric.thread_id)
                for metric in detail_metrics
                if metric.thread_id
            ]

            selected_detail_thread = st.selectbox(
                "Select Investigation",
                detail_options,
                key="investigation_detail_thread",
            )

            selected_metric = next(
                (
                    metric
                    for metric in detail_metrics
                    if str(metric.thread_id)
                    == selected_detail_thread
                ),
                None,
            )

            checkpoint = detail_threads.get(
                selected_detail_thread
            )

            checkpoint_values = {}

            if checkpoint and checkpoint.checkpoint:
                checkpoint_values = checkpoint.checkpoint.get(
                    "channel_values",
                    {},
                )

        # -------------------------
        # Basic investigation data
        # -------------------------

        incident_id = (
            selected_metric.incident_id
            if selected_metric
            else None
        )

        tower_id = (
            selected_metric.tower_id
            if selected_metric
            else None
        ) or checkpoint_values.get(
            "tower_id",
            "N/A",
        )

        execution_time = (
            selected_metric.execution_time_ms
            if selected_metric
            else checkpoint_values.get(
                "execution_time_ms"
            )
        )

        human_review = (
            selected_metric.human_review
            if selected_metric
            else None
        ) or checkpoint_values.get(
            "human_review_status",
            "N/A",
        )

        decision = checkpoint_values.get(
            "network_decision",
            "N/A",
        )

        outcome = (
            selected_metric.outcome
            if selected_metric
            else None
        ) or "N/A"

        case_status = checkpoint_values.get(
            "case_status",
            "NOT CREATED",
        )

        # -------------------------
        # Incident lookup
        # -------------------------
        selected_incident = None

        if incident_id is not None:
            selected_incident = get_incident_by_id(int(incident_id))

        # If the recorded incident ID no longer exists,
        # use the LangGraph thread ID as the reliable fallback.
        if selected_incident is None and selected_detail_thread:

            for incident in detail_incidents:

                if str(incident.thread_id) == str(
                    selected_detail_thread
                ):
                    selected_incident = incident
                    break
        # Use the actual incident status when
        # investigation checkpoint status is unavailable.
        if (
            selected_incident
            and (
                not case_status
                or case_status == "NOT CREATED"
            )
        ):
            case_status = (
                selected_incident.case_status
                or "NOT CREATED"
            )
        # -------------------------
        # Investigation summary
        # -------------------------

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**Thread ID**")
            st.code(selected_detail_thread)

            st.write("**Tower**")

            incident_tower = (
                selected_incident.tower_id
                if selected_incident and selected_incident.tower_id
                else tower_id
                if tower_id
                else "N/A"
            )

        st.write(incident_tower)
        with col2:
            st.write("**Incident ID**")
            st.write(
                incident_id
                if incident_id is not None
                else "N/A"
            )

            st.write("**Investigation Time**")
            st.write(
                f"{execution_time:.0f} ms"
                if execution_time is not None
                else "N/A"
            )

        with col3:
            st.write("**Outcome**")
            st.write(outcome)

            st.write("**Case Status**")
            st.write(case_status)

        st.divider()

        # -------------------------
        # Investigation timeline
        # -------------------------

        st.subheader("🕒 Investigation Timeline")

        timeline = []

        timeline.append(
            {
                "Stage": "Investigation",
                "Status": "Completed",
                "Detail": (
                    f"Execution time: "
                    f"{execution_time:.0f} ms"
                    if execution_time is not None
                    else "Investigation recorded"
                ),
            }
        )

        if human_review not in {
            None,
            "",
            "N/A",
        }:

            timeline.append(
                {
                    "Stage": "Human Review",
                    "Status": human_review,
                    "Detail": (
                        "Human review state recorded "
                        "in investigation metrics."
                    ),
                }
            )

        if decision not in {
            None,
            "",
            "N/A",
        }:

            timeline.append(
                {
                    "Stage": "Network Decision",
                    "Status": decision,
                    "Detail": (
                        "Network investigation decision "
                        "stored in LangGraph checkpoint."
                    ),
                }
            )

        if incident_id is not None:

            timeline.append(
                {
                    "Stage": "Incident",
                    "Status": "Created",
                    "Detail": (
                        f"Incident #{incident_id}"
                    ),
                }
            )

        if case_status not in {
            None,
            "",
            "NOT CREATED",
        }:

            timeline.append(
                {
                    "Stage": "Case",
                    "Status": case_status,
                    "Detail": (
                        "Case lifecycle status "
                        "stored in investigation state."
                    ),
                }
            )

        if outcome not in {
            None,
            "",
            "N/A",
        }:

            timeline.append(
                {
                    "Stage": "Outcome",
                    "Status": outcome,
                    "Detail": (
                        "Final investigation outcome."
                    ),
                }
            )

        timeline_df = pd.DataFrame(timeline)

        st.dataframe(
            timeline_df,
            width="stretch",
            hide_index=True,
        )
        # ---------------------------------------------------------
        # Incident Lifecycle Management
        # ---------------------------------------------------------

        if selected_incident:

            st.divider()
            st.subheader("🔧 Incident Lifecycle Management")

            current_status = (
                selected_incident.case_status
                or "OPEN"
            )

            st.write(
                f"**Current Status:** `{current_status}`"
            )

            # -----------------------------------------------------
            # Assignment
            # -----------------------------------------------------

            st.markdown("### 👤 Incident Assignment")

            engineer_options = [
                "Unassigned",
                "NOC Engineer",
                "Network Engineer",
                "Support Engineer",
                "Fraud Analyst",
            ]

            current_engineer = (
                selected_incident.assigned_to
                or "Unassigned"
            )

            assigned_to = st.selectbox(
                "Assign incident to",
                engineer_options,
                index=(
                    engineer_options.index(current_engineer)
                    if current_engineer in engineer_options
                    else 0
                ),
                key=f"assign_engineer_{selected_incident.id}",
            )

            if assigned_to != current_engineer:

                if st.button(
                    "👤 Update Assignment",
                    key=f"update_assignment_{selected_incident.id}",
                ):

                    success = update_incident_status(
                        selected_incident.id,
                        current_status,
                        assigned_to=assigned_to,
                    )

                    if success:
                        st.success(
                            f"Incident assigned to {assigned_to}."
                        )
                        st.rerun()
                    else:
                        st.error(
                            "Assignment could not be updated."
                        )

        # -----------------------------------------------------
        # Resolution
        # -----------------------------------------------------

            if current_status == "IN_PROGRESS":

                st.markdown("### ✅ Resolve Incident")

                resolution_notes = st.text_area(
                    "Resolution Notes",
                    placeholder=(
                        "Describe the resolution, verification "
                        "performed, and service status."
                    ),
                    key=f"resolution_notes_{selected_incident.id}",
                )

                if st.button(
                    "✅ Mark as Resolved",
                    key=f"resolve_incident_{selected_incident.id}",
                ):

                    if not resolution_notes.strip():

                        st.warning(
                            "Please enter resolution notes "
                            "before resolving the incident."
                        )

                    else:

                        success = update_incident_status(
                            selected_incident.id,
                            "RESOLVED",
                            assigned_to=assigned_to,
                            resolution_notes=resolution_notes,
                        )

                        if success:
                            st.success(
                                f"Incident #{selected_incident.id} "
                                "has been resolved."
                            )
                            st.rerun()
                        else:
                            st.error(
                                "Incident could not be resolved."
                            )

    # -----------------------------------------------------
    # Close
    # -----------------------------------------------------

            elif current_status == "RESOLVED":

                st.markdown("### 🔒 Close Incident")

                close_notes = st.text_area(
                    "Closure Notes",
                    placeholder=(
                        "Confirm that the incident has been "
                        "reviewed and is ready to close."
                    ),
                    key=f"closure_notes_{selected_incident.id}",
                )

                if st.button(
                    "🔒 Close Incident",
                    key=f"close_incident_{selected_incident.id}",
                ):

                    if not close_notes.strip():

                        st.warning(
                            "Please enter closure notes "
                            "before closing the incident."
                        )

                    else:

                        success = update_incident_status(
                            selected_incident.id,
                            "CLOSED",
                            assigned_to=assigned_to,
                            resolution_notes=close_notes,
                        )

                        if success:
                            st.success(
                                f"Incident #{selected_incident.id} "
                                "has been closed."
                            )
                            st.rerun()
                        else:
                            st.error(
                                "Incident could not be closed."
                            )
        # -------------------------
        # Raw checkpoint state
        # -------------------------

        with st.expander(
            "🔐 View LangGraph Checkpoint State"
        ):

            if checkpoint_values:

                st.json(
                    checkpoint_values
                )

            else:

                st.info(
                    "No checkpoint state is available "
                    "for this investigation."
                )

        # -------------------------
        # Incident details
        # -------------------------
        if selected_incident:

            incident_tower = (
                selected_incident.tower_id
                or (
                    selected_metric.tower_id
                    if selected_metric
                    else None
                )
                or "N/A"
            )

            with st.expander(
                "🚨 View Related Incident",
                expanded=True
            ):

                st.markdown("### Incident Summary")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.markdown("**Incident ID**")
                    st.write(
                        f"#{selected_incident.id}"
                    )

                with col2:
                    st.markdown("**Case Status**")
                    st.write(
                        selected_incident.case_status
                        or "N/A"
                    )

                with col3:
                    st.markdown("**Priority**")
                    st.write(
                        selected_incident.priority
                        or "N/A"
                    )

                with col4:
                    st.markdown("**Severity**")
                    st.write(
                        selected_incident.severity
                        or "N/A"
                    )

                st.divider()

                col1, col2 = st.columns(2)

                with col1:

                    st.markdown("**Tower**")
                    st.write(incident_tower)

                    st.markdown("**Assigned To**")
                    st.write(
                        selected_incident.assigned_to
                        or "Unassigned"
                    )

                    st.markdown("**Incident Type**")
                    st.write(
                        selected_incident.incident_type
                        or "N/A"
                    )

                with col2:

                    st.markdown("**Thread ID**")
                    st.code(
                        selected_incident.thread_id
                        or "N/A"
                    )

                    st.markdown("**Created At**")
                    st.write(
                        str(
                            selected_incident.created_at
                        )
                        if selected_incident.created_at
                        else "N/A"
                    )

                st.divider()

                st.markdown("### Reported Issue")
                st.info(
                    selected_incident.issue
                    or "No issue description available."
                )

                st.markdown("### Diagnosis")
                st.write(
                    selected_incident.diagnosis
                    or "No diagnosis recorded."
                )

                st.markdown("### Recommendation")
                st.write(
                    selected_incident.recommendation
                    or "No recommendation recorded."
                )

        else:

            st.info(
                f"Incident #{incident_id} could not be found "
                "in the incident records."
            )

    # -------------------------
    # Investigation Metrics
    # -------------------------
        st.subheader("📊 Investigation Metrics")
        metrics = recent_investigation_metrics(50)

        if metrics:
            metrics_data = [
                {
                    "Thread ID": metric.thread_id,
                    "Incident ID": metric.incident_id,
                    "Tower": metric.tower_id,
                    "Status": metric.status,
                    "Outcome": metric.outcome,
                    "Execution Time (ms)": metric.execution_time_ms,
                    "Human Review": metric.human_review,
                    "Created At": metric.created_at,
                }
                for metric in metrics
            ]

            st.dataframe(
                pd.DataFrame(metrics_data),
                width="stretch",
                hide_index=True,
            )

        else:
            st.info(
                "No investigation metrics recorded yet."
            )

        # -------------------------
        # Investigation Quality Signals
        # -------------------------

        st.subheader("🧪 Investigation Quality Signals")

        if metrics:

            total_metric_records = len(metrics)

            human_review_required = sum(
                1
                for metric in metrics
                if metric.human_review == "REQUIRED"
            )

            human_approved = sum(
                1
                for metric in metrics
                if metric.human_review == "APPROVED"
            )

            resolved_investigations = sum(
                1
                for metric in metrics
                if metric.status == "RESOLVED"
            )

            unresolved_investigations = sum(
                1
                for metric in metrics
                if metric.outcome == "NOT_RECOVERED"
            )

            q1, q2, q3, q4, q5 = st.columns(5)

            with q1:
                st.metric("Tracked Records", total_metric_records)

            with q2:
                st.metric("Review Required", human_review_required)

            with q3:
                st.metric("Human Approved", human_approved)

            with q4:
                st.metric("Resolved", resolved_investigations)

            with q5:
                st.metric("Not Recovered", unresolved_investigations)

        else:
            st.info(
                "No investigation quality signals available yet."
            )

        st.divider()

    except Exception as e:
        pass

# -------------------------
# Load incidents
# -------------------------

    rows = incident_rows(50)

    if rows:

        df_incidents = pd.DataFrame(rows)

        # -------------------------
        # Incident Analytics
        # -------------------------

        st.subheader("Incident Analytics")

        total_incidents = len(df_incidents)

        high_severity = (
            (df_incidents["severity"] == "HIGH").sum()
        )

        medium_confidence = (
            (df_incidents["root_cause_confidence"] == "MEDIUM").sum()
        )

        low_confidence = (
            (df_incidents["root_cause_confidence"] == "LOW").sum()
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Incidents",
                total_incidents
            )

        with col2:
            st.metric(
                "High Severity",
                high_severity
            )

        with col3:
            st.metric(
                "Medium Confidence",
                medium_confidence
            )

        with col4:
            st.metric(
                "Low Confidence",
                low_confidence
            )

        # -------------------------
        # Fraud Review Status
        # -------------------------

        df_incidents["Review Status"] = "Not Applicable"

        fraud_mask = (
            df_incidents["incident_type"] == "FRAUD"
        )

        if fraud_mask.any():

            audit_events = recent_audit_events(100)

            for index, incident in df_incidents[fraud_mask].iterrows():

                transaction_id = None

                if "transaction" in incident["issue"].lower():
                    transaction_id = incident["issue"].split()[-1]

                if transaction_id:

                    matching_events = [
                        event
                        for event in audit_events
                        if event.resource_id == transaction_id
                        and event.action.startswith(
                            "fraud_review_"
                        )
                    ]

                    if matching_events:

                        latest_event = matching_events[0]

                        if "approved" in latest_event.action:
                            df_incidents.at[
                                index,
                                "Review Status"
                            ] = "Approved"

                        elif "rejected" in latest_event.action:
                            df_incidents.at[
                                index,
                                "Review Status"
                            ] = "Rejected"

                        else:
                            df_incidents.at[
                                index,
                                "Review Status"
                            ] = "Pending"

        # -------------------------
        # Fraud Review Overview
        # -------------------------

        st.subheader("Fraud Review Overview")

        review_status_counts = (
            df_incidents["Review Status"]
            .value_counts()
        )

        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric(
                "Pending Reviews",
                int(
                    review_status_counts.get(
                        "Pending",
                        0
                    )
                )
            )

        with r2:
            st.metric(
                "Approved Reviews",
                int(
                    review_status_counts.get(
                        "Approved",
                        0
                    )
                )
            )

        with r3:
            st.metric(
                "Rejected Reviews",
                int(
                    review_status_counts.get(
                        "Rejected",
                        0
                    )
                )
            )

        # -------------------------
        # Filters
        # -------------------------

        st.subheader("Incident Filters")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:

            tower_filter = st.selectbox(
                "Tower",
                ["All"] + sorted(
                    df_incidents["tower_id"]
                    .dropna()
                    .unique()
                    .tolist()
                )
            )

        with col2:

            severity_filter = st.selectbox(
                "Severity",
                ["All"] + sorted(
                    df_incidents["severity"]
                    .dropna()
                    .unique()
                    .tolist()
                )
            )

        with col3:

            confidence_filter = st.selectbox(
                "Root Cause Confidence",
                ["All"] + sorted(
                    df_incidents[
                        "root_cause_confidence"
                    ]
                    .dropna()
                    .unique()
                    .tolist()
                )
            )

        with col4:

            incident_type_filter = st.selectbox(
                "Incident Type",
                ["All"] + sorted(
                    df_incidents[
                        "incident_type"
                    ]
                    .fillna("UNKNOWN")
                    .unique()
                    .tolist()
                )
            )

        with col5:

            review_status_filter = st.selectbox(
                "Review Status",
                [
                    "All",
                    "Pending",
                    "Approved",
                    "Rejected",
                    "Not Applicable",
                ]
            )
        # -------------------------
        # Apply Filters
        # -------------------------

        filtered_incidents = df_incidents.copy()

        if tower_filter != "All":

            filtered_incidents = filtered_incidents[
                filtered_incidents["tower_id"]
                == tower_filter
            ]

        if severity_filter != "All":

            filtered_incidents = filtered_incidents[
                filtered_incidents["severity"]
                == severity_filter
            ]

        if confidence_filter != "All":

            filtered_incidents = filtered_incidents[
                filtered_incidents["root_cause_confidence"]
                == confidence_filter
            ]

        if incident_type_filter != "All":

            filtered_incidents = filtered_incidents[
                filtered_incidents["incident_type"]
                == incident_type_filter
            ]

        if review_status_filter != "All":

            filtered_incidents = filtered_incidents[
                filtered_incidents["Review Status"]
                == review_status_filter
            ]
        # -------------------------
        # Incident Distribution
        # -------------------------

        st.subheader("Incident Distribution")

        col1, col2 = st.columns(2)

        with col1:

            st.caption("Incidents by Severity")

            severity_counts = (
                filtered_incidents["severity"]
                .fillna("UNKNOWN")
                .value_counts()
            )

            st.bar_chart(severity_counts)

        with col2:

            st.caption(
                "Incidents by Root Cause Confidence"
            )

            confidence_counts = (
                filtered_incidents[
                    "root_cause_confidence"
                ]
                .fillna("UNKNOWN")
                .value_counts()
            )

            st.bar_chart(confidence_counts)

        # -------------------------
        # Incident Table
        # -------------------------

        st.subheader("Incident Records")

        st.dataframe(
            filtered_incidents,
            width="stretch",
            hide_index=True
        )
        # -------------------------
        # Resume Investigation
        # -------------------------

        st.subheader("Investigation")

        if filtered_incidents.empty:
            st.info(
                "No incidents match the current filters. "
                "Clear or change the filters to select an incident."
            )
        else:
            selected_incident_id = st.selectbox(
                "Select Incident to Resume",
                filtered_incidents["id"].tolist(),
                key="resume_incident_selection"
            )

            resume_incident = filtered_incidents[
                filtered_incidents["id"] == selected_incident_id
            ].iloc[0]

            incident_type = resume_incident["incident_type"]
            incident_thread_id = resume_incident["thread_id"]

            if incident_type == "NETWORK":
                if incident_thread_id:
                    st.code(
                        incident_thread_id,
                        language="text"
                    )

                    if st.button(
                        "▶️ Resume Investigation",
                        type="primary",
                        key="resume_incident_investigation"
                    ):
                        try:
                            agent = build_agent()

                            config = {
                                "configurable": {
                                    "thread_id": incident_thread_id
                                }
                            }

                            state = agent.get_state(config)

                            if not state.values:
                                st.warning(
                                    "No saved LangGraph investigation was found "
                                    "for this Thread ID."
                                )
                            else:
                                st.session_state.agent_thread_id = (
                                    incident_thread_id
                                )
                                st.session_state.agent = agent
                                st.session_state.agent_config = config
                                st.session_state.agent_result = state.values
                                st.session_state.agent_issue = (
                                    state.values.get("issue", "")
                                )

                                st.success(
                                    "Investigation restored successfully."
                                )

                                st.rerun()

                        except Exception as e:
                            st.error(
                                f"Unable to resume investigation: {e}"
                            )
                else:
                    st.info(
                        "This network incident does not have a linked "
                        "LangGraph investigation."
                    )

            elif incident_type == "FRAUD":
                st.info(
                    "This is a fraud incident. "
                    "It should be handled through Fraud Analyst Review "
                    "rather than Network LangGraph investigation."
                )

            else:
                st.info(
                    "This incident type does not have a linked "
                    "LangGraph investigation."
                )
        # -------------------------
        # Case Status Management
        # -------------------------

        st.subheader("🔄 Case Status Management")

        if not filtered_incidents.empty:

            selected_case = st.selectbox(
                "Select Incident",
                filtered_incidents["id"].tolist(),
                key="incident_case_selection",
            )

            selected_case_row = filtered_incidents[
                filtered_incidents["id"] == selected_case
            ].iloc[0]

            current_case_status = (
                selected_case_row.get(
                    "case_status"
                )
                or "OPEN"
            )

            current_assigned_to = (
                selected_case_row.get(
                    "assigned_to"
                )
                or "Unassigned"
            )

            st.write(
                f"**Current Status:** `{current_case_status}`"
            )

            # -------------------------------------------------
            # Assignment
            # -------------------------------------------------

            engineer_options = [
                "Unassigned",
                "NOC Engineer",
                "Network Engineer",
                "Support Engineer",
                "Fraud Analyst",
            ]

            assigned_to = st.selectbox(
                "Assigned To",
                engineer_options,
                index=(
                    engineer_options.index(
                        current_assigned_to
                    )
                    if current_assigned_to
                    in engineer_options
                    else 0
                ),
                key="incident_center_assignment",
            )

            if assigned_to != current_assigned_to:

                if st.button(
                    "👤 Update Assignment",
                    key="incident_center_update_assignment",
                ):

                    success = update_incident_status(
                        int(selected_case),
                        current_case_status,
                        assigned_to=assigned_to,
                    )

                    if success:

                        st.success(
                            f"Incident #{selected_case} "
                            f"assigned to {assigned_to}."
                        )

                        st.rerun()

                    else:

                        st.error(
                            "Assignment could not be updated."
                        )

            # -------------------------------------------------
            # Resolve
            # -------------------------------------------------

            if current_case_status == "IN_PROGRESS":

                st.markdown("### ✅ Resolve Incident")

                resolution_notes = st.text_area(
                    "Resolution Notes",
                    placeholder=(
                        "Describe the resolution, "
                        "verification performed, "
                        "and service status."
                    ),
                    key="incident_center_resolution_notes",
                )

                if st.button(
                    "✅ Mark as Resolved",
                    type="primary",
                    key="incident_center_resolve",
                ):

                    if not resolution_notes.strip():

                        st.warning(
                            "Please enter resolution notes "
                            "before resolving the incident."
                        )

                    else:

                        success = update_incident_status(
                            int(selected_case),
                            "RESOLVED",
                            assigned_to=assigned_to,
                            resolution_notes=resolution_notes,
                        )

                        if success:

                            st.success(
                                f"Incident #{selected_case} "
                                "has been resolved."
                            )

                            st.rerun()

                        else:

                            st.error(
                                "Incident could not be resolved."
                            )

            # -------------------------------------------------
            # Close
            # -------------------------------------------------

            elif current_case_status == "RESOLVED":

                st.markdown("### 🔒 Close Incident")

                closure_notes = st.text_area(
                    "Closure Notes",
                    placeholder=(
                        "Confirm that the incident has "
                        "been reviewed and is ready to close."
                    ),
                    key="incident_center_closure_notes",
                )

                if st.button(
                    "🔒 Close Incident",
                    type="primary",
                    key="incident_center_close",
                ):

                    if not closure_notes.strip():

                        st.warning(
                            "Please enter closure notes "
                            "before closing the incident."
                        )

                    else:

                        success = update_incident_status(
                            int(selected_case),
                            "CLOSED",
                            assigned_to=assigned_to,
                            resolution_notes=closure_notes,
                        )

                        if success:

                            st.success(
                                f"Incident #{selected_case} "
                                "has been closed."
                            )

                            st.rerun()

                        else:

                            st.error(
                                "Incident could not be closed."
                            )

    elif current_case_status == "OPEN":

        st.info(
            "This incident is open and ready to be assigned "
            "and moved into active investigation."
        )

        if st.button(
            "▶️ Start Investigation",
            type="primary",
            key="incident_center_start",
        ):

            success = update_incident_status(
                int(selected_case),
                "IN_PROGRESS",
                assigned_to=assigned_to,
            )

            if success:

                st.success(
                    f"Incident #{selected_case} "
                    "is now IN_PROGRESS."
                )

                st.rerun()

            else:

                st.error(
                    "Incident could not be moved to IN_PROGRESS."
                )

    elif current_case_status == "CLOSED":

        st.success(
            "This incident is closed. "
            "No further lifecycle actions are available."
        )

        # -------------------------
        # Fraud Analyst Review
        # -------------------------

        if (
            not df_incidents.empty
            and selected_incident["incident_type"] == "FRAUD"
        ):
            st.divider()
            st.subheader("🕵️ Fraud Analyst Review")

            st.warning(
                f"Incident {selected_case} is a fraud case and requires "
                "human fraud analyst review."
            )

            fraud_feedback = st.text_area(
                "Fraud Analyst Feedback",
                placeholder=(
                    "Explain why you approve or reject the fraud assessment."
                ),
                key="fraud_analyst_feedback"
            )

            fraud_decision = st.selectbox(
                "Review Decision",
                [
                    "Pending Review",
                    "Approved",
                    "Rejected"
                ],
                key="fraud_analyst_decision"
            )

            if st.button(
                "Submit Fraud Review",
                type="primary",
                key="submit_fraud_review"
            ):

                if fraud_decision == "Pending Review":

                    st.warning(
                        "Please select Approved or Rejected."
                    )

                else:

                    review_action = (
                        f"fraud_review_{fraud_decision.lower()}"
                    )

                    audit(
                        "fraud-analyst",
                        st.session_state.role,
                        review_action,
                        (
                            f"Fraud incident {selected_case} reviewed. "
                            f"Decision: {fraud_decision}. "
                            f"Feedback: "
                            f"{fraud_feedback or 'No feedback provided'}"
                        ),
                        resource_id=str(selected_case)
                    )

                    if fraud_decision == "Approved":

                        update_case_status(
                            selected_case,
                            "IN_PROGRESS"
                        )

                        st.success(
                            f"Fraud incident {selected_case} approved "
                            "and remains IN_PROGRESS for investigation."
                        )

                    else:

                        update_case_status(
                            selected_case,
                            "CLOSED"
                        )

                        st.warning(
                            f"Fraud incident {selected_case} rejected "
                            "and marked CLOSED."
                        )

                    st.rerun()

        # -------------------------
        # Fraud Review History
        # -------------------------

        st.subheader("Fraud Review History")

        audit_events = recent_audit_events(100)

        fraud_reviews = [
            event
            for event in audit_events
            if event.action.startswith(
                "fraud_review_"
            )
        ]

        if fraud_reviews:

            review_rows = []

            for event in fraud_reviews:

                review_rows.append({
                    "Transaction": event.resource_id,
                    "Decision": event.action.replace(
                        "fraud_review_",
                        ""
                    ).title(),
                    "Reviewer": event.actor,
                    "Role": event.role,
                    "Reviewed At": event.created_at,
                    "Details": event.details,
                })

            review_history = pd.DataFrame(
                review_rows
            )

            st.dataframe(
                review_history,
                width="stretch",
                hide_index=True
            )

        else:

            st.info(
                "No fraud reviews have been recorded yet."
            )


st.divider()

st.caption(
    "TelecomAI Enterprise MVP • Human approval required • "
    "No autonomous production network changes"
)
if page == "🤖 AI Evaluation":

    st.title("🤖 AI Evaluation")
    st.caption(
        "Evaluation and quality monitoring for TelecomAI investigations, "
        "root-cause analysis, evidence, human review, and outcomes."
    )

    st.divider()

    # -------------------------
    # Load evaluation data
    # -------------------------

    quality_metrics = recent_investigation_metrics(100)
    quality_incidents = incident_rows(100)
    # -------------------------
    # Load ground-truth cases
    # -------------------------

    evaluation_cases_path = Path("data/evaluation_cases.json")

    if evaluation_cases_path.exists():
        with open(
            evaluation_cases_path,
            "r",
            encoding="utf-8",
        ) as f:
            evaluation_cases = json.load(f)
    else:
        evaluation_cases = []
    st.subheader("📚 Ground-Truth Evaluation Dataset")

    st.metric(
        "Evaluation Cases",
        len(evaluation_cases),
    )

    if evaluation_cases:
        st.dataframe(
            pd.DataFrame(evaluation_cases),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "No ground-truth evaluation cases available."
        )
    # ---------------------------------
    # Run Ground-Truth Evaluation
    # ---------------------------------

    st.subheader("🧪 Run Ground-Truth Evaluation")

    if evaluation_cases:
        evaluation_case_ids = [
            case["case_id"] for case in evaluation_cases
        ]
        selected_case_id = st.selectbox(
            "Select Evaluation Case",
            evaluation_case_ids,
        )

        selected_case = next(
            case
            for case in evaluation_cases
            if case["case_id"] == selected_case_id
        )

        st.write(f"**Issue:** {selected_case['issue']}")
        st.write(f"**Tower:** {selected_case['tower_id']}")
        quality_threads = list_investigation_threads(
            limit=100
        )

        if st.button(
            "▶️ Evaluate Recorded Investigation",
            type="primary",
        ):
            # Find the latest investigation metric
            # for the selected evaluation tower.
            tower_metrics = [
                metric
                for metric in quality_metrics
                if str(metric.tower_id)
                == str(selected_case["tower_id"])
                and metric.thread_id
            ]

            if not tower_metrics:
                st.warning(
                    "No recorded investigation found for "
                    f"{selected_case['tower_id']}."
                )
            else:

                # Use the most recent investigation.
                latest_metric = max(
                    tower_metrics,
                    key=lambda metric: (
                        metric.created_at
                        if metric.created_at
                        else ""
                    ),
                )

                actual_root_cause = "UNKNOWN"

                actual_root_cause_confidence = (
                    "UNKNOWN"
                )

                actual_outcome = (
                    latest_metric.outcome
                    or "UNKNOWN"
                )

                # Retrieve the latest state for the exact
                # investigation thread.
                thread_id = str(
                    latest_metric.thread_id
                )

                actual_root_cause = "UNKNOWN"
                actual_root_cause_confidence = "UNKNOWN"

                agent = build_agent()

                config = {
                    "configurable": {
                        "thread_id": thread_id
                    }
                }

                snapshot = agent.get_state(config)

                checkpoint_values = (
                    snapshot.values or {}
                )

                actual_root_cause = (
                    checkpoint_values.get(
                        "root_cause_category"
                    )
                    or "UNKNOWN"
                )

                actual_root_cause_confidence = (
                    checkpoint_values.get(
                        "root_cause_confidence"
                    )
                    or "UNKNOWN"
                )
                expected_root_cause = (
                    selected_case.get(
                        "expected_root_cause"
                    )
                    or "UNKNOWN"
                )

                root_cause_match = (
                    actual_root_cause
                    == expected_root_cause
                )

                expected_verification = (
                    selected_case[
                        "expected_verification"
                    ]
                )

                expected_outcome = (
                    selected_case[
                        "expected_outcome"
                    ]
                )
                matching_threads = list_investigation_threads(
                    limit=100
                )
                for thread_id, checkpoint in matching_threads.items():
                    if (
                        not checkpoint
                        or not checkpoint.checkpoint
                    ):
                        continue

                    values = checkpoint.checkpoint.get(
                        "channel_values",
                        {}
                    )

                    if (
                        values.get("tower_id")
                        != selected_case["tower_id"]
                    ):
                        continue

                    if values.get("root_cause_category"):
                        actual_root_cause = values[
                            "root_cause_category"
                        ]
                        actual_root_cause_confidence = (
                            values.get(
                                "root_cause_confidence"
                            )
                            or actual_root_cause_confidence
                        )
                        break
                expected_root_cause = (
                    selected_case.get("expected_root_cause")
                    or "UNKNOWN"
                )

                root_cause_match = (
                    actual_root_cause == expected_root_cause
                )

                expected_verification = selected_case[
                    "expected_verification"
                ]

                expected_outcome = selected_case["expected_outcome"]

                st.markdown("### Evaluation Result")

                r1, r2, r3, r4 = st.columns(4)

                with r1:
                    st.markdown("**Actual Root Cause**")
                    st.info(actual_root_cause)

                r2.metric(
                    "Root Cause Confidence",
                    actual_root_cause_confidence,
                )

                r3.metric(
                    "Actual Outcome",
                    actual_outcome,
                )
                r4.metric(
                    "Expected Outcome",
                    expected_outcome,
                )
                outcome_match = actual_outcome == expected_outcome

                if outcome_match:
                    st.success("Outcome matches the ground-truth case.")
                else:
                    st.warning(
                        "Outcome does not match the "
                        "ground-truth case."
                    )
    # ---------------------------------
    # Investigation Quality
    # ---------------------------------

    st.subheader("Investigation Quality")

    quality_execution_times = [
        metric.execution_time_ms
        for metric in quality_metrics
        if metric.execution_time_ms is not None
    ]

    quality_review_required = sum(
        1
        for metric in quality_metrics
        if metric.human_review == "REQUIRED"
    )

    # Use the latest outcome for each unique investigation
    latest_outcomes = {}
    for metric in quality_metrics:
        if not metric.thread_id:
            continue
        if metric.outcome:
            latest_outcomes[metric.thread_id] = metric.outcome

    quality_completed = sum(
        1
        for outcome in latest_outcomes.values()
        if outcome in {"RESOLVED", "NOT_RECOVERED"}
    )
    quality_recovered = sum(
        1
        for outcome in latest_outcomes.values()
        if outcome == "RESOLVED"
    )
    quality_avg_time = (
        sum(quality_execution_times) / len(quality_execution_times)
        if quality_execution_times
        else 0
    )
    quality_review_rate = (
        quality_review_required / len(quality_metrics) * 100
        if quality_metrics
        else 0
    )
    quality_recovery_rate = (
        quality_recovered / quality_completed * 100
        if quality_completed
        else 0
    )

    q1, q2, q3, q4 = st.columns(4)
    unique_quality_threads = {
        metric.thread_id
        for metric in quality_metrics
        if metric.thread_id
    }

    q1.metric("Investigations Evaluated", len(unique_quality_threads))
    q2.metric("Average Investigation Time", f"{quality_avg_time:.0f} ms")
    q3.metric("Human Review Rate", f"{quality_review_rate:.1f}%")
    q4.metric("Recovery Rate", f"{quality_recovery_rate:.1f}%")

    # ---------------------------------
    # Root Cause Quality
    # ---------------------------------

    st.subheader("Root Cause Quality")

    evaluated_incident_ids = {
        metric.incident_id
        for metric in quality_metrics
        if metric.incident_id is not None
    }

    evaluated_incidents = [
        incident
        for incident in quality_incidents
        if incident.get("id") in evaluated_incident_ids
    ]

    confidence_values = [
        incident.get("root_cause_confidence")
        for incident in evaluated_incidents
        if incident.get("root_cause_confidence")
    ]

    high_confidence = confidence_values.count("HIGH")
    medium_confidence = confidence_values.count("MEDIUM")
    low_confidence = confidence_values.count("LOW")
    unknown_confidence = confidence_values.count("UNKNOWN")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("HIGH Confidence", high_confidence)
    c2.metric("MEDIUM Confidence", medium_confidence)
    c3.metric("LOW Confidence", low_confidence)
    c4.metric("UNKNOWN Confidence", unknown_confidence)

    # ---------------------------------
    # Evidence Quality
    # ---------------------------------

    st.subheader("Evidence Quality")

    quality_threads = list_investigation_threads(
        limit=100
    )

    verification_by_thread = {}

    for metric in quality_metrics:

        if not metric.thread_id:
            continue

        thread_id = str(metric.thread_id)

    # Keep one verification result per investigation
        if thread_id in verification_by_thread:
            continue

        checkpoint = quality_threads.get(
            thread_id
        )

        checkpoint_values = {}

        if checkpoint and checkpoint.checkpoint:
            checkpoint_values = checkpoint.checkpoint.get(
                "channel_values",
                {}
            )

        verification_by_thread[thread_id] = (
            checkpoint_values.get(
                "verification_status"
            )
        )

    verification_values = [
        status
        for status in verification_by_thread.values()
        if status
    ]

    supported_evidence = verification_values.count(
        "SUPPORTED"
    )

    partial_evidence = verification_values.count(
        "PARTIAL_CONFIDENCE"
    )

    insufficient_evidence = verification_values.count(
        "INSUFFICIENT_EVIDENCE"
    )

    unknown_evidence = (
        len(verification_by_thread)
        - len(verification_values)
    )

    e1, e2, e3, e4 = st.columns(4)

    e1.metric("SUPPORTED", supported_evidence)
    e2.metric("PARTIAL CONFIDENCE", partial_evidence)
    e3.metric("INSUFFICIENT EVIDENCE", insufficient_evidence)
    e4.metric("UNKNOWN", unknown_evidence)
    # ---------------------------------
    # Human Review Quality
    # ---------------------------------

    st.subheader("Human Review Quality")
    review_events = recent_audit_events(1000)

# Match review events to the towers belonging to
# the investigations currently being evaluated.
    evaluated_tower_ids = {
        str(incident.get("tower_id"))
        for incident in evaluated_incidents
        if incident.get("tower_id")
    }

    approved_reviews = sum(
        1
        for event in review_events
        if event.action == "network_review_approved"
        and event.resource_id is not None
        and str(event.resource_id) in evaluated_tower_ids
    )

    rejected_reviews = sum(
        1
        for event in review_events
        if event.action == "network_review_rejected"
        and event.resource_id is not None
        and str(event.resource_id) in evaluated_tower_ids
    )

    pending_review_threads = {
        str(metric.thread_id)
        for metric in quality_metrics
        if metric.thread_id
        and metric.human_review == "REQUIRED"
    }

    pending_reviews = len(pending_review_threads)

    total_completed_reviews = (
        approved_reviews
        + rejected_reviews
    )

    review_approval_rate = (
        approved_reviews
        / total_completed_reviews
        * 100
        if total_completed_reviews
        else 0
    )

    h1, h2, h3, h4 = st.columns(4)

    h1.metric("Approved", approved_reviews)
    h2.metric("Rejected", rejected_reviews)
    h3.metric("Pending Review", pending_reviews)
    h4.metric(
        "Approval Rate",
        f"{review_approval_rate:.1f}%",
    )

    # ---------------------------------
    # Quality Trend
    # ---------------------------------

    st.subheader("Quality Trend")

    trend_quality_metrics = [
        metric
        for metric in quality_metrics
        if metric.execution_time_ms is not None
        and metric.created_at is not None
    ]

    if trend_quality_metrics:

        quality_trend_data = pd.DataFrame(
            [
                {
                    "Created At": metric.created_at,
                    "Investigation Time (ms)": metric.execution_time_ms,
                    "Human Review Required": (
                        1
                        if metric.human_review == "REQUIRED"
                        else 0
                    ),
                }
                for metric in trend_quality_metrics
            ]
        )

        quality_trend_data["Created At"] = pd.to_datetime(
            quality_trend_data["Created At"]
        )

        quality_trend_data = (
            quality_trend_data
            .sort_values("Created At")
        )

        st.markdown("#### Investigation Time Trend")

        st.line_chart(
            quality_trend_data.set_index("Created At")[
                ["Investigation Time (ms)"]
            ]
        )

        st.markdown("#### Human Review Requirement Trend")

        st.line_chart(
            quality_trend_data.set_index("Created At")[
                ["Human Review Required"]
            ]
        )

    else:

        st.info(
            "No quality trend data available yet."
        )

    # ---------------------------------
    # Outcome Quality
    # ---------------------------------

    st.subheader("Outcome Quality")

    resolved_outcomes = sum(
        1
        for metric in quality_metrics
        if metric.outcome == "RESOLVED"
    )

    not_recovered_outcomes = sum(
        1
        for metric in quality_metrics
        if metric.outcome == "NOT_RECOVERED"
    )

    other_outcomes = sum(
        1
        for metric in quality_metrics
        if metric.outcome not in {
            "RESOLVED",
            "NOT_RECOVERED",
        }
    )

    o1, o2, o3 = st.columns(3)

    o1.metric("RESOLVED", resolved_outcomes)
    o2.metric("NOT_RECOVERED", not_recovered_outcomes)
    o3.metric("Other / Pending", other_outcomes)

    st.caption(
        "Quality metrics are based on recorded investigation "
        "and incident signals. They are not model accuracy "
        "measurements because ground-truth labels are not "
        "currently available."
    )
elif page == "📊 Observability":

    st.title("📊 TelecomAI Observability")

    # ---------------------------------
    # Model Telemetry
    # ---------------------------------

    st.divider()

    st.subheader("🤖 Model Telemetry")

    model_telemetry = recent_model_telemetry(100)

    # Exclude development/test telemetry
    production_telemetry = [
        item
        for item in model_telemetry
        if item.provider != "TEST"
    ]

    successful_model_calls = sum(
        1
        for item in production_telemetry
        if item.status == "SUCCESS"
    )

    failed_model_calls = sum(
        1
        for item in production_telemetry
        if item.status == "FAILED"
    )

    total_model_calls = (
        successful_model_calls
        + failed_model_calls
    )

    model_success_rate = (
        successful_model_calls
        / total_model_calls
        * 100
        if total_model_calls
        else 0
    )

    model_latencies = [
        item.latency_ms
        for item in production_telemetry
        if item.latency_ms is not None
        and item.status == "SUCCESS"
    ]
    average_model_latency = (
        sum(model_latencies)
        / len(model_latencies)
        if model_latencies
        else 0
    )

    total_tokens_used = sum(
        item.total_tokens or 0
        for item in production_telemetry
    )

    m1, m2, m3, m4, m5 = st.columns(5)

    m1.metric(
        "Model Calls",
        total_model_calls,
    )

    m2.metric(
        "Successful",
        successful_model_calls,
    )

    m3.metric(
        "Failed",
        failed_model_calls,
    )

    m4.metric(
        "Success Rate",
        f"{model_success_rate:.1f}%",
    )

    m5.metric(
        "Total Tokens",
        total_tokens_used,
    )

    st.markdown("### ⏱️ Model Latency")

    st.metric(
        "Average Model Latency",
        f"{average_model_latency:.0f} ms",
    )

    # ---------------------------------
    # Model Usage Table
    # ---------------------------------

    st.markdown("### 📋 Model Usage")

    if production_telemetry:

        model_rows = []

        for item in production_telemetry:

            model_rows.append(
                {
                    "Provider": item.provider,
                    "Model": item.model_name,
                    "Operation": item.operation,
                    "Input Tokens": item.input_tokens,
                    "Output Tokens": item.output_tokens,
                    "Total Tokens": item.total_tokens,
                    "Latency (ms)": item.latency_ms,
                    "Status": item.status,
                    "Error": item.error_message,
                    "Created At": item.created_at,
                }
            )

        model_df = pd.DataFrame(model_rows)

        st.dataframe(
            model_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "No production model telemetry recorded yet."
        )

    st.caption(
        "Model telemetry is based on recorded provider "
        "responses. Development test records are excluded."
    )
    st.divider()

    # -------------------------
    # Load observability data
    # -------------------------

    observability_metrics = recent_investigation_metrics(100)
    observability_events = recent_audit_events(1000)

    # -------------------------
    # Agent Performance
    # -------------------------

    st.subheader("⚙️ Agent Performance")

    execution_times = [
        metric.execution_time_ms
        for metric in observability_metrics
        if metric.execution_time_ms is not None
    ]

    average_time = (
        sum(execution_times) / len(execution_times)
        if execution_times
        else 0
    )

    sorted_times = sorted(execution_times)

    if sorted_times:
        p95_index = int(len(sorted_times) * 0.95) - 1
        p95_index = max(0, min(
            p95_index,
            len(sorted_times) - 1
        ))
        p95_time = sorted_times[p95_index]
    else:
        p95_time = 0

    completed_investigations = sum(
        1
        for metric in observability_metrics
        if metric.outcome in {
            "RESOLVED",
            "NOT_RECOVERED",
        }
    )

    active_investigations = sum(
        1
        for metric in observability_metrics
        if metric.outcome not in {
            "RESOLVED",
            "NOT_RECOVERED",
        }
    )

    o1, o2, o3, o4 = st.columns(4)

    unique_threads = {
        metric.thread_id
        for metric in observability_metrics
        if metric.thread_id
    }

    o1.metric(
        "Investigations",
        len(unique_threads),
    )
    o2.metric(
        "Average Latency",
        f"{average_time:.0f} ms",
    )

    o3.metric(
        "P95 Latency",
        f"{p95_time:.0f} ms",
    )

    o4.metric(
        "Active / Pending",
        active_investigations,
    )

    # -------------------------
    # Investigation Latency
    # -------------------------

    st.markdown("### ⏱️ Investigation Latency")

    latency_metrics = [
        metric
        for metric in observability_metrics
        if metric.execution_time_ms is not None
        and metric.created_at is not None
    ]

    if latency_metrics:

        latency_data = pd.DataFrame(
            [
                {
                    "Created At": metric.created_at,
                    "Execution Time (ms)": metric.execution_time_ms,
                }
                for metric in latency_metrics
            ]
        )

        latency_data["Created At"] = pd.to_datetime(
            latency_data["Created At"]
        )

        latency_data = latency_data.sort_values(
            "Created At"
        )

        st.line_chart(
            latency_data.set_index("Created At")[
                ["Execution Time (ms)"]
            ]
        )

    else:

        st.info(
            "No investigation latency data available yet."
        )

    # -------------------------
    # Investigation Outcomes
    # -------------------------

    st.markdown("### 🎯 Investigation Outcomes")

    resolved_count = sum(
        1
        for metric in observability_metrics
        if metric.outcome == "RESOLVED"
    )

    not_recovered_count = sum(
        1
        for metric in observability_metrics
        if metric.outcome == "NOT_RECOVERED"
    )

    pending_count = sum(
        1
        for metric in observability_metrics
        if metric.outcome not in {
            "RESOLVED",
            "NOT_RECOVERED",
        }
    )

    r1, r2, r3 = st.columns(3)

    with r1:
        st.metric(
            "Resolved",
            resolved_count,
        )

    with r2:
        st.metric(
            "Not Recovered",
            not_recovered_count,
        )

    with r3:
        st.metric(
            "Pending",
            pending_count,
        )

    # ---------------------------------
    # Outcome Comparison
    # ---------------------------------

    outcome_match = (
        actual_outcome
        == expected_outcome
    )

    if outcome_match:
        st.success(
            "✅ Outcome matches the ground-truth case."
        )
    else:
        st.warning(
            "⚠️ Outcome does not match the "
            "ground-truth case."
        )
    # -------------------------
    # Human Review Monitoring
    # -------------------------

    st.markdown("### 👤 Human Review Monitoring")

    required_reviews = sum(
        1
        for metric in observability_metrics
        if metric.human_review == "REQUIRED"
    )

    approved_reviews = sum(
        1
        for event in observability_events
        if event.action == "human_review_approved"
    )

    rejected_reviews = sum(
        1
        for event in observability_events
        if event.action == "human_review_rejected"
    )

    h1, h2, h3 = st.columns(3)

    h1.metric(
        "Review Required",
        required_reviews,
    )

    h2.metric(
        "Approved",
        approved_reviews,
    )

    h3.metric(
        "Rejected",
        rejected_reviews,
    )

    # -------------------------
    # Audit / Failure Signals
    # -------------------------

    st.markdown("### 🚨 Operational Signals")

    error_events = [
        event
        for event in observability_events
        if "error" in str(event.action).lower()
        or "failed" in str(event.action).lower()
        or "failure" in str(event.action).lower()
    ]

    warning_events = [
        event
        for event in observability_events
        if "warning" in str(event.action).lower()
        or "rejected" in str(event.action).lower()
    ]

    s1, s2, s3 = st.columns(3)

    s1.metric(
        "Audit Events",
        len(observability_events),
    )

    s2.metric(
        "Error / Failure Signals",
        len(error_events),
    )

    s3.metric(
        "Warning / Rejection Signals",
        len(warning_events),
    )

    # -------------------------
    # LangGraph Execution
    # -------------------------

    st.markdown("### 🔄 LangGraph Execution")

    graph_threads = list_investigation_threads(
        limit=100
    )

    graph_rows = []
    for thread_id, checkpoint in graph_threads.items():
        values = {}

        if checkpoint and checkpoint.checkpoint:
            values = checkpoint.checkpoint.get("channel_values", {})

    # Skip incomplete / empty checkpoints
        if not values.get("tower_id"):
            continue

        metric = metric_lookup.get(str(thread_id))

        incident = (
            incident_lookup.get(str(metric.incident_id))
            if metric and metric.incident_id
            else None
        )
        metric = metric_lookup.get(str(thread_id))

        graph_rows.append({
            "Thread ID": thread_id,

            "Tower": (
                values.get("tower_id")
                or (metric.tower_id if metric else None)
                or (incident.get("tower_id") if incident else None)
                or "N/A"
            ),

            "Case Status": values.get(
                "case_status",
                "N/A"
            ),

            "Root Cause Confidence": values.get(
                "root_cause_confidence",
                "UNKNOWN"
            ),

            "Verification Status": values.get(
                "verification_status",
                "UNKNOWN"
            ),

            "Human Review": values.get(
                "human_review_status",
                "N/A"
            ),

            "Network Decision": values.get(
                "network_decision",
                "N/A"
            ),
        })

    if graph_rows:
        st.dataframe(
            pd.DataFrame(graph_rows),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info(
            "No LangGraph checkpoint data available yet."
        )

    # -------------------------
    # Recent Operational Events
    # -------------------------

    st.markdown("### 📝 Recent Operational Events")

    if observability_events:
        event_rows = [
            {
                "Time": event.created_at,
                "Action": event.action,
                "Actor": event.actor,
                "Role": event.role,
                "Resource": event.resource_id,
                "Details": event.details,
            }
            for event in observability_events[:50]
        ]

        st.dataframe(
            pd.DataFrame(event_rows),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info(
            "No operational events recorded yet."
        )

        st.caption(
            "Observability currently uses recorded investigation "
            "metrics, LangGraph checkpoints, and audit events. "
            "Model, token, and provider-level telemetry will be "
            "added in the production observability phase."
        )
