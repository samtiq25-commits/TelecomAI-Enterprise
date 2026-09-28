import sqlite3
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt
from src.database import (
    audit,
    save_incident, update_case_status, save_investigation_metric,
    find_similar_historical_incidents,

)
from .complaint_analyzer import classify_complaint
from .rag import retrieve
from .llm import generate_explanation
from .data_generator import load_data
from src.fraud_detection import (
    get_transaction,
    predict_fraud,
)
from langgraph.types import interrupt
from langchain_core.tools import tool


@tool
def get_tower_kpi(tower_id: str):
    """
    Read the latest available KPI record for a telecom tower.
    """
    network, _, _ = load_data()

    tower_data = (
        network[
            network["tower_id"] == tower_id
        ]
        .sort_values("timestamp")
        .tail(1)
    )

    if tower_data.empty:
        return {
            "tower_id": tower_id,
            "found": False,
            "message": (
                f"No KPI record was found for tower {tower_id}."
            ),
        }

    record = tower_data.iloc[0].to_dict()

    return {
        "tower_id": tower_id,
        "found": True,
        "kpi": {
            key: (
                str(value)
                if key == "timestamp"
                else value
            )
            for key, value in record.items()
        },
    }


@tool
def get_equipment_health(equipment_id: str):
    """
    Retrieve the latest available health metrics for a telecom equipment unit.
    """
    import pandas as pd

    equipment = pd.read_csv("data/equipment_data.csv")

    equipment_data = equipment[
        equipment["equipment_id"] == equipment_id
    ]

    if equipment_data.empty:
        return {
            "equipment_id": equipment_id,
            "found": False,
            "message": (
                f"No equipment record was found for equipment {equipment_id}."
            ),
        }

    record = equipment_data.iloc[-1].to_dict()

    return {
        "equipment_id": equipment_id,
        "found": True,
        "health": record,
    }


@tool
def find_equipment_for_investigation(
    limit: int = 5,
    tower_id: str = ""
):
    """
    Return equipment records that may require investigation.
    When tower_id is provided, only equipment attached to that tower
    is considered.
    """
    import pandas as pd

    equipment = pd.read_csv("data/equipment_data.csv")

    if tower_id:
        equipment = equipment[
            equipment["tower_id"] == tower_id
        ]

    candidates = (
        equipment.sort_values(
            ["error_count", "cpu_usage"],
            ascending=[False, False]
        )
        .head(limit)
    )

    return {
        "found": not candidates.empty,
        "count": len(candidates),
        "tower_id": tower_id,
        "equipment": candidates.to_dict("records"),
    }


class State(TypedDict, total=False):
    request_type: str
    selected_agent: str
    routing_reason: str
    actor: str
    actor_role: str
    reviewer_actor: str
    reviewer_role: str
    source_channel: str
    audience: str
    issue: str
    tower_id: str
    transaction_id: str
    execution_time_ms: int
    fraud_probability: float
    fraud_risk_level: str
    fraud_detected: bool
    fraud_reasons: list
    historical_incidents: list
    historical_guidance: list
    fraud_decision: str
    anomaly: dict

    complaint_analysis: dict
    retrieved_docs: list

    kpi_status: str
    investigation: str
    diagnosis: str
    recommendation: str
    explanation: str
    incident_report: str
    root_cause: str
    root_cause_category: str
    possible_causes: list
    cause_analysis: list
    root_cause_confidence: str
    root_cause_evidence: list
    contradicting_evidence: list
    rag_guidance: list
    verification_status: str
    verification_action: str
    service_recovery_status: str
    service_recovery_reason: str
    recovery_kpi: dict
    investigation_attempts: int
    additional_evidence: list
    human_review_status: str
    human_review_feedback: str
    recovery_baseline: dict
    incident_id: int
    case_status: str
    network_decision: str
    network_decision_reason: str
    thread_id: str


def analyze_complaint(state: State):
    analysis = classify_complaint(state["issue"])

    return {
        "complaint_analysis": analysis
    }


def check_kpis(state: State):
    tower_id = state.get("tower_id")

    # Use the Network KPI tool to retrieve the latest tower data.
    if tower_id:
        tower_result = get_tower_kpi.invoke({
            "tower_id": tower_id
        })

        if not tower_result.get("found"):
            return {
                "kpi_status": "UNKNOWN",
                "investigation": tower_result.get(
                    "message",
                    f"No KPI record was found for tower {tower_id}."
                )
            }

        anomaly = tower_result.get("kpi", {})
    else:
        anomaly = state.get("anomaly", {})

    if not anomaly:
        return {
            "kpi_status": "UNKNOWN",
            "investigation": (
                "No recent KPI record was available for the selected tower."
            )
        }

    checks = []

    latency = float(anomaly.get("latency", 0))
    packet_loss = float(anomaly.get("packet_loss", 0))
    cpu = float(anomaly.get("cpu_usage", 0))
    memory = float(anomaly.get("memory_usage", 0))
    call_drop = float(anomaly.get("call_drop_rate", 0))
    signal = float(anomaly.get("signal_strength", 0))

    if latency > 30:
        checks.append(f"High latency detected ({latency:.2f} ms)")

    if packet_loss > 2:
        checks.append(f"Elevated packet loss detected ({packet_loss:.2f}%)")

    if cpu > 80:
        checks.append(f"High CPU utilization detected ({cpu:.1f}%)")

    if memory > 80:
        checks.append(f"High memory utilization detected ({memory:.1f}%)")

    if call_drop > 2:
        checks.append(f"Elevated call-drop rate detected ({call_drop:.2f}%)")

    if signal < -95:
        checks.append(f"Weak signal detected ({signal:.2f} dBm)")

    if anomaly.get("is_anomaly"):
        checks.append("ML anomaly detector flagged this KPI record.")

    if checks:
        return {
            "kpi_status": "ABNORMAL",
            "investigation": "\n".join(checks),
            "anomaly": anomaly,
        }

    return {
        "kpi_status": "NORMAL",
        "investigation": (
            "The latest available KPI record does not show a major "
            "network abnormality."
        ),
        "anomaly": anomaly,
    }


def verify_service_recovery(state: State):
    print("========== RECOVERY STATE DEBUG ==========")
    print("ADDITIONAL EVIDENCE COUNT:",
          len(state.get("additional_evidence", [])))
    print("RECOVERY BASELINE:",
          state.get("recovery_baseline"))
    print("========== SERVICE RECOVERY DEBUG ==========")
    print("verify_service_recovery() EXECUTED")
    print("Tower:", state.get("tower_id"))

    tower_id = state.get("tower_id")

    if not tower_id:
        result = {
            "service_recovery_status": "UNKNOWN",
            "service_recovery_reason": (
                "Service recovery could not be verified because "
                "no tower ID was available."
            )
        }

        print("RECOVERY RESULT:", result)
        return result

    tower_result = get_tower_kpi.invoke({
        "tower_id": tower_id
    })

    print("TOWER KPI RESULT:", tower_result)

    if not tower_result.get("found"):
        result = {
            "service_recovery_status": "UNKNOWN",
            "service_recovery_reason": (
                f"Current KPI data was unavailable for tower {tower_id}."
            )
        }

        print("RECOVERY RESULT:", result)
        return result

    import math

    kpi = tower_result.get("kpi", {})

    required_kpis = [
        "latency",
        "packet_loss",
        "call_drop_rate",
    ]

    missing_kpis = [
        name
        for name in required_kpis
        if kpi.get(name) is None
    ]

    if missing_kpis:
        return {
            "service_recovery_status": "NOT_RECOVERED",
            "service_recovery_reason": (
                "Service recovery cannot be confirmed. "
                "Missing KPI values: "
                + ", ".join(missing_kpis)
            ),
            "recovery_kpi": kpi,
        }

    try:
        latency = float(kpi["latency"])
        packet_loss = float(kpi["packet_loss"])
        call_drop = float(kpi["call_drop_rate"])

    except (TypeError, ValueError):
        return {
            "service_recovery_status": "NOT_RECOVERED",
            "service_recovery_reason": (
                "Service recovery cannot be confirmed "
                "because KPI values are invalid."
            ),
            "recovery_kpi": kpi,
        }

    if not all(
        math.isfinite(value)
        for value in [
            latency,
            packet_loss,
            call_drop,
        ]
    ):
        return {
            "service_recovery_status": "NOT_RECOVERED",
            "service_recovery_reason": (
                "Service recovery cannot be confirmed "
                "because KPI values are not finite."
            ),
            "recovery_kpi": kpi,
        }

    recovery_baseline = state.get("recovery_baseline", {})
    print("========== RECOVERY STATE DEBUG ==========")
    print("RECOVERY BASELINE:", recovery_baseline)
    print("ADDITIONAL EVIDENCE COUNT:", len(
        state.get("additional_evidence", [])))
    print("ADDITIONAL EVIDENCE:", state.get("additional_evidence", []))
    print("========== RECOVERY BASELINE DEBUG ==========")
    print("RECOVERY BASELINE:", recovery_baseline)
    print("CURRENT LATENCY:", latency)
    print("CURRENT PACKET LOSS:", packet_loss)

    avg_latency = recovery_baseline.get("avg_latency")
    avg_packet_loss = recovery_baseline.get("avg_packet_loss")

    failed_checks = []
    baseline_checks = []

    # ---------------------------------------------------------
    # 1. Current KPI threshold checks
    # ---------------------------------------------------------

    if latency > 30:
        failed_checks.append(
            f"latency remains elevated at {latency:.2f} ms"
        )

    if packet_loss > 2:
        failed_checks.append(
            f"packet loss remains elevated at {packet_loss:.2f}%"
        )

    if call_drop > 2:
        failed_checks.append(
            f"call-drop rate remains elevated at {call_drop:.2f}%"
        )

    # ---------------------------------------------------------
    # 2. Historical baseline comparison
    # ---------------------------------------------------------

    if avg_latency is not None:
        latency_change = latency - float(avg_latency)

        baseline_checks.append(
            f"current latency {latency:.2f} ms vs "
            f"recent average {float(avg_latency):.2f} ms"
        )

        if latency_change > 5:
            failed_checks.append(
                f"latency is {latency_change:.2f} ms above "
                f"the recent tower average"
            )

    if avg_packet_loss is not None:
        packet_loss_change = (
            packet_loss - float(avg_packet_loss)
        )

        baseline_checks.append(
            f"current packet loss {packet_loss:.2f}% vs "
            f"recent average {float(avg_packet_loss):.2f}%"
        )

        if packet_loss_change > 0.5:
            failed_checks.append(
                f"packet loss is {packet_loss_change:.2f} percentage "
                f"points above the recent tower average"
            )

    # ---------------------------------------------------------
    # 3. Determine recovery status
    # ---------------------------------------------------------

    if failed_checks:
        result = {
            "service_recovery_status": "NOT_RECOVERED",
            "service_recovery_reason": (
                "Service recovery has not been confirmed: "
                + "; ".join(failed_checks)
                + "."
            ),
            "recovery_kpi": kpi
        }

        print("BASELINE CHECKS:", baseline_checks)
        print("RECOVERY RESULT:", result)

        return result

    result = {
        "service_recovery_status": "RECOVERED",
        "service_recovery_reason": (
            "Current service KPIs are within the configured "
            "recovery thresholds and are consistent with the "
            "recent tower baseline."
        ),
        "recovery_kpi": kpi
    }

    print("BASELINE CHECKS:", baseline_checks)
    print("FAILED CHECKS:", failed_checks)
    print("FINAL RECOVERY STATUS:", result["service_recovery_status"])
    print("FINAL RECOVERY REASON:", result["service_recovery_reason"])
    print("RECOVERY RESULT:", result)

    return result


def collect_additional_evidence(state: State):
    from .data_generator import load_data

    network, equipment, complaints = load_data()
    from .config import NETWORK_EVENTS_CSV
    import pandas as pd

    network_events = pd.read_csv(
        NETWORK_EVENTS_CSV,
        parse_dates=["timestamp"]
    )
    tower_id = state.get("tower_id")
    anomaly = state.get("anomaly", {})

    additional_evidence = []

    # ---------------------------------------------------------
    # 1. Current network traffic
    # ---------------------------------------------------------
    traffic = float(anomaly.get("network_traffic", 0))

    if traffic > 80:
        additional_evidence.append(
            f"Current network traffic is {traffic:.1f}%, "
            "which may indicate traffic saturation."
        )
    else:
        additional_evidence.append(
            f"Current network traffic is {traffic:.1f}%."
        )

    # ---------------------------------------------------------
    # 2. Recent tower-specific network history
    # ---------------------------------------------------------
    tower_history = network[
        network["tower_id"] == tower_id
    ].sort_values("timestamp").tail(10)

    if not tower_history.empty:

        avg_latency = tower_history["latency"].mean()
        avg_packet_loss = tower_history["packet_loss"].mean()
        avg_traffic = tower_history["network_traffic"].mean()
        recovery_baseline = {
            "avg_latency": float(avg_latency),
            "avg_packet_loss": float(avg_packet_loss),
            "avg_traffic": float(avg_traffic),
            "sample_size": int(len(tower_history))
        }

        additional_evidence.append(
            f"Recent average latency for {tower_id}: "
            f"{avg_latency:.2f} ms."
        )

        additional_evidence.append(
            f"Recent average packet loss for {tower_id}: "
            f"{avg_packet_loss:.2f}%."
        )

        additional_evidence.append(
            f"Recent average network traffic for {tower_id}: "
            f"{avg_traffic:.1f}%."
        )

    # ---------------------------------------------------------
    # 3. Equipment evidence
    # ---------------------------------------------------------
    if not equipment.empty:

        error_threshold = equipment["error_count"].quantile(0.75)

        elevated_errors = equipment[
            equipment["error_count"] > error_threshold
        ]

        if not elevated_errors.empty:
            additional_evidence.append(
                f"{len(elevated_errors)} equipment records show "
                "elevated error counts across the available "
                "equipment data."
            )
        else:
            additional_evidence.append(
                "No elevated equipment error counts were found "
                "in the available equipment data."
            )

    else:
        additional_evidence.append(
            "No equipment data is available for analysis."
        )

    # ---------------------------------------------------------
    # 4. Comparative tower analysis
    # ---------------------------------------------------------
    if "tower_id" in network.columns and tower_id:

        other_towers = network[
            network["tower_id"] != tower_id
        ]

        neighboring_tower_count = (
            other_towers["tower_id"].nunique()
        )

        additional_evidence.append(
            f"{neighboring_tower_count} other towers are available "
            "for comparative analysis."
        )

        # -----------------------------------------------------
        # Calculate network-wide baseline
        # -----------------------------------------------------
        if not other_towers.empty and not tower_history.empty:

            comparison_latency = other_towers["latency"].mean()
            comparison_packet_loss = other_towers[
                "packet_loss"
            ].mean()
            comparison_traffic = other_towers[
                "network_traffic"
            ].mean()

            additional_evidence.append(
                f"Network comparison baseline: "
                f"average latency {comparison_latency:.2f} ms, "
                f"average packet loss "
                f"{comparison_packet_loss:.2f}%, "
                f"average traffic "
                f"{comparison_traffic:.1f}%."
            )

            # -------------------------------------------------
            # Compare affected tower against baseline
            # -------------------------------------------------
            tower_latency = tower_history["latency"].mean()
            tower_packet_loss = tower_history[
                "packet_loss"
            ].mean()
            tower_traffic = tower_history[
                "network_traffic"
            ].mean()

            latency_difference = (
                tower_latency - comparison_latency
            )

            packet_loss_difference = (
                tower_packet_loss - comparison_packet_loss
            )

            traffic_difference = (
                tower_traffic - comparison_traffic
            )

            # -------------------------------------------------
            # Latency comparison
            # -------------------------------------------------
            if latency_difference > 5:
                additional_evidence.append(
                    f"{tower_id} has significantly higher latency "
                    f"than the comparison baseline: "
                    f"{tower_latency:.2f} ms vs "
                    f"{comparison_latency:.2f} ms."
                )
            else:
                additional_evidence.append(
                    f"{tower_id} latency is close to the "
                    f"network comparison baseline: "
                    f"{tower_latency:.2f} ms vs "
                    f"{comparison_latency:.2f} ms."
                )

            # -------------------------------------------------
            # Packet loss comparison
            # -------------------------------------------------
            if packet_loss_difference > 0.5:
                additional_evidence.append(
                    f"{tower_id} has higher packet loss than "
                    f"the comparison baseline: "
                    f"{tower_packet_loss:.2f}% vs "
                    f"{comparison_packet_loss:.2f}%."
                )
            else:
                additional_evidence.append(
                    f"{tower_id} packet loss is close to the "
                    f"network comparison baseline: "
                    f"{tower_packet_loss:.2f}% vs "
                    f"{comparison_packet_loss:.2f}%."
                )

            # -------------------------------------------------
            # Traffic comparison
            # -------------------------------------------------
            if abs(traffic_difference) > 10:
                additional_evidence.append(
                    f"{tower_id} traffic differs from the "
                    f"network comparison baseline: "
                    f"{tower_traffic:.1f}% vs "
                    f"{comparison_traffic:.1f}%."
                )
            else:
                additional_evidence.append(
                    f"{tower_id} traffic is close to the "
                    f"network comparison baseline: "
                    f"{tower_traffic:.1f}% vs "
                    f"{comparison_traffic:.1f}%."
                )
            # ---------------------------------------------------------
            # 5. Backhaul, maintenance, and configuration investigation
            # ---------------------------------------------------------
            if not network_events.empty and tower_id:
                tower_events = network_events[
                    network_events["tower_id"] == tower_id
                ].sort_values("timestamp")

                if not tower_events.empty:
                    # -------------------------------------------------
                    # Backhaul investigation
                    # -------------------------------------------------
                    backhaul_events = tower_events[
                        tower_events["event_type"] == "BACKHAUL"
                    ]

                    if not backhaul_events.empty:
                        latest_backhaul = backhaul_events.iloc[-1]
                        backhaul_utilization = float(
                            latest_backhaul["backhaul_utilization"]
                        )

                        if backhaul_utilization >= 85:
                            additional_evidence.append(
                                f"Backhaul investigation: {tower_id} "
                                f"latest utilization is "
                                f"{backhaul_utilization:.1f}%, "
                                "indicating high backhaul utilization."
                            )
                        else:
                            additional_evidence.append(
                                f"Backhaul investigation: {tower_id} "
                                f"latest utilization is "
                                f"{backhaul_utilization:.1f}%, "
                                "with no high-utilization condition detected."
                            )
                    else:
                        additional_evidence.append(
                            f"No backhaul events were found for {tower_id}."
                        )

                    # -------------------------------------------------
                    # Maintenance investigation
                    # -------------------------------------------------
                    maintenance_events = tower_events[
                        tower_events["event_type"] == "MAINTENANCE"
                    ]

                    if not maintenance_events.empty:
                        latest_maintenance = maintenance_events.iloc[-1]
                        additional_evidence.append(
                            f"Maintenance investigation: latest maintenance "
                            f"event for {tower_id} was recorded at "
                            f"{latest_maintenance['timestamp']} "
                            f"({latest_maintenance['event_name']})."
                        )
                    else:
                        additional_evidence.append(
                            f"No maintenance events were found for {tower_id}."
                        )

                    # -------------------------------------------------
                    # Configuration investigation
                    # -------------------------------------------------
                    configuration_events = tower_events[
                        tower_events["event_type"] == "CONFIGURATION"
                    ]

                    if not configuration_events.empty:
                        latest_configuration = configuration_events.iloc[-1]
                        additional_evidence.append(
                            f"Configuration investigation: latest configuration "
                            f"event for {tower_id} was recorded at "
                            f"{latest_configuration['timestamp']} "
                            f"({latest_configuration['event_name']})."
                        )
                    else:
                        additional_evidence.append(
                            f"No configuration events were found for {tower_id}."
                        )
                else:
                    additional_evidence.append(
                        f"No network operational events were found for {tower_id}."
                    )

    else:
        additional_evidence.append(
            "Network operational event data is unavailable."
        )
    return {
        "additional_evidence": additional_evidence,
        "recovery_baseline": recovery_baseline
    }


def route_after_kpi_check(state: State):
    status = state.get("kpi_status", "UNKNOWN")

    if status == "ABNORMAL":
        return "investigate"

    return "retrieve"


def investigate(state: State):
    analysis = state.get("complaint_analysis", {})
    investigation = state.get("investigation", "")
    attempts = state.get("investigation_attempts", 0) + 1

    category = analysis.get("category", "Unknown")
    anomaly = state.get("anomaly", {})
    kpi_status = state.get("kpi_status", "UNKNOWN")
    root_cause = state.get("root_cause", "")

    evidence = [investigation]

    # Always initialize these values so they are available
    # throughout the function and safely returned to state.
    additional_evidence = []
    recovery_baseline = {}

    # Decide whether supplementary equipment evidence is needed.
    equipment_needed = (
        kpi_status == "ABNORMAL"
        and not root_cause
    )

    tool_decision = (
        "Equipment investigation selected because the KPI state is "
        "ABNORMAL and a verified root cause is not available."
        if equipment_needed
        else
        "Equipment investigation not required at this stage."
    )

    if equipment_needed:
        # Use the equipment investigation tool to collect additional evidence.
        affected_tower_id = state.get("tower_id", "")

        equipment_result = find_equipment_for_investigation.invoke({
            "limit": 5,
            "tower_id": affected_tower_id
        })

        equipment_records = (
            equipment_result.get("equipment", [])
            if equipment_result.get("found")
            else []
        )
    else:
        equipment_records = []

    affected_tower_id = state.get("tower_id")

    if affected_tower_id:
        associated_equipment = [
            item
            for item in equipment_records
            if item.get("tower_id") == affected_tower_id
        ]

        unmapped_equipment = [
            item
            for item in equipment_records
            if item.get("tower_id") != affected_tower_id
        ]
    else:
        associated_equipment = []
        unmapped_equipment = equipment_records

    def format_equipment(item):
        return (
            f"{item['equipment_id']}: "
            f"errors={item['error_count']}, "
            f"CPU={item['cpu_usage']:.1f}%, "
            f"memory={item['memory_usage']:.1f}%, "
            f"temperature={item['temperature']:.1f}, "
            f"failure={item['failure']}"
        )

    if associated_equipment:
        equipment_evidence = [
            format_equipment(item)
            for item in associated_equipment
        ]

        evidence.append(
            f"Equipment health evidence for affected tower "
            f"{affected_tower_id}:\n"
            + "\n".join(equipment_evidence)
        )

    if unmapped_equipment:
        unmapped_evidence = [
            format_equipment(item)
            for item in unmapped_equipment
        ]

        evidence.append(
            "Additional equipment health candidates "
            "(not directly associated with the affected tower):\n"
            + "\n".join(unmapped_evidence)
        )

    # Perform additional evidence collection during retry.
    if attempts >= 2:
        additional_result = collect_additional_evidence(state)

        additional_evidence = additional_result.get(
            "additional_evidence",
            []
        )

        recovery_baseline = additional_result.get(
            "recovery_baseline",
            {}
        )

        evidence.extend(additional_evidence)

    diagnosis = (
        f"Potential network degradation related to {category}. "
        f"KPI investigation found:\n"
        + "\n".join(evidence)
    )

    return {
        "diagnosis": diagnosis,
        "investigation": "\n".join(evidence),
        "additional_evidence": additional_evidence,
        "recovery_baseline": recovery_baseline,
        "investigation_attempts": attempts,
    }


def retrieve_node(state: State):
    print("========== RETRIEVE NODE DEBUG ==========")
    print("Tower:", state.get("tower_id"))
    print(
        "Historical incidents found:",
        len(
            find_similar_historical_incidents(
                tower_id=state.get("tower_id"),
                issue=state.get("issue"),
                limit=5,
            )
        )
    )
    query = (
        state["issue"]
        + " "
        + state.get("diagnosis", "")
        + " "
        + state.get("investigation", "")
    )

    # ---------------------------------------------------------
    # 1. Retrieve telecom knowledge-base guidance
    # ---------------------------------------------------------

    retrieved_docs = retrieve(query)

    rag_guidance = []

    for doc in retrieved_docs:
        content = str(doc.get("content", ""))
        source = doc.get(
            "source",
            "knowledge_base"
        )

        if not content:
            continue

        rag_guidance.append(
            f"{source}: {content.strip()}"
        )

    # ---------------------------------------------------------
    # 2. Retrieve historical incidents
    # ---------------------------------------------------------

    historical_incidents = (
        find_similar_historical_incidents(
            tower_id=state.get("tower_id"),
            issue=state.get("issue"),
            limit=5,
        )
    )

    historical_guidance = []

    if historical_incidents:
        historical_guidance.append(
            f"Found {len(historical_incidents)} "
            "resolved historical incidents for "
            f"tower {state.get('tower_id', 'unknown')}."
        )

        for incident in historical_incidents:
            historical_guidance.append(
                f"Incident #{incident['incident_id']}: "
                f"status={incident['case_status']}, "
                f"root-cause confidence="
                f"{incident['root_cause_confidence']}, "
                f"diagnosis="
                f"{incident['diagnosis'] or 'Not recorded'}."
            )
    else:
        historical_guidance.append(
            "No resolved historical incidents were found "
            "for this tower."
        )

    # ---------------------------------------------------------
    # 3. Return both evidence sources
    # ---------------------------------------------------------

    return {
        "retrieved_docs": retrieved_docs,
        "rag_guidance": rag_guidance,
        "historical_incidents": historical_incidents,
        "historical_guidance": historical_guidance,
    }


def diagnose_from_evidence(state: State):
    analysis = state.get("complaint_analysis", {})
    kpi_status = state.get("kpi_status", "UNKNOWN")
    investigation = state.get("investigation", "")

    category = analysis.get("category", "Unknown")

    if kpi_status == "NORMAL":
        diagnosis = (
            f"Likely {category} based on the customer complaint. "
            "The latest available tower KPIs do not show a major anomaly."
        )

    elif kpi_status == "UNKNOWN":
        diagnosis = (
            f"Likely {category}. "
            "Network diagnosis is limited because recent KPI data "
            "was unavailable."
        )

    else:
        diagnosis = (
            f"Likely {category} with supporting evidence of network degradation."
        )

        # Add concise interpretation of important investigation findings.
        investigation_lower = investigation.lower()

        key_findings = []

        if "packet loss" in investigation_lower:
            key_findings.append(
                "Elevated packet loss is present."
            )
        if "high latency" in investigation_lower:
            key_findings.append(
                "Elevated latency is present."
            )
        if "equipment health evidence" in investigation_lower:
            key_findings.append(
                "Multiple equipment indicators require attention."
            )

        if "no high-utilization condition detected" in investigation_lower:
            key_findings.append(
                "Backhaul utilization does not indicate high-utilization congestion."
            )

        if "traffic is close to the network comparison baseline" in investigation_lower:
            key_findings.append(
                "Network traffic is close to the comparison baseline."
            )

        if "configuration event" in investigation_lower:
            key_findings.append(
                "Recent configuration activity may be a contributing factor."
            )

        if key_findings:
            diagnosis += "\nKey findings:\n" + "\n".join(
                f"- {finding}" for finding in key_findings
            )

        diagnosis += (
            "\nThe available evidence supports further investigation "
            "of equipment and configuration-related causes, but does not "
            "establish a definitive single root cause."
        )

    return {
        "diagnosis": diagnosis
    }


def calculate_evidence_strength(
    supporting_count: int,
    contradicting_count: int,
) -> str:
    net_evidence = supporting_count - contradicting_count

    if net_evidence >= 3:
        return "HIGH"

    if net_evidence >= 1:
        return "MEDIUM"

    if supporting_count > 0:
        return "LOW"

    return "UNSUPPORTED"


def root_cause_analysis(state: State):
    print("========== ROOT CAUSE STATE DEBUG ==========")
    print("ADDITIONAL EVIDENCE COUNT:",
          len(state.get("additional_evidence", [])))
    print("RECOVERY BASELINE:",
          state.get("recovery_baseline"))
    """
    RAG-aware root cause analysis.

    Combines:
    - Live network KPI evidence
    - Equipment health evidence
    - Retrieved knowledge-base guidance

    RAG guidance is treated as investigation guidance,
    not as proof of a root cause.
    """

    anomaly = state.get("anomaly", {})
    kpi_status = state.get("kpi_status", "UNKNOWN")
    additional_evidence = state.get("additional_evidence", [])
    investigation = state.get("investigation", "")
    retrieved_docs = state.get("retrieved_docs", [])

    possible_causes = []
    cause_analysis = []
    cause_evidence_map = {}
    supporting_evidence = []
    rag_guidance = []
    evidence_count = 0

    # ---------------------------------------------------------
    # 1. Network KPI values
    # ---------------------------------------------------------

    latency = float(anomaly.get("latency", 0))
    packet_loss = float(anomaly.get("packet_loss", 0))
    cpu = float(anomaly.get("cpu_usage", 0))
    memory = float(anomaly.get("memory_usage", 0))
    call_drop = float(anomaly.get("call_drop_rate", 0))
    signal = float(anomaly.get("signal_strength", 0))
    traffic = float(anomaly.get("network_traffic", 0))

    # ---------------------------------------------------------
    # 2. Extract RAG knowledge-base guidance
    # ---------------------------------------------------------

    for doc in retrieved_docs:
        content = str(doc.get("content", ""))
        source = doc.get("source", "knowledge_base")

        if not content:
            continue

        rag_guidance.append(
            f"{source}: {content.strip()}"
        )

    # ---------------------------------------------------------
    # 3. Analyze live network evidence
    # ---------------------------------------------------------

    if kpi_status == "ABNORMAL":

        if latency > 30:
            evidence_count += 1
            supporting_evidence.append(
                f"Live KPI evidence: latency is elevated at "
                f"{latency:.2f} ms."
            )

        if packet_loss > 2:
            evidence_count += 1
            supporting_evidence.append(
                f"Live KPI evidence: packet loss is elevated at "
                f"{packet_loss:.2f}%."
            )

        if traffic > 80:
            evidence_count += 1
            supporting_evidence.append(
                f"Live KPI evidence: network traffic is high at "
                f"{traffic:.1f}%."
            )

        if cpu > 80:
            evidence_count += 1
            supporting_evidence.append(
                f"Live KPI evidence: CPU utilization is high at "
                f"{cpu:.1f}%."
            )

        if memory > 80:
            evidence_count += 1
            supporting_evidence.append(
                f"Live KPI evidence: memory utilization is high at "
                f"{memory:.1f}%."
            )

        if call_drop > 2:
            evidence_count += 1
            supporting_evidence.append(
                f"Live KPI evidence: call-drop rate is elevated at "
                f"{call_drop:.2f}%."
            )

        if signal < -95:
            evidence_count += 1
            supporting_evidence.append(
                f"Live KPI evidence: signal strength is weak at "
                f"{signal:.1f}."
            )

    # ---------------------------------------------------------
    # 4. Network root-cause hypotheses
    # ---------------------------------------------------------

    if kpi_status == "ABNORMAL":

        if latency > 30 and packet_loss > 2:
            possible_causes.append(
                "Potential network congestion or transport degradation"
            )

        if traffic > 80 and latency > 30:
            possible_causes.append(
                "Potential network traffic saturation"
            )

        if cpu > 80:
            possible_causes.append(
                "Potential network equipment processing overload"
            )

        if memory > 80:
            possible_causes.append(
                "Potential equipment memory pressure"
            )

        if call_drop > 2:
            possible_causes.append(
                "Potential radio or network quality degradation"
            )

        if signal < -95:
            possible_causes.append(
                "Potential weak signal or radio coverage issue"
            )

    # ---------------------------------------------------------
    # 5. Analyze equipment evidence
    # ---------------------------------------------------------

    combined_evidence = (
        str(investigation)
        + "\n"
        + "\n".join(
            str(item)
            for item in additional_evidence
        )
    )

    equipment_records = []

    for line in combined_evidence.splitlines():

        line = line.strip()

        if not line.lower().startswith("eq-"):
            continue

        equipment_records.append(line)

        try:
            equipment_id = line.split(":")[0].strip()

            errors = None
            equipment_cpu = None
            equipment_memory = None
            temperature = None
            failure = None

            for part in line.split(","):

                part = part.strip()

                if "errors=" in part.lower():
                    errors = float(
                        part.split("=", 1)[1]
                    )

                elif "cpu=" in part.lower():
                    equipment_cpu = float(
                        part.split("=", 1)[1]
                        .replace("%", "")
                    )

                elif "memory=" in part.lower():
                    equipment_memory = float(
                        part.split("=", 1)[1]
                        .replace("%", "")
                    )

                elif "temperature=" in part.lower():
                    temperature = float(
                        part.split("=", 1)[1]
                    )

                elif "failure=" in part.lower():
                    failure = int(
                        float(part.split("=", 1)[1])
                    )

            # Error count
            if errors is not None and errors >= 10:
                evidence_count += 1

                supporting_evidence.append(
                    f"{equipment_id} has an elevated error count "
                    f"of {int(errors)}."
                )

                possible_causes.append(
                    f"Potential equipment degradation involving "
                    f"{equipment_id}"
                )

            # Temperature
            if temperature is not None and temperature >= 60:
                evidence_count += 1

                supporting_evidence.append(
                    f"{equipment_id} has elevated temperature "
                    f"of {temperature:.1f}°C."
                )

                possible_causes.append(
                    f"Potential thermal stress involving "
                    f"{equipment_id}"
                )

            # Memory
            if equipment_memory is not None and equipment_memory >= 70:
                evidence_count += 1

                supporting_evidence.append(
                    f"{equipment_id} has elevated memory utilization "
                    f"of {equipment_memory:.1f}%."
                )

                possible_causes.append(
                    f"Potential memory pressure involving "
                    f"{equipment_id}"
                )

            # CPU
            if equipment_cpu is not None and equipment_cpu >= 70:
                evidence_count += 1

                supporting_evidence.append(
                    f"{equipment_id} has elevated CPU utilization "
                    f"of {equipment_cpu:.1f}%."
                )

                possible_causes.append(
                    f"Potential processing load involving "
                    f"{equipment_id}"
                )

            # Recorded failure
            if failure is not None and failure > 0:
                evidence_count += 1

                supporting_evidence.append(
                    f"{equipment_id} has a recorded equipment failure."
                )

                possible_causes.append(
                    f"Potential equipment failure involving "
                    f"{equipment_id}"
                )

        except (ValueError, IndexError):
            continue

    # ---------------------------------------------------------
    # 6. Historical / supplementary evidence
    # ---------------------------------------------------------

    for item in additional_evidence:

        text = str(item).lower()

        if "average latency" in text:
            supporting_evidence.append(
                "Recent latency trend supports persistent degradation."
            )

        if "average packet loss" in text:
            supporting_evidence.append(
                "Recent packet-loss trend supports persistent degradation."
            )

        if "average traffic" in text:
            supporting_evidence.append(
                "Recent traffic trend provides additional network-load context."
            )

    # ---------------------------------------------------------
    # 7. Remove duplicate causes/evidence
    # ---------------------------------------------------------

    possible_causes = list(
        dict.fromkeys(possible_causes)
    )

    supporting_evidence = list(
        dict.fromkeys(supporting_evidence)
    )

    rag_guidance = list(
        dict.fromkeys(rag_guidance)
    )
    # ---------------------------------------------------------
    # 8. Determine confidence and separate supporting/
    #    contradicting evidence
    # ---------------------------------------------------------

    has_network_degradation = (
        latency > 30
        and packet_loss > 2
    )

    has_equipment_indicator = (
        len(equipment_records) > 0
        and any(
            (
                "elevated error count" in item.lower()
                or "elevated temperature" in item.lower()
                or "elevated memory" in item.lower()
                or "elevated cpu" in item.lower()
            )
            for item in supporting_evidence
        )
    )

    has_historical_support = any(
        "recent latency trend" in item.lower()
        or "recent packet-loss trend" in item.lower()
        for item in supporting_evidence
    )

    # ---------------------------------------------------------
    # Comparative tower evidence
    # ---------------------------------------------------------

    comparative_evidence = [
        str(item).lower()
        for item in additional_evidence
    ]

    has_comparative_support = any(
        (
            "significantly higher latency" in item
            or "higher packet loss" in item
            or "traffic differs from" in item
        )
        for item in comparative_evidence
    )

    # ---------------------------------------------------------
    # Operational event evidence
    # ---------------------------------------------------------

    has_high_backhaul = any(
        "high backhaul utilization" in item
        for item in comparative_evidence
    )

    has_normal_backhaul = any(
        "no high-utilization condition detected" in item
        for item in comparative_evidence
    )

    has_recent_maintenance = any(
        "maintenance investigation:" in item
        for item in comparative_evidence
    )

    has_recent_configuration = any(
        "configuration investigation:" in item
        for item in comparative_evidence
    )

    # ---------------------------------------------------------
    # Network congestion assessment
    # ---------------------------------------------------------

    traffic_supports_congestion = traffic > 80

    traffic_matches_baseline = any(
        "traffic is close to the network comparison baseline"
        in item
        for item in comparative_evidence
    )

    latency_below_or_close_to_baseline = any(
        "latency is close to the network comparison baseline"
        in item
        for item in comparative_evidence
    )

    # ---------------------------------------------------------
    # Contradicting evidence
    # ---------------------------------------------------------

    contradicting_evidence = []

    if has_normal_backhaul:
        contradicting_evidence.append(
            "Backhaul utilization is normal and does not indicate "
            "high-utilization congestion."
        )

    if traffic_matches_baseline:
        contradicting_evidence.append(
            "Current network traffic is close to the network "
            "comparison baseline."
        )

    if latency_below_or_close_to_baseline:
        contradicting_evidence.append(
            "Recent tower latency is close to or below the "
            "network comparison baseline."
        )

    # ---------------------------------------------------------
    # Root-cause interpretation
    # ---------------------------------------------------------

    if has_network_degradation:

        # Strong congestion evidence
        if has_high_backhaul or traffic_supports_congestion:

            root_cause = (
                "Potential network congestion or transport degradation"
            )

        # Evidence weakens congestion and points toward
        # equipment/configuration contribution
        elif (
            has_normal_backhaul
            and traffic_matches_baseline
            and latency_below_or_close_to_baseline
        ):

            congestion_cause = (
                "Potential network congestion or transport degradation"
            )

            possible_causes = [
                cause
                for cause in possible_causes
                if cause != congestion_cause
            ]

            root_cause = (
                "Potential equipment/configuration-related "
                "network degradation"
            )

        else:
            root_cause = (
                "Potential network degradation requiring "
                "additional investigation"
            )

    else:
        root_cause = (
            "No confirmed root cause identified from current evidence"
        )

    # ---------------------------------------------------------
    # Add configuration-related cause when supported
    # ---------------------------------------------------------

    if has_recent_configuration:

        configuration_cause = (
            "Configuration-related contribution — recent "
            "configuration event"
        )

        if not any(
            "Configuration-related contribution" in cause
            for cause in possible_causes
        ):
            possible_causes.append(configuration_cause)

        supporting_evidence.append(
            "Operational evidence: a recent configuration event "
            "was recorded for the affected tower."
        )

    # ---------------------------------------------------------
    # Add maintenance evidence
    # ---------------------------------------------------------

    if has_recent_maintenance:

        supporting_evidence.append(
            "Operational evidence: a recent maintenance event "
            "was recorded for the affected tower."
        )

    # ---------------------------------------------------------
    # Add backhaul evidence
    # ---------------------------------------------------------

    if has_normal_backhaul:

        supporting_evidence.append(
            "Operational evidence: latest backhaul utilization "
            "does not indicate high utilization."
        )

    if has_high_backhaul:

        supporting_evidence.append(
            "Operational evidence: latest backhaul utilization "
            "indicates high utilization."
        )

    # ---------------------------------------------------------
    # Remove duplicate evidence
    # ---------------------------------------------------------

    supporting_evidence = list(
        dict.fromkeys(supporting_evidence)
    )

    contradicting_evidence = list(
        dict.fromkeys(contradicting_evidence)
    )

    possible_causes = list(
        dict.fromkeys(possible_causes)
    )

    # ---------------------------------------------------------
    # Overall root-cause confidence
    # Derive confidence from evidence-weighted cause analysis
    # ---------------------------------------------------------

    cause_strength_scores = {
        "HIGH": 3,
        "MEDIUM": 2,
        "LOW": 1,
        "UNSUPPORTED": 0,
        "UNASSESSED": 0,
    }

    if not possible_causes:
        possible_causes.append(
            "No clear root cause identified from the available evidence."
        )

    # ---------------------------------------------------------
    # CAUSE-SPECIFIC EVIDENCE ANALYSIS
    # ---------------------------------------------------------
    cause_analysis = []
    cause_evidence_map = {}

    # Initialize an evidence container for every possible cause
    for cause in possible_causes:
        cause_evidence_map[cause] = {
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "evidence_strength": "UNASSESSED",
        }
    # Equipment degradation
    for cause in possible_causes:
        if "equipment degradation" in cause.lower():
            equipment_id = cause.split("involving")[-1].strip()

            for line in supporting_evidence:
                if equipment_id in line:
                    cause_evidence_map[cause]["supporting_evidence"].append(
                        line)

            for line in contradicting_evidence:
                if equipment_id in line:
                    cause_evidence_map[cause]["contradicting_evidence"].append(
                        line)

    # Thermal stress
    for cause in possible_causes:
        if "thermal stress" in cause.lower():
            equipment_id = cause.split("involving")[-1].strip()

            for line in supporting_evidence:
                if equipment_id in line:
                    cause_evidence_map[cause]["supporting_evidence"].append(
                        line)

            for line in contradicting_evidence:
                if equipment_id in line:
                    cause_evidence_map[cause]["contradicting_evidence"].append(
                        line)

    # Processing load
    for cause in possible_causes:
        if "processing load" in cause.lower():
            equipment_id = cause.split("involving")[-1].strip()

            for line in supporting_evidence:
                if equipment_id in line:
                    cause_evidence_map[cause]["supporting_evidence"].append(
                        line)

            for line in contradicting_evidence:
                if equipment_id in line:
                    cause_evidence_map[cause]["contradicting_evidence"].append(
                        line)

    # Memory pressure
    for cause in possible_causes:
        if "memory pressure" in cause.lower():
            equipment_id = cause.split("involving")[-1].strip()

            for line in supporting_evidence:
                if equipment_id in line:
                    cause_evidence_map[cause]["supporting_evidence"].append(
                        line)

            for line in contradicting_evidence:
                if equipment_id in line:
                    cause_evidence_map[cause]["contradicting_evidence"].append(
                        line)

        # Configuration contribution
    for cause in possible_causes:
        if "configuration" in cause.lower():
            for line in supporting_evidence:
                if "configuration" in line.lower():
                    cause_evidence_map[cause]["supporting_evidence"].append(
                        line)

            for line in contradicting_evidence:
                if "configuration" in line.lower():
                    cause_evidence_map[cause]["contradicting_evidence"].append(
                        line)

    # ---------------------------------------------------------
    # Calculate evidence strength for every possible cause
    # ---------------------------------------------------------

    cause_analysis = []

    for cause in possible_causes:

        evidence = cause_evidence_map.get(
            cause,
            {
                "supporting_evidence": [],
                "contradicting_evidence": [],
                "evidence_strength": "UNASSESSED",
            },
        )

        supporting_items = evidence["supporting_evidence"]
        contradicting_items = evidence["contradicting_evidence"]

        evidence_score = 0

        for item in supporting_items:

            text = item.lower()

            if "recorded equipment failure" in text:
                evidence_score += 3

            elif "elevated error count" in text:
                evidence_score += 2

            elif "elevated temperature" in text:
                evidence_score += 2

            elif "elevated cpu utilization" in text:
                evidence_score += 1

            elif "elevated memory utilization" in text:
                evidence_score += 1

            elif "configuration event" in text:
                evidence_score += 1

            elif "recent latency trend" in text:
                evidence_score += 1

            elif "recent packet-loss trend" in text:
                evidence_score += 1

            else:
                evidence_score += 1

        for item in contradicting_items:
            evidence_score -= 1

        if evidence_score >= 3:
            strength = "HIGH"

        elif evidence_score >= 2:
            strength = "MEDIUM"

        elif evidence_score >= 1:
            strength = "LOW"

        else:
            strength = "UNSUPPORTED"
        # Store the calculated strength back into the evidence map
        cause_evidence_map[cause]["evidence_strength"] = strength

        cause_analysis.append(
            {
                "cause": cause,
                "supporting_evidence": supporting_items,
                "contradicting_evidence": contradicting_items,
                "evidence_strength": strength,
            }
        )

    # ---------------------------------------------------------
    # Overall root-cause confidence
    # ---------------------------------------------------------

    cause_strength_scores = {
        "HIGH": 3,
        "MEDIUM": 2,
        "LOW": 1,
        "UNSUPPORTED": 0,
        "UNASSESSED": 0,
    }

    strongest_score = 0

    for cause in possible_causes:

        evidence = cause_evidence_map.get(
            cause,
            {
                "evidence_strength": "UNASSESSED",
            },
        )

        strength = evidence.get(
            "evidence_strength",
            "UNASSESSED",
        )

        score = cause_strength_scores.get(
            strength,
            0,
        )

        if score > strongest_score:
            strongest_score = score

    if strongest_score >= 3:
        confidence = "HIGH"

    elif strongest_score >= 2:
        confidence = "MEDIUM"

    elif strongest_score >= 1:
        confidence = "LOW"

    else:
        confidence = "UNKNOWN"
        # ---------------------------------------------------------
    # Verification status
    # ---------------------------------------------------------

    if confidence == "HIGH":
        verification_status = "SUPPORTED"
        verification_action = (
            "Root cause has sufficient supporting evidence; "
            "engineer validation is recommended before final closure."
        )

    elif confidence == "MEDIUM":
        verification_status = "PARTIAL_CONFIDENCE"
        verification_action = (
            "Evidence supports the potential root cause, "
            "but engineer validation is recommended."
        )

    elif confidence == "LOW":
        verification_status = "INSUFFICIENT_EVIDENCE"
        verification_action = (
            "Additional investigation and evidence collection "
            "are required."
        )

    else:
        verification_status = "UNVERIFIED"
        verification_action = (
            "Root cause cannot currently be established from "
            "available evidence."
        )
    root_cause_category = "network_degradation"


    if any(
        "equipment" in str(cause).lower()
        or "configuration" in str(cause).lower()
        for cause in possible_causes
    ):
        root_cause_category = "equipment_or_configuration_degradation"

    elif any(
        "congestion" in str(cause).lower()
        or "traffic" in str(cause).lower()
        for cause in possible_causes
    ):
        root_cause_category = "network_congestion"

    elif any(
        "backhaul" in str(cause).lower()
        for cause in possible_causes
    ):
        root_cause_category = "backhaul_degradation"
    return {
        "root_cause": root_cause,
        "possible_causes": possible_causes,
        "cause_analysis": cause_analysis,
        "root_cause_evidence": supporting_evidence,
        "contradicting_evidence": contradicting_evidence,
        "root_cause_confidence": confidence,
        "root_cause_category": root_cause_category,
        "verification_status": verification_status,
        "verification_action": verification_action,
        "rag_guidance": rag_guidance,
    }


def route_after_verification(state: State):
    status = state.get(
        "verification_status",
        "UNKNOWN"
    )

    attempts = state.get(
        "investigation_attempts",
        0
    )

    if (
        status in (
            "INSUFFICIENT_EVIDENCE",
            "PARTIAL_CONFIDENCE",
        )
        and attempts < 2
    ):
        return "investigate"

    return "recommend"


def recommend(state: State):
    analysis = state.get(
        "complaint_analysis",
        {}
    )

    severity = analysis.get(
        "severity",
        "MEDIUM"
    )

    kpi_status = state.get(
        "kpi_status",
        "UNKNOWN"
    )

    incident_id = state.get(
        "incident_id"
    )

    if incident_id:

        recommendation_items = []

        cause_analysis = state.get("cause_analysis", [])

        for cause in cause_analysis:
            cause_name = cause.get("cause", "")
            strength = cause.get("evidence_strength", "UNASSESSED")

            if strength in ["HIGH", "MEDIUM"]:

                if "equipment degradation" in cause_name.lower():
                    equipment_name = cause_name.replace(
                        "Potential equipment degradation involving ",
                        ""
                    )

                    recommendation_items.append(
                        f"Inspect {equipment_name} for elevated error activity."
                    )

                elif "thermal stress" in cause_name.lower():
                    equipment_name = cause_name.replace(
                        "Potential thermal stress involving ",
                        ""
                    )

                    recommendation_items.append(
                        f"Inspect {equipment_name} for possible thermal stress."
                    )

        for cause in cause_analysis:
            cause_name = cause.get("cause", "").lower()

            if "configuration" in cause_name:
                recommendation_items.append(
                    "Review the recent configuration change for the affected tower."
                )
                break

        if state.get("kpi_status") == "ABNORMAL":
            recommendation_items.append(
                "Continue monitoring latency and packet loss for the affected tower."
            )

        recommendation_items.append(
            "Validate the suspected causes with a Network Engineer before closure."
        )

        recommendation = (
            f"Incident {incident_id} is now in progress.\n\n"
            "Recommended Actions:\n"
            + "\n".join(
                f"{index}. {item}"
                for index, item in enumerate(
                    recommendation_items,
                    start=1
                )
            )
            + "\n\n"
            "The incident should remain in progress until the suspected "
            "causes are verified and service stability is confirmed."
        )

    elif severity in ("HIGH", "CRITICAL") or kpi_status == "ABNORMAL":
        recommendation = (
            "Escalate to the network operations team and "
            "open an incident for engineer review."
        )

    else:
        recommendation = (
            "Run KPI checks and route the case to support."
        )

    # ---------------------------------------------------------
    # Persist final recommendation to the existing incident
    # ---------------------------------------------------------

    if incident_id:
        save_incident({
            "incident_type": "NETWORK",
            "case_status": "IN_PROGRESS",
            "tower_id": state.get("tower_id"),
            "thread_id": state.get("thread_id"),
            "diagnosis": state.get("diagnosis"),
            "recommendation": recommendation,
            "root_cause_confidence": state.get(
                "root_cause_confidence",
                "UNKNOWN"
            ),
        })

    return {
        "recommendation": recommendation
    }


def explain(state: State):
    evidence = "\n".join(
        doc["content"]
        for doc in state.get("retrieved_docs", [])
    )

    return {
        "explanation": generate_explanation(
            state.get("diagnosis", ""),
            evidence,
            state.get("recommendation", "")
        )
    }


def report(state: State):
    print("========== REPORT STATE DEBUG ==========")
    print("service_recovery_status:",
          state.get("service_recovery_status"))
    print("service_recovery_reason:",
          state.get("service_recovery_reason"))
    print("recovery_kpi:",
          state.get("recovery_kpi"))
    print("case_status:",
          state.get("case_status"))
    possible_causes = state.get("possible_causes", [])
    cause_analysis = state.get("cause_analysis", [])
    additional_evidence = state.get("additional_evidence", [])
    root_cause_evidence = state.get("root_cause_evidence", [])
    contradicting_evidence = state.get("contradicting_evidence", [])
    rag_guidance = state.get("rag_guidance", [])
    print("========== HISTORICAL DEBUG ==========")
    print("historical_incidents:", state.get("historical_incidents"))
    print("historical_guidance:", state.get("historical_guidance"))
    historical_incidents = state.get(
        "historical_incidents",
        []
    )

    historical_guidance = state.get(
        "historical_guidance",
        []
    )

    service_recovery_status = state.get(
        "service_recovery_status",
        "UNKNOWN"
    )

    service_recovery_reason = state.get(
        "service_recovery_reason",
        "Service recovery has not been evaluated."
    )
    causes_text = "\n".join(
        f"- {cause}" for cause in possible_causes
    )
    cause_analysis_parts = []

    for item in cause_analysis:
        cause = item.get("cause", "Unknown cause")
        supporting = item.get("supporting_evidence", [])
        contradicting = item.get("contradicting_evidence", [])
        strength = item.get("evidence_strength", "UNASSESSED")

    cause_analysis_parts.append(
        f"Cause: {cause}\n"
        "Supporting Evidence:\n"
        + (
            "\n".join(f"- {e}" for e in supporting)
            if supporting
            else "- None recorded."
        )
        + "\n"
        "Contradicting Evidence:\n"
        + (
            "\n".join(f"- {e}" for e in contradicting)
            if contradicting
            else "- None recorded."
        )
        + f"\nEvidence Strength: {strength}"
    )

    cause_analysis_text = "\n\n".join(cause_analysis_parts)

    if not cause_analysis_text:
        cause_analysis_text = "No cause-specific analysis available."
# Build primary supporting causes section
    primary_supporting_causes = []

    for item in cause_analysis:
        strength = item.get("evidence_strength", "UNASSESSED")

        if strength in ["HIGH", "MEDIUM", "LOW"]:
            primary_supporting_causes.append(
                f"- {item.get('cause', 'Unknown cause')} — {strength}"
            )

    primary_supporting_causes_text = "\n".join(
        primary_supporting_causes
    )

    if not primary_supporting_causes_text:
        primary_supporting_causes_text = (
            "No primary supporting causes identified."
        )
    evidence_text = "\n".join(
        f"- {evidence}"
        for evidence in additional_evidence
    )
    root_cause_evidence_text = "\n".join(
        f"- {evidence}"
        for evidence in root_cause_evidence
    )

    contradicting_evidence_text = "\n".join(
        f"- {evidence}"
        for evidence in contradicting_evidence
    )

    rag_guidance_text = "\n".join(
        f"- {guidance}"
        for guidance in rag_guidance
    )
    historical_guidance_text = "\n".join(
        f"- {guidance}"
        for guidance in historical_guidance
    )

    if not historical_guidance_text:
        historical_guidance_text = (
            "No historical incident intelligence available."
        )

    # Prevent empty sections from making the report confusing.
    if not evidence_text:
        evidence_text = "No additional evidence collected."

    if not root_cause_evidence_text:
        root_cause_evidence_text = (
            "No explicit root-cause evidence recorded."
        )

    if not contradicting_evidence_text:
        contradicting_evidence_text = (
            "No contradicting evidence recorded."
        )

    if not rag_guidance_text:
        rag_guidance_text = (
            "No relevant knowledge-base guidance retrieved."
        )

    human_review_status = state.get(
        "human_review_status"
    )

    if (
        state.get("request_type") == "FRAUD"
        and state.get("fraud_risk_level") in ["LOW", "MEDIUM"]
        and not human_review_status
    ):
        human_review_status = "AUTO_APPROVED"

    if not human_review_status:
        human_review_status = "PENDING"

    return {
        "human_review_status": human_review_status,
        "incident_report": (
            "INCIDENT REPORT\n"
            "==============================\n"
            f"Issue: {state['issue']}\n"
            f"Tower: {state.get('tower_id', 'N/A')}\n"
            f"KPI Status: {state.get('kpi_status', 'UNKNOWN')}\n"
            "\n"

            "INVESTIGATION\n"
            "------------------------------\n"
            f"{state.get('investigation', '')}\n"
            "\n"

            "ADDITIONAL EVIDENCE\n"
            "------------------------------\n"
            f"{evidence_text}\n"
            "\n"
            "RAG KNOWLEDGE-BASE GUIDANCE\n"
            "------------------------------\n"
            f"{rag_guidance_text}\n"
            "\n"

            "HISTORICAL INCIDENT INTELLIGENCE\n"
            "------------------------------\n"
            f"{historical_guidance_text}\n"
            "\n"

            "DIAGNOSIS\n"

            "SELECTED ROOT CAUSE\n"
            "------------------------------\n"
            f"{state.get('root_cause', '')}\n"
            "\n"

            "PRIMARY SUPPORTING CAUSES\n"
            "------------------------------\n"
            f"{primary_supporting_causes_text}\n"
            "\n"

            "POSSIBLE CAUSES\n"
            "------------------------------\n"
            f"{causes_text}\n"
            "\n"
            "CAUSE ANALYSIS\n"
            "------------------------------\n"
            f"{cause_analysis_text}\n"
            "\n"

            "ROOT CAUSE EVIDENCE\n"
            "------------------------------\n"
            f"{root_cause_evidence_text}\n"
            "\n"

            "CONTRADICTING EVIDENCE\n"
            "------------------------------\n"
            f"{contradicting_evidence_text}\n"
            "\n"

            "ROOT CAUSE CONFIDENCE\n"
            "------------------------------\n"
            f"{state.get('root_cause_confidence', 'UNKNOWN')}\n"
            "\n"

            "VERIFICATION\n"
            "------------------------------\n"
            f"Status: "
            f"{state.get('verification_status', 'UNKNOWN')}\n"
            f"Action: "
            f"{state.get('verification_action', '')}\n"
            f"Investigation Attempts: "
            f"{state.get('investigation_attempts', 0)}\n"
            "\n"
            "SERVICE RECOVERY VERIFICATION\n"
            "------------------------------\n"
            f"Status: {service_recovery_status}\n"
            f"Finding: {service_recovery_reason}\n"
            "\n"
            "CASE LIFECYCLE STATUS\n"
            "------------------------------\n"
            f"Current Case Status: {state.get('case_status', 'UNKNOWN')}\n"
            "\n"
            "RECOMMENDATION\n"
            "------------------------------\n"
            f"{state.get('recommendation', '')}\n"
            "\n"

            "HUMAN REVIEW\n"
            "------------------------------\n"
            f"Status: {human_review_status}\n"
            f"Engineer Feedback: "
            f"{state.get('human_review_feedback', 'No feedback provided')}\n"
            f"Approval: HUMAN ENGINEER "
            f"{human_review_status.upper()}"
        )
    }


def verify_root_cause(state: State):
    confidence = state.get("root_cause_confidence", "UNKNOWN")

    if confidence == "LOW":
        return {
            "verification_status": "INSUFFICIENT_EVIDENCE",
            "verification_action": (
                "Additional investigation required before confirming "
                "the potential root cause."
            ),
            "network_decision": (
                "INVESTIGATE"
            ),
            "network_decision_reason": (
                "Root cause confidence is LOW. "
                "Additional investigation is required before "
                "requesting human engineer approval."
            ),
        }

    if confidence == "MEDIUM":
        attempts = state.get("investigation_attempts", 0)

        if attempts < 2:
            return {
                "verification_status": "PARTIAL_CONFIDENCE",
                "verification_action": (
                    "Evidence supports the potential root cause, "
                    "but additional investigation is required "
                    "before requesting engineer validation."
                ),
                "network_decision": (
                    "INVESTIGATE"
                ),
                "network_decision_reason": (
                    "Verification status is PARTIAL_CONFIDENCE and "
                    f"investigation attempt {attempts} is below "
                    "the maximum retry limit."
                ),
            }

        return {
            "verification_status": "PARTIAL_CONFIDENCE",
            "verification_action": (
                "Evidence supports the potential root cause, "
                "but engineer validation is recommended."
            ),
            "network_decision": (
                "HUMAN_REVIEW"
            ),
            "network_decision_reason": (
                "Verification status is PARTIAL_CONFIDENCE and "
                "the maximum investigation attempts have been reached. "
                "Human network engineer validation is required."
            ),
        }

    if confidence == "HIGH":
        return {
            "verification_status": "SUPPORTED",
            "verification_action": (
                "Available KPI evidence strongly supports the "
                "potential root cause."
            ),
            "network_decision": (
                "HUMAN_REVIEW"
            ),
            "network_decision_reason": (
                "Verification status is SUPPORTED. "
                "Human network engineer validation is required "
                "before the incident recommendation is finalized."
            ),
        }

    return {
        "verification_status": "UNKNOWN",
        "verification_action": (
            "Root cause could not be verified from available evidence."
        ),
        "network_decision": (
            "HUMAN_REVIEW"
        ),
        "network_decision_reason": (
            "Verification status is UNKNOWN. "
            "Human network engineer review is required because "
            "the available evidence is inconclusive."
        ),
    }


def network_engineer_review(state: State):
    decision = interrupt({
        "message": (
            "Human network engineer review required "
            "before recommendation."
        ),
        "tower_id": state.get("tower_id", "N/A"),
        "root_cause": state.get("root_cause", ""),
        "confidence": state.get(
            "root_cause_confidence",
            "UNKNOWN"
        ),
        "verification_status": state.get(
            "verification_status",
            "UNKNOWN"
        ),
        "verification_action": state.get(
            "verification_action",
            ""
        ),
        "diagnosis": state.get(
            "diagnosis",
            ""
        ),
        "investigation_attempts": state.get(
            "investigation_attempts",
            0
        ),
        "additional_evidence": state.get(
            "additional_evidence",
            []
        ),
        "actor": state.get(
            "actor",
            "unknown"
        ),
        "actor_role": state.get(
            "actor_role",
            "Network Engineer"
        ),
    })

    if isinstance(decision, dict):
        review_status = decision.get(
            "decision",
            "rejected"
        )
        feedback = decision.get(
            "feedback",
            ""
        )
        reviewer_actor = decision.get(
            "reviewer_actor",
            state.get("actor", "unknown")
        )
        reviewer_role = decision.get(
            "reviewer_role",
            "Network Engineer"
        )
    else:
        review_status = decision
        feedback = ""
        reviewer_actor = state.get(
            "actor",
            "unknown"
        )
        reviewer_role = "Network Engineer"

    audit(
        reviewer_actor,
        reviewer_role,
        f"network_review_{review_status}",
        (
            f"Network engineer review decision: "
            f"{review_status}. "
            f"Feedback: {feedback}. "
            f"Thread ID: {state.get('thread_id', 'unknown')}. "
            f"Tower: {state.get('tower_id', 'N/A')}. "
            f"Root cause confidence: "
            f"{state.get('root_cause_confidence', 'UNKNOWN')}."
        ),
        resource_id=str(
            state.get("tower_id", "N/A")
        ),
    )

    return {
        "human_review_status": review_status,
        "human_review_feedback": feedback,
        "reviewer_actor": reviewer_actor,
        "reviewer_role": reviewer_role,
        "network_decision": (
            "APPROVED"
            if review_status == "approved"
            else "REJECTED"
        ),
    }


def human_engineer_review(state: State):
    decision = interrupt({
        "message": "Human engineer review required before recommendation.",
        "transaction_id": state.get(
            "transaction_id",
            "N/A"
        ),
        "fraud_decision": state.get(
            "fraud_decision",
            "UNKNOWN"
        ),
        "fraud_risk_level": state.get(
            "fraud_risk_level",
            "UNKNOWN"
        ),
        "fraud_probability": state.get(
            "fraud_probability",
            0.0
        ),
        "fraud_reasons": state.get(
            "fraud_reasons",
            []
        ),
        "tower_id": state.get("tower_id", "N/A"),
        "root_cause": state.get("root_cause", ""),
        "confidence": state.get(
            "root_cause_confidence",
            "UNKNOWN"
        ),
        "verification_status": state.get(
            "verification_status",
            "UNKNOWN"
        ),
        "verification_action": state.get(
            "verification_action",
            ""
        ),
    })

    audit(
        state.get("actor", "unknown"),
        state.get("actor_role", "unknown"),
        f"fraud_review_{decision}",
        (
            f"Human review decision: {decision}. "
            f"Transaction: "
            f"{state.get('transaction_id', 'N/A')}. "
            f"Risk level: "
            f"{state.get('fraud_risk_level', 'UNKNOWN')}. "
            f"Fraud probability: "
            f"{state.get('fraud_probability', 0.0):.4f}."
        ),
        resource_id=str(
            state.get("transaction_id", "N/A")
        ),
    )
    return {
        "human_review_status": decision,
        "investigation": state.get("investigation", ""),
        "fraud_probability": state.get(
            "fraud_probability",
            0.0
        ),
        "fraud_risk_level": state.get(
            "fraud_risk_level",
            "UNKNOWN"
        ),
        "fraud_detected": state.get(
            "fraud_detected",
            False
        ),
        "fraud_reasons": state.get(
            "fraud_reasons",
            []
        ),
        "fraud_decision": state.get(
            "fraud_decision",
            "UNKNOWN"
        ),
    }


def create_fraud_incident(state: State):
    if state.get("request_type") != "FRAUD":
        return {}

    if state.get("human_review_status") != "approved":
        return {}

    transaction_id = state.get("transaction_id", "UNKNOWN")
    risk_level = state.get("fraud_risk_level", "UNKNOWN")
    probability = state.get("fraud_probability", 0.0)
    reasons = state.get("fraud_reasons", [])

    incident_id = save_incident({
        "incident_type": "FRAUD",
        "case_status": "OPEN",
        "priority": risk_level,
        "assigned_to": "NOC Engineer",
        "issue": (
            f"Suspicious billing transaction {transaction_id}"
        ),
        "tower_id": None,
        "diagnosis": (
            f"Fraud Investigation Agent classified the transaction "
            f"as {risk_level} risk with "
            f"{probability:.4f} fraud probability."
        ),
        "recommendation": (
            "Escalate the transaction for fraud investigation."
        ),
        "complaint_analysis": {
            "severity": risk_level
        },
        "root_cause_confidence": "HIGH",
    })
    audit(
        state.get("actor", "agentic-fraud-agent"),
        state.get("actor_role", "Fraud Analyst"),
        "fraud_incident_created",
        (
            f"Fraud incident created after human approval. "
            f"Risk level: {risk_level}. "
            f"Fraud probability: {probability:.4f}. "
            f"Transaction: {transaction_id}."
        ),
        resource_id=str(incident_id),
    )
    return {
        "incident_report": (
            f"Fraud incident {incident_id} created for "
            f"transaction {transaction_id}.\n"
            f"Risk Level: {risk_level}\n"
            f"Fraud Probability: {probability:.4f}\n"
            "Risk Indicators:\n"
            + "\n".join(
                f"- {reason}" for reason in reasons
            )
        )
    }


def create_network_incident(state: State):
    if state.get("request_type") != "NETWORK":
        return {}

    if state.get("human_review_status") != "approved":
        return {}

    incident_id = save_incident({
        "incident_type": "NETWORK",
        "thread_id": state.get(
            "thread_id",
            ""
        ),
        "case_status": "OPEN",
        "priority": state.get(
            "root_cause_confidence",
            "MEDIUM"
        ),
        "assigned_to": "NOC Engineer",

        "issue": state.get("issue", ""),
        "tower_id": state.get("tower_id", ""),
        "diagnosis": state.get("diagnosis", ""),
        "recommendation": state.get(
            "recommendation",
            ""
        ),
        "severity": state.get(
            "complaint_analysis",
            {}
        ).get("severity", "MEDIUM"),
        "root_cause_confidence": state.get(
            "root_cause_confidence",
            "UNKNOWN"
        ),
    })
    save_investigation_metric(
        thread_id=state.get("thread_id", ""),
        incident_id=incident_id,
        tower_id=state.get("tower_id"),
        status="INCIDENT_CREATED",
        outcome="HUMAN_APPROVED",
        execution_time_ms=state.get(
            "execution_time_ms"
        ),
        human_review="APPROVED",
    )

    audit(
        state.get("actor", "unknown"),
        state.get("actor_role", "unknown"),
        "network_incident_created",
        (
            f"Network incident created for tower "
            f"{state.get('tower_id', 'N/A')}."
        ),
        resource_id=str(incident_id),
    )

    return {"incident_id": incident_id
            }


def move_incident_to_in_progress(state: State):
    incident_id = state.get("incident_id")

    if not incident_id:
        return {}

    success = update_case_status(
        incident_id,
        "IN_PROGRESS",
        actor=state.get("actor", "unknown"),
        role=state.get("actor_role", "unknown"),
    )
    print("========== STATE BEFORE RECOVERY ==========")
    print(
        "ADDITIONAL EVIDENCE:",
        state.get("additional_evidence", [])
    )
    print(
        "RECOVERY BASELINE:",
        state.get("recovery_baseline")
    )
    return {
        "case_status": "IN_PROGRESS"
    } if success else {}


def evaluate_incident_resolution(state: State):
    incident_id = state.get("incident_id")

    if not incident_id:
        return {}

    current_status = state.get(
        "case_status",
        "IN_PROGRESS"
    )

    service_recovery_status = state.get(
        "service_recovery_status",
        "UNKNOWN"
    )

    if service_recovery_status == "RECOVERED":
        new_status = "RESOLVED"
    else:
        new_status = "IN_PROGRESS"

    if new_status != current_status:
        success = update_case_status(
            incident_id,
            new_status,
            actor=state.get("actor", "unknown"),
            role=state.get("actor_role", "unknown"),
        )

        if not success:
            return {}

    save_investigation_metric(
        thread_id=state.get("thread_id", ""),
        incident_id=incident_id,
        tower_id=state.get("tower_id"),
        status=new_status,
        outcome=(
            "RESOLVED"
            if new_status == "RESOLVED"
            else "NOT_RECOVERED"
        ),
        human_review=(
            "APPROVED"
            if state.get("human_review_status") == "approved"
            else "NOT_REQUIRED"
        ),
    )

    return {
        "case_status": new_status
    }


def route_after_human_review(state: State):
    status = state.get(
        "human_review_status",
        "rejected"
    )

    if status == "approved":
        return "recommend"

    return "rejected"


def route_after_network_review(state: State):
    status = state.get(
        "human_review_status",
        "rejected"
    )

    if status == "approved":
        return "recommend"

    return "rejected"


def rejected_report(state: State):
    possible_causes = state.get("possible_causes", [])
    additional_evidence = state.get(
        "additional_evidence",
        []
    )

    causes_text = "\n".join(
        f"- {cause}"
        for cause in possible_causes
    )

    evidence_text = "\n".join(
        f"- {evidence}"
        for evidence in additional_evidence
    )

    return {
        "incident_report": (
            "INCIDENT REPORT\n"
            f"Issue: {state.get('issue', '')}\n"
            f"Tower: {state.get('tower_id', 'N/A')}\n"
            f"KPI Status: "
            f"{state.get('kpi_status', 'UNKNOWN')}\n"
            "\n"
            f"Investigation:\n"
            f"{state.get('investigation', '')}\n"
            "\n"
            f"Additional Evidence:\n"
            f"{evidence_text}\n"
            "\n"
            f"Diagnosis:\n"
            f"{state.get('diagnosis', '')}\n"
            "\n"
            f"Root Cause: "
            f"{state.get('root_cause', '')}\n"
            f"Possible Causes:\n"
            f"{causes_text}\n"
            f"Root Cause Confidence: "
            f"{state.get('root_cause_confidence', 'UNKNOWN')}\n"
            "\n"
            f"Verification Status: "
            f"{state.get('verification_status', 'UNKNOWN')}\n"
            f"Verification Action: "
            f"{state.get('verification_action', '')}\n"
            "\n"
            f"Human Review Status: "
            f"{state.get('human_review_status', 'UNKNOWN')}\n"
            "\n"
            "Recommendation: NOT GENERATED\n"
            "\n"
            "Approval: HUMAN ENGINEER REJECTED\n"
            "\n"
            "The proposed root cause was rejected by the "
            "human engineer. The technical evidence has "
            "been preserved for further investigation."
        )
    }


def route_request(state):
    existing_request_type = state.get("request_type")

    if existing_request_type in (
        "NETWORK",
        "FRAUD",
        "CUSTOMER",
    ):
        return {
            "request_type": existing_request_type,
            "selected_agent": (
                "Network Investigation Agent"
                if existing_request_type == "NETWORK"
                else (
                    "Fraud Investigation Agent"
                    if existing_request_type == "FRAUD"
                    else "Customer Intelligence Agent"
                )
            ),
            "routing_reason": (
                "The request type was already determined by "
                "the upstream orchestration workflow."
            )
        }

    issue = state.get("issue", "").lower()

    if any(
        keyword in issue
        for keyword in [
            "fraud",
            "billing",
            "transaction",
            "payment"
        ]
    ):
        return {
            "request_type": "FRAUD",
            "selected_agent": "Fraud Investigation Agent",
            "routing_reason": (
                "The request contains billing or transaction-related indicators."
            )
        }
    # Knowledge-related requests take priority when
    # the user is explicitly asking for documentation,
    # procedures, policies, or instructions.
    if any(
        keyword in issue
        for keyword in [
            "how do i",
            "how can i",
            "what is",
            "procedure",
            "policy",
            "manual",
            "guide",
            "sop",
        ]
    ):
        return {
            "request_type": "GENERAL",
            "selected_agent": "Knowledge Retrieval Agent",
            "routing_reason": (
                "The request is asking for telecom knowledge, "
                "procedure, policy, or documentation."
            ),
        }

    if any(
        keyword in issue
        for keyword in [
            "customer",
            "complaint",
            "subscription",
            "churn"
        ]
    ):
        return {
            "request_type": "CUSTOMER",
            "selected_agent": "Customer Intelligence Agent",
            "routing_reason": (
                "The request contains customer or service-related indicators."
            )
        }

    if any(
        keyword in issue
        for keyword in [
            "network",
            "tower",
            "latency",
            "packet loss",
            "signal",
            "outage",
            "connection"
            "internet",
            "slow internet",
            "call drop",
            "calls keep dropping",
            "throughput",
            "4g",
            "5g",
            "coverage"
        ]
    ):
        return {
            "request_type": "NETWORK",
            "selected_agent": "Network Investigation Agent",
            "routing_reason": (
                "The request contains network or connectivity-related indicators."
            )
        }

    return {
        "request_type": "GENERAL",
        "selected_agent": "Knowledge Retrieval Agent",
        "routing_reason": (
            "No specialized domain was identified, so the knowledge agent "
            "will handle the request."
        )
    }


def supervisor_agent(state: State):
    """
    Supervisor Agent responsible for deciding which
    specialist agent should handle the request.
    """

    routing = route_request(state)

    return {
        "request_type": routing.get(
            "request_type",
            "GENERAL"
        ),
        "selected_agent": routing.get(
            "selected_agent",
            "Knowledge Retrieval Agent"
        ),
        "routing_reason": routing.get(
            "routing_reason",
            ""
        ),
    }


def route_to_agent(state: State):
    request_type = state.get(
        "request_type",
        "GENERAL"
    )

    if request_type == "NETWORK":
        return "network"

    if request_type == "FRAUD":
        return "fraud"

    if request_type == "CUSTOMER":
        return "customer"

    return "general"


def fraud_agent(state: State):
    transaction = state.get("transaction", {})
    transaction_id = state.get("transaction_id")

    if not transaction and transaction_id:
        transaction = get_transaction(transaction_id)

    if not transaction:
        return {
            "investigation": (
                "Fraud Investigation Agent could not evaluate "
                "the request because transaction data was not provided."
            )
        }

    fraud_result = predict_fraud(transaction)

    return {
        "transaction_id": transaction.get(
            "transaction_id",
            transaction_id
        ),
        "fraud_probability": fraud_result["fraud_probability"],
        "fraud_risk_level": fraud_result["risk_level"],
        "fraud_detected": fraud_result["is_fraud"],
        "fraud_reasons": fraud_result["reasons"],
        "fraud_decision": (
            "HUMAN_REVIEW_REQUIRED"
            if fraud_result["risk_level"] in ["HIGH", "CRITICAL"]
            else "AUTO_APPROVED"
        ),
        "investigation": (
            "Fraud Investigation Agent analyzed the transaction.\n"
            f"Transaction ID: {transaction.get('transaction_id', transaction_id)}\n"
            f"Fraud Probability: "
            f"{fraud_result['fraud_probability']:.4f}\n"
            f"Risk Level: {fraud_result['risk_level']}\n"
            f"Fraud Detected: {fraud_result['is_fraud']}\n"
            "Risk Indicators:\n"
            + "\n".join(
                f"- {reason}"
                for reason in fraud_result["reasons"]
            )
        )
    }


def route_after_fraud(state: State):
    risk_level = state.get("fraud_risk_level", "LOW")

    if risk_level in ["HIGH", "CRITICAL"]:
        state["fraud_decision"] = "HUMAN_REVIEW_REQUIRED"
        return "human_review"

    state["fraud_decision"] = "AUTO_APPROVED"
    return "report"


def customer_agent(state: State):
    return {
        "investigation": (
            "Customer Intelligence Agent activated. "
            "The request will be evaluated using customer "
            "profile, usage, complaint, and service evidence."
        )
    }


def general_agent(state: State):
    return {
        "investigation": (
            "Knowledge Retrieval Agent activated. "
            "The request will be handled using the "
            "telecom knowledge base."
        )
    }


def network_tool_decision(state: State):
    """
    Decide whether additional equipment investigation is needed
    based on the current investigation state.
    """

    # Stop the equipment branch from repeating.
    equipment_completed = state.get(
        "equipment_investigation_completed",
        False
    )

    if equipment_completed:
        return {
            "tool_decision": "skip",
            "tool_decision_reason": (
                "Equipment investigation has already been completed, "
                "so the workflow should continue to root-cause verification."
            ),
        }

    kpi_status = state.get("kpi_status", "UNKNOWN")
    possible_causes = state.get("possible_causes", [])
    equipment_id = state.get("equipment_id")

    equipment_related = any(
        any(
            keyword in str(cause).lower()
            for keyword in [
                "equipment",
                "cpu",
                "memory",
                "processing overload",
            ]
        )
        for cause in possible_causes
    )

    if (
        kpi_status == "ABNORMAL"
        and equipment_related
        and equipment_id
    ):
        return {
            "tool_decision": "equipment_health",
            "tool_decision_reason": (
                f"An equipment-related hypothesis was identified and "
                f"equipment {equipment_id} is available for direct health "
                f"inspection."
            ),
        }

    if kpi_status == "ABNORMAL":
        return {
            "tool_decision": "equipment_candidates",
            "tool_decision_reason": (
                "Network degradation is present and additional equipment "
                "evidence may help validate the current hypotheses."
            ),
        }

    return {
        "tool_decision": "skip",
        "tool_decision_reason": (
            "The current network state does not require additional "
            "equipment investigation."
        ),
    }


def equipment_investigation(state: State):
    """
    Execute the equipment investigation selected by the
    Network Tool Decision node while preserving existing evidence.
    """

    existing_evidence = state.get(
        "additional_evidence",
        []
    )

    tool_decision = state.get(
        "tool_decision",
        "equipment_candidates"
    )

    if tool_decision == "equipment_health":
        equipment_id = state.get("equipment_id")

        if not equipment_id:
            equipment_text = (
                "Direct equipment health inspection was selected, "
                "but no equipment identifier is available."
            )

            return {
                "additional_evidence": (
                    existing_evidence + [equipment_text]
                ),
                "investigation": equipment_text,
            }

        equipment_result = get_equipment_health.invoke({
            "equipment_id": equipment_id
        })

        if not equipment_result.get("found"):
            equipment_text = equipment_result.get(
                "message",
                f"No equipment record was found for {equipment_id}."
            )

            return {
                "additional_evidence": (
                    existing_evidence + [equipment_text]
                ),
                "investigation": equipment_text,
                "equipment_investigation_completed": True,
            }

        health = equipment_result.get("health", {})

        equipment_text = (
            f"Direct equipment health inspection for {equipment_id}:\n"
            f"Temperature: "
            f"{float(health.get('temperature', 0)):.1f}\n"
            f"Uptime: "
            f"{float(health.get('uptime', 0)):.1f}\n"
            f"CPU usage: "
            f"{float(health.get('cpu_usage', 0)):.1f}%\n"
            f"Memory usage: "
            f"{float(health.get('memory_usage', 0)):.1f}%\n"
            f"Error count: {health.get('error_count', 0)}\n"
            f"Maintenance count: "
            f"{health.get('maintenance_count', 0)}\n"
            f"Failure flag: {health.get('failure', 0)}"
        )

        return {
            "additional_evidence": (
                existing_evidence + [equipment_text]
            ),
            "investigation": equipment_text,
            "equipment_investigation_completed": True,
        }

    if tool_decision == "equipment_candidates":
        tower_id = state.get("tower_id", "")

        equipment_result = find_equipment_for_investigation.invoke({
            "limit": 5,
            "tower_id": tower_id
        })

        if not equipment_result.get("found"):
            equipment_text = (
                "No supplementary equipment candidates were found."
            )

            return {
                "additional_evidence": (
                    existing_evidence + [equipment_text]
                ),
                "investigation": equipment_text,
                "equipment_investigation_completed": True,
            }

        equipment_records = equipment_result.get(
            "equipment",
            []
        )

        equipment_evidence = [
            (
                f"{item['equipment_id']}: "
                f"errors={item['error_count']}, "
                f"CPU={item['cpu_usage']:.1f}%, "
                f"memory={item['memory_usage']:.1f}%, "
                f"temperature={item['temperature']:.1f}, "
                f"failure={item['failure']}"
            )
            for item in equipment_records
        ]

        equipment_text = (
            f"Equipment health candidates for tower {tower_id}:\n"
            + "\n".join(equipment_evidence)
        )

        return {
            "additional_evidence": (
                existing_evidence + [equipment_text]
            ),
            "investigation": equipment_text,
            "equipment_investigation_completed": True,
        }

    equipment_text = (
        "No equipment investigation was selected."
    )

    return {
        "additional_evidence": (
            existing_evidence + [equipment_text]
        ),
        "investigation": equipment_text,
    }


def route_after_network_tool_decision(state: State):
    return state.get("tool_decision", "skip")

_CHECKPOINT_DB_PATH = (
    r"D:\TelecomAI_Enterprise_Voice_Customer_Employee\telecomai.db"
)

_checkpointer = SqliteSaver(
    sqlite3.connect(
        _CHECKPOINT_DB_PATH,
        check_same_thread=False
    )
)
def build_agent():
    graph = StateGraph(State)
    graph.add_node(
        "supervisor",
        supervisor_agent
    )
    graph.add_node("fraud", fraud_agent)
    graph.add_node(
        "create_fraud_incident",
        create_fraud_incident
    )
    graph.add_node("customer", customer_agent)
    graph.add_node("general", general_agent)
    graph.add_node("analyze", analyze_complaint)
    graph.add_node("check_kpis", check_kpis)
    graph.add_node("network_tool_decision", network_tool_decision)
    graph.add_node("equipment_investigation", equipment_investigation)
    graph.add_node("investigate", investigate)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("diagnose", diagnose_from_evidence)
    graph.add_node("root_cause_analysis", root_cause_analysis)
    graph.add_node("verify_root_cause", verify_root_cause)
    graph.add_node("network_engineer_review", network_engineer_review)
    graph.add_node("create_network_incident", create_network_incident)
    graph.add_node(
        "move_incident_to_in_progress",
        move_incident_to_in_progress
    )
    graph.add_node(
        "evaluate_incident_resolution",
        evaluate_incident_resolution
    )
    graph.add_node(
        "verify_service_recovery",
        verify_service_recovery
    )
    graph.add_node("human_engineer_review", human_engineer_review)
    graph.add_node("rejected_report", rejected_report)
    graph.add_node("recommend", recommend)
    graph.add_node("explain", explain)
    graph.add_node("report", report)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route_to_agent,
                                {
                                    "network": "analyze",
                                    "fraud": "fraud",
                                    "customer": "customer",
                                    "general": "general",
                                }
                                )
    graph.add_edge("analyze", "check_kpis")
    graph.add_conditional_edges(
        "check_kpis",
        route_after_kpi_check,
        {
            "investigate": "investigate",
            "retrieve": "retrieve",
        }
    )
    graph.add_edge("investigate", "retrieve")
    graph.add_edge("equipment_investigation", "retrieve")
    graph.add_edge("retrieve", "diagnose")

    graph.add_edge("diagnose", "root_cause_analysis")

    graph.add_edge(
        "root_cause_analysis",
        "network_tool_decision"
    )

    graph.add_conditional_edges(
        "network_tool_decision",
        route_after_network_tool_decision,
        {
            "equipment_candidates": "equipment_investigation",
            "equipment_health": "equipment_investigation",
            "skip": "verify_root_cause",
        }
    )

    graph.add_conditional_edges(
        "verify_root_cause",
        route_after_verification,
        {
            "investigate": "investigate",
            "recommend": "network_engineer_review",
        }
    )
    graph.add_conditional_edges(
        "network_engineer_review",
        route_after_network_review,
        {
            "recommend":  "create_network_incident",
            "rejected": "rejected_report",
        }
    )
    graph.add_edge("create_network_incident", "move_incident_to_in_progress")
    graph.add_edge("move_incident_to_in_progress", "verify_service_recovery")
    graph.add_edge("verify_service_recovery", "evaluate_incident_resolution")
    graph.add_edge("evaluate_incident_resolution", "recommend")
    graph.add_conditional_edges(
        "human_engineer_review",
        route_after_human_review,
        {
            "recommend":  "create_fraud_incident",
            "rejected": "rejected_report",
        }
    )
    graph.add_edge("create_fraud_incident", "recommend")
    graph.add_edge("rejected_report", END)

    graph.add_edge("recommend", "explain")

    graph.add_edge("explain", "report")
    graph.add_conditional_edges(
        "fraud",
        route_after_fraud,
        {
            "human_review": "human_engineer_review",
            "report": "report",
        }
    )

    graph.add_edge("report", END)

    checkpointer = SqliteSaver(
        sqlite3.connect(
            r"D:\TelecomAI_Enterprise_Voice_Customer_Employee\telecomai.db",
            check_same_thread=False
        )
    )

    return graph.compile(
        checkpointer=checkpointer
    )


def list_investigation_threads(limit=50):
    """
    Return saved LangGraph investigation checkpoints.
    """

    agent = build_agent()

    checkpoints = agent.checkpointer.list(
        None,
        limit=limit
    )

    threads = {}

    for checkpoint in checkpoints:
        config = checkpoint.config or {}
        configurable = config.get(
            "configurable",
            {}
        )

        thread_id = configurable.get(
            "thread_id"
        )

        if thread_id:
            threads[thread_id] = checkpoint

    return threads


def get_investigation_history(limit: int = 50):
    """
    Return metadata for persisted investigation threads.
    """
    agent = build_agent()
    threads = list_investigation_threads(limit=limit)

    history = []

    for thread_id in threads:
        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        snapshot = agent.get_state(config)
        state = snapshot.values or {}

        history.append(
            {
                "thread_id": thread_id,
                "source_channel": state.get("source_channel"),
                "audience": state.get("audience"),
                "network_decision": state.get("network_decision"),
                "human_review_status": state.get("human_review_status"),
                "incident_id": state.get("incident_id"),
                "case_status": state.get("case_status"),
                "pending": bool(snapshot.next),
                "next_nodes": list(snapshot.next or []),
            }
        )

    return history


def get_investigation_for_incident(incident_id: int):
    """
    Find the persisted investigation associated with an incident.
    """

    history = get_investigation_history(limit=200)

    for item in history:
        if item.get("incident_id") == incident_id:
            return item

    return None


def get_investigation_summary(thread_id: str):
    """
    Return the operational context of a persisted investigation.
    """

    data = get_investigation_state(thread_id)
    state = data["state"]

    return {
        "thread_id": thread_id,
        "tower_id": state.get("tower_id"),
        "issue": state.get("issue"),
        "source_channel": state.get("source_channel"),
        "audience": state.get("audience"),
        "kpi_status": state.get("kpi_status"),
        "diagnosis": state.get("diagnosis"),
        "root_cause": state.get("root_cause"),
        "root_cause_confidence": state.get(
            "root_cause_confidence"
        ),
        "verification_status": state.get(
            "verification_status"
        ),
        "human_review_status": state.get(
            "human_review_status"
        ),
        "reviewer_actor": state.get(
            "reviewer_actor"
        ),
        "reviewer_role": state.get(
            "reviewer_role"
        ),
        "network_decision": state.get(
            "network_decision"
        ),
        "incident_id": state.get("incident_id"),
        "case_status": state.get("case_status"),
        "pending": data["pending"],
        "next_nodes": data["next"],
    }


def get_resumable_investigations(limit: int = 50):
    """
    Return investigations that are currently paused and can be resumed.
    """
    history = get_investigation_history(limit=limit)

    return [
        item
        for item in history
        if item["pending"] is True
    ]


def continue_investigation(
    thread_id: str,
    decision: str,
    feedback: str = "",
    reviewer_actor: str = "unknown",
    reviewer_role: str = "Network Engineer",
):
    """
    Resume a paused investigation from its persisted LangGraph checkpoint.
    """

    if decision not in ("approved", "rejected"):
        raise ValueError(
            "Decision must be either 'approved' or 'rejected'."
        )

    agent = build_agent()

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    snapshot = agent.get_state(config)

    if not snapshot.values:
        raise ValueError(
            f"No persisted investigation found for thread: {thread_id}"
        )

    if not snapshot.next:
        raise ValueError(
            f"Investigation '{thread_id}' is not currently pending."
        )

    from langgraph.types import Command
    import time
    investigation_start = time.perf_counter()
    result = agent.invoke(
        Command(
            resume={
                "decision": decision,
                "feedback": feedback,
                "reviewer_actor": reviewer_actor,
                "reviewer_role": reviewer_role,
            }
        ),
        config=config,
    )
    investigation_elapsed_ms = int(
        (time.perf_counter() - investigation_start) * 1000
    )
    save_investigation_metric(
        thread_id=config["configurable"]["thread_id"],
        incident_id=None,
        tower_id=tower,
        status="STARTED",
        outcome="PAUSED_FOR_REVIEW" if getattr(
            result, "next", None) else "COMPLETED",
        execution_time_ms=investigation_elapsed_ms,
        human_review="REQUIRED" if getattr(
            result, "next", None) else "NOT_REQUIRED",
    )
    final_snapshot = agent.get_state(config)
    final_state = final_snapshot.values or {}

    return {
        "thread_id": thread_id,
        "network_decision": final_state.get("network_decision"),
        "human_review_status": final_state.get(
            "human_review_status"
        ),
        "incident_id": final_state.get("incident_id"),
        "case_status": final_state.get("case_status"),
        "pending": bool(final_snapshot.next),
        "next_nodes": list(final_snapshot.next or []),
        "state": final_state,
        "result": result,
    }


def get_investigation_state(thread_id: str):
    """
    Load the latest persisted state for an investigation thread.
    """
    agent = build_agent()

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    snapshot = agent.get_state(config)

    return {
        "thread_id": thread_id,
        "state": snapshot.values or {},
        "next": list(snapshot.next or []),
        "pending": bool(snapshot.next),
    }
