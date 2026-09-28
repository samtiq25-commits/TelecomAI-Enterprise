import re
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from .complaint_analyzer import classify_complaint
from .rag import retrieve
from .agent import build_agent
from .fraud_detection import predict_fraud
from .llm import extract_transaction_details
from src.database import (
    audit,
    save_incident,
    save_investigation_metric,
)
import uuid
import time


class VoiceState(TypedDict, total=False):
    text: str
    audience: str
    tower_id: str
    anomaly: dict
    thread_id: str

    intent: str
    evidence: str

    diagnosis: str
    recommendation: str
    network_decision: str
    human_review_status: str
    kpi_status: str
    root_cause: str
    possible_causes: list
    root_cause_confidence: str
    verification_status: str
    response: str
    transaction: dict
    transaction_missing_fields: list
    transaction_extracted: dict
    fraud_review_status: str
    fraud_review_feedback: str
    fraud_incident_id: int
    fraud_probability: float
    fraud_risk_level: str
    fraud_detected: bool
    fraud_reasons: list
    fraud_decision: str


def voice_route(state: VoiceState):
    intent = route_voice_intent(
        state.get("text", ""),
        state.get("audience", "customer")
    )

    return {
        "intent": intent
    }


def route_voice_graph(state: VoiceState):
    intent = state.get(
        "intent",
        "CUSTOMER_SUPPORT"
    )

    if intent == "NETWORK":
        return "network"

    if intent == "BILLING_FRAUD":
        return "fraud"

    if intent == "KNOWLEDGE":
        return "knowledge"

    return "customer"


def voice_knowledge_node(state: VoiceState):
    text = state.get("text", "")

    docs = retrieve(text)

    evidence = "\n".join(
        d["content"]
        for d in docs[:2]
    )

    if evidence:
        response = (
            "I found relevant telecom guidance for your request. "
            f"{evidence[:1000]}"
        )
    else:
        response = (
            "I could not find relevant telecom documentation "
            "for that request."
        )

    return {
        "evidence": evidence,
        "response": response,
    }


def voice_network_node(state: VoiceState):
    text = state.get("text", "")
    tower_id = state.get("tower_id", "TWR-1024")
    anomaly = state.get("anomaly", {})
    thread_id = state.get("thread_id", "")

    config = {
        "configurable": {
            "thread_id": thread_id

        }
    }

    agent = build_agent()

    investigation_start = time.perf_counter()

    result = agent.invoke(
        {
            "request_type": "NETWORK",
            "issue": text,
            "tower_id": tower_id,
            "thread_id": thread_id,
            "anomaly": anomaly,
            "source_channel": "VOICE",
            "audience": state.get("audience", "customer"),
        },
        config=config,
    )

    investigation_elapsed_ms = int(
        (time.perf_counter() - investigation_start) * 1000
    )

    snapshot = agent.get_state(config)

    pending_review = bool(snapshot.next)

    save_investigation_metric(
        thread_id=thread_id,
        incident_id=None,
        tower_id=tower_id,
        status="STARTED",
        outcome="PAUSED_FOR_REVIEW" if pending_review else "COMPLETED",
        execution_time_ms=investigation_elapsed_ms,
        human_review="REQUIRED" if pending_review else "NOT_REQUIRED",
    )

    agent_state = snapshot.values or result

    network_decision = agent_state.get(
        "network_decision",
        "",
    )

    human_review_status = agent_state.get(
        "human_review_status",
        "",
    )

    if pending_review:
        network_decision = "HUMAN_REVIEW"

    return {
        "intent": "NETWORK",

        "diagnosis": agent_state.get(
            "diagnosis",
            "",
        ),

        "recommendation": agent_state.get(
            "recommendation",
            "",
        ),

        "kpi_status": agent_state.get(
            "kpi_status",
            "",
        ),

        "root_cause": agent_state.get(
            "root_cause",
            "",
        ),

        "possible_causes": agent_state.get(
            "possible_causes",
            [],
        ),

        "root_cause_confidence": agent_state.get(
            "root_cause_confidence",
            "",
        ),

        "verification_status": agent_state.get(
            "verification_status",
            "",
        ),

        "network_decision": network_decision,

        "human_review_status": human_review_status,

        "evidence": agent_state.get(
            "investigation",
            "",
        ),

        "response": agent_state.get(
            "incident_report",
            "",
        ),

        "thread_id": thread_id,
    }


def voice_customer_node(state: VoiceState):
    """
    Handle customer-support requests through the voice orchestrator.
    """

    text = state.get("text", "")

    analysis = classify_complaint(text)

    category = analysis.get(
        "category",
        "General Support Request"
    )

    severity = analysis.get(
        "severity",
        "MEDIUM"
    )

    return {
        "intent": "CUSTOMER_SUPPORT",
        "diagnosis": (
            f"Customer complaint classified as "
            f"{category} with {severity} severity."
        ),
        "recommendation": (
            "Provide customer support based on the complaint "
            "category and severity."
        ),
        "evidence": "",
    }


def voice_fraud_node(state: VoiceState):
    """
    Handle billing and fraud-related voice requests.

    Extracts transaction information available in the request,
    identifies missing model inputs, and only runs the fraud
    model when all required fields are available.
    """

    text = state.get("text", "")

    transaction = state.get("transaction", {})

    if not transaction:
        transaction = extract_transaction_details(text)

    required_fields = [
        "amount",
        "previous_transactions",
        "transaction_velocity",
        "is_new_device",
        "hour",
        "country",
        "payment_type",
        "device_type",
    ]

    missing_fields = [
        field for field in required_fields
        if field not in transaction
    ]

    if missing_fields:
        return {
            "intent": "BILLING_FRAUD",
            "transaction": transaction,
            "transaction_extracted": transaction,
            "transaction_missing_fields": missing_fields,
            "diagnosis": (
                "A billing or fraud-related request was identified. "
                f"Transaction information received: {transaction or 'none'}. "
                f"Missing required fields: {', '.join(missing_fields)}."
            ),
            "recommendation": (
                "Request the missing transaction details before "
                "performing fraud assessment."
            ),
            "evidence": text,
            "fraud_decision": "DETAILS_REQUIRED",
        }

    fraud_result = predict_fraud(transaction)

    probability = fraud_result["fraud_probability"]
    risk_level = fraud_result["risk_level"]
    is_fraud = fraud_result["is_fraud"]
    reasons = fraud_result["reasons"]

    return {
        "intent": "BILLING_FRAUD",
        "transaction_extracted": transaction,
        "transaction_missing_fields": [],
        "fraud_probability": probability,
        "fraud_risk_level": risk_level,
        "fraud_detected": is_fraud,
        "fraud_reasons": reasons,
        "diagnosis": (
            f"Transaction fraud assessment completed with "
            f"{risk_level} risk."
        ),
        "recommendation": (
            "Review the transaction according to the detected "
            "fraud risk level."
        ),
        "evidence": ", ".join(reasons),
        "fraud_decision": "ASSESSMENT_COMPLETE",
    }


def build_voice_orchestrator():
    graph = StateGraph(VoiceState)

    graph.add_node(
        "route",
        voice_route
    )

    graph.add_node(
        "knowledge",
        voice_knowledge_node
    )

    graph.add_node(
        "network",
        voice_network_node
    )

    graph.add_node(
        "customer",
        voice_customer_node
    )
    graph.add_node(
        "fraud",
        voice_fraud_node
    )

    graph.add_edge(
        START,
        "route"
    )

    graph.add_conditional_edges(
        "route",
        route_voice_graph,
        {
            "network": "network",
            "knowledge": "knowledge",
            "customer": "customer",
            "fraud": "fraud",
        }

    )

    graph.add_edge(
        "network",
        END
    )

    graph.add_edge(
        "knowledge",
        END
    )

    graph.add_edge(
        "customer",
        END
    )
    graph.add_edge(
        "fraud",
        END
    )
    return graph.compile()


def _customer_response(analysis, diagnosis, recommendation):
    if analysis["severity"] == "CRITICAL":
        return (
            f"I’m sorry you’re experiencing this {analysis['category'].lower()} issue. "
            "We’ve marked it as critical and recommend immediate escalation to our network team. "
            "Please keep your phone available in case support needs additional information."
        )
    if analysis["severity"] == "HIGH":
        return (
            f"I’m sorry for the trouble with your {analysis['category'].lower()} issue. "
            "We’ve identified this as a high-priority case and it should be escalated to our network support team. "
            "Our team will investigate the affected service and work toward restoring it."
        )
    return (
        f"Thank you for reporting the issue. I’ve classified it as {analysis['category'].lower()}. "
        "I’ve captured the details for our support workflow. "
        "The next step is to run the relevant checks and provide you with an update."
    )


def _employee_response(analysis, diagnosis, recommendation, evidence, network_decision):
    if network_decision == "HUMAN_REVIEW":
        return (
            f"Employee support assessment: "
            f"{analysis['category']} with "
            f"{analysis['severity']} severity. "
            f"Diagnosis: {diagnosis}. "
            "The investigation requires human network engineer "
            "review before a final recommendation is issued."
        )
    return (
        f"Employee support assessment: {analysis['category']} with {analysis['severity']} severity. "
        f"Diagnosis: {diagnosis}. "
        f"Recommended action: {recommendation}. "
        f"Relevant operational guidance: {evidence[:600]} "
        "Human approval is required before any production network change."
    )


def route_voice_intent(text, audience="customer"):
    """
    Determine which TelecomAI capability should handle
    the voice request.
    """

    if not text or not text.strip():
        return "CUSTOMER_SUPPORT"

    text_lower = text.lower().strip()
    text_lower = re.sub(r"[^\w\s]", " ", text_lower)
    text_lower = re.sub(r"\s+", " ", text_lower)

    audience = audience.lower()
    # Account and package knowledge requests
    account_knowledge_keywords = [
        "internet packages",
        "data packages",
        "available packages",
        "package details",
        "package price",
        "package validity",
        "change my package",
        "change package",
        "activate package",
        "check my balance",
        "check balance",
        "account balance",
        "remaining balance",
        "data balance",
        "remaining data",
    ]

    if any(
        keyword in text_lower
        for keyword in account_knowledge_keywords
    ):
        return "KNOWLEDGE"
    network_keywords = [
        "network",
        "signal",
        "tower",
        "latency",
        "packet loss",
        "internet",
        "slow internet",
        "slow connection",
        "connection problem",
        "no connection",
        "outage",
        "4g",
        "5g",
        "call drop",
        "call drops",
        "calls keep dropping",
        "dropped calls",
        "coverage",
        "throughput",
        "weak signal",
        "poor signal",
        "network issue",
        "network problem",
        "no service",
        "internet not working",
        "wifi not working",
        "buffering",
        "network congestion",
    ]

    billing_keywords = [
        "bill",
        "billing",
        "charged",
        "charge",
        "payment",
        "invoice",
        "fraud",
        "transaction",
        "subscription",
        "deduction",
        "wrong bill",
        "double charged",
        "balance was deducted",
        "unexpected deduction",
        "incorrect balance",
        "wrong balance",
        "unauthorized deduction",
    ]

    knowledge_keywords = [
        "how do i",
        "how can i",
        "what is",
        "policy",
        "procedure",
        "manual",
        "guide",
        "sop",
        "explain",
        "information about",
    ]
    # Billing and fraud requests
    if any(keyword in text_lower for keyword in billing_keywords):
        return "BILLING_FRAUD"

    # Account and package knowledge requests
    if any(
        keyword in text_lower
        for keyword in account_knowledge_keywords
    ):
        return "KNOWLEDGE"

    # Network complaints
    if any(keyword in text_lower for keyword in network_keywords):
        return "NETWORK"

    # General knowledge requests
    if any(
        keyword in text_lower
        for keyword in knowledge_keywords
    ):
        return "KNOWLEDGE"

    return "CUSTOMER_SUPPORT"


def _employee_response(
    analysis,
    diagnosis,
    recommendation,
    evidence,
    network_decision=None,
):
    if network_decision == "HUMAN_REVIEW":
        return (
            f"Employee support assessment: "
            f"{analysis['category']} with {analysis['severity']} severity. "
            f"Diagnosis: {diagnosis}. "
            "The investigation requires human network engineer review "
            "before a final recommendation is issued. "
            "Human approval is required before any production network change."
        )

    return (
        f"Employee support assessment: "
        f"{analysis['category']} with {analysis['severity']} severity. "
        f"Diagnosis: {diagnosis}. "
        f"Recommended action: {recommendation}. "
        f"Relevant operational guidance: {evidence[:600]} "
        "Human approval is required before any production network change."
    )


def resume_voice_investigation(
    thread_id,
    decision,
    feedback="",
    reviewer_actor="unknown",
    reviewer_role="Network Engineer",
    on_incident_created=None,
):
    import time

    from langgraph.types import Command

    if not thread_id:
        raise ValueError(
            "A voice investigation thread ID is required."
        )

    if decision not in ("approved", "rejected"):
        raise ValueError(
            "Decision must be 'approved' or 'rejected'."
        )

    agent = build_agent()

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = {}

    print("\n=== RESUMING NETWORK INVESTIGATION ===")

    start_time = time.perf_counter()

    for event in agent.stream(
        Command(
            resume={
                "decision": decision,
                "feedback": feedback,
                "reviewer_actor": reviewer_actor,
                "reviewer_role": reviewer_role,
            }
        ),
        config=config,
        stream_mode="updates",
    ):
        elapsed = time.perf_counter() - start_time

        print(
            f"[{elapsed:.2f}s] "
            f"LANGGRAPH NODE UPDATE: {event}",
            flush=True,
        )

        if isinstance(event, dict):

            # Notify the interface immediately when the
            # incident creation node returns an incident ID.
            incident_output = event.get(
                "create_network_incident"
            )

            if (
                isinstance(incident_output, dict)
                and incident_output.get("incident_id")
                and on_incident_created is not None
            ):
                on_incident_created(
                    incident_output["incident_id"]
                )

            # Preserve the existing result aggregation.
            for node_output in event.values():
                if isinstance(node_output, dict):
                    result.update(node_output)
    snapshot = agent.get_state(config)

    if snapshot and snapshot.values:
        result = snapshot.values

    print(
        "=== NETWORK INVESTIGATION FINISHED ===",
        flush=True,
    )

    return result


def classify_complaint(text):
    """Classify a telecom customer request."""

    text = (text or "").lower().strip()

    network_keywords = [
        "slow internet",
        "internet is slow",
        "no signal",
        "weak signal",
        "call drop",
        "calls keep dropping",
        "packet loss",
        "high latency",
        "network outage",
        "poor coverage",
        "connection problem",
        "my internet is very slow",
    ]

    billing_keywords = [
        "charged twice",
        "double charged",
        "unexpected deduction",
        "wrong bill",
        "billing issue",
        "fraudulent transaction",
        "unauthorized transaction",
    ]

    account_keywords = [
        "check my balance",
        "account balance",
        "remaining balance",
        "change my package",
        "change package",
        "account details",
        "check my account",
        "data balance",
        "remaining data",
    ]

    if any(word in text for word in network_keywords):
        category = "Network Issue"
        severity = "HIGH"

    elif any(word in text for word in billing_keywords):
        category = "Billing Issue"
        severity = "MEDIUM"

    elif any(word in text for word in account_keywords):
        category = "Account and Package Inquiry"
        severity = "LOW"

    else:
        category = "General Support Request"
        severity = "MEDIUM"

    return {
        "category": category,
        "severity": severity,
        "original_text": text,
    }


def process_voice_request(
    text,
    audience="customer",
    tower_id="TWR-1024",
    anomaly=None,
    thread_id=None,
    transaction=None
):
    """
    Process a spoken request through the TelecomAI agent workflow.

    The LangGraph router decides which workflow should handle
    the request.
    """

    analysis = classify_complaint(text)

    orchestrator = build_voice_orchestrator()

    if not thread_id:
        thread_id = f"voice-{uuid.uuid4().hex}"

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    voice_state = orchestrator.invoke(
        {
            "text": text,
            "audience": audience,
            "tower_id": tower_id,
            "anomaly": anomaly or {},
            "transaction": transaction or {},
            "thread_id": thread_id,
        },
        config=config,
    )
    # Initialize request state BEFORE routing branches
    request_type = voice_state.get(
    "intent",
    "CUSTOMER_SUPPORT"
)
# Preserve the complete LangGraph state
    result = dict(voice_state)

    evidence = voice_state.get(
    "evidence",
    ""
)

    diagnosis = voice_state.get(
    "diagnosis",
    ""
)

    recommendation = voice_state.get(
    "recommendation",
    ""
)


# Knowledge requests should use retrieved
# telecom documentation rather than exposing
# internal network diagnostics.
    if request_type == "KNOWLEDGE":

        if evidence:
            response = (
                "I found relevant telecom guidance for your request. "
                f"{evidence[:1000]}"
            )
        else:
            response = (
                "I could not find relevant telecom documentation "
                "for that request."
            )
    elif request_type == "NETWORK":

        if audience == "employee":
            response = _employee_response(
                analysis,
                diagnosis,
                recommendation,
                evidence,
                result.get("network_decision"),
            )
        else:
            response = _customer_response(
                analysis,
                diagnosis,
                recommendation,
            )

    elif request_type == "BILLING_FRAUD":

        if result.get("fraud_decision") == "DETAILS_REQUIRED":
            response = (
                "I identified this as a billing fraud request. "
                f"{diagnosis} "
                f"{recommendation}"
            )
        else:
            response = (
                "Billing fraud assessment completed successfully. "
                f"{diagnosis} "
                f"{recommendation}"
            )

    elif request_type == "CUSTOMER_SUPPORT":

        response = _customer_response(
            analysis,
            diagnosis,
            recommendation
        )

    else:

        response = (
            "I have received your request and routed it through "
            "the appropriate TelecomAI workflow."
        )

    result["audience"] = audience
    result["complaint_analysis"] = analysis
    result["voice_response"] = response
    result["voice_request_type"] = request_type

    result["transaction"] = voice_state.get(
        "transaction",
        {}
    )

    result["transaction_missing_fields"] = voice_state.get(
        "transaction_missing_fields",
        []
    )

    return result, response


def process_voice_text(
    text,
    audience="customer",
    tower_id="TWR-1024",
    anomaly=None,
    thread_id=None
):
    return process_voice_request(

        text=text,
        audience=audience,
        tower_id=tower_id,
        anomaly=anomaly,
        thread_id=thread_id
    )
