KEYWORDS = {
    "Internet Speed": ["slow", "speed", "internet", "data"],
    "Call Drops": ["call drop", "disconnect", "dropped"],
    "No Network": ["no network", "no service", "signal"],
    "SMS Failure": ["sms", "text message"],
    "SIM Problems": ["sim", "sim card"],
    "Billing": ["bill", "charged", "billing"],
    "Package Issues": ["package", "bundle", "benefits"],
    "Network Outage": ["outage", "area", "unavailable"],
}


def classify_complaint(text):
    t = text.lower()

    scores = {
        category: sum(word in t for word in keywords)
        for category, keywords in KEYWORDS.items()
    }

    cat = max(scores, key=scores.get)

    if scores[cat] == 0:
        cat = "General Network Issue"

    if any(
        word in t
        for word in [
            "emergency",
            "completely",
            "critical",
            "whole area",
            "no service",
        ]
    ):
        sev = "CRITICAL"
    elif any(
        word in t
        for word in [
            "very slow",
            "cannot",
            "can't",
            "stopped",
            "dropping",
        ]
    ):
        sev = "HIGH"
    elif cat in ["No Network", "Network Outage"]:
        sev = "HIGH"
    else:
        sev = "MEDIUM"

    if any(
        word in t
        for word in [
            "great",
            "good",
            "excellent",
            "thank",
            "working well",
        ]
    ):
        sentiment = "POSITIVE"
    elif any(
        word in t
        for word in [
            "slow",
            "bad",
            "poor",
            "problem",
            "issue",
            "dropping",
            "can't",
            "cannot",
            "failed",
            "failure",
        ]
    ):
        sentiment = "NEGATIVE"
    else:
        sentiment = "NEUTRAL"

    if sev == "CRITICAL":
        priority = "URGENT"
    elif sev == "HIGH":
        priority = "HIGH"
    else:
        priority = "NORMAL"

    return {
        "category": cat,
        "severity": sev,
        "sentiment": sentiment,
        "priority": priority,
        "keywords": [
            word
            for word in KEYWORDS.get(cat, [])
            if word in t
        ],
    }