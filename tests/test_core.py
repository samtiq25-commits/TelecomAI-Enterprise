from src.complaint_analyzer import classify_complaint
from src.rag import retrieve


def test_complaint():
    result = classify_complaint("My internet is very slow")
    assert result["category"] == "Internet Speed"


def test_rag():
    docs = retrieve("How do I troubleshoot high latency?")
    assert len(docs) > 0


def test_complaint_returns_sentiment():
    result = classify_complaint(
        "My internet is very slow and calls keep dropping"
    )

    assert "sentiment" in result
    assert result["sentiment"] == "NEGATIVE"


def test_complaint_returns_priority():
    result = classify_complaint(
        "My internet is very slow and calls keep dropping"
    )

    assert "priority" in result
    assert result["priority"] == "HIGH"


def test_complaint_analysis_contract():
    result = classify_complaint(
        "My internet is very slow and calls keep dropping"
    )

    assert result["category"] == "Internet Speed"
    assert result["severity"] == "HIGH"
    assert result["sentiment"] == "NEGATIVE"
    assert result["priority"] == "HIGH"
    assert "keywords" in result