from src.complaint_analyzer import classify_complaint
from src.rag import retrieve

def test_complaint():
    result = classify_complaint("My internet is very slow")
    assert result["category"] == "Internet Speed"

def test_rag():
    docs = retrieve("How do I troubleshoot high latency?")
    assert len(docs) > 0
