import sys
from pathlib import Path

# Add the TelecomAI project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from src.voice_assistant import route_voice_intent


@pytest.mark.parametrize(
    "text, expected",
    [
        ("What is my account balance?", "KNOWLEDGE"),
        ("How can I change my package?", "KNOWLEDGE"),
        ("What internet packages are available?", "KNOWLEDGE"),
        ("I was charged twice for my bill.", "BILLING_FRAUD"),
        (
            "My internet is slow and calls keep dropping.",
            "NETWORK",
        ),
    ],
)
def test_voice_intent_routing(text, expected):
    assert route_voice_intent(text) == expected


def test_empty_request():
    assert route_voice_intent("") == "CUSTOMER_SUPPORT"


def test_billing_priority_over_network():
    assert (
        route_voice_intent(
            "My network bill has an incorrect charge."
        )
        == "BILLING_FRAUD"
    )
