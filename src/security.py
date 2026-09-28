import os
import hmac
from dotenv import load_dotenv

load_dotenv()
def _digest(value: str) -> str:
    return value

ROLES = {
    "admin": {
        "dashboard",
        "noc",
        "network",
        "maintenance",
        "customer",
        "fraud",
        "rag",
        "agent",
        "voice",
        "incidents",
        "evaluation",
        "observability",
    },
    "noc_engineer": {
        "dashboard",
        "network",
        "maintenance",
        "fraud",
        "rag",
        "agent",
        "voice",
        "incidents",
    },
    "support": {
        "dashboard",
        "customer",
        "fraud",
        "rag",
        "voice",
    },
    "viewer": {
        "dashboard",
    },
}
def roles():
    return list(ROLES.keys())

def verify_password(password: str, role: str) -> bool:
    passwords = {
        "admin": os.getenv("TELECOM_ADMIN_PASSWORD", ""),
        "noc_engineer": os.getenv("TELECOM_NOC_PASSWORD", ""),
        "support": os.getenv("TELECOM_SUPPORT_PASSWORD", ""),
        "viewer": os.getenv("TELECOM_VIEWER_PASSWORD", ""),
    }

    expected = passwords.get(role, "")

    if not expected:
        return False

    return hmac.compare_digest(
        _digest(password),
        _digest(expected),
    )
def can_access(role, page):
    return page in ROLES.get(role, set())