from .config import GOOGLE_API_KEY, GEMINI_MODEL

import re
import time

from .database import save_model_telemetry
def _record_model_telemetry(
    *,
    operation,
    started_at,
    response=None,
    status="SUCCESS",
    error_message=None,
):
    """
    Record real Gemini model telemetry.

    Telemetry failures must never break the application.
    """

    try:

        latency_ms = int(
            (time.perf_counter() - started_at) * 1000
        )

        input_tokens = None
        output_tokens = None
        total_tokens = None

        if response is not None:

            usage = getattr(
                response,
                "usage_metadata",
                None,
            )

            if usage:

                input_tokens = usage.get(
                    "input_tokens"
                )

                output_tokens = usage.get(
                    "output_tokens"
                )

                total_tokens = usage.get(
                    "total_tokens"
                )

            if (
                input_tokens is None
                or output_tokens is None
                or total_tokens is None
            ):

                metadata = getattr(
                    response,
                    "response_metadata",
                    {}
                ) or {}

                usage = metadata.get(
                    "usage_metadata"
                ) or metadata.get(
                    "token_usage"
                ) or {}

                input_tokens = (
                    input_tokens
                    if input_tokens is not None
                    else usage.get("input_tokens")
                    or usage.get("prompt_tokens")
                )

                output_tokens = (
                    output_tokens
                    if output_tokens is not None
                    else usage.get("output_tokens")
                    or usage.get("completion_tokens")
                )

                total_tokens = (
                    total_tokens
                    if total_tokens is not None
                    else usage.get("total_tokens")
                )

        if (
            total_tokens is None
            and input_tokens is not None
            and output_tokens is not None
        ):
            total_tokens = (
                input_tokens
                + output_tokens
            )

        save_model_telemetry(
            provider="Google",
            model_name=GEMINI_MODEL,
            operation=operation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            status=status,
            error_message=error_message,
        )

    except Exception:
        # Telemetry must never break the main application.
        pass
def generate_explanation(
    issue,
    evidence,
    recommendation,
):

    if not GOOGLE_API_KEY:

        return (
            f"Assessment: {issue}\n\n"
            f"Grounded evidence: {evidence[:700]}\n\n"
            f"Recommendation: {recommendation}\n\n"
            "Human engineer approval is required "
            "before production changes."
        )

    started_at = time.perf_counter()

    try:

        from langchain_google_genai import (
            ChatGoogleGenerativeAI
        )

        from langchain_core.messages import (
            HumanMessage
        )

        model = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.2,
        )

        response = model.invoke(
            [
                HumanMessage(
                    content=(
                        "You are a telecom network assistant. "
                        "Explain this issue using only the "
                        "supplied evidence.\n\n"
                        f"Issue: {issue}\n"
                        f"Evidence: {evidence}\n"
                        f"Recommendation: {recommendation}"
                    )
                )
            ]
        )

        _record_model_telemetry(
            operation="generate_explanation",
            started_at=started_at,
            response=response,
            status="SUCCESS",
        )

        return response.content

    except Exception as e:

        _record_model_telemetry(
            operation="generate_explanation",
            started_at=started_at,
            status="FAILED",
            error_message=str(e),
        )

        return (
            f"Local fallback: {issue}. "
            f"Recommendation: {recommendation}. "
            f"Gemini unavailable: "
            f"{type(e).__name__}."
        )
def extract_transaction_details(text):
    """
    Extract transaction fields explicitly present in the request.

    Uses deterministic extraction first for simple fields.
    Gemini is used only when available for additional fields.
    """

    transaction = {}
    text_lower = text.lower()

    # -------------------------
    # Amount
    # -------------------------
    amount_match = re.search(
        r"(?:rs\.?|pkr|rupees?|₨)\s*([0-9]+(?:\.[0-9]+)?)"
        r"|([0-9]+(?:\.[0-9]+)?)\s*(?:rs\.?|pkr|rupees?|₨)",
        text_lower
    )

    if amount_match:
        amount_value = (
            amount_match.group(1)
            or amount_match.group(2)
        )
        transaction["amount"] = float(amount_value)

    # -------------------------
    # New device
    # -------------------------
    if "new device" in text_lower:
        transaction["is_new_device"] = 1

    elif any(
        phrase in text_lower
        for phrase in [
            "old device",
            "known device",
            "existing device",
            "trusted device",
        ]
    ):
        transaction["is_new_device"] = 0

    # -------------------------
    # Hour
    # -------------------------
    time_match = re.search(
        r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b",
        text_lower
    )

    if time_match:
        hour = int(time_match.group(1))
        period = time_match.group(3)

        if period == "pm" and hour != 12:
            hour += 12

        elif period == "am" and hour == 12:
            hour = 0

        transaction["hour"] = hour

    # -------------------------
    # Use Gemini for additional
    # explicitly stated fields
    # -------------------------
    if GOOGLE_API_KEY:

      started_at = time.perf_counter()

      try:

          from langchain_google_genai import (
            ChatGoogleGenerativeAI
        )

          from langchain_core.messages import (
            HumanMessage
        )

          import json

          model = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0,
        )

          prompt = f"""
Extract ONLY these transaction fields if explicitly stated:

previous_transactions

transaction_velocity

country

payment_type

device_type

Rules:

- Never guess.
- Never invent values.
- Return ONLY valid JSON.
- Omit fields that are not explicitly stated.

User request:

{text}
"""

          response = model.invoke(
            [
                HumanMessage(
                    content=prompt
                )
            ]
        )

          _record_model_telemetry(
            operation="extract_transaction_details",
            started_at=started_at,
            response=response,
            status="SUCCESS",
        )

          raw = response.content.strip()

          if raw.startswith("```"):

            raw = raw.replace(
                "```json",
                ""
            )

            raw = raw.replace(
                "```",
                ""
            )

            raw = raw.strip()

          data = json.loads(raw)

          allowed_fields = {
            "previous_transactions",
            "transaction_velocity",
            "country",
            "payment_type",
            "device_type",
        }

          for key, value in data.items():

            if key in allowed_fields:
                transaction[key] = value

      except Exception as e:

        _record_model_telemetry(
            operation="extract_transaction_details",
            started_at=started_at,
            status="FAILED",
            error_message=str(e),
        )

        pass

    return transaction