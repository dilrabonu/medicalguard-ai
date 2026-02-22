from dataclasses import dataclass


@dataclass
class SafetyResult:
    allowed: bool
    message: str


def safety_guard(user_query: str) -> SafetyResult:
    """
    Medical safety guardrails:
    - No personalized dosing or diagnosis.
    - Allow general educational information + safe next steps.
    """
    q = (user_query or "").lower().strip()

    # Requests that can lead to medical harm if answered directly
    blocked_phrases = [
        "how many mg", "dosage", "dose", "take per day", "times a day",
        "for my child", "for my baby", "pregnant", "breastfeeding",
        "mix with alcohol", "overdose", "replace my medicine",
        "stop taking", "should i take", "can i take", "inject",
        "without doctor", "without prescription"
    ]

    if any(p in q for p in blocked_phrases):
        return SafetyResult(
            allowed=False,
            message=(
                "I can share general information, but I can’t provide personalized dosing, "
                "diagnosis, or medical decisions. For exact use, please consult a pharmacist/doctor. "
                "You can ask: what it is used for, general cautions, and when to seek help."
            )
        )

    return SafetyResult(allowed=True, message="OK")
