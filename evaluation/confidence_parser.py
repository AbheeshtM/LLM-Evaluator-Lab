import re

# -----------------------------------
# Allowed confidence labels
# -----------------------------------
CONFIDENCE_LEVELS = {"LOW", "MEDIUM", "HIGH"}

def parse_confidence(text: str) -> str:
    """
    Extract confidence level from LLM output text.

    Expected formats (examples):
    - Confidence: HIGH
    - confidence = medium
    - CONFIDENCE : Low

    Returns:
        "LOW", "MEDIUM", or "HIGH"
        Defaults to "MEDIUM" if not found
    """
    if not isinstance(text, str):
        return "MEDIUM"

    match = re.search(
        r"confidence\s*[:=]\s*(low|medium|high)",
        text,
        re.IGNORECASE
    )

    if match:
        return match.group(1).upper()

    # Fallback (safe default)
    return "MEDIUM"
