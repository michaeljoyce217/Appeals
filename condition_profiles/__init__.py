# condition_profiles/__init__.py


# Required attributes that every condition profile MUST define.
# These are accessed directly (no hasattr/getattr guard) by at least one base engine file.
REQUIRED_ATTRIBUTES = [
    # Identity & Paths
    "CONDITION_NAME",
    "CONDITION_DISPLAY_NAME",
    "GOLD_LETTERS_PATH",
    "PROPEL_DATA_PATH",
    "DEFAULT_TEMPLATE_PATH",
    "CLINICAL_SCORES_TABLE_NAME",  # Can be None, but must be defined
    # Denial Parser
    "DENIAL_CONDITION_QUESTION",
    "DENIAL_CONDITION_FIELD",
    # Note Extraction
    "NOTE_EXTRACTION_TARGETS",
    # Structured Data
    "STRUCTURED_DATA_CONTEXT",
    # Conflict Detection
    "CONFLICT_EXAMPLES",
    # Writer Prompt
    "WRITER_SCORING_INSTRUCTIONS",
    # Assessment Labels
    "ASSESSMENT_CONDITION_LABEL",
    "ASSESSMENT_CRITERIA_LABEL",
]


def validate_profile(profile):
    """
    Check that a condition profile module has all required attributes.
    Call immediately after importlib.import_module() for a clear error on misconfigured profiles.

    Raises:
        AttributeError: listing all missing required attributes.
    """
    missing = [attr for attr in REQUIRED_ATTRIBUTES if not hasattr(profile, attr)]
    if missing:
        name = getattr(profile, '__name__', str(profile))
        raise AttributeError(
            f"Condition profile '{name}' is missing required attributes:\n"
            + "\n".join(f"  - {attr}" for attr in missing)
            + "\n\nSee condition_profiles/TEMPLATE.py for the full interface."
        )
