# condition_profiles/TEMPLATE.py
# Template for creating a new condition profile for the DRG Appeal Engine.
#
# To add a new condition:
# 1. Copy this file to condition_profiles/<condition_name>.py
# 2. Fill in all required constants and functions
# 3. Set CONDITION_PROFILE = "<condition_name>" in each notebook
#
# See sepsis.py for a complete working example.

# =============================================================================
# REQUIRED: Identity & Paths
# =============================================================================

CONDITION_NAME = ""           # e.g., "respiratory_failure" — used for Propel table lookup
CONDITION_DISPLAY_NAME = ""   # e.g., "Respiratory Failure" — used in print statements
DRG_CODES = []                # e.g., ["003", "004"] — DRG codes for this condition

GOLD_LETTERS_PATH = ""        # Workspace path to gold standard appeal letter PDFs
PROPEL_DATA_PATH = ""         # Workspace path to Propel clinical definition PDFs
DEFAULT_TEMPLATE_PATH = ""    # Workspace path to default appeal template DOCX

# Clinical scores table name suffix (base engine prepends {catalog}.fin_ds.)
# Set to None if this condition has no clinical scoring system.
CLINICAL_SCORES_TABLE_NAME = None  # e.g., "fudgesicle_case_resp_scores"

# =============================================================================
# REQUIRED: Denial Parser
# =============================================================================

# Question added to the denial parser prompt to detect this condition
DENIAL_CONDITION_QUESTION = ""  # e.g., "5. IS RESPIRATORY FAILURE RELATED - does this denial involve respiratory failure?"

# Field name in the parser output (uppercase, no spaces)
DENIAL_CONDITION_FIELD = ""     # e.g., "IS_RESPIRATORY_FAILURE"

# =============================================================================
# REQUIRED: Note Extraction Targets
# =============================================================================

# Fallback extraction targets when Propel definition_summary is unavailable.
# Used in the note extraction prompt to tell the LLM what clinical data to extract.
NOTE_EXTRACTION_TARGETS = ""

# =============================================================================
# REQUIRED: Structured Data Context
# =============================================================================

# Instructions for the structured data extraction LLM about what to prioritize.
# This replaces the condition-specific portion of the structured data extraction prompt.
STRUCTURED_DATA_CONTEXT = ""

# Optional: Custom system message for structured data extraction.
# Defaults to a generic "clinical data analyst" message if not set.
STRUCTURED_DATA_SYSTEM_MESSAGE = ""  # e.g., "You are a clinical data analyst specializing in respiratory failure cases."

# =============================================================================
# REQUIRED: Conflict Detection Examples
# =============================================================================

# Condition-specific examples of conflicts between notes and structured data.
CONFLICT_EXAMPLES = ""

# =============================================================================
# REQUIRED: Numeric Cross-Check Parameter Mapping
# =============================================================================

# Maps parameter names (as LLM might extract them from notes) to internal categories
# used by LAB_VITAL_MATCHERS. Only parameters relevant to this condition.
PARAM_TO_CATEGORY = {}

# =============================================================================
# REQUIRED: Writer Prompt — Scoring Instructions
# =============================================================================

# Condition-specific instructions for the letter writer about how to reference
# clinical scores in the appeal letter. Inserted into the writer prompt.
WRITER_SCORING_INSTRUCTIONS = ""

# =============================================================================
# REQUIRED: Assessment Labels
# =============================================================================

# Labels used in the assessment prompt
ASSESSMENT_CONDITION_LABEL = ""   # e.g., "respiratory failure DRG appeal letter"
ASSESSMENT_CRITERIA_LABEL = ""    # e.g., "PROPEL RESPIRATORY FAILURE CRITERIA"

# =============================================================================
# OPTIONAL: Clinical Scorer
# =============================================================================

# If this condition has a clinical scoring system (like SOFA for sepsis),
# implement these two functions. If not, omit them — the base engine checks
# with hasattr() before calling.

# Also provide LAB_VITAL_MATCHERS if your scorer or numeric cross-check needs them.
# LAB_VITAL_MATCHERS = {}

# def calculate_clinical_scores(account_id, spark, tables):
#     """
#     Calculate condition-specific clinical scores from raw structured data.
#
#     Args:
#         account_id: The hospital account ID
#         spark: SparkSession instance
#         tables: dict with keys "labs", "vitals", "meds" → full table names
#
#     Returns:
#         dict with at minimum:
#             "organ_scores": dict of organ → {"score": int, "value": str, "timestamp": str}
#             "total_score": int
#             "organs_scored": int
#     """
#     pass

# def write_clinical_scores_table(account_id, scores_result, spark, table_name):
#     """
#     Write clinical scores to a Delta table.
#
#     Args:
#         account_id: The hospital account ID
#         scores_result: dict returned by calculate_clinical_scores
#         spark: SparkSession instance
#         table_name: Full table name to write to
#     """
#     pass

# =============================================================================
# OPTIONAL: DOCX Rendering
# =============================================================================

# If this condition has clinical scores to render in the DOCX output,
# implement these functions. If not, omit them — the base engine checks
# with hasattr() before calling.

# def format_scores_for_prompt(scores_data):
#     """Format clinical scores as text for inclusion in the writer/assessment prompts."""
#     pass

# def render_scores_status_note(doc, scores_data):
#     """Render a status note about clinical scores in the internal review section."""
#     pass

# def render_scores_in_docx(doc, scores_data):
#     """Render clinical scores table/visualization in the DOCX appendix."""
#     pass
