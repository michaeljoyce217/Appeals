# poc/config.py
# Configuration for Rebuttal Engine v2 POC
#
# SCOPE FILTERS: Change these to expand beyond sepsis
# All code below should be general-purpose; these constants control the filter.

# =============================================================================
# SCOPE FILTER - Set to "all" to process any denial type
# =============================================================================
SCOPE_FILTER = "sepsis"  # "sepsis" | "all"

# =============================================================================
# SEPSIS-SPECIFIC CONSTANTS (only used when SCOPE_FILTER = "sepsis")
# =============================================================================
SEPSIS_DRG_CODES = ["870", "871", "872"]
# 870 = Septic Shock
# 871 = Severe Sepsis (with organ dysfunction)
# 872 = Sepsis (without organ dysfunction)

SEPSIS_ICD10_CODES = [
    "A41.9",   # Sepsis, unspecified organism
    "A41.0",   # Sepsis due to Staphylococcus aureus
    "A41.1",   # Sepsis due to other specified staphylococcus
    "A41.2",   # Sepsis due to unspecified staphylococcus
    "A41.50",  # Gram-negative sepsis, unspecified
    "A41.51",  # Sepsis due to Escherichia coli
    "A41.52",  # Sepsis due to Pseudomonas
    "A41.53",  # Sepsis due to Serratia
    "R65.20",  # Severe sepsis without septic shock
    "R65.21",  # Severe sepsis with septic shock
]

# =============================================================================
# EVIDENCE PRIORITY (highest to lowest importance)
# =============================================================================
EVIDENCE_PRIORITY = [
    "provider_notes",      # Discharge summary, H&P - BEST source
    "structured_data",     # Labs, vitals from Clarity - backup
    "inference",           # Our conclusions from structured data - LEAST important
]

# =============================================================================
# AZURE OPENAI CONFIGURATION
# =============================================================================
# These are retrieved from Databricks secrets in production
# For local testing, set environment variables or modify these
AZURE_OPENAI_CONFIG = {
    "secret_scope": "idp_etl",
    "api_key_secret": "az-openai-key1",
    "endpoint_secret": "az-openai-base",
    "api_version": "2024-10-21",
    "model": "gpt-4.1",
}

# =============================================================================
# AZURE AI DOCUMENT INTELLIGENCE CONFIGURATION
# =============================================================================
AZURE_DOC_INTELLIGENCE_CONFIG = {
    "secret_scope": "idp_etl",
    "api_key_secret": "az-doc-intelligence-key",  # Update with actual secret name
    "endpoint_secret": "az-doc-intelligence-endpoint",  # Update with actual secret name
}

# =============================================================================
# RETRIEVAL SETTINGS
# =============================================================================
VECTOR_SEARCH_TOP_K = 10  # Number of candidates to retrieve before LLM eval
MATCH_SCORE_THRESHOLD = 0.7  # Below this, fall back to template

# =============================================================================
# TABLE NAMES
# =============================================================================
def get_table_names(catalog: str = "dev"):
    """Return table names for the given catalog."""
    return {
        "inference": f"{catalog}.fin_ds.fudgesicle_inference",
        "inference_score": f"{catalog}.fin_ds.fudgesicle_inference_score",
        "reference_documents": f"{catalog}.fin_ds.fudgesicle_reference_documents",
        "gold_letters": f"{catalog}.fin_ds.fudgesicle_gold_letters",
    }
