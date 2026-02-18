# inference_standalone.py
# ==============================================================================
# COMPLETELY SELF-CONTAINED - NO EXTERNAL IMPORTS FROM poc/
# ==============================================================================
#
# This script runs the 4-agent pipeline to generate rebuttal letters:
#
#   1. SCAN AGENT: Reads denial letter text, extracts structured info
#      - What condition is being denied?
#      - What are the payer's arguments?
#      - What DRGs are involved?
#
#   2. RETRIEVAL AGENT: Finds relevant clinical criteria
#      - Queries clinical_criteria table using denial signals
#      - Returns Propel/ACDIS guidance to support the rebuttal
#
#   3. TEMPLATE MATCH AGENT: Finds best-matching gold standard letter
#      - Currently returns fallback template (no gold letters yet)
#      - Will use vector search when gold_templates is populated
#
#   4. WRITER AGENT: Generates the final rebuttal letter
#      - Uses denial info, criteria, template, and clinical notes
#      - Outputs structured JSON that renders to letter text
#
# ==============================================================================

# =============================================================================
# CONFIGURATION - EDIT THESE VALUES
# =============================================================================

# Database settings
CATALOG = "dev"  # Change to "test" or "prod"
SCHEMA = "clncl_ds"  # Your schema

# Feature flags
USE_GOLD_TEMPLATES = False  # Set True when gold_templates table is populated
USE_VECTOR_SEARCH = False   # Set True when vector search index is created

# Output settings
# "table" = write to draft_letters Delta table
# "file"  = write to local files (for testing)
OUTPUT_MODE = "table"
OUTPUT_FOLDER = "/dbfs/tmp/rebuttal_output"  # Only used if OUTPUT_MODE="file"

# LLM settings
AZURE_OPENAI_MODEL = "gpt-4.1"  # Your deployment name
AZURE_OPENAI_API_VERSION = "2024-10-21"

# =============================================================================
# IMPORTS
# =============================================================================

import os
import json
import uuid
import time
from datetime import datetime, date
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

# Databricks/Spark imports
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, FloatType,
    TimestampType, IntegerType, ArrayType
)

# =============================================================================
# GET SPARK AND DBUTILS
# =============================================================================

spark = SparkSession.builder.getOrCreate()

try:
    # Databricks environment - method 1
    dbutils = spark._jvm.com.databricks.service.DBUtils(spark._jsparkSession)
except:
    # Databricks environment - method 2
    from pyspark.dbutils import DBUtils
    dbutils = DBUtils(spark)

print(f"Catalog: {CATALOG}")
print(f"Schema: {SCHEMA}")

# =============================================================================
# TABLE NAMES - Must match featurization_standalone.py
# =============================================================================

DENIAL_LETTERS_TABLE = f"{CATALOG}.{SCHEMA}.fudgesicle_denial_letters"
DENIAL_EXTRACTS_TABLE = f"{CATALOG}.{SCHEMA}.fudgesicle_denial_extracts"
CLINICAL_CRITERIA_TABLE = f"{CATALOG}.{SCHEMA}.fudgesicle_clinical_criteria"
SOFA_THRESHOLDS_TABLE = f"{CATALOG}.{SCHEMA}.fudgesicle_sofa_thresholds"
GOLD_TEMPLATES_TABLE = f"{CATALOG}.{SCHEMA}.fudgesicle_gold_templates"
DRAFT_LETTERS_TABLE = f"{CATALOG}.{SCHEMA}.fudgesicle_draft_letters"

# =============================================================================
# AZURE CREDENTIALS FROM DATABRICKS SECRETS
# =============================================================================

print("\n[SETUP] Loading Azure credentials from secrets...")

# Azure OpenAI credentials (for LLM calls)
AZURE_OPENAI_KEY = dbutils.secrets.get(scope="idp_etl", key="az-openai-key1")
AZURE_OPENAI_ENDPOINT = dbutils.secrets.get(scope="idp_etl", key="az-openai-base")

print("  Azure OpenAI credentials loaded")

# =============================================================================
# CREATE AZURE OPENAI CLIENT
# =============================================================================

from openai import AzureOpenAI

# This client is used by all agents for LLM calls
openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

print("  Azure OpenAI client initialized")

# =============================================================================
# DATA CLASSES
# =============================================================================
# These define the structured output from each agent.
# Using dataclasses makes the code cleaner and more type-safe.
# =============================================================================

@dataclass
class DenialExtract:
    """
    Output from SCAN AGENT.

    Contains all structured information extracted from a denial letter.
    The denial_signals field is critical - it's used to filter criteria retrieval.
    """
    extract_id: str                        # Unique ID for this extraction
    letter_id: str                         # Links back to denial_letters table

    # Core extraction - WHAT is being denied
    condition_denied: str                  # sepsis, severe_sepsis, septic_shock, etc.
    denial_category: str                   # clinical_validation, coding_error, etc.
    denial_subcategory: Optional[str] = None

    # Payer's reasoning - WHY they're denying
    denial_rationale: str = ""             # Full text of their reasoning
    denial_summary: str = ""               # Our 1-2 sentence summary
    payer_quotes: List[str] = field(default_factory=list)  # Direct quotes to cite back

    # Signals for retrieval filtering
    # These map to denial_relevance in clinical_criteria table
    denial_signals: List[str] = field(default_factory=list)

    # DRG info - WHAT codes are involved
    denied_drg: Optional[str] = None       # DRG payer wants to change TO
    claimed_drg: Optional[str] = None      # DRG originally claimed

    # Reviewer info - WHO reviewed
    reviewer_name: Optional[str] = None
    reviewer_credentials: Optional[str] = None
    review_date: Optional[str] = None

    # Metadata
    confidence_score: float = 0.0          # Model's confidence (0-1)
    processing_time_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Spark DataFrame."""
        return asdict(self)


@dataclass
class RetrievedCriteria:
    """
    A single clinical criteria chunk from RETRIEVAL AGENT.

    These come from the clinical_criteria table (populated from Propel).
    """
    criteria_id: str
    criteria_type: str           # clinical_definition, diagnostic_criteria, cdi_guidance, etc.
    criteria_subtype: Optional[str]
    source_citation: str         # e.g., "Propel ACDIS Sepsis Reference, CDI Critical Thinking"
    content: str                 # The actual text content
    relevance_score: float       # How relevant to this denial (0-1)
    denial_relevance: List[str] = field(default_factory=list)


@dataclass
class MatchedTemplate:
    """
    Output from TEMPLATE MATCH AGENT.

    Either a gold standard letter or the fallback template.
    """
    template_id: str
    condition: str
    denial_category: str
    denial_pattern: Optional[str]
    template_text: str
    template_summary: Optional[str]
    key_arguments: List[str]
    similarity_score: float
    win_rate: Optional[float] = None
    times_used: int = 0

# =============================================================================
# AGENT 1: SCAN AGENT
# =============================================================================
# Purpose: Extract structured denial information from raw denial letter text.
#
# The LLM reads the denial letter and outputs JSON with:
# - What condition is being denied
# - Why the payer is denying (their arguments)
# - Key quotes we can cite in our rebuttal
# - "Signals" that help us find relevant criteria
#
# The denial_signals are especially important - they tell the Retrieval Agent
# what type of criteria to look for (e.g., "negative_cultures" means we need
# guidance about why negative cultures don't rule out sepsis).
# =============================================================================

SCAN_AGENT_SYSTEM_PROMPT = """You are a Clinical Documentation Improvement (CDI) specialist analyzing insurance denial letters.

Your task: Extract structured information from denial letters to enable targeted rebuttal generation.

## EXTRACTION FIELDS

1. **condition_denied**: The primary clinical condition being denied or downgraded
   - Values: sepsis, severe_sepsis, septic_shock, pneumonia, heart_failure, aki, respiratory_failure, other

2. **denial_category**: High-level denial type
   - clinical_validation: Payer claims clinical criteria not met
   - coding_error: Payer claims wrong codes assigned
   - medical_necessity: Payer claims treatment not medically necessary
   - coverage: Payer claims service not covered
   - other: Doesn't fit above

3. **denial_subcategory**: More specific denial type (freeform)
   - Examples: "sepsis_criteria_not_met", "drg_downgrade_871_to_872"

4. **denial_rationale**: Full text of payer's reasoning (copy verbatim if possible)

5. **denial_summary**: Your 1-2 sentence summary of WHY they're denying

6. **payer_quotes**: Array of direct quotes from the letter stating the denial reason
   - CRITICAL for rebuttals - we cite their own words back

7. **denial_signals**: Array of retrieval filter signals. Use these exact values:
   - lacks_clinical_indicators: Payer says clinical criteria not documented
   - negative_cultures: Payer cites negative blood cultures
   - sofa_threshold_not_met: Payer says SOFA/organ dysfunction not met
   - documentation_insufficient: Payer says documentation doesn't support
   - coding_sequence_error: Payer says codes sequenced incorrectly
   - severity_not_supported: Payer says severity level not supported
   - sepsis_vs_sirs: Payer argues this is SIRS, not sepsis
   - timing_of_diagnosis: Payer disputes when condition was present
   - treatment_inconsistent: Payer says treatment doesn't match diagnosis

8. **DRG info**: denied_drg (what they want), claimed_drg (what we claimed)

9. **Reviewer info**: reviewer_name, reviewer_credentials, review_date (YYYY-MM-DD)

10. **confidence_score**: Your confidence in extraction (0.0-1.0)

## OUTPUT FORMAT
Return a JSON object with all fields. Use null for fields you can't extract."""


def run_scan_agent(letter_id: str, letter_text: str) -> DenialExtract:
    """
    SCAN AGENT: Extract denial information from a denial letter.

    This is the first step in the pipeline. It reads the raw denial letter
    text and extracts structured information that downstream agents use.

    Args:
        letter_id: Unique identifier for the denial letter
        letter_text: Full text of the denial letter

    Returns:
        DenialExtract with structured denial information
    """
    print(f"    [SCAN] Processing letter {letter_id[:8]}...")
    start_time = time.time()

    # Call the LLM to extract information
    response = openai_client.chat.completions.create(
        model=AZURE_OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SCAN_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract denial information from this letter:\n\n{letter_text}"}
        ],
        temperature=0.1,  # Low temperature for consistent extraction
        max_tokens=2000,
        response_format={"type": "json_object"}  # Force JSON output
    )

    processing_time_ms = int((time.time() - start_time) * 1000)

    # Parse the JSON response
    result = json.loads(response.choices[0].message.content)

    # Build the DenialExtract dataclass
    extract = DenialExtract(
        extract_id=str(uuid.uuid4()),
        letter_id=letter_id,
        condition_denied=result.get("condition_denied", "unknown"),
        denial_category=result.get("denial_category", "other"),
        denial_subcategory=result.get("denial_subcategory"),
        denial_rationale=result.get("denial_rationale", ""),
        denial_summary=result.get("denial_summary", ""),
        payer_quotes=result.get("payer_quotes", []),
        denial_signals=result.get("denial_signals", []),
        denied_drg=result.get("denied_drg"),
        claimed_drg=result.get("claimed_drg"),
        reviewer_name=result.get("reviewer_name"),
        reviewer_credentials=result.get("reviewer_credentials"),
        review_date=result.get("review_date"),
        confidence_score=result.get("confidence_score", 0.0),
        processing_time_ms=processing_time_ms
    )

    print(f"    [SCAN] Done in {processing_time_ms}ms. Signals: {extract.denial_signals}")
    return extract

# =============================================================================
# AGENT 2: RETRIEVAL AGENT
# =============================================================================
# Purpose: Find relevant clinical criteria from our knowledge base.
#
# Uses the denial_signals from Scan Agent to query the clinical_criteria table.
# The criteria help us build strong arguments in the rebuttal.
#
# Two retrieval modes:
# 1. SQL fallback (current): Simple query matching denial_relevance array
# 2. Vector search (future): Semantic similarity using embeddings
# =============================================================================

def run_retrieval_agent(denial_extract: DenialExtract, top_k: int = 5) -> List[RetrievedCriteria]:
    """
    RETRIEVAL AGENT: Find relevant clinical criteria for the denial.

    This queries the clinical_criteria table to find Propel/ACDIS guidance
    that helps rebut the payer's specific arguments.

    Args:
        denial_extract: Output from Scan Agent
        top_k: Number of criteria chunks to retrieve

    Returns:
        List of RetrievedCriteria with relevant guidance
    """
    print(f"    [RETRIEVAL] Looking for criteria matching: {denial_extract.denial_signals}")
    start_time = time.time()

    # Check if clinical_criteria table has data
    try:
        count = spark.sql(f"SELECT COUNT(*) as cnt FROM {CLINICAL_CRITERIA_TABLE}").collect()[0].cnt
        if count == 0:
            print(f"    [RETRIEVAL] Warning: {CLINICAL_CRITERIA_TABLE} is empty. Using mock criteria.")
            return get_mock_criteria(denial_extract.denial_signals, top_k)
    except Exception as e:
        print(f"    [RETRIEVAL] Warning: Could not query {CLINICAL_CRITERIA_TABLE}: {e}")
        return get_mock_criteria(denial_extract.denial_signals, top_k)

    # Build SQL query with array overlap filter
    # This finds criteria where denial_relevance contains any of our denial_signals
    denial_signals = denial_extract.denial_signals
    if denial_signals:
        signal_list = ", ".join([f"'{s}'" for s in denial_signals])
        filter_clause = f"""
        WHERE EXISTS (
            SELECT 1 FROM (
                SELECT explode(denial_relevance) as rel
            ) WHERE rel IN ({signal_list})
        )
        """
    else:
        filter_clause = ""

    query = f"""
    SELECT
        criteria_id,
        criteria_type,
        criteria_subtype,
        source_citation,
        content,
        denial_relevance
    FROM {CLINICAL_CRITERIA_TABLE}
    {filter_clause}
    LIMIT {top_k}
    """

    try:
        results = spark.sql(query).collect()

        criteria = []
        for row in results:
            criteria.append(RetrievedCriteria(
                criteria_id=row.criteria_id,
                criteria_type=row.criteria_type,
                criteria_subtype=row.criteria_subtype,
                source_citation=row.source_citation,
                content=row.content,
                denial_relevance=row.denial_relevance or [],
                relevance_score=0.8  # Fixed score for SQL fallback
            ))

        processing_time_ms = int((time.time() - start_time) * 1000)
        print(f"    [RETRIEVAL] Found {len(criteria)} criteria in {processing_time_ms}ms")
        return criteria

    except Exception as e:
        print(f"    [RETRIEVAL] Query failed: {e}. Using mock criteria.")
        return get_mock_criteria(denial_signals, top_k)


def get_mock_criteria(denial_signals: List[str], top_k: int) -> List[RetrievedCriteria]:
    """
    Return mock criteria for testing when clinical_criteria table is empty.

    These are real examples from Propel documentation.
    """
    mock_data = [
        RetrievedCriteria(
            criteria_id="mock-001",
            criteria_type="clinical_definition",
            criteria_subtype="sepsis",
            source_citation="Propel ACDIS Sepsis Reference",
            content="Sepsis is a systemic, deleterious host response to infection. Widespread inflammation occurs in reaction to chemicals released into the blood. If left untreated, this can lead to decreased organ perfusion, organ failure, and shock.",
            denial_relevance=["lacks_clinical_indicators", "sepsis_vs_sirs"],
            relevance_score=0.95
        ),
        RetrievedCriteria(
            criteria_id="mock-002",
            criteria_type="cdi_guidance",
            criteria_subtype="negative_cultures",
            source_citation="Propel ACDIS Sepsis Reference, CDI Critical Thinking",
            content="Negative or inconclusive blood cultures do not preclude a diagnosis of sepsis in patients with clinical evidence of the condition. Only 30-40% of sepsis cases yield positive blood cultures. Clinical diagnosis remains appropriate when other criteria are met.",
            denial_relevance=["negative_cultures", "documentation_insufficient"],
            relevance_score=0.92
        ),
        RetrievedCriteria(
            criteria_id="mock-003",
            criteria_type="diagnostic_criteria",
            criteria_subtype="sepsis2_general_parameters",
            source_citation="Propel ACDIS Sepsis Reference, Sepsis-2",
            content="Sepsis-2 General Parameters (2+ indicates sepsis when attributed to infection): Temperature >38.3C or <36.0C, Heart rate >90 bpm, Respiratory rate >20 or PaCO2 <32 mmHg, WBC >12,000 or <4,000 or >10% bands.",
            denial_relevance=["lacks_clinical_indicators"],
            relevance_score=0.88
        ),
        RetrievedCriteria(
            criteria_id="mock-004",
            criteria_type="sofa_scoring",
            criteria_subtype="sofa2_overview",
            source_citation="Propel ACDIS Sepsis Reference, SOFA-2",
            content="Suspected or documented infection and an acute increase of >=2 points from baseline score using the SOFA-2 scale indicates sepsis. SOFA-2 (October 2025 consensus) includes: Respiratory (PaO2/FiO2), Coagulation (Platelets), Liver (Bilirubin), Cardiovascular (MAP/vasopressors), CNS (GCS), Renal (Creatinine/UOP).",
            denial_relevance=["sofa_threshold_not_met", "severity_not_supported"],
            relevance_score=0.85
        ),
    ]

    # Filter by denial_signals if provided
    if denial_signals:
        mock_data = [
            c for c in mock_data
            if any(s in c.denial_relevance for s in denial_signals)
        ]

    return mock_data[:top_k]

# =============================================================================
# AGENT 3: TEMPLATE MATCH AGENT
# =============================================================================
# Purpose: Find the best-matching gold standard rebuttal letter.
#
# Currently returns a fallback template since gold_templates table is empty.
# When gold letters are available:
# 1. Filter by condition and denial_category
# 2. Vector search for semantic similarity
# 3. Optionally use LLM to evaluate fit
# =============================================================================

# Fallback template - used when no gold letters are available
FALLBACK_TEMPLATE = """Dear Medical Director,

We respectfully appeal the denial of [DRG] for the above-referenced patient, [PATIENT NAME], Date of Service [DOS].

Upon review of the medical record, we believe the clinical documentation supports the diagnosis of [CONDITION] based on the following evidence:

CLINICAL PRESENTATION:
[INSERT CLINICAL EVIDENCE FROM NOTES]

DIAGNOSTIC CRITERIA MET:
[INSERT RELEVANT CRITERIA - SEPSIS-2 OR SEPSIS-3]

ORGAN DYSFUNCTION EVIDENCE:
[INSERT SOFA SCORING OR ORGAN DYSFUNCTION MARKERS]

TREATMENT COURSE:
[INSERT TREATMENT SUPPORTING DIAGNOSIS]

Based on the above clinical documentation and established diagnostic criteria, we respectfully request reconsideration of the DRG assignment.

Please contact our office if additional information is needed.

Respectfully submitted,

[PHYSICIAN NAME], [CREDENTIALS]
[FACILITY NAME]
[CONTACT INFORMATION]
"""


def run_template_match_agent(denial_extract: DenialExtract) -> MatchedTemplate:
    """
    TEMPLATE MATCH AGENT: Find the best-matching gold standard letter.

    Currently always returns the fallback template since we don't have
    gold standard letters yet. When USE_GOLD_TEMPLATES=True, this will
    search the gold_templates table.

    Args:
        denial_extract: Output from Scan Agent

    Returns:
        MatchedTemplate with the best template to use
    """
    print(f"    [TEMPLATE] Finding template for {denial_extract.condition_denied}...")

    # For now, always use fallback
    # TODO: When gold_templates is populated, add vector search here
    if USE_GOLD_TEMPLATES:
        # Future: query gold_templates table
        pass

    # Return fallback template
    template = MatchedTemplate(
        template_id="fallback-sepsis-001",
        condition=denial_extract.condition_denied,
        denial_category=denial_extract.denial_category,
        denial_pattern="generic",
        template_text=FALLBACK_TEMPLATE,
        template_summary="Generic sepsis rebuttal template",
        key_arguments=[
            "Present clinical evidence from medical record",
            "Cite relevant diagnostic criteria (Sepsis-2 or Sepsis-3)",
            "Document organ dysfunction",
            "Reference treatment course supporting diagnosis"
        ],
        similarity_score=0.5,
        win_rate=0.55,
        times_used=0
    )

    print(f"    [TEMPLATE] Using fallback template")
    return template

# =============================================================================
# AGENT 4: WRITER AGENT
# =============================================================================
# Purpose: Generate the final rebuttal letter.
#
# Takes all the inputs from previous agents and generates a complete letter
# following the Mercy Hospital template structure.
#
# Evidence priority:
# 1. Provider notes (Discharge Summary, H&P) - BEST
# 2. Structured data (labs, vitals) - backup
# 3. Inference - LEAST important
# =============================================================================

WRITER_AGENT_PROMPT = '''You are a clinical coding expert writing a DRG validation appeal letter for Mercy Hospital.

# Task
Write a complete appeal letter using the Mercy Hospital template format.

# Denial Information
{denial_info_json}

# Clinical Criteria (from Propel guidelines)
{criteria_json}

# Reference Template (use as style/structure guide)
{template_text}

# Patient Clinical Notes
{clinical_notes}

# Instructions
1. ADDRESS EACH DENIAL ARGUMENT specifically - quote the payer's argument, then refute it
2. CITE CLINICAL EVIDENCE from the notes
3. Use the criteria from Propel to support your arguments
4. Follow the reference template structure
5. Include specific values (lactate 2.4, MAP 62, etc.) whenever available
6. Be thorough but concise - every statement should support the appeal

# Output
Return the complete letter as plain text. Start with "Dear" and end with signature block.
Do NOT return JSON - return the letter text only.'''


def run_writer_agent(
    denial_extract: DenialExtract,
    criteria: List[RetrievedCriteria],
    template: MatchedTemplate,
    clinical_notes: str = ""
) -> str:
    """
    WRITER AGENT: Generate the final rebuttal letter.

    This is the final step. It takes all the gathered information and
    generates a complete, ready-to-review rebuttal letter.

    Args:
        denial_extract: Output from Scan Agent
        criteria: Output from Retrieval Agent
        template: Output from Template Match Agent
        clinical_notes: Patient's clinical notes (if available)

    Returns:
        Complete letter text
    """
    print(f"    [WRITER] Generating rebuttal letter...")
    start_time = time.time()

    # Format denial info for prompt
    denial_info = {
        "condition": denial_extract.condition_denied,
        "category": denial_extract.denial_category,
        "summary": denial_extract.denial_summary,
        "payer_quotes": denial_extract.payer_quotes,
        "signals": denial_extract.denial_signals,
        "claimed_drg": denial_extract.claimed_drg,
        "denied_drg": denial_extract.denied_drg,
        "reviewer": denial_extract.reviewer_name
    }

    # Format criteria for prompt
    criteria_text = []
    for c in criteria:
        criteria_text.append(f"""
SOURCE: {c.source_citation}
TYPE: {c.criteria_type}
CONTENT: {c.content}
""")

    # Build the prompt
    prompt = WRITER_AGENT_PROMPT.format(
        denial_info_json=json.dumps(denial_info, indent=2),
        criteria_json="\n---\n".join(criteria_text) if criteria_text else "No criteria available",
        template_text=template.template_text,
        clinical_notes=clinical_notes if clinical_notes else "No clinical notes available"
    )

    # Call the LLM
    response = openai_client.chat.completions.create(
        model=AZURE_OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a clinical coding expert writing DRG validation appeal letters. "
                    "Be thorough, specific, and cite clinical values. "
                    "Return only the letter text, no JSON or markdown."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,  # Slightly creative for better writing
        max_tokens=3000
    )

    letter_text = response.choices[0].message.content.strip()

    # Clean up any markdown if present
    if letter_text.startswith("```"):
        letter_text = letter_text.split("```")[1]
        if letter_text.startswith("text") or letter_text.startswith("\n"):
            letter_text = letter_text.split("\n", 1)[1] if "\n" in letter_text else letter_text

    processing_time_ms = int((time.time() - start_time) * 1000)
    print(f"    [WRITER] Generated letter ({len(letter_text)} chars) in {processing_time_ms}ms")

    return letter_text

# =============================================================================
# MAIN PIPELINE
# =============================================================================
# This orchestrates the 4 agents:
#   1. Read unprocessed denial letters from table
#   2. For each letter: Scan -> Retrieve -> Match -> Write
#   3. Save results to draft_letters table
# =============================================================================

def run_pipeline(limit: int = None):
    """
    Run the complete 4-agent pipeline.

    Args:
        limit: Optional limit on number of letters to process (for testing)
    """
    print("=" * 60)
    print("INFERENCE PIPELINE (STANDALONE)")
    print("=" * 60)

    # Step 1: Get unprocessed denial letters
    print("\n[STEP 1] Loading unprocessed denial letters...")

    query = f"""
    SELECT dl.letter_id, dl.letter_text
    FROM {DENIAL_LETTERS_TABLE} dl
    LEFT JOIN {DRAFT_LETTERS_TABLE} dr ON dl.letter_id = dr.letter_id
    WHERE dr.letter_id IS NULL
      AND dl.letter_text IS NOT NULL
      AND LENGTH(dl.letter_text) > 100
    """
    if limit:
        query += f" LIMIT {limit}"

    letters_df = spark.sql(query)
    letters = letters_df.collect()

    if not letters:
        print("  No unprocessed letters found. Pipeline complete.")
        return

    print(f"  Found {len(letters)} letters to process")

    # Step 2: Process each letter through the pipeline
    print("\n[STEP 2] Running 4-agent pipeline...")

    results = []
    for i, row in enumerate(letters, 1):
        print(f"\n  Processing letter {i}/{len(letters)}...")

        letter_id = row.letter_id
        letter_text = row.letter_text

        try:
            # Agent 1: Scan
            denial_extract = run_scan_agent(letter_id, letter_text)

            # Agent 2: Retrieve
            criteria = run_retrieval_agent(denial_extract)

            # Agent 3: Template Match
            template = run_template_match_agent(denial_extract)

            # Agent 4: Write
            # Note: In production, we'd also pass clinical notes from Epic
            draft_text = run_writer_agent(
                denial_extract=denial_extract,
                criteria=criteria,
                template=template,
                clinical_notes=""  # Would come from Epic Clarity
            )

            # Build result
            result = {
                "draft_id": str(uuid.uuid4()),
                "letter_id": letter_id,
                "extract_id": denial_extract.extract_id,
                "template_id": template.template_id,
                "draft_text": draft_text,
                "draft_json": json.dumps(denial_extract.to_dict()),
                "evidence_sources": json.dumps({
                    "criteria_used": [c.criteria_id for c in criteria],
                    "template_used": template.template_id
                }),
                "citations_used": json.dumps([c.source_citation for c in criteria]),
                "confidence_score": denial_extract.confidence_score,
                "status": "pending_review",
                "reviewer_notes": None,
                "revision_count": 0,
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }
            results.append(result)

            print(f"    SUCCESS - Generated {len(draft_text)} char letter")

        except Exception as e:
            print(f"    ERROR processing {letter_id}: {e}")
            continue

    # Step 3: Save results
    if not results:
        print("\n  No letters were successfully processed.")
        return

    print(f"\n[STEP 3] Saving {len(results)} draft letters...")

    if OUTPUT_MODE == "table":
        # Write to Delta table
        schema = StructType([
            StructField("draft_id", StringType(), False),
            StructField("letter_id", StringType(), False),
            StructField("extract_id", StringType(), True),
            StructField("template_id", StringType(), True),
            StructField("draft_text", StringType(), False),
            StructField("draft_json", StringType(), True),
            StructField("evidence_sources", StringType(), True),
            StructField("citations_used", StringType(), True),
            StructField("confidence_score", FloatType(), True),
            StructField("status", StringType(), True),
            StructField("reviewer_notes", StringType(), True),
            StructField("revision_count", IntegerType(), True),
            StructField("created_at", TimestampType(), True),
            StructField("updated_at", TimestampType(), True),
        ])

        df = spark.createDataFrame(results, schema)
        df.write.mode("append").saveAsTable(DRAFT_LETTERS_TABLE)
        print(f"  Wrote {len(results)} drafts to {DRAFT_LETTERS_TABLE}")

    else:
        # Write to local files
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        for result in results:
            file_path = os.path.join(OUTPUT_FOLDER, f"{result['letter_id'][:8]}_rebuttal.txt")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(result['draft_text'])
            print(f"  Wrote: {file_path}")

    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Processed: {len(results)} letters")
    if OUTPUT_MODE == "table":
        print(f"  Results in: {DRAFT_LETTERS_TABLE}")
        print(f"\n  View with: SELECT * FROM {DRAFT_LETTERS_TABLE} ORDER BY created_at DESC")
    else:
        print(f"  Results in: {OUTPUT_FOLDER}")

# =============================================================================
# RUN THE PIPELINE
# =============================================================================

# Process all unprocessed letters (or set limit for testing)
run_pipeline(limit=None)
