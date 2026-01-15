# model/inference.py
# Appeal Engine v2 - Inference (Letter Generation)
#
# PIPELINE OVERVIEW:
# 1. Parse denial letter → Extract structured info via LLM (DRG, payor, denial reasons)
# 2. Vector search → Compare pre-computed embeddings (apples-to-apples)
# 3. Generate appeal → Use winning appeal as template, clinical notes as evidence
# 4. Output → DOCX files for POC, Delta table for production
#
# KEY INSIGHT:
# We match new denials to PAST denials (not appeals) because denial letters
# from the same payor with similar arguments tend to need similar appeals.
# The gold letter's winning appeal becomes our "template to learn from."
#
# RIGOROUS ARCHITECTURE:
# Both new denials AND gold letter denials are embedded using the SAME
# generate_embedding() function in featurization.py. The denial_embedding
# column in the inference table is pre-computed, ensuring true apples-to-apples
# comparison with gold letter embeddings.
#
# Run on Databricks Runtime 15.4 LTS ML

# =============================================================================
# CELL 1: Install Dependencies (run once per cluster)
# =============================================================================
# Only openai is needed here - Document Intelligence was used in featurization.
# The denial text is already extracted and stored in the inference table.
#
# %pip install openai python-docx
# dbutils.library.restartPython()

# =============================================================================
# CELL 2: Imports and Configuration
# =============================================================================
import os
import json
import math
from datetime import datetime, date
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp

# Get or create Spark session (already exists in Databricks notebooks)
spark = SparkSession.builder.getOrCreate()

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# SCOPE_FILTER: Which denials to process
# - "sepsis": Only process sepsis-related denials (our POC scope)
# - "all": Process all denial types (future expansion)
SCOPE_FILTER = "sepsis"

# DRG codes for sepsis cases:
# - 870: Sepsis with MV >96 hours (highest severity/reimbursement)
# - 871: Sepsis without MV >96 hours
# - 872: Sepsis without MCC (lowest severity)
# Payors often try to downgrade from 871 to 872 - that's what we're fighting.
SEPSIS_DRG_CODES = ["870", "871", "872"]

# Minimum cosine similarity to use a gold letter as reference.
# 0.7 means the denial must be reasonably similar.
# Lower = more matches but potentially less relevant templates.
# Higher = fewer matches but higher quality templates.
MATCH_SCORE_THRESHOLD = 0.7

# Default template path - used when no gold letter matches well enough
# This should be a DOCX file in the gold_standard_appeals folder
DEFAULT_TEMPLATE_PATH = "/Workspace/Repos/mijo8881@mercy.net/fudgesicle/utils/gold_standard_appeals/default_sepsis_appeal_template.docx"

# =============================================================================
# CELL 3: Environment Setup
# =============================================================================
# Determine which Unity Catalog to use.
# Same logic as featurization.py - dev defaults to prod catalog.
trgt_cat = os.environ.get('trgt_cat', 'dev')
if trgt_cat == 'dev':
    spark.sql('USE CATALOG prod;')
else:
    spark.sql(f'USE CATALOG {trgt_cat};')

# Table names - must match featurization.py
GOLD_LETTERS_TABLE = f"{trgt_cat}.fin_ds.fudgesicle_gold_letters"
INFERENCE_TABLE = f"{trgt_cat}.fin_ds.fudgesicle_inference"
INFERENCE_SCORE_TABLE = f"{trgt_cat}.fin_ds.fudgesicle_inference_score"
PROPEL_DATA_TABLE = f"{trgt_cat}.fin_ds.fudgesicle_propel_data"

print(f"Catalog: {trgt_cat}")

# =============================================================================
# CELL 3B: Validation Checkpoints
# =============================================================================
# These checkpoints verify that prerequisites are met before processing.

def checkpoint_table_exists(table_name, min_rows=1, required_columns=None):
    """
    CHECKPOINT: Verify a table exists and meets minimum requirements.
    Returns (success, row_count, message)
    """
    try:
        count = spark.sql(f"SELECT COUNT(*) as cnt FROM {table_name}").collect()[0]["cnt"]

        if count < min_rows:
            return (False, count, f"Expected at least {min_rows} rows, found {count}")

        if required_columns:
            df = spark.sql(f"SELECT * FROM {table_name} LIMIT 1")
            missing = set(required_columns) - set(df.columns)
            if missing:
                return (False, count, f"Missing columns: {missing}")

        return (True, count, "OK")
    except Exception as e:
        return (False, 0, str(e))


def checkpoint_prerequisites():
    """
    CHECKPOINT: Verify all required tables exist before running inference.
    """
    print("\n" + "="*60)
    print("CHECKPOINT: Prerequisite Validation")
    print("="*60)

    all_valid = True

    # Check inference table (input)
    success, count, msg = checkpoint_table_exists(
        INFERENCE_TABLE,
        min_rows=1,
        required_columns=["hsp_account_id", "denial_letter_text", "denial_embedding", "is_sepsis"]
    )
    if success:
        print(f"[OK] Inference table: {count} rows")
    else:
        print(f"[FAIL] Inference table: {msg}")
        all_valid = False

    # Check gold letters table
    success, count, msg = checkpoint_table_exists(
        GOLD_LETTERS_TABLE,
        min_rows=1,
        required_columns=["letter_id", "denial_embedding", "rebuttal_text"]
    )
    if success:
        print(f"[OK] Gold letters table: {count} rows")
    else:
        print(f"[FAIL] Gold letters table: {msg}")
        all_valid = False

    # Check propel data table
    success, count, msg = checkpoint_table_exists(
        PROPEL_DATA_TABLE,
        min_rows=0,  # Optional - can run without it
        required_columns=["condition_name", "definition_summary"]
    )
    if success:
        print(f"[OK] Propel data table: {count} rows")
    else:
        print(f"[WARN] Propel data table: {msg} (will proceed without official definitions)")

    # Check sepsis cases exist
    try:
        sepsis_count = spark.sql(f"""
            SELECT COUNT(*) as cnt FROM {INFERENCE_TABLE}
            WHERE is_sepsis = true
        """).collect()[0]["cnt"]
        if sepsis_count > 0:
            print(f"[OK] Sepsis cases to process: {sepsis_count}")
        else:
            print(f"[WARN] No sepsis cases found in inference table")
    except Exception as e:
        print(f"[WARN] Could not count sepsis cases: {e}")

    if all_valid:
        print("\n[CHECKPOINT PASSED] Prerequisites met - ready for inference")
    else:
        print("\n[CHECKPOINT FAILED] Run featurization.py first to populate tables")

    return all_valid


def checkpoint_generation_results(results, expected_min=1):
    """
    CHECKPOINT: Verify letter generation produced expected results.
    """
    print("\n" + "="*60)
    print("CHECKPOINT: Generation Results")
    print("="*60)

    total = len(results)
    successful = sum(1 for r in results if r.get("generated_letter"))
    with_gold_match = sum(1 for r in results if r.get("gold_letter_source_file"))

    print(f"Total cases processed: {total}")
    print(f"Letters generated: {successful}")
    print(f"With gold letter match: {with_gold_match}")

    if successful >= expected_min:
        print(f"\n[CHECKPOINT PASSED] Generated {successful} letters")
        return True
    else:
        print(f"\n[CHECKPOINT FAILED] Expected at least {expected_min} letters, got {successful}")
        return False


# Run prerequisite checkpoint on load
prerequisites_met = checkpoint_prerequisites()

# =============================================================================
# CELL 4: Azure OpenAI Setup
# =============================================================================
from openai import AzureOpenAI

# Load credentials from Databricks secrets
api_key = dbutils.secrets.get(scope='idp_etl', key='az-openai-key1')
azure_endpoint = dbutils.secrets.get(scope='idp_etl', key='az-openai-base')
api_version = '2024-10-21'

# Model names deployed in our Azure OpenAI resource
model = 'gpt-4.1'                    # For parsing denial info and letter generation
# NOTE: Embedding generation moved to featurization.py for rigorous architecture.
# The denial_embedding is pre-computed using text-embedding-ada-002 (1536 dims).

# Initialize the client
client = AzureOpenAI(api_key=api_key, azure_endpoint=azure_endpoint, api_version=api_version)
print(f"Azure OpenAI client initialized (model: {model})")

# =============================================================================
# CELL 4b: Clinical Note Extraction (Pre-summarization)
# =============================================================================
# For long clinical notes, we extract relevant data with timestamps rather than
# passing the entire note to the writer. This:
# 1. Reduces token usage and cost
# 2. Focuses on clinically relevant information
# 3. Preserves important timestamps for each data point
#
# Only notes exceeding NOTE_EXTRACTION_THRESHOLD are extracted; shorter notes
# are passed through unchanged.

NOTE_EXTRACTION_THRESHOLD = 8000  # Characters - notes longer than this get extracted
EXTRACTION_MODEL = 'gpt-4.1'      # Could use cheaper model like gpt-35-turbo

NOTE_EXTRACTION_PROMPT = '''Extract clinically relevant information from this {note_type}.

CRITICAL: For EVERY piece of information you extract, include the associated date/time if available.
Format timestamps consistently as: MM/DD/YYYY HH:MM or MM/DD/YYYY if time not available.

# Clinical Note
{note_text}

# What to Extract (with timestamps)

## Vital Signs (with date/time for each)
- Temperature
- Heart rate
- Blood pressure / MAP
- Respiratory rate
- SpO2 / oxygen requirements

## Laboratory Values (with date/time for each)
- Lactate (CRITICAL for sepsis)
- WBC count
- Creatinine / BUN
- Bilirubin
- Platelet count
- Procalcitonin
- Blood cultures (when drawn, results if available)
- Other relevant labs

## Infection Evidence (with date/time)
- Suspected or confirmed infection source
- Organisms identified
- Antibiotics started (which ones, when)

## Organ Dysfunction (with date/time)
- Cardiovascular (hypotension, vasopressor use)
- Respiratory (mechanical ventilation, oxygen needs)
- Renal (AKI, urine output)
- Hepatic (elevated bilirubin)
- Hematologic (thrombocytopenia, coagulopathy)
- Neurologic (altered mental status, GCS)

## Clinical Events (with date/time)
- Rapid responses or code events
- ICU transfers
- Significant clinical changes
- Procedures performed

## Physician Assessments (with date/time)
- Sepsis mentioned in differential or diagnosis
- Severity assessments
- Clinical reasoning about infection/sepsis

# Output Format
Return a structured summary with timestamps. Example format:

VITAL SIGNS:
- 03/15/2024 08:00: Temp 38.9°C, HR 112, BP 85/52 (MAP 63), RR 24
- 03/15/2024 14:00: Temp 37.8°C, HR 98, BP 95/60 (MAP 72), RR 20

LABS:
- 03/15/2024 06:30: Lactate 4.2, WBC 18.5, Creatinine 2.1
- 03/15/2024 12:00: Lactate 2.8 (improving)

INFECTION:
- 03/15/2024 07:00: Blood cultures drawn
- 03/15/2024 08:30: Started on vancomycin and piperacillin-tazobactam
- 03/16/2024: Cultures positive for E. coli

[Continue for other sections...]

Only include sections that have relevant data. Be thorough but concise.'''


def extract_clinical_data(note_text, note_type):
    """
    Extract clinically relevant data with timestamps from a long clinical note.
    Returns extracted summary if note is long, otherwise returns original text.
    """
    if not note_text or note_text == "Not available" or note_text == "No Note Available":
        return note_text

    # Only extract from notes exceeding threshold
    if len(note_text) < NOTE_EXTRACTION_THRESHOLD:
        return note_text

    try:
        extraction_prompt = NOTE_EXTRACTION_PROMPT.format(
            note_type=note_type,
            note_text=note_text
        )

        response = client.chat.completions.create(
            model=EXTRACTION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a clinical data extraction specialist. "
                        "Extract relevant medical information with precise timestamps. "
                        "Be thorough - do not miss important clinical values."
                    )
                },
                {"role": "user", "content": extraction_prompt}
            ],
            temperature=0.1,  # Low temperature for consistent extraction
            max_tokens=3000   # Extracted summaries should be concise
        )

        extracted = response.choices[0].message.content.strip()
        print(f"    Extracted {note_type}: {len(note_text)} chars → {len(extracted)} chars")
        return extracted

    except Exception as e:
        print(f"    Warning: Extraction failed for {note_type}: {e}")
        # Fall back to truncation if extraction fails
        return note_text[:NOTE_EXTRACTION_THRESHOLD] + "\n\n[Note truncated due to length]"


def extract_notes_for_case(row):
    """
    Extract clinical data from all long notes for a case.
    Returns dict of note_type -> extracted/original text.
    14 sepsis-relevant note types.
    """
    note_types = {
        # Key must match the placeholder name in WRITER_PROMPT
        "discharge_summary": ("discharge_summary_text", "Discharge Summary"),
        "hp_note": ("hp_note_text", "History & Physical"),
        "progress_note": ("progress_note_text", "Progress Notes"),
        "consult_note": ("consult_note_text", "Consult Notes"),
        "ed_notes": ("ed_notes_text", "ED Notes"),
        "initial_assessment": ("initial_assessment_text", "Initial Assessments"),
        "ed_triage": ("ed_triage_text", "ED Triage Notes"),
        "ed_provider_note": ("ed_provider_note_text", "ED Provider Notes"),
        "addendum_note": ("addendum_note_text", "Addendum Note"),
        "hospital_course": ("hospital_course_text", "Hospital Course"),
        "subjective_objective": ("subjective_objective_text", "Subjective & Objective"),
        "assessment_plan": ("assessment_plan_text", "Assessment & Plan Note"),
        "nursing_note": ("nursing_note_text", "Nursing Note"),
        "code_documentation": ("code_documentation_text", "Code Documentation"),
    }

    extracted_notes = {}
    notes_to_extract = []

    # First pass: identify which notes need extraction
    for key, (col_name, display_name) in note_types.items():
        note_text = row.get(col_name, "Not available")
        if note_text and note_text not in ("Not available", "No Note Available"):
            if len(note_text) >= NOTE_EXTRACTION_THRESHOLD:
                notes_to_extract.append((key, col_name, display_name, note_text))
            else:
                extracted_notes[key] = note_text
        else:
            extracted_notes[key] = "Not available"

    # Second pass: extract from long notes
    # Note: Could parallelize this with concurrent.futures for speed
    if notes_to_extract:
        print(f"  Extracting from {len(notes_to_extract)} long notes...")
        for key, col_name, display_name, note_text in notes_to_extract:
            extracted_notes[key] = extract_clinical_data(note_text, display_name)

    return extracted_notes


# =============================================================================
# CELL 5: Create Score Table (run once)
# =============================================================================
# This table stores the generated appeal letters along with metadata
# about how they were generated (which gold letter was used, etc.)
#
# Schema:
# - hsp_account_id: Links back to the original denial case
# - letter_text: The generated appeal letter
# - gold_letter_used: Which gold letter we learned from (for auditability)
# - gold_letter_score: Cosine similarity score (how good was the match?)
# - denial_info_json: Parsed denial info (for debugging/analysis)
create_score_table_sql = f"""
CREATE TABLE IF NOT EXISTS {INFERENCE_SCORE_TABLE} (
    hsp_account_id STRING,
    pat_mrn_id STRING,
    formatted_name STRING,
    discharge_summary_note_id STRING,
    discharge_note_csn_id STRING,
    hp_note_id STRING,
    hp_note_csn_id STRING,
    letter_type STRING,
    letter_text STRING,
    letter_curated_date DATE,
    payor STRING,
    original_drg STRING,
    proposed_drg STRING,
    gold_letter_used STRING,
    gold_letter_score FLOAT,
    pipeline_version STRING,
    insert_tsp TIMESTAMP
)
USING DELTA
COMMENT 'Generated appeal letters'
"""

spark.sql(create_score_table_sql)
print(f"Table {INFERENCE_SCORE_TABLE} ready")

# =============================================================================
# CELL 6: Check for New Records to Process
# =============================================================================
# Find records that haven't been processed yet.
# A record needs processing if:
# 1. It was inserted after the last processing run, OR
# 2. It has never been processed (not in score table)

# Get timestamp of last processing run
try:
    last_processed_ts = spark.sql(f"""
        SELECT COALESCE(MAX(insert_tsp), TIMESTAMP'2020-01-01 00:00:00') AS last_ts
        FROM {INFERENCE_SCORE_TABLE}
    """).collect()[0]["last_ts"]
except Exception:
    # Table might not exist yet on first run
    last_processed_ts = "2020-01-01 00:00:00"

# Count new rows needing processing
n_new_rows = spark.sql(f"""
    WITH scored_accounts AS (
        -- Get all accounts that have already been processed
        SELECT DISTINCT hsp_account_id
        FROM {INFERENCE_SCORE_TABLE}
    )
    SELECT COUNT(*) AS cnt
    FROM {INFERENCE_TABLE} src
    LEFT JOIN scored_accounts sa ON src.hsp_account_id = sa.hsp_account_id
    WHERE src.insert_tsp > TIMESTAMP'{last_processed_ts}'  -- New since last run
       OR sa.hsp_account_id IS NULL                        -- Never processed
""").collect()[0]["cnt"]

print(f"New rows to process: {n_new_rows}")

# =============================================================================
# CELL 7: Load Gold Letters into Memory (for vector search)
# =============================================================================
# We load all gold letters into memory for fast similarity search.
# With ~30 letters, this is efficient. For 1000s, we'd need a vector DB.
print("\nLoading gold standard letters...")

try:
    gold_letters_df = spark.sql(f"""
        SELECT letter_id, source_file, payor, denial_text, appeal_text, denial_embedding, metadata
        FROM {GOLD_LETTERS_TABLE}
    """)
    gold_letters = gold_letters_df.collect()

    # Convert Spark Rows to Python dicts for easier access
    gold_letters_cache = [
        {
            "letter_id": row["letter_id"],
            "source_file": row["source_file"],
            "payor": row["payor"],
            "denial_text": row["denial_text"],
            "appeal_text": row["appeal_text"],
            # Convert Spark array to Python list
            "denial_embedding": list(row["denial_embedding"]) if row["denial_embedding"] else None,
            "metadata": dict(row["metadata"]) if row["metadata"] else {},
        }
        for row in gold_letters
    ]
    print(f"Loaded {len(gold_letters_cache)} gold standard letters")
except Exception as e:
    print(f"Warning: Could not load gold letters: {e}")
    print("Will generate letters without gold letter reference.")
    gold_letters_cache = []

# =============================================================================
# CELL 7A: Load Default Template (fallback when no good match)
# =============================================================================
# The default template is used when vector search doesn't find a good match.
# It provides a generic appeal structure for the condition.
print("\nLoading default template...")

default_template = None

try:
    if os.path.exists(DEFAULT_TEMPLATE_PATH):
        from docx import Document as DocxDocument
        doc = DocxDocument(DEFAULT_TEMPLATE_PATH)
        template_text = "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])

        # Create a pseudo gold letter entry for the default template
        default_template = {
            "letter_id": "default_template",
            "source_file": os.path.basename(DEFAULT_TEMPLATE_PATH),
            "payor": "Generic",
            "denial_text": None,
            "appeal_text": template_text,
            "denial_embedding": None,  # No embedding - not used for matching
            "metadata": {"is_default_template": "true"},
        }
        print(f"  Loaded default template: {len(template_text)} chars")
    else:
        print(f"  Warning: Default template not found at {DEFAULT_TEMPLATE_PATH}")
        print("  Will generate letters without template fallback")
except Exception as e:
    print(f"  Warning: Could not load default template: {e}")

# =============================================================================
# CELL 7B: Load Propel Definitions (official clinical criteria)
# =============================================================================
# Uses definition_summary (LLM-extracted key criteria) for prompts.
# Falls back to definition_text if summary not available.
print("\nLoading Propel clinical definitions...")

propel_definitions = {}

try:
    propel_df = spark.sql(f"""
        SELECT condition_name, definition_summary, definition_text
        FROM {PROPEL_DATA_TABLE}
    """)
    propel_rows = propel_df.collect()

    for row in propel_rows:
        # Prefer summary (concise), fall back to full text if not available
        definition = row["definition_summary"] or row["definition_text"]
        propel_definitions[row["condition_name"]] = definition
        # Log which version we're using
        if row["definition_summary"]:
            print(f"  {row['condition_name']}: using summary ({len(definition)} chars)")
        else:
            print(f"  {row['condition_name']}: using full text ({len(definition)} chars)")

    print(f"Loaded {len(propel_definitions)} definitions: {list(propel_definitions.keys())}")
except Exception as e:
    print(f"Warning: Could not load propel definitions: {e}")
    print("Will generate letters without official definitions.")

# =============================================================================
# CELL 8: [REMOVED - Parsing now done in featurization.py]
# =============================================================================
# Denial parsing (DRG codes, sepsis flag) is now pre-computed in featurization.py
# using a simple key-value format instead of JSON. This eliminates JSON parsing
# errors and saves one LLM call per case at inference time.

# =============================================================================
# CELL 9: Writer Prompt - Generate the Appeal Letter
# =============================================================================
# This is the main generation prompt. Key design decisions:
#
# 1. CLINICAL NOTES FIRST: We prioritize clinical evidence from the actual
#    patient encounter. This is the most defensible evidence.
#
# 2. GOLD LETTER AS GUIDE: If we found a similar past denial, we include
#    the winning appeal as a template. The LLM learns the style/approach
#    but adapts it to THIS patient's specific clinical data.
#
# 3. TEMPLATE STRUCTURE: We enforce the Mercy Hospital letter format
#    for consistency and professional appearance.
WRITER_PROMPT = '''You are a clinical coding expert writing a DRG validation appeal letter for Mercy Hospital.

# Original Denial Letter (read this carefully)
{denial_letter_text}

# Clinical Notes (PRIMARY EVIDENCE - use these first)
# Note: Long notes have been pre-processed to extract relevant clinical data WITH TIMESTAMPS.
# Use these timestamps when citing evidence (e.g., "On 03/15/2024 at 08:00, lactate was 4.2")

## Discharge Summary (Code 5)
{discharge_summary}

## H&P Note (Code 4 - History & Physical - Admission Assessment)
{hp_note}

## Progress Notes (Code 1 - Daily Physician Documentation)
{progress_note}

## Consult Notes (Code 2 - Specialist Consultations)
{consult_note}

## ED Notes (Code 6 - Emergency Department Notes)
{ed_notes}

## Initial Assessments (Code 7 - Early Clinical Picture)
{initial_assessment}

## ED Triage Notes (Code 8 - Arrival Vitals, Chief Complaint)
{ed_triage}

## ED Provider Notes (Code 19 - ED Physician Assessment)
{ed_provider_note}

## Addendum Note (Code 29 - Updates/Corrections)
{addendum_note}

## Hospital Course (Code 32 - Timeline Narrative)
{hospital_course}

## Subjective & Objective (Code 33 - Clinical Findings)
{subjective_objective}

## Assessment & Plan Note (Code 38 - Physician Reasoning)
{assessment_plan}

## Nursing Note (Code 70 - Vital Signs, Observations)
{nursing_note}

## Code Documentation (Code 10000 - Code Events)
{code_documentation}

# Official Clinical Definition (USE THIS - not your general knowledge)
{clinical_definition_section}

# Gold Standard Letter (WINNING REBUTTAL - learn from this)
{gold_letter_section}

# Patient Information
Name: {patient_name}
DOB: {patient_dob}
Hospital Account #: {hsp_account_id}
Date of Service: {date_of_service}
Facility: {facility_name}
Original DRG: {original_drg}
Proposed DRG: {proposed_drg}
Payor: {payor}

# Instructions
{gold_letter_instructions}
1. READ THE DENIAL LETTER - extract the payor address, reviewer name, claim numbers
2. ADDRESS EACH DENIAL ARGUMENT - quote the payer, then refute
3. CITE CLINICAL EVIDENCE from provider notes FIRST (best source)
4. INCLUDE TIMESTAMPS with clinical values (e.g., "On 03/15/2024 at 08:00, lactate was 4.2 mmol/L")
5. Follow the Mercy Hospital template structure exactly
6. Include specific clinical values with units (lactate 2.4 mmol/L, MAP 62 mmHg, etc.)
7. DELETE sections that don't apply to this patient

# Template Structure
Return the complete letter text following this structure:

Mercy Hospital
Payor Audits & Denials Dept
ATTN: Compliance Manager
2115 S Fremont Ave - Ste LL1
Springfield, MO 65804

{current_date}

[PAYOR ADDRESS - extract from denial letter]

First Level Appeal

Beneficiary Name: {patient_name}
DOB: {patient_dob}
Claim reference #: [EXTRACT FROM DENIAL LETTER]
Hospital Account #: {hsp_account_id}
Date of Service: {date_of_service}

Dear [REVIEWER - extract from denial letter]:

[Opening paragraph about receiving DRG review...]

Justification for Appeal:
[Why we disagree...]

Rationale:
[Quote payer's argument, then provide clinical evidence...]

Infection Source: [...]
Organ Dysfunction: [List each with values...]
SIRS Criteria Met: [List each with values...]
[Other relevant sections...]

Hospital Course:
[Narrative from clinical notes...]

[Summary paragraph...]

Conclusion:
We anticipate our original DRG of {original_drg} will be approved.

[Contact info and signature...]

Return ONLY the letter text, no JSON wrapper.'''

# =============================================================================
# CELL 10: Main Processing Loop
# =============================================================================
# Process each denial case one at a time.
# Parsing was moved to featurization.py - we use pre-computed columns here.
# For each case:
# 1. Check if it's in scope (using pre-computed is_sepsis flag)
# 2. Find the best matching gold letter (vector search)
# 3. Generate the appeal

if n_new_rows == 0:
    print("No new rows to process")
else:
    print(f"\nProcessing {n_new_rows} rows...")

    # Pull unprocessed rows from the inference table
    df = spark.sql(f"""
        WITH scored_accounts AS (
            SELECT DISTINCT hsp_account_id
            FROM {INFERENCE_SCORE_TABLE}
        )
        SELECT src.*
        FROM {INFERENCE_TABLE} src
        LEFT JOIN scored_accounts sa ON src.hsp_account_id = sa.hsp_account_id
        WHERE src.insert_tsp > TIMESTAMP'{last_processed_ts}'
           OR sa.hsp_account_id IS NULL
    """).toPandas()

    print(f"Pulled {len(df)} rows for processing")

    # Collect results for all processed rows
    results = []

    # Process each row individually
    for idx, row in df.iterrows():
        hsp_account_id = row.get("hsp_account_id", "unknown")
        print(f"\n{'='*60}")
        print(f"Processing {idx+1}/{len(df)}: {hsp_account_id}")
        print(f"{'='*60}")

        # Initialize result dict for this case
        result = {
            "hsp_account_id": hsp_account_id,
            "status": None,
            "letter_text": None,
            "gold_letter_used": None,
            "gold_letter_score": None,
        }

        # Get pre-computed values from featurization
        denial_text = row.get("denial_letter_text", "")
        original_drg = row.get("original_drg", "")
        proposed_drg = row.get("proposed_drg", "")
        is_sepsis = row.get("is_sepsis", False)
        payor = row.get("payor", "Unknown")

        print(f"  DRG: {original_drg} → {proposed_drg} | Payor: {payor} | Sepsis: {is_sepsis}")

        # Skip if no denial text (can't process without it)
        if not denial_text or str(denial_text).strip() == "" or denial_text is None:
            result["status"] = "no_denial_text"
            result["error"] = "No denial letter text"
            results.append(result)
            print("  SKIP: No denial letter text")
            continue

        # ---------------------------------------------------------------------
        # STEP 1: Check if case is in scope (using pre-computed flag)
        # ---------------------------------------------------------------------
        if SCOPE_FILTER == "sepsis":
            if not is_sepsis:
                result["status"] = "out_of_scope"
                result["reason"] = "Not sepsis-related"
                results.append(result)
                print("  SKIP: Not sepsis-related (pre-computed)")
                continue

        # ---------------------------------------------------------------------
        # STEP 2: Vector search for best matching gold letter
        # ---------------------------------------------------------------------
        # RIGOROUS ARCHITECTURE:
        # The denial_embedding was pre-computed in featurization.py using the
        # SAME generate_embedding() function as gold letters. This ensures
        # apples-to-apples comparison (no embedding generation here).
        print("  Step 1: Finding similar gold standard letter...")

        gold_letter = None        # Will hold the best match (if found)
        gold_letter_score = 0.0   # Cosine similarity score

        if gold_letters_cache:
            # Get pre-computed embedding from inference table (computed in featurization.py)
            query_embedding = row.get("denial_embedding")

            if query_embedding is None:
                print("  WARNING: No denial_embedding in inference table - skipping vector search")
                print("  (Re-run featurization.py to compute embeddings)")
            else:
                # Convert to list if it's a Spark array
                if hasattr(query_embedding, 'tolist'):
                    query_embedding = query_embedding.tolist()
                elif not isinstance(query_embedding, list):
                    query_embedding = list(query_embedding)

                print(f"  Using pre-computed embedding ({len(query_embedding)} dims)")

                # Compare against all gold letters using cosine similarity
                best_score = 0.0
                best_match = None

                for letter in gold_letters_cache:
                    if letter["denial_embedding"]:
                        # Cosine similarity = dot(a,b) / (||a|| * ||b||)
                        vec1 = query_embedding
                        vec2 = letter["denial_embedding"]

                        # Dot product
                        dot_product = sum(a * b for a, b in zip(vec1, vec2))

                        # Vector norms (magnitudes)
                        norm1 = math.sqrt(sum(a * a for a in vec1))
                        norm2 = math.sqrt(sum(b * b for b in vec2))

                        # Cosine similarity (avoid div by zero)
                        if norm1 > 0 and norm2 > 0:
                            similarity = dot_product / (norm1 * norm2)
                            if similarity > best_score:
                                best_score = similarity
                                best_match = letter

                gold_letter_score = best_score

                # Only use the match if it meets our quality threshold
                if best_match and best_score >= MATCH_SCORE_THRESHOLD:
                    gold_letter = best_match
                    print(f"  Found match: {gold_letter['source_file']} | Payor: {gold_letter.get('payor', 'Unknown')} | Score: {best_score:.3f}")
                else:
                    # No good match - use default template if available
                    if default_template:
                        gold_letter = default_template
                        gold_letter_score = 0.0  # Indicate this is a fallback, not a match
                        print(f"  No good match (best: {best_score:.3f}) - using default template")
                    else:
                        print(f"  No good match (best score: {best_score:.3f}, threshold: {MATCH_SCORE_THRESHOLD})")

        else:
            # No gold letters loaded - try default template
            if default_template:
                gold_letter = default_template
                gold_letter_score = 0.0
                print("  No gold letters loaded - using default template")
            else:
                print("  No gold letters loaded and no default template")

        # Record which gold letter we used (for auditability)
        result["gold_letter_used"] = gold_letter["letter_id"] if gold_letter else None
        result["gold_letter_source_file"] = gold_letter["source_file"] if gold_letter else None
        result["gold_letter_score"] = gold_letter_score

        # ---------------------------------------------------------------------
        # STEP 3: Generate the appeal letter
        # ---------------------------------------------------------------------
        # Combine everything into the Writer prompt and generate.
        print("  Step 2: Generating appeal letter...")

        current_date_str = date.today().strftime("%m/%d/%Y")

        # Build the gold letter section for the prompt
        # Differentiate between matched gold letter vs default template
        if gold_letter:
            is_default = gold_letter.get("metadata", {}).get("is_default_template") == "true"

            if is_default:
                # Using default template (no good match found)
                gold_letter_section = f"""## APPEAL TEMPLATE - USE AS STRUCTURAL GUIDE
Source: {gold_letter.get('source_file', 'Default Template')}

### Template Structure:
{gold_letter['appeal_text']}
"""
                gold_letter_instructions = """**NOTE: No closely matching past appeal was found. A default template is provided above.**
- Use the template structure and formatting as a guide
- Adapt the language and arguments to this specific patient's clinical situation
- Focus on the clinical evidence from this patient's notes

"""
            else:
                # Using matched gold letter
                gold_letter_section = f"""## THIS LETTER WON A SIMILAR APPEAL - LEARN FROM IT
Source: {gold_letter.get('source_file', 'Unknown')}
Payor: {gold_letter.get('payor', 'Unknown')}
Match Score: {gold_letter_score:.3f}

### Winning Appeal Text:
{gold_letter['appeal_text']}
"""
                gold_letter_instructions = """**CRITICAL: A gold standard letter that won a similar appeal is provided above.**
- Study how it structures arguments and presents clinical evidence
- Emulate its persuasive techniques and medical reasoning
- Adapt its successful patterns to this patient's specific situation
- Do NOT copy verbatim - adapt the approach with this patient's actual clinical data

"""
        else:
            gold_letter_section = "No similar winning appeal or template available. Generate using standard medical appeal format."
            gold_letter_instructions = ""

        # Build the clinical definition section
        # For sepsis cases, include the official Propel definition
        if is_sepsis and "sepsis" in propel_definitions:
            clinical_definition_section = f"""## OFFICIAL SEPSIS DEFINITION (from Propel)
Use these criteria when arguing the patient meets sepsis diagnosis:

{propel_definitions['sepsis']}
"""
        else:
            clinical_definition_section = "No specific clinical definition loaded for this condition."

        # ---------------------------------------------------------------------
        # STEP 2b: Extract clinical data from long notes (with timestamps)
        # ---------------------------------------------------------------------
        # For notes exceeding NOTE_EXTRACTION_THRESHOLD, we extract relevant
        # clinical data with timestamps rather than passing the entire note.
        # This reduces token usage and focuses on clinically relevant info.
        extracted_notes = extract_notes_for_case(row)

        # Assemble the full Writer prompt (using extracted/processed notes)
        # All 14 sepsis-relevant note types
        writer_prompt = WRITER_PROMPT.format(
            denial_letter_text=denial_text,
            # 14 note types
            discharge_summary=extracted_notes.get("discharge_summary", "Not available"),
            hp_note=extracted_notes.get("hp_note", "Not available"),
            progress_note=extracted_notes.get("progress_note", "Not available"),
            consult_note=extracted_notes.get("consult_note", "Not available"),
            ed_notes=extracted_notes.get("ed_notes", "Not available"),
            initial_assessment=extracted_notes.get("initial_assessment", "Not available"),
            ed_triage=extracted_notes.get("ed_triage", "Not available"),
            ed_provider_note=extracted_notes.get("ed_provider_note", "Not available"),
            addendum_note=extracted_notes.get("addendum_note", "Not available"),
            hospital_course=extracted_notes.get("hospital_course", "Not available"),
            subjective_objective=extracted_notes.get("subjective_objective", "Not available"),
            assessment_plan=extracted_notes.get("assessment_plan", "Not available"),
            nursing_note=extracted_notes.get("nursing_note", "Not available"),
            code_documentation=extracted_notes.get("code_documentation", "Not available"),
            # Other prompt fields
            clinical_definition_section=clinical_definition_section,
            gold_letter_section=gold_letter_section,
            gold_letter_instructions=gold_letter_instructions,
            patient_name=row.get("formatted_name", ""),
            patient_dob=row.get("formatted_birthdate", ""),
            hsp_account_id=hsp_account_id,
            date_of_service=row.get("formatted_date_of_service", ""),
            facility_name=row.get("facility_name", "Mercy Hospital"),
            original_drg=original_drg or "Unknown",
            proposed_drg=proposed_drg or "Unknown",
            payor=payor,
            current_date=current_date_str,
        )

        # Generate the appeal letter
        writer_response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a clinical coding expert writing DRG appeal letters. "
                        "Prioritize evidence from provider notes. Be thorough and specific."
                    )
                },
                {"role": "user", "content": writer_prompt}
            ],
            temperature=0.2,  # Slight variation for natural language
            max_tokens=4000   # Letters can be long
        )

        letter_text = writer_response.choices[0].message.content.strip()
        result["status"] = "success"
        result["letter_text"] = letter_text
        results.append(result)

        print(f"  SUCCESS: Generated {len(letter_text)} character letter")

    # -------------------------------------------------------------------------
    # SUMMARY - Print processing stats
    # -------------------------------------------------------------------------
    success = sum(1 for r in results if r["status"] == "success")
    out_of_scope = sum(1 for r in results if r["status"] == "out_of_scope")
    errors = sum(1 for r in results if r["status"] not in ("success", "out_of_scope"))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Success: {success}")
    print(f"  Out of Scope: {out_of_scope}")
    print(f"  Errors/No Data: {errors}")

# =============================================================================
# CELL 11: Write Results to Delta Table
# =============================================================================
# For production: Set WRITE_TO_TABLE = True to persist results.
# The score table becomes the source of truth for generated letters.
WRITE_TO_TABLE = False

if n_new_rows > 0 and 'results' in dir() and len(results) > 0:
    # Build output rows from successful results
    output_rows = []

    for r, orig_row in zip(results, df.itertuples()):
        if r["status"] == "success":
            output_rows.append({
                "hsp_account_id": r["hsp_account_id"],
                "pat_mrn_id": getattr(orig_row, 'pat_mrn_id', None),
                "formatted_name": getattr(orig_row, 'formatted_name', None),
                "discharge_summary_note_id": getattr(orig_row, 'discharge_summary_note_id', None),
                "discharge_note_csn_id": getattr(orig_row, 'discharge_note_csn_id', None),
                "hp_note_id": getattr(orig_row, 'hp_note_id', None),
                "hp_note_csn_id": getattr(orig_row, 'hp_note_csn_id', None),
                "letter_type": "Sepsis_v2",  # Identifies this pipeline version
                "letter_text": r["letter_text"],
                "letter_curated_date": datetime.now().date(),
                "payor": getattr(orig_row, 'payor', 'Unknown'),
                "original_drg": getattr(orig_row, 'original_drg', None),
                "proposed_drg": getattr(orig_row, 'proposed_drg', None),
                "gold_letter_used": r.get("gold_letter_used"),
                "gold_letter_source_file": r.get("gold_letter_source_file"),
                "gold_letter_score": r.get("gold_letter_score"),
                "pipeline_version": "appeal_engine_v2",
            })

    if output_rows:
        print(f"\n{len(output_rows)} letters ready to write")

        if WRITE_TO_TABLE:
            import pandas as pd
            output_df = pd.DataFrame(output_rows)
            spark_df = spark.createDataFrame(output_df)
            spark_df = spark_df.withColumn("insert_tsp", current_timestamp())
            spark_df.write.mode("append").saveAsTable(INFERENCE_SCORE_TABLE)
            print(f"Wrote {len(output_df)} letters to {INFERENCE_SCORE_TABLE}")
        else:
            print("To write to table, set WRITE_TO_TABLE = True")

        # Preview the first letter (truncated for display)
        print(f"\n{'='*60}")
        print("PREVIEW: First Generated Letter")
        print(f"{'='*60}")
        print(output_rows[0]["letter_text"][:2000])
        print("...")
    else:
        print("No successful letters to write")

    # Validate generation results
    checkpoint_generation_results(output_rows, expected_min=1)

# =============================================================================
# CELL 12: POC - Export to DOCX for User Feedback
# =============================================================================
# For the POC phase, we export letters as Word documents.
# This lets users review and provide feedback on quality.
# In production, we'd skip this and just write to the Delta table.
EXPORT_TO_DOCX = True
DOCX_OUTPUT_BASE = "/Workspace/Repos/mijo8881@mercy.net/fudgesicle/utils/outputs"

if EXPORT_TO_DOCX and 'output_rows' in dir() and len(output_rows) > 0:
    print(f"\n{'='*60}")
    print("EXPORTING TO DOCX")
    print(f"{'='*60}")

    # Import python-docx for Word document creation
    from docx import Document
    from docx.shared import Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    # Create timestamped output folder for this run
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    DOCX_OUTPUT_PATH = os.path.join(DOCX_OUTPUT_BASE, f"output_{run_timestamp}")
    os.makedirs(DOCX_OUTPUT_PATH, exist_ok=True)
    print(f"Output folder: output_{run_timestamp}")

    # Create one DOCX per letter
    for row in output_rows:
        hsp_account_id = row["hsp_account_id"]
        patient_name = row.get("formatted_name", "Unknown")
        letter_text = row["letter_text"]

        # Create new Word document
        doc = Document()

        # Add title
        title = doc.add_heading(f'Appeal Letter', level=1)

        # Add metadata section (for reviewer context - remove before sending)
        meta = doc.add_paragraph()
        meta.add_run("Generated: ").bold = True
        meta.add_run(f"{datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        meta.add_run("Patient: ").bold = True
        meta.add_run(f"{patient_name}\n")
        meta.add_run("Payor: ").bold = True
        meta.add_run(f"{row.get('payor', 'Unknown')}\n")
        meta.add_run("DRG: ").bold = True
        meta.add_run(f"{row.get('original_drg', '?')} → {row.get('proposed_drg', '?')}\n")
        meta.add_run("Gold Letter: ").bold = True
        if row.get("gold_letter_source_file"):
            meta.add_run(f"{row['gold_letter_source_file']} (score: placeholder)\n")
        else:
            meta.add_run("None\n")

        # Visual separator
        doc.add_paragraph("─" * 60)

        # Add the actual letter content (split by double newlines for paragraphs)
        for paragraph in letter_text.split('\n\n'):
            if paragraph.strip():
                p = doc.add_paragraph(paragraph.strip())
                p.paragraph_format.space_after = Pt(12)

        # Generate safe filename (remove special characters)
        safe_name = "".join(c for c in patient_name if c.isalnum() or c in (' ', '-', '_')).strip()
        filename = f"{hsp_account_id}_{safe_name}_appeal.docx"
        filepath = os.path.join(DOCX_OUTPUT_PATH, filename)

        # Save the document
        doc.save(filepath)
        print(f"  Saved: {filename}")

    print(f"\nDOCX files saved to: {DOCX_OUTPUT_PATH}")

# =============================================================================
# CELL FINAL: Summary
# =============================================================================
print("\n" + "="*60)
print("INFERENCE COMPLETE")
print("="*60)

if 'output_rows' in dir():
    successful = len([r for r in output_rows if r.get('letter_text')])
    print(f"Letters generated: {successful}")
    if EXPORT_TO_DOCX and 'DOCX_OUTPUT_PATH' in dir():
        print(f"DOCX files exported to: {DOCX_OUTPUT_PATH}")
    if WRITE_TO_TABLE:
        print(f"Results written to: {INFERENCE_SCORE_TABLE}")
else:
    print("No letters were generated - check prerequisites and input data")

print("\nInference complete.")
