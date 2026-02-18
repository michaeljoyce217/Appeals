# poc/inference.py
# Rebuttal Engine v2 - Inference (Letter Generation)
#
# MLFlow Structure: All inference and output happens here
# Input: fudgesicle_inference table (from featurization.py)
# Output: fudgesicle_inference_score table with generated letters
#
# Pipeline: Parser Agent → Research Agent → Reference Agent → Writer Agent

import os
import json
from datetime import datetime, date
from typing import Dict, Any, List, Optional

# =============================================================================
# CELL 1: Install Dependencies (run once)
# =============================================================================
# %pip install openai
# %restart_python

# =============================================================================
# CELL 2: Imports and Configuration
# =============================================================================
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp
from openai import AzureOpenAI

spark = SparkSession.builder.getOrCreate()

# -----------------------------------------------------------------------------
# SCOPE FILTER CONSTANTS - Must match featurization.py
# -----------------------------------------------------------------------------
SCOPE_FILTER = "sepsis"  # "sepsis" | "all"

SEPSIS_DRG_CODES = ["870", "871", "872"]
SEPSIS_ICD10_CODES = [
    "A41.9", "A41.0", "A41.1", "A41.2", "A41.50", "A41.51", "A41.52", "A41.53",
    "R65.20", "R65.21",
]

# Evidence priority (highest to lowest importance)
EVIDENCE_PRIORITY = [
    "provider_notes",      # Discharge summary, H&P - BEST source
    "structured_data",     # Labs, vitals - backup
    "inference",           # Our conclusions - LEAST important
]

# =============================================================================
# CELL 3: Environment Configuration
# =============================================================================
trgt_cat = os.environ.get('trgt_cat', 'dev')
spark.sql('use catalog prod;') if trgt_cat == 'dev' else spark.sql(f'use catalog {trgt_cat};')

# Azure OpenAI setup
api_key = dbutils.secrets.get(scope='idp_etl', key='az-openai-key1')
azure_endpoint = dbutils.secrets.get(scope='idp_etl', key='az-openai-base')
api_version = '2024-10-21'
model = 'gpt-4.1'

client = AzureOpenAI(api_key=api_key, azure_endpoint=azure_endpoint, api_version=api_version)
print(f"Azure OpenAI client initialized (model: {model})")

# =============================================================================
# CELL 4: Check for New Records
# =============================================================================
last_processed_ts = spark.sql(f"""
    SELECT COALESCE(MAX(insert_tsp), TIMESTAMP'2025-01-01 00:00:00') AS last_ts
    FROM {trgt_cat}.fin_ds.fudgesicle_inference_score
""").collect()[0]["last_ts"]

n_new_rows = spark.sql(f"""
    WITH scored_accounts AS (
        SELECT DISTINCT hsp_account_id
        FROM {trgt_cat}.fin_ds.fudgesicle_inference_score
    )
    SELECT COUNT(*) AS cnt
    FROM {trgt_cat}.fin_ds.fudgesicle_inference src
    LEFT JOIN scored_accounts sa ON src.hsp_account_id = sa.hsp_account_id
    WHERE src.insert_tsp > TIMESTAMP'{last_processed_ts}'
       OR sa.hsp_account_id IS NULL
""").collect()[0]["cnt"]

print(f"New rows to process: {n_new_rows}")

# =============================================================================
# CELL 5: Parser Agent - Extract Denial Info (LLM, not regex!)
# =============================================================================
PARSER_PROMPT = '''You are a medical billing expert extracting information from denial letters.

# Task
Extract ALL relevant information from this denial letter into structured JSON.

# Denial Letter
{denial_letter_text}

# Output Format
Return ONLY valid JSON (no markdown):
{{
  "denial_date": "YYYY-MM-DD or null",
  "payer_name": "Insurance company name",
  "payer_address": "Full mailing address or null",
  "reviewer_name": "Name and credentials or null",
  "original_drg": "Billed DRG (e.g., '871')",
  "proposed_drg": "Payer's proposed DRG (e.g., '872')",
  "administrative_data": {{
    "claim_reference_number": "Primary claim/reference number",
    "member_id": "Patient member ID",
    "authorization_number": "Prior auth number or null",
    "date_of_service": "Admission date or date range",
    "other_identifiers": {{}}
  }},
  "denial_reasons": [{{
    "type": "clinical_validation | medical_necessity | level_of_care | coding | other",
    "summary": "Brief summary",
    "specific_arguments": ["Each specific argument made"],
    "payer_quote": "Direct quote if available"
  }}],
  "is_sepsis_related": true/false,
  "is_single_issue": true/false
}}'''


def parse_denial_letter(denial_text: str) -> Dict[str, Any]:
    """Parse denial letter using LLM to extract structured info."""
    if not denial_text or denial_text.strip() == "":
        return {"error": "No denial letter text provided", "is_sepsis_related": None}

    prompt = PARSER_PROMPT.format(denial_letter_text=denial_text)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Extract information accurately. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=2000
    )

    raw = response.choices[0].message.content.strip()

    # Clean JSON from markdown
    json_str = raw
    if "```json" in json_str:
        json_str = json_str.split("```json")[1].split("```")[0].strip()
    elif "```" in json_str:
        json_str = json_str.split("```")[1].split("```")[0].strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse failed: {e}", "raw_response": raw}


def check_scope(denial_info: Dict[str, Any]) -> tuple:
    """Check if denial is in scope. Returns (in_scope, reason)."""
    if SCOPE_FILTER == "all":
        return True, None

    if SCOPE_FILTER == "sepsis":
        if denial_info.get("is_sepsis_related"):
            return True, None
        else:
            return False, "Not sepsis-related (scope_filter='sepsis')"

    return True, None


# =============================================================================
# CELL 6: Research Agent - Find Relevant Criteria & Filter Structured Data
# =============================================================================
# Default criteria by denial/condition type (used when Propel not available)
# These are GENERAL - not sepsis-specific
DEFAULT_CRITERIA_BY_TYPE = {
    "sepsis": {
        "labs": ["LACTATE", "WBC", "PROCALCITONIN", "CRP", "CREATININE", "BILIRUBIN", "PLATELETS", "INR", "APTT"],
        "vitals": ["TEMPERATURE", "HEART RATE", "RESPIRATORY RATE", "MAP", "SYSTOLIC BP", "O2 SAT", "GCS"],
        "meds": ["ANTIBIOTIC", "VASOPRESSOR", "NOREPINEPHRINE", "EPINEPHRINE", "FLUID"],
        "procedures": ["BLOOD CULTURE", "CENTRAL LINE", "ARTERIAL LINE"],
        "thresholds": {
            "LACTATE": {"threshold": 2.0, "unit": "mmol/L", "direction": ">"},
            "WBC": {"threshold_high": 12000, "threshold_low": 4000, "unit": "/uL"},
            "TEMPERATURE": {"threshold_high": 38.3, "threshold_low": 36.0, "unit": "°C"},
            "MAP": {"threshold": 65, "unit": "mmHg", "direction": "<"},
            "CREATININE": {"threshold": 2.0, "unit": "mg/dL", "direction": ">"},
        }
    },
    "pneumonia": {
        "labs": ["WBC", "PROCALCITONIN", "CRP", "LACTATE", "BNP"],
        "vitals": ["TEMPERATURE", "RESPIRATORY RATE", "O2 SAT", "HEART RATE"],
        "meds": ["ANTIBIOTIC", "OXYGEN", "BRONCHODILATOR"],
        "procedures": ["CHEST X-RAY", "CT CHEST", "SPUTUM CULTURE", "BLOOD CULTURE"],
        "thresholds": {}
    },
    "cardiac": {
        "labs": ["TROPONIN", "BNP", "CK-MB", "CREATININE", "POTASSIUM"],
        "vitals": ["HEART RATE", "SYSTOLIC BP", "DIASTOLIC BP", "O2 SAT"],
        "meds": ["ANTICOAGULANT", "BETA BLOCKER", "ACE INHIBITOR", "STATIN"],
        "procedures": ["EKG", "ECHO", "CARDIAC CATH", "STRESS TEST"],
        "thresholds": {}
    },
    "default": {
        "labs": ["WBC", "HEMOGLOBIN", "CREATININE", "GLUCOSE", "LACTATE"],
        "vitals": ["TEMPERATURE", "HEART RATE", "RESPIRATORY RATE", "SYSTOLIC BP", "O2 SAT"],
        "meds": [],
        "procedures": [],
        "thresholds": {}
    }
}


def determine_condition_type(denial_info: Dict[str, Any]) -> str:
    """
    Determine the condition type from denial info.
    Used to select appropriate criteria when Propel not available.
    """
    # Check if sepsis-related
    if denial_info.get("is_sepsis_related"):
        return "sepsis"

    # Check DRG codes
    original_drg = denial_info.get("original_drg", "")
    proposed_drg = denial_info.get("proposed_drg", "")

    # Sepsis DRGs
    if original_drg in ["870", "871", "872"] or proposed_drg in ["870", "871", "872"]:
        return "sepsis"

    # Pneumonia DRGs (examples)
    if original_drg in ["177", "178", "179", "193", "194", "195"]:
        return "pneumonia"

    # Cardiac DRGs (examples)
    if original_drg in ["280", "281", "282", "283", "284", "285"]:
        return "cardiac"

    # Check denial reasons for keywords
    denial_reasons = denial_info.get("denial_reasons", [])
    for reason in denial_reasons:
        summary = (reason.get("summary", "") + " " + " ".join(reason.get("specific_arguments", []))).lower()
        if "sepsis" in summary or "septic" in summary:
            return "sepsis"
        if "pneumonia" in summary:
            return "pneumonia"
        if "cardiac" in summary or "heart" in summary or "mi" in summary:
            return "cardiac"

    return "default"


def research_criteria(denial_info: Dict[str, Any], clinical_notes: str,
                      structured_data_json: Optional[str] = None) -> Dict[str, Any]:
    """
    Research relevant clinical criteria and filter structured data.

    Strategy:
    1. If Propel docs available: Query for specific criteria (FUTURE)
    2. If no Propel: Use default criteria based on condition type
    3. Filter structured data to what's relevant for this denial

    Args:
        denial_info: Parsed denial information
        clinical_notes: Clinical notes text
        structured_data_json: JSON string of all structured data from featurization

    Returns:
        - relevant_criteria: Criteria that apply to this denial
        - filtered_structured_data: Only the relevant labs/vitals/meds
        - suggested_arguments: Arguments to make based on criteria
    """
    # Determine condition type from denial
    condition_type = determine_condition_type(denial_info)

    # Get criteria for this condition type
    # FUTURE: Query Propel/reference_documents table here
    criteria = DEFAULT_CRITERIA_BY_TYPE.get(condition_type, DEFAULT_CRITERIA_BY_TYPE["default"])

    # Parse and filter structured data
    filtered_data = {"labs": [], "vitals": [], "meds": [], "procedures": []}

    if structured_data_json:
        try:
            import json
            all_data = json.loads(structured_data_json)

            for record in all_data:
                data_type = record.get("data_type", "")

                if data_type == "labs":
                    component = record.get("component_name", "")
                    if any(c in component.upper() for c in criteria.get("labs", [])):
                        filtered_data["labs"].append(record)

                elif data_type == "vitals":
                    vital = record.get("vital_name", "")
                    if any(v in vital.upper() for v in criteria.get("vitals", [])):
                        filtered_data["vitals"].append(record)

                elif data_type == "meds":
                    med = record.get("medication_name", "")
                    if any(m in med.upper() for m in criteria.get("meds", [])):
                        filtered_data["meds"].append(record)

                elif data_type == "procedures":
                    proc = record.get("proc_name", "")
                    if any(p in proc.upper() for p in criteria.get("procedures", [])):
                        filtered_data["procedures"].append(record)

        except Exception as e:
            print(f"Warning: Failed to parse structured data: {e}")

    return {
        "condition_type": condition_type,
        "relevant_criteria": criteria,
        "filtered_structured_data": filtered_data,
        "suggested_arguments": [],  # FUTURE: Generate from Propel criteria
        "_status": "using_default_criteria" if not False else "using_propel",  # FUTURE: Check Propel availability
        "_propel_available": False,  # FUTURE: Set based on table check
    }


# =============================================================================
# CELL 7: Reference Agent (STUB - falls back to template)
# =============================================================================
def find_gold_letters(denial_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Find matching gold standard letters.
    STUB: Returns fallback to template until gold letters are loaded.
    """
    return {
        "matched_letters": [],
        "use_template_fallback": True,
        "fallback_reason": "No gold standard letters loaded yet",
        "_status": "stub - no gold letters loaded"
    }


# =============================================================================
# CELL 8: Writer Agent - Generate Rebuttal Letter
# =============================================================================
WRITER_PROMPT = '''You are a clinical coding expert writing a DRG validation appeal letter for Mercy Hospital.

# Denial Information
{denial_info_json}

# Clinical Notes (PRIMARY EVIDENCE - use these first)
## Discharge Summary
{discharge_summary}

## H&P Note
{hp_note}

# Patient Information
{patient_info_json}

# Instructions
1. ADDRESS EACH DENIAL ARGUMENT - quote the payer, then refute
2. CITE CLINICAL EVIDENCE from provider notes FIRST (best source)
3. Follow the Mercy Hospital template structure exactly
4. Include specific clinical values (lactate 2.4, MAP 62, etc.)
5. DELETE sections that don't apply to this patient

# Template Structure
Return the complete letter text following this structure:

Mercy Hospital
Payor Audits & Denials Dept
ATTN: Compliance Manager
2115 S Fremont Ave - Ste LL1
Springfield, MO 65804

{current_date}

[PAYOR ADDRESS]

First Level Appeal

Beneficiary Name: [NAME]
DOB: [DOB]
Claim reference #: [CLAIM_REF]
Hospital Account #: [HSP_ACCOUNT_ID]
Date of Service: [DOS]

Dear [REVIEWER]:

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
We anticipate our original DRG of [X] will be approved.

[Contact info and signature...]

Return ONLY the letter text, no JSON wrapper.'''


def generate_letter(denial_info: Dict[str, Any], row: Dict[str, Any]) -> str:
    """Generate rebuttal letter using LLM."""
    current_date = date.today().strftime("%m/%d/%Y")

    patient_info = {
        "formatted_name": row.get("formatted_name", ""),
        "formatted_birthdate": row.get("formatted_birthdate", ""),
        "hsp_account_id": row.get("hsp_account_id", ""),
        "claim_number": row.get("claim_number", ""),
        "formatted_date_of_service": row.get("formatted_date_of_service", ""),
        "facility_name": row.get("facility_name", ""),
        "number_of_midnights": row.get("number_of_midnights", ""),
        "code": row.get("code", ""),
        "dx_name": row.get("dx_name", ""),
    }

    prompt = WRITER_PROMPT.format(
        denial_info_json=json.dumps(denial_info, indent=2, default=str),
        discharge_summary=row.get("discharge_summary_text", "Not available"),
        hp_note=row.get("hp_note_text", "Not available"),
        patient_info_json=json.dumps(patient_info, indent=2),
        current_date=current_date,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a clinical coding expert writing DRG appeal letters. "
                    "Prioritize evidence from provider notes. Be thorough and specific."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=4000
    )

    return response.choices[0].message.content.strip()


# =============================================================================
# CELL 9: Process Single Row
# =============================================================================
def process_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single row through the full pipeline.

    Pipeline: Parser → Research → Reference → Writer
    """
    hsp_account_id = row.get("hsp_account_id", "unknown")
    result = {
        "hsp_account_id": hsp_account_id,
        "status": None,
        "letter_text": None,
        "denial_info": None,
    }

    # Step 1: Parse denial letter
    denial_text = row.get("denial_letter_text", "")
    denial_info = parse_denial_letter(denial_text)

    if "error" in denial_info:
        result["status"] = "parse_error"
        result["error"] = denial_info["error"]
        return result

    result["denial_info"] = denial_info

    # Step 2: Check scope
    in_scope, scope_reason = check_scope(denial_info)
    if not in_scope:
        result["status"] = "out_of_scope"
        result["reason"] = scope_reason
        return result

    # Step 3: Research criteria and filter structured data
    criteria = research_criteria(
        denial_info,
        row.get("discharge_summary_text", ""),
        row.get("structured_data_json")  # Pass structured data for filtering
    )
    print(f"    Condition type: {criteria.get('condition_type', 'unknown')}")

    # Step 4: Find gold letters (stub)
    gold_letters = find_gold_letters(denial_info)

    # Step 5: Generate letter
    try:
        letter_text = generate_letter(denial_info, row)
        result["status"] = "success"
        result["letter_text"] = letter_text
    except Exception as e:
        result["status"] = "generation_error"
        result["error"] = str(e)

    return result


# =============================================================================
# CELL 10: Pull and Process Data
# =============================================================================
if n_new_rows == 0:
    print("No new rows to process")
else:
    print(f"Processing {n_new_rows} rows...")

    # Pull unprocessed rows
    df = spark.sql(f"""
        WITH scored_accounts AS (
            SELECT DISTINCT hsp_account_id
            FROM {trgt_cat}.fin_ds.fudgesicle_inference_score
        )
        SELECT src.*
        FROM {trgt_cat}.fin_ds.fudgesicle_inference src
        LEFT JOIN scored_accounts sa ON src.hsp_account_id = sa.hsp_account_id
        WHERE src.insert_tsp > TIMESTAMP'{last_processed_ts}'
           OR sa.hsp_account_id IS NULL
    """).toPandas()

    print(f"Pulled {len(df)} rows for processing")

    # Process each row
    results = []
    for idx, row in df.iterrows():
        print(f"\nProcessing {idx+1}/{len(df)}: {row['hsp_account_id']}")
        result = process_row(row.to_dict())
        print(f"  Status: {result['status']}")
        results.append(result)

    # Summary
    success = sum(1 for r in results if r["status"] == "success")
    out_of_scope = sum(1 for r in results if r["status"] == "out_of_scope")
    errors = sum(1 for r in results if r["status"] not in ("success", "out_of_scope"))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Success: {success}")
    print(f"  Out of Scope: {out_of_scope}")
    print(f"  Errors: {errors}")

# =============================================================================
# CELL 11: Write Results to Table
# =============================================================================
if n_new_rows > 0 and 'results' in dir():
    # Build output dataframe
    output_rows = []
    for r, orig_row in zip(results, df.itertuples()):
        if r["status"] == "success":
            output_rows.append({
                "workqueue_entry_date": getattr(orig_row, 'workqueue_entry_date', None),
                "hsp_account_id": r["hsp_account_id"],
                "pat_mrn_id": getattr(orig_row, 'pat_mrn_id', None),
                "discharge_summary_note_id": getattr(orig_row, 'discharge_summary_note_id', None),
                "discharge_note_csn_id": getattr(orig_row, 'discharge_note_csn_id', None),
                "hp_note_id": getattr(orig_row, 'hp_note_id', None),
                "hp_note_csn_id": getattr(orig_row, 'hp_note_csn_id', None),
                "letter_type": "Sepsis_v2",
                "letter_text": r["letter_text"],
                "letter_curated_date": datetime.now().date(),
                "denial_info_json": json.dumps(r.get("denial_info", {}), default=str),
                "pipeline_version": "rebuttal_engine_v2",
            })

    if output_rows:
        output_df = pd.DataFrame(output_rows)
        spark_df = spark.createDataFrame(output_df)
        spark_df = spark_df.withColumn("insert_tsp", current_timestamp())

        # Uncomment to write:
        # spark_df.write.mode("append").saveAsTable(f"{trgt_cat}.fin_ds.fudgesicle_inference_score")
        # print(f"Wrote {len(output_df)} letters to {trgt_cat}.fin_ds.fudgesicle_inference_score")

        print(f"\n{len(output_df)} letters ready to write")
        print("Uncomment write statement to save to table")
    else:
        print("No successful letters to write")

print("\nInference complete.")
