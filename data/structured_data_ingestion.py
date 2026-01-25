# data/structured_data_ingestion.py
# Sepsis Appeal Engine - Structured Data Ingestion
#
# STRUCTURED DATA PIPELINE:
# 1. Query Clarity for labs, vitals, meds, procedures, ICD-10 for a single account
# 2. Write each to intermediate tables (herald_train_*)
# 3. Merge all into chronological timeline
# 4. LLM extracts sepsis-relevant data
#
# Run on Databricks Runtime 15.4 LTS ML

# =============================================================================
# CELL 1: Configuration
# =============================================================================
import os
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, to_timestamp, coalesce

spark = SparkSession.builder.getOrCreate()

# Environment
trgt_cat = os.environ.get('trgt_cat', 'dev')
if trgt_cat == 'dev':
    spark.sql('USE CATALOG prod;')
else:
    spark.sql(f'USE CATALOG {trgt_cat};')

# -----------------------------------------------------------------------------
# Intermediate Table Names
# -----------------------------------------------------------------------------
LABS_TABLE = f"{trgt_cat}.fin_ds.herald_train_labs"
VITALS_TABLE = f"{trgt_cat}.fin_ds.herald_train_vitals"
MEDS_TABLE = f"{trgt_cat}.fin_ds.herald_train_meds"
PROCEDURES_TABLE = f"{trgt_cat}.fin_ds.herald_train_procedures"
ICD10_TABLE = f"{trgt_cat}.fin_ds.herald_train_icd10"

print(f"Catalog: {trgt_cat}")
print(f"Tables: herald_train_*")

# =============================================================================
# CELL 2: Table Schemas (Run once to create tables)
# =============================================================================
# Uncomment and run once to create the intermediate tables

CREATE_TABLES = False  # Set to True to create tables

if CREATE_TABLES:
    # Labs table
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {LABS_TABLE} (
            hsp_account_id STRING,
            event_timestamp TIMESTAMP,
            event_type STRING,
            lab_name STRING,
            lab_value STRING,
            lab_units STRING,
            reference_range STRING,
            abnormal_flag STRING,
            ordering_provider STRING,
            ingested_at TIMESTAMP
        )
        USING DELTA
    """)
    print(f"Created {LABS_TABLE}")

    # Vitals table
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {VITALS_TABLE} (
            hsp_account_id STRING,
            event_timestamp TIMESTAMP,
            event_type STRING,
            vital_name STRING,
            vital_value STRING,
            vital_units STRING,
            measurement_source STRING,
            ingested_at TIMESTAMP
        )
        USING DELTA
    """)
    print(f"Created {VITALS_TABLE}")

    # Medications table
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {MEDS_TABLE} (
            hsp_account_id STRING,
            event_timestamp TIMESTAMP,
            event_type STRING,
            med_name STRING,
            med_dose STRING,
            med_units STRING,
            med_route STRING,
            med_frequency STRING,
            admin_action STRING,
            ordering_provider STRING,
            ingested_at TIMESTAMP
        )
        USING DELTA
    """)
    print(f"Created {MEDS_TABLE}")

    # Procedures table
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {PROCEDURES_TABLE} (
            hsp_account_id STRING,
            event_timestamp TIMESTAMP,
            event_type STRING,
            procedure_name STRING,
            procedure_code STRING,
            procedure_type STRING,
            performing_provider STRING,
            ingested_at TIMESTAMP
        )
        USING DELTA
    """)
    print(f"Created {PROCEDURES_TABLE}")

    # ICD-10 codes table
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {ICD10_TABLE} (
            hsp_account_id STRING,
            event_timestamp TIMESTAMP,
            event_type STRING,
            icd10_code STRING,
            icd10_description STRING,
            diagnosis_type STRING,
            diagnosis_priority INT,
            diagnosing_provider STRING,
            ingested_at TIMESTAMP
        )
        USING DELTA
    """)
    print(f"Created {ICD10_TABLE}")

# =============================================================================
# CELL 3: Query Functions - Labs
# =============================================================================

def query_labs_for_account(account_id):
    """
    Query Clarity for lab results for a single account.
    Writes to herald_train_labs table.

    TODO: Replace with actual Clarity table/column names
    """
    print(f"  Querying labs for account {account_id}...")

    # ---------------------------------------------------------------------
    # TODO: UPDATE THIS QUERY WITH ACTUAL CLARITY TABLE/COLUMN NAMES
    # ---------------------------------------------------------------------
    # Key labs for sepsis:
    # - Lactate (CRITICAL - all values)
    # - CBC: WBC, platelets, bands/differential
    # - BMP: Creatinine, BUN, glucose, electrolytes
    # - LFTs: Bilirubin (total), AST, ALT
    # - Coagulation: PT/INR, PTT, fibrinogen, D-dimer
    # - Infection markers: Procalcitonin, CRP
    # - Blood gases: PaO2, PaCO2, pH, FiO2
    # - Cardiac: Troponin, BNP
    # - Cultures: Blood, urine, other
    # ---------------------------------------------------------------------

    query = f"""
        INSERT INTO {LABS_TABLE}
        SELECT
            '{account_id}' AS hsp_account_id,
            -- TODO: result_timestamp column
            CAST(NULL AS TIMESTAMP) AS event_timestamp,
            'LAB' AS event_type,
            -- TODO: lab_name/component_name column
            'TODO_LAB_NAME' AS lab_name,
            -- TODO: result_value column
            'TODO_VALUE' AS lab_value,
            -- TODO: units column
            'TODO_UNITS' AS lab_units,
            -- TODO: reference_range column
            NULL AS reference_range,
            -- TODO: abnormal_flag column (H/L/Critical)
            NULL AS abnormal_flag,
            -- TODO: ordering_provider column
            NULL AS ordering_provider,
            current_timestamp() AS ingested_at
        FROM
            -- TODO: clarity_cur.order_results or similar
            (SELECT 1) AS placeholder
        WHERE
            -- TODO: join to get account_id
            1 = 0  -- Placeholder - remove when query is complete
    """

    # Uncomment when query is ready:
    # spark.sql(query)
    # print(f"  Labs written to {LABS_TABLE}")

    print(f"  [PLACEHOLDER] Labs query not yet configured")
    return None


# =============================================================================
# CELL 4: Query Functions - Vitals
# =============================================================================

def query_vitals_for_account(account_id):
    """
    Query Clarity for vital signs for a single account.
    Writes to herald_train_vitals table.

    TODO: Replace with actual Clarity table/column names
    """
    print(f"  Querying vitals for account {account_id}...")

    # ---------------------------------------------------------------------
    # TODO: UPDATE THIS QUERY WITH ACTUAL CLARITY TABLE/COLUMN NAMES
    # ---------------------------------------------------------------------
    # Key vitals for sepsis:
    # - Temperature (for SIRS: <36°C or >38°C)
    # - Heart rate (for SIRS: >90 bpm)
    # - Respiratory rate (for qSOFA: ≥22/min, SIRS: >20/min)
    # - Blood pressure: Systolic (qSOFA: ≤100), MAP (shock: <65)
    # - SpO2 / Oxygen saturation
    # - GCS / Mental status (qSOFA: <15)
    # - Urine output (for AKI: <0.5 mL/kg/h)
    # - FiO2 (for P/F ratio calculation)
    # ---------------------------------------------------------------------

    query = f"""
        INSERT INTO {VITALS_TABLE}
        SELECT
            '{account_id}' AS hsp_account_id,
            -- TODO: recorded_timestamp column
            CAST(NULL AS TIMESTAMP) AS event_timestamp,
            'VITAL' AS event_type,
            -- TODO: vital_name/flowsheet_name column
            'TODO_VITAL_NAME' AS vital_name,
            -- TODO: vital_value column
            'TODO_VALUE' AS vital_value,
            -- TODO: units column
            NULL AS vital_units,
            -- TODO: measurement_source (device, manual, etc.)
            NULL AS measurement_source,
            current_timestamp() AS ingested_at
        FROM
            -- TODO: clarity_cur.ip_flwsht_rec or similar
            (SELECT 1) AS placeholder
        WHERE
            1 = 0  -- Placeholder
    """

    print(f"  [PLACEHOLDER] Vitals query not yet configured")
    return None


# =============================================================================
# CELL 5: Query Functions - Medications
# =============================================================================

def query_meds_for_account(account_id):
    """
    Query Clarity for medication administrations for a single account.
    Writes to herald_train_meds table.

    TODO: Replace with actual Clarity table/column names
    """
    print(f"  Querying medications for account {account_id}...")

    # ---------------------------------------------------------------------
    # TODO: UPDATE THIS QUERY WITH ACTUAL CLARITY TABLE/COLUMN NAMES
    # ---------------------------------------------------------------------
    # Key medications for sepsis:
    # - Antibiotics (ALL - timing critical for SEP-1 bundle)
    # - Vasopressors: Norepinephrine, vasopressin, dopamine, dobutamine,
    #                 epinephrine, phenylephrine (with doses in µg/kg/min)
    # - IV Fluids: Crystalloids (NS, LR), colloids, volume administered
    # - Sedatives/paralytics (may affect GCS interpretation)
    # ---------------------------------------------------------------------

    query = f"""
        INSERT INTO {MEDS_TABLE}
        SELECT
            '{account_id}' AS hsp_account_id,
            -- TODO: admin_timestamp column
            CAST(NULL AS TIMESTAMP) AS event_timestamp,
            'MED' AS event_type,
            -- TODO: medication_name column
            'TODO_MED_NAME' AS med_name,
            -- TODO: dose column
            'TODO_DOSE' AS med_dose,
            -- TODO: dose_units column
            NULL AS med_units,
            -- TODO: route column (IV, PO, etc.)
            NULL AS med_route,
            -- TODO: frequency column
            NULL AS med_frequency,
            -- TODO: admin_action (Given, Held, etc.)
            'Given' AS admin_action,
            -- TODO: ordering_provider column
            NULL AS ordering_provider,
            current_timestamp() AS ingested_at
        FROM
            -- TODO: clarity_cur.mar_admin_info or similar
            (SELECT 1) AS placeholder
        WHERE
            1 = 0  -- Placeholder
    """

    print(f"  [PLACEHOLDER] Medications query not yet configured")
    return None


# =============================================================================
# CELL 6: Query Functions - Procedures
# =============================================================================

def query_procedures_for_account(account_id):
    """
    Query Clarity for procedures for a single account.
    Writes to herald_train_procedures table.

    TODO: Replace with actual Clarity table/column names
    """
    print(f"  Querying procedures for account {account_id}...")

    # ---------------------------------------------------------------------
    # TODO: UPDATE THIS QUERY WITH ACTUAL CLARITY TABLE/COLUMN NAMES
    # ---------------------------------------------------------------------
    # Key procedures for sepsis:
    # - Central line placement (for vasopressors, CVP monitoring)
    # - Arterial line placement (for continuous BP/MAP)
    # - Intubation / Mechanical ventilation
    # - Dialysis / CRRT
    # - Blood cultures collection
    # - Source control procedures (drainage, debridement)
    # ---------------------------------------------------------------------

    query = f"""
        INSERT INTO {PROCEDURES_TABLE}
        SELECT
            '{account_id}' AS hsp_account_id,
            -- TODO: procedure_timestamp column
            CAST(NULL AS TIMESTAMP) AS event_timestamp,
            'PROCEDURE' AS event_type,
            -- TODO: procedure_name column
            'TODO_PROCEDURE_NAME' AS procedure_name,
            -- TODO: procedure_code column (CPT or internal)
            NULL AS procedure_code,
            -- TODO: procedure_type column
            NULL AS procedure_type,
            -- TODO: performing_provider column
            NULL AS performing_provider,
            current_timestamp() AS ingested_at
        FROM
            -- TODO: clarity_cur.or_log or similar
            (SELECT 1) AS placeholder
        WHERE
            1 = 0  -- Placeholder
    """

    print(f"  [PLACEHOLDER] Procedures query not yet configured")
    return None


# =============================================================================
# CELL 7: Query Functions - ICD-10 Codes
# =============================================================================

def query_icd10_for_account(account_id):
    """
    Query Clarity for ICD-10 diagnosis codes for a single account.
    Writes to herald_train_icd10 table.

    TODO: Replace with actual Clarity table/column names
    """
    print(f"  Querying ICD-10 codes for account {account_id}...")

    # ---------------------------------------------------------------------
    # TODO: UPDATE THIS QUERY WITH ACTUAL CLARITY TABLE/COLUMN NAMES
    # ---------------------------------------------------------------------
    # Key ICD-10 codes for sepsis:
    # - A41.x (Sepsis)
    # - R65.20 (Severe sepsis without shock)
    # - R65.21 (Severe sepsis with septic shock)
    # - A40.x (Streptococcal sepsis)
    # - Infection source codes (pneumonia, UTI, cellulitis, etc.)
    # - Organ dysfunction codes
    # ---------------------------------------------------------------------

    query = f"""
        INSERT INTO {ICD10_TABLE}
        SELECT
            '{account_id}' AS hsp_account_id,
            -- TODO: diagnosis_timestamp column (or admission date if not available)
            CAST(NULL AS TIMESTAMP) AS event_timestamp,
            'DIAGNOSIS' AS event_type,
            -- TODO: icd10_code column
            'TODO_ICD10_CODE' AS icd10_code,
            -- TODO: icd10_description column
            'TODO_DESCRIPTION' AS icd10_description,
            -- TODO: diagnosis_type (Admitting, Principal, Secondary)
            NULL AS diagnosis_type,
            -- TODO: diagnosis_priority/sequence
            NULL AS diagnosis_priority,
            -- TODO: diagnosing_provider column
            NULL AS diagnosing_provider,
            current_timestamp() AS ingested_at
        FROM
            -- TODO: clarity_cur.hsp_acct_dx_list or similar
            (SELECT 1) AS placeholder
        WHERE
            1 = 0  -- Placeholder
    """

    print(f"  [PLACEHOLDER] ICD-10 query not yet configured")
    return None


# =============================================================================
# CELL 8: Ingest All Structured Data for Account
# =============================================================================

def ingest_structured_data_for_account(account_id):
    """
    Run all structured data queries for a single account.
    Each writes to its respective herald_train_* table.
    """
    print(f"\n{'='*60}")
    print(f"INGESTING STRUCTURED DATA FOR ACCOUNT {account_id}")
    print(f"{'='*60}")

    query_labs_for_account(account_id)
    query_vitals_for_account(account_id)
    query_meds_for_account(account_id)
    query_procedures_for_account(account_id)
    query_icd10_for_account(account_id)

    print(f"\nStructured data ingestion complete for {account_id}")


# =============================================================================
# CELL 9: Merge and Order by Timestamp
# =============================================================================

def get_merged_timeline_for_account(account_id):
    """
    Merge all structured data for an account into a single chronological timeline.
    Returns a DataFrame ordered by event_timestamp.
    """
    print(f"  Merging structured data timeline for {account_id}...")

    # Labs
    labs_df = spark.sql(f"""
        SELECT
            hsp_account_id,
            event_timestamp,
            event_type,
            CONCAT(lab_name, ': ', lab_value,
                   COALESCE(CONCAT(' ', lab_units), ''),
                   COALESCE(CONCAT(' [', abnormal_flag, ']'), '')) AS event_detail,
            lab_name AS detail_name,
            lab_value AS detail_value,
            lab_units AS detail_units,
            abnormal_flag AS detail_flag
        FROM {LABS_TABLE}
        WHERE hsp_account_id = '{account_id}'
    """)

    # Vitals
    vitals_df = spark.sql(f"""
        SELECT
            hsp_account_id,
            event_timestamp,
            event_type,
            CONCAT(vital_name, ': ', vital_value,
                   COALESCE(CONCAT(' ', vital_units), '')) AS event_detail,
            vital_name AS detail_name,
            vital_value AS detail_value,
            vital_units AS detail_units,
            NULL AS detail_flag
        FROM {VITALS_TABLE}
        WHERE hsp_account_id = '{account_id}'
    """)

    # Medications
    meds_df = spark.sql(f"""
        SELECT
            hsp_account_id,
            event_timestamp,
            event_type,
            CONCAT(med_name, ' ', med_dose,
                   COALESCE(CONCAT(' ', med_units), ''),
                   COALESCE(CONCAT(' ', med_route), ''),
                   COALESCE(CONCAT(' [', admin_action, ']'), '')) AS event_detail,
            med_name AS detail_name,
            med_dose AS detail_value,
            med_units AS detail_units,
            admin_action AS detail_flag
        FROM {MEDS_TABLE}
        WHERE hsp_account_id = '{account_id}'
    """)

    # Procedures
    procedures_df = spark.sql(f"""
        SELECT
            hsp_account_id,
            event_timestamp,
            event_type,
            procedure_name AS event_detail,
            procedure_name AS detail_name,
            procedure_code AS detail_value,
            procedure_type AS detail_units,
            NULL AS detail_flag
        FROM {PROCEDURES_TABLE}
        WHERE hsp_account_id = '{account_id}'
    """)

    # ICD-10
    icd10_df = spark.sql(f"""
        SELECT
            hsp_account_id,
            event_timestamp,
            event_type,
            CONCAT(icd10_code, ': ', icd10_description,
                   COALESCE(CONCAT(' [', diagnosis_type, ']'), '')) AS event_detail,
            icd10_code AS detail_name,
            icd10_description AS detail_value,
            diagnosis_type AS detail_units,
            NULL AS detail_flag
        FROM {ICD10_TABLE}
        WHERE hsp_account_id = '{account_id}'
    """)

    # Union all and order by timestamp
    merged_df = (
        labs_df
        .union(vitals_df)
        .union(meds_df)
        .union(procedures_df)
        .union(icd10_df)
        .orderBy("event_timestamp")
    )

    row_count = merged_df.count()
    print(f"  Merged {row_count} events into timeline")

    return merged_df


def format_timeline_for_llm(merged_df):
    """
    Format the merged timeline DataFrame into a string for LLM input.
    Groups by date for readability.
    """
    rows = merged_df.collect()

    if not rows:
        return "No structured data available for this account."

    lines = []
    current_date = None

    for row in rows:
        ts = row['event_timestamp']
        if ts:
            date_str = ts.strftime('%Y-%m-%d')
            time_str = ts.strftime('%H:%M')

            # Add date header when date changes
            if date_str != current_date:
                if current_date is not None:
                    lines.append("")  # Blank line between dates
                lines.append(f"=== {date_str} ===")
                current_date = date_str

            event_type = row['event_type']
            event_detail = row['event_detail']
            lines.append(f"  {time_str} [{event_type}] {event_detail}")
        else:
            # No timestamp - group at end
            event_type = row['event_type']
            event_detail = row['event_detail']
            lines.append(f"  [NO TIME] [{event_type}] {event_detail}")

    return "\n".join(lines)


# =============================================================================
# CELL 10: LLM Extraction for Sepsis-Relevant Data
# =============================================================================

STRUCTURED_DATA_EXTRACTION_PROMPT = '''Extract sepsis-relevant clinical data from this patient timeline.

# Patient Timeline (Chronological)
{timeline_text}

# What to Extract

## SOFA SCORE COMPONENTS (calculate points where possible)
For each organ system, extract values and assign SOFA points (0-4):

RESPIRATORY (PaO2/FiO2 ratio):
- P/F ≥400 = 0, <400 = 1, <300 = 2, <200 with vent = 3, <100 with vent = 4

COAGULATION (Platelets × 10³/µL):
- ≥150 = 0, <150 = 1, <100 = 2, <50 = 3, <20 = 4

LIVER (Bilirubin mg/dL):
- <1.2 = 0, 1.2-1.9 = 1, 2.0-5.9 = 2, 6.0-11.9 = 3, >12 = 4

CARDIOVASCULAR (MAP and vasopressors):
- MAP ≥70 = 0, <70 = 1, dopamine <5 or dobutamine = 2
- dopamine 5-15 or norepi ≤0.1 = 3, dopamine >15 or norepi >0.1 = 4

CNS (GCS):
- 15 = 0, 13-14 = 1, 10-12 = 2, 6-9 = 3, <6 = 4

RENAL (Creatinine mg/dL or urine output):
- Cr <1.2 = 0, 1.2-1.9 = 1, 2.0-3.4 = 2, 3.5-4.9 = 3, >5.0 = 4

## LACTATE (ALL values with timestamps)
- Note initial value, peak value, and clearance trend
- Flag if >2 mmol/L (sepsis marker) or >4 mmol/L (severe)

## SEP-1 BUNDLE COMPLIANCE
3-Hour Bundle (check timing from severe sepsis recognition):
- Lactate measured?
- Blood cultures drawn (before antibiotics)?
- Broad-spectrum antibiotics given?
- 30 mL/kg crystalloid if hypotensive or lactate ≥4?

6-Hour Bundle:
- Repeat lactate if initial ≥2?
- Vasopressors if fluid-refractory hypotension?
- Volume status reassessment documented?

## INFECTION EVIDENCE
- Culture results (blood, urine, other)
- Procalcitonin, CRP values
- WBC, bands
- Infection source identified?

## VASOPRESSOR DETAILS
- Drug, dose (µg/kg/min), start time, duration
- Dose changes and timing
- Was vasopressin added? At what NE dose?

## KEY ICD-10 CODES
- Sepsis codes (A41.x, R65.20, R65.21)
- Infection source codes
- Organ dysfunction codes

# Output Format
Return a structured summary organized by category. Include timestamps for all events.
Flag any data that strongly supports or contradicts sepsis diagnosis.
Note any missing data that would be important for the appeal.'''


def extract_sepsis_data_from_timeline(timeline_text, openai_client):
    """
    Use LLM to extract sepsis-relevant data from the merged timeline.
    Returns extracted summary text.
    """
    print("  Extracting sepsis-relevant data via LLM...")

    if not timeline_text or timeline_text == "No structured data available for this account.":
        return "No structured data available for extraction."

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a clinical data extraction specialist focused on sepsis criteria. Extract and organize relevant data with precise timestamps."},
                {"role": "user", "content": STRUCTURED_DATA_EXTRACTION_PROMPT.format(timeline_text=timeline_text)}
            ],
            temperature=0.1,
            max_tokens=4000
        )

        extracted = response.choices[0].message.content.strip()
        print(f"  Extracted {len(extracted)} chars of sepsis-relevant data")
        return extracted

    except Exception as e:
        print(f"  Warning: Structured data extraction failed: {e}")
        return f"Extraction failed: {str(e)}"


# =============================================================================
# CELL 11: Main Function - Get Structured Data for Account
# =============================================================================

def get_structured_data_for_account(account_id, openai_client):
    """
    Complete pipeline: query, merge, format, extract.
    Returns extracted sepsis-relevant structured data as text.

    This function is called from inference.py as Step 5b.
    """
    print(f"\n--- Structured Data Pipeline for {account_id} ---")

    # Step 1: Get merged timeline from intermediate tables
    merged_df = get_merged_timeline_for_account(account_id)

    # Step 2: Format for LLM
    timeline_text = format_timeline_for_llm(merged_df)

    # Step 3: Extract sepsis-relevant data
    extracted_data = extract_sepsis_data_from_timeline(timeline_text, openai_client)

    return extracted_data


# =============================================================================
# CELL 12: Utility - Clear Account Data from Intermediate Tables
# =============================================================================

def clear_account_from_tables(account_id):
    """
    Remove all data for an account from intermediate tables.
    Useful for re-processing.
    """
    tables = [LABS_TABLE, VITALS_TABLE, MEDS_TABLE, PROCEDURES_TABLE, ICD10_TABLE]

    for table in tables:
        spark.sql(f"DELETE FROM {table} WHERE hsp_account_id = '{account_id}'")
        print(f"  Cleared {account_id} from {table}")


print("Structured data ingestion functions loaded")
print("\nNOTE: Query placeholders need to be updated with actual Clarity table/column names")
