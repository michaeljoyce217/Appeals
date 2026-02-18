# data/structured_data_ingestion.py
# Structured Data Ingestion - Linear Script
#
# Run this notebook to gather structured data for a single account.
# Will be merged into featurization.py later.
#
# Run on Databricks Runtime 15.4 LTS ML

# =============================================================================
# CELL 1: Configuration
# =============================================================================
import os
from datetime import datetime
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# -----------------------------------------------------------------------------
# INPUT: Set the account ID to process
# -----------------------------------------------------------------------------
HSP_ACCOUNT_ID = 204200688858  

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
trgt_cat = os.environ.get('trgt_cat', 'dev')

spark.sql('USE CATALOG prod;')

# -----------------------------------------------------------------------------
# SINGLE LOOKUP: Map HSP_ACCOUNT_ID → PAT_ENC_CSN_ID(s)
# This view is used by all downstream queries
# -----------------------------------------------------------------------------
# =========================================================================
# STEP 1: Get the target encounter details (start/end times)
# =========================================================================

spark.sql(f"""
CREATE OR REPLACE TEMP VIEW target_encounter AS
SELECT 
    peh.HSP_ACCOUNT_ID,
    peh.PAT_ENC_CSN_ID,
    peh.PAT_ID,
    peh.ADM_DATE_TIME AS ENCOUNTER_START,
    peh.DISCH_DATE_TIME AS ENCOUNTER_END
FROM prod.clarity_cur.pat_enc_hsp_har_enh peh
WHERE peh.HSP_ACCOUNT_ID = {HSP_ACCOUNT_ID}
""")


# Cache it since we'll hit it repeatedly
spark.sql("CACHE TABLE target_encounter")

# Table names
LABS_TABLE = f"{trgt_cat}.fin_ds.fudgesicle_labs"
VITALS_TABLE = f"{trgt_cat}.fin_ds.fudgesicle_vitals"
MEDS_TABLE = f"{trgt_cat}.fin_ds.fudgesicle_meds"
PROCEDURES_TABLE = f"{trgt_cat}.fin_ds.fudgesicle_procedures"
ICD10_TABLE = f"{trgt_cat}.fin_ds.fudgesicle_icd10"
MERGED_TABLE = f"{trgt_cat}.fin_ds.fudgesicle_structured_timeline"
LLM_TIMELINE_TABLE = f"{trgt_cat}.fin_ds.fudgesicle_llm_timeline_{HSP_ACCOUNT_ID}"

print(f"Account ID: {HSP_ACCOUNT_ID}")
print(f"Catalog: {trgt_cat}")
# =============================================================================
# STEP 2: Create Tables (run once)
# =============================================================================
CREATE_TABLES = False  # Set to True on first run

if CREATE_TABLES:
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {LABS_TABLE} (
            HSP_ACCOUNT_ID BIGINT,
            PAT_ENC_CSN_ID BIGINT,
            EVENT_TIMESTAMP TIMESTAMP,
            LAB_NAME STRING,
            lab_value STRING,
            lab_units STRING,
            reference_range STRING,
            abnormal_flag STRING
        ) USING DELTA
    """)

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {VITALS_TABLE} (
            HSP_ACCOUNT_ID BIGINT,
            PAT_ENC_CSN_ID BIGINT,
            EVENT_TIMESTAMP TIMESTAMP,
            VITAL_NAME STRING,
            vital_value STRING,
            vital_units STRING
        ) USING DELTA
    """)

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {MEDS_TABLE} (
            HSP_ACCOUNT_ID BIGINT,
            PAT_ENC_CSN_ID BIGINT,
            EVENT_TIMESTAMP TIMESTAMP,
            MED_NAME STRING,
            MED_DOSE STRING,
            MED_UNITS STRING,
            MED_ROUTE STRING,
            ADMIN_ACTION STRING
        ) USING DELTA
    """)

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {PROCEDURES_TABLE} (
            HSP_ACCOUNT_ID BIGINT,
            PAT_ENC_CSN_ID BIGINT,
            EVENT_TIMESTAMP TIMESTAMP,
            PROCEDURE_NAME STRING,
            PROCEDURE_CODE STRING
        ) USING DELTA
    """)

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {ICD10_TABLE} (
            HSP_ACCOUNT_ID BIGINT,           
            PAT_ENC_CSN_ID BIGINT,
            ICD10_CODE STRING,
            icd10_description STRING,         
            TIMING_CATEGORY STRING,
            availability_time STRING,
            EVENT_TIMESTAMP TIMESTAMP,
            source STRING
        ) USING DELTA
    """)

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {MERGED_TABLE} (
            HSP_ACCOUNT_ID BIGINT,
            PAT_ENC_CSN_ID BIGINT,
            EVENT_TIMESTAMP TIMESTAMP,
            event_type STRING,
            event_detail STRING
        ) USING DELTA
    """)

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {LLM_TIMELINE_TABLE} (
            HSP_ACCOUNT_ID BIGINT,
            PAT_ENC_CSN_ID BIGINT,
            section_type STRING,
            sort_order INT,
            event_timestamp TIMESTAMP,
            content STRING
        ) USING DELTA
    """)

    print("Tables created")

# =============================================================================
# STEP 3: Query All Labs
# =============================================================================
print(f"\n{'='*60}")
print(f"QUERYING ALL LABS FOR {HSP_ACCOUNT_ID}")
print(f"{'='*60}")

spark.sql(f"""
CREATE OR REPLACE TABLE {LABS_TABLE} AS
SELECT
    t.HSP_ACCOUNT_ID AS HSP_ACCOUNT_ID,
    t.PAT_ENC_CSN_ID AS PAT_ENC_CSN_ID,
    CAST(res_comp.COMP_VERIF_DTTM AS TIMESTAMP) AS EVENT_TIMESTAMP,
    cc.NAME AS LAB_NAME,
    CAST(REGEXP_REPLACE(res_comp.COMPONENT_VALUE, '>', '') AS STRING) AS lab_value,
    res_comp.component_units AS lab_units,
    NULL AS reference_range,
    zsab.NAME AS abnormal_flag

FROM target_encounter t

INNER JOIN prod.clarity.order_proc op
    ON t.PAT_ENC_CSN_ID = op.PAT_ENC_CSN_ID

INNER JOIN prod.clarity.RES_DB_MAIN rdm
    ON rdm.RES_ORDER_ID = op.ORDER_PROC_ID

INNER JOIN prod.clarity.res_components res_comp
    ON res_comp.result_id = rdm.result_id

INNER JOIN prod.clarity.clarity_component cc
    ON cc.component_id = res_comp.component_id

LEFT JOIN prod.clarity.zc_stat_abnorms zsab
    ON zsab.stat_abnorms_c = res_comp.component_abn_c

WHERE op.order_status_c = 5
  AND op.lab_status_c IN (3, 5)
  AND rdm.res_val_status_c = 9
  AND res_comp.COMPONENT_VALUE IS NOT NULL
  AND res_comp.COMPONENT_VALUE <> '-1'

ORDER BY res_comp.COMP_VERIF_DTTM ASC
""")

print(f"All labs written to {LABS_TABLE}")

# =============================================================================
# STEP 4: Query Vitals
# =============================================================================
print(f"\n{'='*60}")
print(f"QUERYING VITALS FOR {HSP_ACCOUNT_ID}")
print(f"{'='*60}")

spark.sql(f"""
CREATE OR REPLACE TABLE {VITALS_TABLE} AS
SELECT
    t.HSP_ACCOUNT_ID AS HSP_ACCOUNT_ID,
    t.PAT_ENC_CSN_ID AS PAT_ENC_CSN_ID,
    CAST(to_timestamp(substring(v.RECORDED_TIME, 1, 19), 'yyyy-MM-dd HH:mm:ss') AS TIMESTAMP) AS EVENT_TIMESTAMP,
    v.FLO_MEAS_NAME AS VITAL_NAME,
    v.MEAS_VALUE AS vital_value
    -- Removed vital_units from here
FROM target_encounter t
INNER JOIN prod.clarity_cur.ip_flwsht_rec_enh v
    ON t.PAT_ENC_CSN_ID = v.IP_DATA_STORE_EPT_CSN
WHERE v.FLO_MEAS_ID IN ('5', '6', '8', '9', '10', '11', '14')
  AND v.MEAS_VALUE IS NOT NULL
ORDER BY EVENT_TIMESTAMP ASC
""")

print(f"Vitals written to {VITALS_TABLE}")


# =============================================================================
# STEP 5: Query Medications
# =============================================================================
print(f"\n{'='*60}")
print(f"QUERYING ALL MEDICATIONS FOR {HSP_ACCOUNT_ID}")
print(f"{'='*60}")

spark.sql(f"""
CREATE OR REPLACE TABLE {MEDS_TABLE} AS
SELECT
    t.HSP_ACCOUNT_ID AS HSP_ACCOUNT_ID,
    t.PAT_ENC_CSN_ID AS PAT_ENC_CSN_ID,
    CAST(mar.TAKEN_TIME AS TIMESTAMP) AS EVENT_TIMESTAMP,
    om.SIMPLE_GENERIC_NAME AS MED_NAME,
    CAST(om.HV_DISCRETE_DOSE AS STRING) AS MED_DOSE,
    om.DOSE_UNIT AS MED_UNITS,
    mar.ROUTE AS MED_ROUTE,
    mar.ACTION AS ADMIN_ACTION

FROM target_encounter t

INNER JOIN prod.clarity_cur.order_med_enh om
    ON t.PAT_ENC_CSN_ID = om.PAT_ENC_CSN_ID

INNER JOIN prod.clarity_cur.mar_admin_info_enh mar
    ON om.ORDER_MED_ID = mar.ORDER_MED_ID

WHERE mar.ACTION IN (
    'Given',
    'Patient/Family Admin',
    'Given-See Override',
    'Admin by Another Clinician (Comment)',
    'New Bag',
    'Bolus',
    'Push',
    'Started by Another Clinician',
    'Bag Switched',
    'Clinic Sample Administered',
    'Applied',
    'Feeding Started',
    'Acknowledged',
    'Contrast Given',
    'New Bag-See Override',
    'Bolus from Bag'
)

ORDER BY EVENT_TIMESTAMP ASC
""")

print(f"All medications written to {MEDS_TABLE}")

# =============================================================================
# STEP 6: Query Procedures
# =============================================================================
print(f"\n{'='*60}")
print(f"QUERYING PROCEDURES FOR {HSP_ACCOUNT_ID}")
print(f"{'='*60}")

spark.sql(f"""
CREATE OR REPLACE TABLE {PROCEDURES_TABLE} AS
SELECT
    t.HSP_ACCOUNT_ID AS HSP_ACCOUNT_ID,
    t.PAT_ENC_CSN_ID AS PAT_ENC_CSN_ID,
    CAST(op.PROC_START_TIME AS TIMESTAMP) AS EVENT_TIMESTAMP,
    op.PROC_NAME AS PROCEDURE_NAME,
    CAST(op.ORDER_PROC_ID AS STRING) AS PROCEDURE_CODE

FROM target_encounter t

INNER JOIN prod.clarity_cur.order_proc_enh op
    ON t.PAT_ENC_CSN_ID = op.PAT_ENC_CSN_ID

WHERE op.PROC_NAME IN (
    'DIET NPO',
    'DIET TUBE FEEDING',
    'DIET CALORIE CONTROLLED',
    'DIET COMPLEX CARB',
    'DIET KETOGENIC',
    'DIET DIABETIC'
)
AND op.ORDER_STATUS_C = 5
AND op.PROC_START_TIME IS NOT NULL

ORDER BY EVENT_TIMESTAMP ASC
""")

print(f"Procedures written to {PROCEDURES_TABLE}")


# =============================================================================
# STEP 7: Query ICD-10 Codes
# =============================================================================
print(f"\n{'='*60}")
print(f"QUERYING ALL ICD-10 CODES FOR {HSP_ACCOUNT_ID}")
print(f"{'='*60}")

spark.sql(f"""
CREATE OR REPLACE TEMP VIEW encounter_icd10_codes AS

-- SOURCE 1: Outpatient encounter diagnoses
SELECT
    te.HSP_ACCOUNT_ID,
    te.PAT_ENC_CSN_ID,
    dd.ICD10_CODE,
    edg.DX_NAME AS icd10_description,
    CAST(pe.CONTACT_DATE AS TIMESTAMP) AS EVENT_TIMESTAMP,
    CASE
        WHEN pe.CONTACT_DATE < te.ENCOUNTER_START THEN 'BEFORE'
        WHEN pe.CONTACT_DATE >= te.ENCOUNTER_START
             AND (te.ENCOUNTER_END IS NULL OR pe.CONTACT_DATE <= te.ENCOUNTER_END) THEN 'DURING'
        ELSE 'AFTER'
    END AS TIMING_CATEGORY,
    'OUTPATIENT_ENC_DX' AS source
FROM target_encounter te
JOIN prod.clarity_cur.pat_enc_dx_enh dd
    ON dd.PAT_ENC_CSN_ID = te.PAT_ENC_CSN_ID
JOIN prod.clarity_cur.pat_enc_enh pe
    ON pe.PAT_ENC_CSN_ID = dd.PAT_ENC_CSN_ID
LEFT JOIN prod.clarity.clarity_edg edg
    ON dd.DX_ID = edg.DX_ID
WHERE dd.ICD10_CODE IS NOT NULL

UNION ALL

-- SOURCE 2: Inpatient hospital account diagnoses
SELECT
    te.HSP_ACCOUNT_ID,
    te.PAT_ENC_CSN_ID,
    dx.CODE AS ICD10_CODE,
    edg.DX_NAME AS icd10_description,
    CAST(ha.DISCH_DATE_TIME AS TIMESTAMP) AS EVENT_TIMESTAMP,
    CASE
        WHEN ha.DISCH_DATE_TIME < te.ENCOUNTER_START THEN 'BEFORE'
        WHEN ha.DISCH_DATE_TIME >= te.ENCOUNTER_START
             AND (te.ENCOUNTER_END IS NULL OR ha.DISCH_DATE_TIME <= te.ENCOUNTER_END) THEN 'DURING'
        ELSE 'AFTER'
    END AS TIMING_CATEGORY,
    'INPATIENT_ACCT_DX' AS source
FROM target_encounter te
JOIN prod.clarity_cur.hsp_acct_dx_list_enh dx
    ON dx.PAT_ID = te.PAT_ID
JOIN prod.clarity_cur.pat_enc_hsp_har_enh ha
    ON ha.HSP_ACCOUNT_ID = dx.HSP_ACCOUNT_ID
LEFT JOIN prod.clarity.clarity_edg edg
    ON dx.DX_ID = edg.DX_ID
WHERE dx.CODE IS NOT NULL

UNION ALL

-- SOURCE 3: Problem list history
SELECT
    te.HSP_ACCOUNT_ID,
    te.PAT_ENC_CSN_ID,
    phx.HX_PROBLEM_ICD10_CODE AS ICD10_CODE,
    phx.HX_PROBLEM_DX_NAME AS icd10_description,  -- Use the name already in the table
    CAST(phx.HX_DATE_OF_ENTRY AS TIMESTAMP) AS EVENT_TIMESTAMP,
    CASE
        WHEN phx.HX_DATE_OF_ENTRY < te.ENCOUNTER_START THEN 'BEFORE'
        WHEN phx.HX_DATE_OF_ENTRY >= te.ENCOUNTER_START
             AND (te.ENCOUNTER_END IS NULL OR phx.HX_DATE_OF_ENTRY <= te.ENCOUNTER_END) THEN 'DURING'
        ELSE 'AFTER'
    END AS TIMING_CATEGORY,
    'PROBLEM_LIST' AS source
FROM target_encounter te
JOIN prod.clarity_cur.problem_list_hx_enh phx
    ON phx.PAT_ID = te.PAT_ID
WHERE phx.HX_PROBLEM_ICD10_CODE IS NOT NULL
  AND phx.HX_STATUS = 'Active'
""")

# Final output - exclude AFTER, format timing appropriately
spark.sql(f"""
CREATE OR REPLACE TABLE {ICD10_TABLE} AS
SELECT
    HSP_ACCOUNT_ID,
    PAT_ENC_CSN_ID,
    ICD10_CODE,
    icd10_description,
    TIMING_CATEGORY,
    CASE
        WHEN TIMING_CATEGORY = 'BEFORE' THEN 'Available at encounter start'
        WHEN TIMING_CATEGORY = 'DURING' THEN CAST(EVENT_TIMESTAMP AS STRING)
    END AS availability_time,
    EVENT_TIMESTAMP,
    source
FROM encounter_icd10_codes
WHERE TIMING_CATEGORY != 'AFTER'
ORDER BY
    TIMING_CATEGORY DESC,
    EVENT_TIMESTAMP ASC
""")

print(f"All ICD-10 codes written to {ICD10_TABLE}")


# =============================================================================
# STEP 8: Merge All Events into Chronological Timeline (with deduplication)
# =============================================================================
print(f"\n{'='*60}")
print(f"CREATING UNIFIED TIMELINE FOR {HSP_ACCOUNT_ID} WITH DEDUPLICATION")
print(f"{'='*60}")

spark.sql(f"""
CREATE OR REPLACE TABLE {MERGED_TABLE} AS
WITH RawMergedEvents AS (
    -- Labs
    SELECT
        HSP_ACCOUNT_ID,
        PAT_ENC_CSN_ID,
        EVENT_TIMESTAMP,
        'LAB' AS event_type,
        CONCAT(
            LAB_NAME, ': ', lab_value, ' ', COALESCE(lab_units, ''),
            CASE WHEN abnormal_flag IS NOT NULL THEN CONCAT(' (', abnormal_flag, ')') ELSE '' END
        ) AS event_detail,
        CASE WHEN abnormal_flag IS NOT NULL THEN 1 ELSE 0 END AS is_abnormal
    FROM {LABS_TABLE}

    UNION ALL

    -- Vitals
    SELECT
        HSP_ACCOUNT_ID,
        PAT_ENC_CSN_ID,
        EVENT_TIMESTAMP,
        'VITAL' AS event_type,
        CONCAT(VITAL_NAME, ': ', vital_value) AS event_detail,
        0 AS is_abnormal
    FROM {VITALS_TABLE}

    UNION ALL

    -- Medications
    SELECT
        HSP_ACCOUNT_ID,
        PAT_ENC_CSN_ID,
        EVENT_TIMESTAMP,
        'MEDICATION' AS event_type,
        CONCAT(
            MED_NAME, ' ', COALESCE(MED_DOSE, ''), ' ', COALESCE(MED_UNITS, ''),
            ' via ', COALESCE(MED_ROUTE, 'unknown route'),
            ' - ', ADMIN_ACTION
        ) AS event_detail,
        0 AS is_abnormal
    FROM {MEDS_TABLE}

    UNION ALL

    -- Procedures
    SELECT
        HSP_ACCOUNT_ID,
        PAT_ENC_CSN_ID,
        EVENT_TIMESTAMP,
        'PROCEDURE' AS event_type,
        PROCEDURE_NAME AS event_detail,
        0 AS is_abnormal
    FROM {PROCEDURES_TABLE}

    UNION ALL

    -- ICD-10 codes (only those DURING encounter with timestamps)
    SELECT
        HSP_ACCOUNT_ID,
        PAT_ENC_CSN_ID,
        EVENT_TIMESTAMP,
        'DIAGNOSIS' AS event_type,
        CONCAT(
            ICD10_CODE, ' - ', COALESCE(icd10_description, 'Unknown'),
            ' (', source, ')'
        ) AS event_detail,
        0 AS is_abnormal
    FROM {ICD10_TABLE}
    WHERE TIMING_CATEGORY = 'DURING'
      AND EVENT_TIMESTAMP IS NOT NULL
),
DeduplicatedEvents AS (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY HSP_ACCOUNT_ID, PAT_ENC_CSN_ID, EVENT_TIMESTAMP, event_type, event_detail ORDER BY EVENT_TIMESTAMP) as rn
    FROM RawMergedEvents
)
SELECT
    HSP_ACCOUNT_ID,
    PAT_ENC_CSN_ID,
    EVENT_TIMESTAMP,
    event_type,
    event_detail,
    is_abnormal
FROM DeduplicatedEvents
WHERE rn = 1
ORDER BY EVENT_TIMESTAMP ASC
""")

print(f"Unified timeline (deduplicated) written to {MERGED_TABLE}")


# =============================================================================
# STEP 9: Create LLM-Ready Timeline Table (with all pre-existing diagnoses and their original timestamps)
# =============================================================================

spark.sql(f"""
CREATE OR REPLACE TABLE {LLM_TIMELINE_TABLE} AS
WITH encounter_context AS (
    SELECT
        HSP_ACCOUNT_ID,
        PAT_ENC_CSN_ID,
        ENCOUNTER_START,
        ENCOUNTER_END,
        'ENCOUNTER_INFO' AS section_type,
        0 AS sort_order,
        NULL AS event_timestamp,
        CONCAT(
            'Account ID: ', HSP_ACCOUNT_ID, '\n',
            'Encounter: ', PAT_ENC_CSN_ID, '\n',
            'Admission: ', ENCOUNTER_START, '\n',
            'Discharge: ', COALESCE(CAST(ENCOUNTER_END AS STRING), 'Still admitted')
        ) AS content
    FROM target_encounter
),

pre_existing_dx AS (
    SELECT
        icd.HSP_ACCOUNT_ID,
        icd.PAT_ENC_CSN_ID,
        'PRE_EXISTING_DX' AS section_type,
        1 AS sort_order,
        -- Use the original event timestamp from the ICD10 table
        icd.EVENT_TIMESTAMP AS event_timestamp,
        CONCAT(
            ICD10_CODE, ': ',
            COALESCE(icd.icd10_description, 'Unknown'),
            ' (', source, ')'
        ) AS content
    FROM {ICD10_TABLE} icd
    -- We still join to target_encounter to ensure we're only looking at diagnoses
    -- relevant to the specific encounter's patient, though ENCOUNTER_START isn't used for filtering here.
    JOIN target_encounter te
        ON icd.PAT_ENC_CSN_ID = te.PAT_ENC_CSN_ID
    WHERE icd.TIMING_CATEGORY = 'BEFORE'
    -- Removed the recency filter here to include all 'BEFORE' diagnoses
),

timeline_events AS (
    SELECT
        m.HSP_ACCOUNT_ID,
        m.PAT_ENC_CSN_ID,
        'TIMELINE_EVENT' AS section_type,
        2 AS sort_order,
        m.EVENT_TIMESTAMP AS event_timestamp,
        CONCAT(
            '[', CAST(m.EVENT_TIMESTAMP AS STRING), '] ',
            UPPER(m.event_type), ': ',
            m.event_detail
        ) AS content
    FROM {MERGED_TABLE} m
    -- IMPORTANT: Keep this exclusion filter to prevent duplication.
    -- Diagnoses originally 'BEFORE' are now handled by the pre_existing_dx CTE.
    WHERE NOT (m.event_type = 'DIAGNOSIS' AND m.event_detail LIKE '% - Pre-existing%')
)

SELECT
    HSP_ACCOUNT_ID,
    PAT_ENC_CSN_ID,
    section_type,
    sort_order,
    event_timestamp,
    content
FROM encounter_context

UNION ALL

SELECT * FROM pre_existing_dx

UNION ALL

SELECT * FROM timeline_events

ORDER BY sort_order, event_timestamp NULLS FIRST
""")

print(f"LLM-ready timeline (with all pre-existing diagnoses) written to {LLM_TIMELINE_TABLE}")




# =============================================================================
# STEP 10: View Results
# =============================================================================
print(f"\n{'='*60}")
print(f"RESULTS FOR {HSP_ACCOUNT_ID}")
print(f"{'='*60}")

# Uncomment to view results:
display(spark.sql(f"SELECT * FROM dev.fin_ds.llm_timeline_{HSP_ACCOUNT_ID} ORDER BY EVENT_TIMESTAMP"))

print("Done.")
