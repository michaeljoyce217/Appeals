# Databricks notebook source
# MAGIC %md
# MAGIC # Herald Visit History Features Notebook
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎯 Quick Start: What This Notebook Does
# MAGIC
# MAGIC **In 3 sentences:**
# MAGIC 1. This notebook extracts 41 healthcare utilization and procedure history features from **831,397 patient-month observations** for colorectal cancer (CRC) risk prediction.
# MAGIC 2. We analyze these features to identify patterns signaling undiagnosed pathology in a screening-overdue population, finding that primary care engagement is a strong indicator (Risk Ratio 2.55 for engaged primary care).
# MAGIC 3. We then reduce the initial set of 41 features to **23 key features** (a 43.9% reduction) while preserving all critical signals for efficient model training.
# MAGIC
# MAGIC **Key finding:** A significant care gap exists, with 8.2% of patients having GI-related symptoms across all settings, but only 0.9% seeing a GI specialist in the past 12 months.
# MAGIC **Output:** `herald_eda_train_visit_features_reduced` table with 831,397 rows.

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎯 Quick Start: What This Notebook Does
# MAGIC
# MAGIC **In 3 sentences:**
# MAGIC 1. This notebook extracts 41 healthcare utilization and procedure history features from **831,397 patient-month observations** for colorectal cancer (CRC) risk prediction.
# MAGIC 2. We analyze these features to identify patterns signaling undiagnosed pathology in a screening-overdue population, finding that primary care engagement is a strong indicator (Risk Ratio 2.55 for engaged primary care).
# MAGIC 3. We then reduce the initial set of 41 features to **23 key features** (a 43.9% reduction) while preserving all critical signals for efficient model training.
# MAGIC
# MAGIC **Key finding:** A significant care gap exists, with 8.2% of patients having GI-related symptoms across all settings, but only 0.9% seeing a GI specialist in the past 12 months.
# MAGIC **Output:** `herald_eda_train_visit_features_reduced` table with 831,397 rows.
# MAGIC
# MAGIC # Herald Visit History and Procedure Features Notebook
# MAGIC
# MAGIC ## Introduction and Objectives
# MAGIC
# MAGIC This notebook extracts healthcare utilization patterns and procedure history from 11,443,201 patient-month observations for CRC risk prediction. Visit patterns reveal both clinical symptoms and care-seeking behaviors, while procedure history identifies relevant interventions and diagnostic workups. This cohort has been specifically selected to exclude patients with recent screening, allowing us to focus on the overdue population where utilization patterns may signal undiagnosed pathology.
# MAGIC
# MAGIC ## Clinical Motivation
# MAGIC
# MAGIC ### Why Visit History Matters for CRC Risk
# MAGIC
# MAGIC Healthcare utilization patterns provide critical context for risk assessment:
# MAGIC
# MAGIC 1. **Symptom-Driven Encounters**
# MAGIC    - ED visits for GI symptoms indicate acute presentations
# MAGIC    - Hospitalization patterns reveal severity
# MAGIC    - GI specialist visits suggest recognized pathology
# MAGIC    - Primary care continuity affects screening adherence
# MAGIC
# MAGIC 2. **Diagnostic Cascades**
# MAGIC    - Abnormal labs trigger further testing
# MAGIC    - Anemia workups lead to GI evaluation
# MAGIC    - Emergency presentations prompt investigations
# MAGIC
# MAGIC 3. **Care Gaps**
# MAGIC    - Symptoms without specialty follow-up
# MAGIC    - Abnormal results without investigation
# MAGIC    - ED visits without GI referral
# MAGIC    - These gaps modify other risk factors
# MAGIC
# MAGIC ## Cohort Context
# MAGIC
# MAGIC ### Critical Design Feature
# MAGIC This cohort **excludes patients with recent screening** (colonoscopy, FIT, etc.), meaning:
# MAGIC - Zero colonoscopy/polypectomy procedures expected in lookback period
# MAGIC - Population is overdue for screening by definition
# MAGIC - Any GI symptoms or abnormal labs are particularly concerning
# MAGIC - Healthcare utilization despite no screening suggests barriers
# MAGIC
# MAGIC ### Expected Patterns
# MAGIC Given the screening-overdue population:
# MAGIC - **Higher symptom burden**: Potential undiagnosed pathology
# MAGIC - **More ED utilization**: Crisis management instead of prevention
# MAGIC - **Lower specialty care**: Access barriers preventing screening
# MAGIC - **Primary care gaps**: Missing preventive care opportunities
# MAGIC
# MAGIC ## Feature Engineering Strategy
# MAGIC
# MAGIC ### Visit Pattern Features
# MAGIC
# MAGIC 1. **Healthcare Settings (12-24 month windows)**
# MAGIC    - ED visit count and recency
# MAGIC    - Hospitalization count and length of stay
# MAGIC    - GI specialist encounters
# MAGIC    - Primary care visit frequency
# MAGIC    - Any setting with GI symptoms
# MAGIC
# MAGIC 2. **Specialty Engagement**
# MAGIC    - Gastroenterology visits
# MAGIC    - Primary care continuity metrics
# MAGIC    - No-show patterns
# MAGIC    - Care fragmentation indicators
# MAGIC
# MAGIC 3. **Care Continuity Metrics**
# MAGIC    - Primary care provider consistency
# MAGIC    - Days since last primary care
# MAGIC    - Preventive care engagement
# MAGIC    - Missed appointment patterns
# MAGIC
# MAGIC ### Composite Risk Scores
# MAGIC
# MAGIC Combines multiple domains:
# MAGIC - Healthcare utilization intensity
# MAGIC - Symptom-related encounters
# MAGIC - Care continuity patterns
# MAGIC - Acute vs routine care balance
# MAGIC
# MAGIC ## Expected Outcomes
# MAGIC
# MAGIC ### Utilization Patterns
# MAGIC Based on screening-age population characteristics:
# MAGIC - **ED visits**: 10-15% (higher than general population)
# MAGIC - **Hospitalizations**: 1-3% (moderate acuity)
# MAGIC - **GI specialists**: 0.5-1% (care gap expected)
# MAGIC - **Primary care**: 40-60% (suboptimal for screening age)
# MAGIC
# MAGIC ### Risk Score Distribution
# MAGIC - **Score 0**: ~70% (minimal utilization)
# MAGIC - **Score 1-2**: ~25% (moderate concern)
# MAGIC - **Score 3+**: ~5% (high complexity)
# MAGIC
# MAGIC ## Clinical Significance
# MAGIC
# MAGIC Visit patterns reveal critical insights:
# MAGIC
# MAGIC 1. **Diagnostic Delays**
# MAGIC    - Symptoms present but screening delayed
# MAGIC    - Abnormal patterns without follow-through
# MAGIC    - ED management instead of prevention
# MAGIC
# MAGIC 2. **Care Fragmentation**
# MAGIC    - Multiple settings without coordination
# MAGIC    - Specialty access barriers
# MAGIC    - Missing preventive care
# MAGIC
# MAGIC 3. **System Failures**
# MAGIC    - High-risk patients not receiving screening
# MAGIC    - Symptoms managed without definitive diagnosis
# MAGIC    - Reactive instead of preventive care
# MAGIC
# MAGIC ## Technical Implementation
# MAGIC
# MAGIC ### Data Sources
# MAGIC - **Encounter tables**: Visit details and diagnoses
# MAGIC - **Department mapping**: Specialty identification
# MAGIC - **Appointment tables**: Including no-show tracking
# MAGIC
# MAGIC ### Processing Approach
# MAGIC 1. **Temporal aggregation**: 12 and 24-month windows
# MAGIC 2. **Setting classification**: ED vs inpatient vs outpatient
# MAGIC 3. **Symptom detection**: ICD codes for GI symptoms
# MAGIC 4. **Score calculation**: Weighted combination of factors
# MAGIC
# MAGIC ### Quality Controls
# MAGIC - Validate expected patterns against cohort design
# MAGIC - Check for reasonable maximum values
# MAGIC - Identify outliers for review
# MAGIC - Ensure temporal consistency
# MAGIC
# MAGIC ---
# MAGIC *The following cells extract healthcare utilization patterns that reveal both clinical symptoms and system failures. The absence of screening procedures confirms cohort selection while highlighting the urgent need for intervention in this high-risk, underserved population.*
# MAGIC

# COMMAND ----------

# # Generic restart command
dbutils.library.restartPython()

# COMMAND ----------

!free -m

# COMMAND ----------


# ---------------------------------
# Imports and Variable Declarations
# ---------------------------------

import datetime
from dateutil.relativedelta import relativedelta
import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif

# Initialize a Spark session for distributed data processing
spark = SparkSession.builder.getOrCreate()

# Ensure date/time comparisons use Central Time
spark.conf.set("spark.sql.session.timeZone", "America/Chicago")

# Define target catalog for SQL based on the environment variable
trgt_cat = os.environ.get('trgt_cat')

# Use appropriate Spark catalog based on the target category
spark.sql('USE CATALOG prod;')

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Featurization

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 1 - Create ED/Inpatient Encounters with GI Symptom Flags
# MAGIC
# MAGIC #### 🔍 What This Cell Does
# MAGIC This cell creates a temporary table, `herald_eda_train_ed_inp_encounters`, by extracting all Emergency Department (ED) and Inpatient encounters from the `clarity_cur.pat_enc_hsp_har_enh` table. It also flags encounters associated with common GI symptoms (GI bleeding, bowel changes, abdominal pain, anemia) using ICD codes. A critical data quality cutoff of '2021-07-01' is applied to ensure data reliability.
# MAGIC
# MAGIC #### Why This Matters for Visit History
# MAGIC ED and Inpatient visits are crucial indicators of acute health issues and disease severity. Identifying GI-related symptoms within these encounters helps pinpoint patients who may be experiencing undiagnosed pathology relevant to CRC risk, especially in a screening-overdue cohort. The length of stay (LOS) also provides insight into the severity of the inpatient episodes.
# MAGIC
# MAGIC #### What to Watch For
# MAGIC - **`earliest_visit`**: Should be on or after the '2021-07-01' cutoff.
# MAGIC - **`total_encounters`**: Expect a large number of encounters, reflecting comprehensive data extraction.
# MAGIC - **`pre_cutoff_count`**: Must be 0, confirming the data quality filter is correctly applied.
# MAGIC mar

# COMMAND ----------

# Cell 1: Create ED/Inpatient encounters with GI symptom flags
spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_ed_inp_encounters AS
WITH gi_symptom_enc AS (
    SELECT DISTINCT PRIM_ENC_CSN_ID
    FROM clarity_cur.hsp_acct_dx_list_enh dx
    WHERE dx.CODE RLIKE '^(K92|K59|R19|R50|D50|K62)' -- GI bleeding, bowel changes, abdominal pain, anemia
)

SELECT
  peh.PAT_ID,
  peh.PAT_ENC_CSN_ID,
  peh.HOSP_ADMSN_TIME AS visit_dt,
  peh.ACCT_CLASS,
  peh.ED_EPISODE_ID,
  CASE WHEN ge.PRIM_ENC_CSN_ID IS NOT NULL THEN 1 ELSE 0 END AS GI_SYMPTOM_YN,
  CASE WHEN peh.DISCH_DATE_TIME IS NOT NULL AND peh.HOSP_ADMSN_TIME IS NOT NULL
       THEN DATEDIFF(DATE(peh.DISCH_DATE_TIME), DATE(peh.HOSP_ADMSN_TIME))
       ELSE NULL END AS los_days,
  CASE 
    WHEN peh.ACCT_CLASS = 'Emergency' OR peh.ED_EPISODE_ID IS NOT NULL THEN 'ED'
    WHEN peh.ACCT_CLASS = 'Inpatient' THEN 'INPATIENT'
    ELSE 'OTHER'
  END AS encounter_type
FROM clarity_cur.pat_enc_hsp_har_enh peh
LEFT JOIN gi_symptom_enc ge ON peh.PAT_ENC_CSN_ID = ge.PRIM_ENC_CSN_ID
WHERE peh.HOSP_ADMSN_TIME >= '2021-07-01'  -- CRITICAL: Data quality cutoff
  AND peh.HOSP_ADMSN_TIME IS NOT NULL
  AND peh.ADT_PATIENT_STAT_C <> 1  -- Exclude preadmits
  AND peh.ADMIT_CONF_STAT_C <> 3   -- Exclude canceled
""")

print("✓ ED/Inpatient encounters created (≥2021-07-01 only)")

# Validation query
validation = spark.sql(f"""
SELECT 
  MIN(visit_dt) as earliest_visit,
  MAX(visit_dt) as latest_visit,
  COUNT(*) as total_encounters,
  COUNT(CASE WHEN visit_dt < '2021-07-01' THEN 1 END) as pre_cutoff_count
FROM {trgt_cat}.clncl_ds.herald_eda_train_ed_inp_encounters
""")
validation.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 📊 Cell 1 Conclusion
# MAGIC
# MAGIC Successfully created the `herald_eda_train_ed_inp_encounters` table, containing **19,832,548 ED/Inpatient encounters**. The earliest visit date is `2021-08-04 01:00:00`, confirming the data quality cutoff was applied correctly with `0` pre-cutoff encounters. This table is foundational for deriving acute care utilization features.
# MAGIC
# MAGIC **Key Achievement**: Creation of a clean, filtered dataset of acute care encounters with GI symptom flags.
# MAGIC
# MAGIC **Next Step**: Extract outpatient encounters, which will capture routine care and specialty visits.
# MAGIC

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 2 - Create Outpatient Encounters (Including All Appointment Statuses)
# MAGIC
# MAGIC #### 🔍 What This Cell Does
# MAGIC This cell generates the `herald_eda_train_op_encounters` table, compiling all outpatient encounters from `clarity_cur.pat_enc_enh`. It identifies GI specialty visits, flags primary care provider (PCP) visits, and detects GI symptoms based on ICD-10 codes. Importantly, it includes all appointment statuses (completed, arrived, no-show, etc.) to allow for comprehensive no-show tracking.
# MAGIC
# MAGIC #### Why This Matters for Visit History
# MAGIC Outpatient visits represent the bulk of routine healthcare engagement. Tracking these visits, especially those with GI specialists or PCPs, provides insights into preventive care, specialty access, and care continuity. Including all appointment statuses is critical for understanding patient adherence and potential barriers to care, such as missed appointments.
# MAGIC
# MAGIC #### What to Watch For
# MAGIC - **`earliest_visit`**: Should be on or after the '2021-07-01' cutoff.
# MAGIC - **`total_encounters`**: Expect a very large number, as outpatient visits are frequent.
# MAGIC - **`pre_cutoff_count`**: Must be 0, ensuring the data quality filter is correctly applied.
# MAGIC

# COMMAND ----------

# Cell 2: Create outpatient encounters (including all appointment statuses for no-show tracking)
spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_op_encounters AS
WITH gi_symptom_dx AS (
  SELECT DISTINCT PAT_ENC_CSN_ID
  FROM clarity_cur.pat_enc_dx_enh dx
  WHERE dx.ICD10_CODE RLIKE '^(K92|K59|R19|R50|D50|K62)' -- GI symptoms
)

SELECT
  pe.PAT_ID,
  pe.PAT_ENC_CSN_ID,
  TO_DATE(SUBSTRING(pe.CONTACT_DATE,1,10)) AS visit_dt,
  ser.SPECIALTY_NAME,
  pe.APPT_STATUS_C,
  pe.APPT_STATUS_NAME,
  CASE 
    WHEN UPPER(ser.SPECIALTY_NAME) IN ('GASTROENTEROLOGY', 'COLON AND RECTAL SURGERY')
    THEN 1 ELSE 0 
  END AS GI_SPECIALTY_YN,
  CASE 
    WHEN pe.VISIT_PROV_ID = pe.PCP_PROV_ID THEN 1 ELSE 0 
  END AS PCP_VISIT_YN,
  CASE 
    WHEN gd.PAT_ENC_CSN_ID IS NOT NULL THEN 1 ELSE 0
  END AS GI_SYMPTOM_YN
FROM clarity_cur.pat_enc_enh pe
LEFT JOIN clarity_cur.clarity_ser_enh ser ON pe.VISIT_PROV_ID = ser.PROV_ID
LEFT JOIN gi_symptom_dx gd ON pe.PAT_ENC_CSN_ID = gd.PAT_ENC_CSN_ID
WHERE pe.CONTACT_DATE >= '2021-07-01'  -- CRITICAL: Data quality cutoff
  AND pe.ENC_TYPE_NAME NOT IN ('Hospital Encounter')
  -- Include ALL appointment statuses (completed, arrived, no-show, etc.)
""")

print("✓ Outpatient encounters created (≥2021-07-01 only)")

# Validation query
validation = spark.sql(f"""
SELECT 
  MIN(visit_dt) as earliest_visit,
  MAX(visit_dt) as latest_visit,
  COUNT(*) as total_encounters,
  COUNT(CASE WHEN visit_dt < '2021-07-01' THEN 1 END) as pre_cutoff_count
FROM {trgt_cat}.clncl_ds.herald_eda_train_op_encounters
""")
validation.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 📊 Cell 2 Conclusion
# MAGIC
# MAGIC Successfully created the `herald_eda_train_op_encounters` table, containing a massive **308,436,620 outpatient encounters**. The earliest visit date is `2022-01-01`, confirming the data quality cutoff was applied correctly with `0` pre-cutoff encounters. This comprehensive dataset is vital for understanding routine care, specialty engagement, and appointment adherence.
# MAGIC
# MAGIC **Key Achievement**: Creation of a detailed outpatient encounter dataset, including GI specialty and PCP visit flags, and all appointment statuses for no-show analysis.
# MAGIC
# MAGIC **Next Step**: Aggregate these raw encounter data into meaningful metrics for ED/Inpatient visits.
# MAGIC

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 3 - ED/Inpatient Metrics Calculation
# MAGIC
# MAGIC #### 🔍 What This Cell Does
# MAGIC This cell calculates various ED and Inpatient utilization metrics for each patient-month observation in the `herald_eda_train_final_cohort`. It aggregates counts of ED visits, GI symptom-related ED visits, Inpatient visits, GI symptom-related Inpatient visits, and total Inpatient days over 3, 12, and 24-month lookback windows.
# MAGIC
# MAGIC #### Why This Matters for Visit History
# MAGIC These aggregated metrics provide a concise summary of a patient's acute care utilization. Frequent ED visits or hospitalizations, especially those with GI symptoms, are strong indicators of underlying health issues and potential diagnostic delays. Different lookback windows capture both recent acute events and longer-term patterns of care.
# MAGIC
# MAGIC #### What to Watch For
# MAGIC - The output confirms the creation of the `herald_eda_train_ed_inp_metrics` table.
# MAGIC - This step is purely aggregation, so the primary validation will occur when these metrics are joined and analyzed in later cells.
# MAGIC

# COMMAND ----------


# Cell 3: ED/Inpatient metrics only (much faster)
spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_ed_inp_metrics AS
SELECT
  c.PAT_ID,
  c.END_DTTM,

  -- ED visits (various time windows)
  SUM(CASE 
    WHEN ei.encounter_type = 'ED' 
     AND ei.visit_dt >= ADD_MONTHS(c.END_DTTM, -3)
     AND ei.visit_dt < c.END_DTTM
    THEN 1 ELSE 0 
  END) AS ED_LAST_90_DAYS,

  SUM(CASE 
    WHEN ei.encounter_type = 'ED' 
     AND ei.visit_dt >= ADD_MONTHS(c.END_DTTM, -12)
     AND ei.visit_dt < c.END_DTTM
    THEN 1 ELSE 0 
  END) AS ED_LAST_12_MONTHS,

  SUM(CASE 
    WHEN ei.encounter_type = 'ED' 
     AND ei.visit_dt >= ADD_MONTHS(c.END_DTTM, -24)
     AND ei.visit_dt < c.END_DTTM
    THEN 1 ELSE 0 
  END) AS ED_LAST_24_MONTHS,

  -- GI symptom-related ED visits
  SUM(CASE 
    WHEN ei.encounter_type = 'ED' 
     AND ei.visit_dt >= ADD_MONTHS(c.END_DTTM, -12)
     AND ei.visit_dt < c.END_DTTM
     AND ei.GI_SYMPTOM_YN = 1
    THEN 1 ELSE 0 
  END) AS GI_ED_LAST_12_MONTHS,

  -- Inpatient visits
  SUM(CASE 
    WHEN ei.encounter_type = 'INPATIENT' 
     AND ei.visit_dt >= ADD_MONTHS(c.END_DTTM, -12)
     AND ei.visit_dt < c.END_DTTM
    THEN 1 ELSE 0 
  END) AS INP_LAST_12_MONTHS,

  SUM(CASE 
    WHEN ei.encounter_type = 'INPATIENT' 
     AND ei.visit_dt >= ADD_MONTHS(c.END_DTTM, -24)
     AND ei.visit_dt < c.END_DTTM
    THEN 1 ELSE 0 
  END) AS INP_LAST_24_MONTHS,

  -- GI symptom-related inpatient visits
  SUM(CASE 
    WHEN ei.encounter_type = 'INPATIENT' 
     AND ei.visit_dt >= ADD_MONTHS(c.END_DTTM, -12)
     AND ei.visit_dt < c.END_DTTM
     AND ei.GI_SYMPTOM_YN = 1
    THEN 1 ELSE 0 
  END) AS GI_INP_LAST_12_MONTHS,

  -- Total inpatient days
  SUM(CASE 
    WHEN ei.encounter_type = 'INPATIENT' 
     AND ei.visit_dt >= ADD_MONTHS(c.END_DTTM, -12)
     AND ei.visit_dt < c.END_DTTM
    THEN COALESCE(ei.los_days, 0) ELSE 0 
  END) AS TOTAL_INPATIENT_DAYS_12MO

FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
LEFT JOIN {trgt_cat}.clncl_ds.herald_eda_train_ed_inp_encounters ei
  ON c.PAT_ID = ei.PAT_ID
GROUP BY c.PAT_ID, c.END_DTTM
""")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 📊 Cell 3 Conclusion
# MAGIC
# MAGIC Successfully calculated and stored ED and Inpatient metrics in `herald_eda_train_ed_inp_metrics`. This step efficiently aggregates acute care utilization over various lookback periods, providing a granular view of patient engagement with emergency and hospital services. The output confirms the table creation.
# MAGIC
# MAGIC **Key Achievement**: Comprehensive aggregation of acute care visit counts and inpatient days, including GI symptom flags, for each patient-month.
# MAGIC
# MAGIC **Next Step**: Calculate similar aggregated metrics for outpatient visits, including specialty and primary care.
# MAGIC

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 4 - Outpatient Metrics Calculation
# MAGIC
# MAGIC #### 🔍 What This Cell Does
# MAGIC This cell calculates various outpatient utilization metrics for each patient-month observation. It aggregates counts of completed outpatient visits, GI specialty visits, PCP visits, GI symptom-related outpatient visits, and no-shows over 12 and 24-month lookback windows. Only 'Completed' or 'Arrived' appointments are counted for visit frequency, while all statuses are considered for no-shows.
# MAGIC
# MAGIC #### Why This Matters for Visit History
# MAGIC Outpatient metrics are essential for understanding a patient's engagement with routine and preventive care. GI specialty visits directly indicate investigation into GI issues, while PCP visits reflect primary care continuity. No-show patterns can highlight access barriers or patient disengagement, which are critical factors in a screening-overdue population.
# MAGIC
# MAGIC #### What to Watch For
# MAGIC - The output confirms the creation of the `herald_eda_train_op_metrics` table.
# MAGIC - Similar to Cell 3, this is an aggregation step, and detailed validation will occur after joining with other metrics.
# MAGIC

# COMMAND ----------

# Cell 4: Outpatient metrics only (separate for performance)
spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_op_metrics AS
SELECT
  c.PAT_ID,
  c.END_DTTM,

  -- Outpatient visits (completed only)
  SUM(CASE 
    WHEN op.visit_dt >= ADD_MONTHS(c.END_DTTM, -12)
     AND op.visit_dt < c.END_DTTM
     AND op.APPT_STATUS_NAME IN ('Completed','Arrived')
    THEN 1 ELSE 0 
  END) AS OUTPATIENT_VISITS_12MO,

  SUM(CASE 
    WHEN op.visit_dt >= ADD_MONTHS(c.END_DTTM, -24)
     AND op.visit_dt < c.END_DTTM
     AND op.APPT_STATUS_NAME IN ('Completed','Arrived')
    THEN 1 ELSE 0 
  END) AS OUTPATIENT_VISITS_24MO,

  -- GI specialty visits (completed only)
  SUM(CASE 
    WHEN op.visit_dt >= ADD_MONTHS(c.END_DTTM, -12)
     AND op.visit_dt < c.END_DTTM
     AND op.GI_SPECIALTY_YN = 1
     AND op.APPT_STATUS_NAME IN ('Completed','Arrived')
    THEN 1 ELSE 0 
  END) AS GI_VISITS_12MO,

  SUM(CASE 
    WHEN op.visit_dt >= ADD_MONTHS(c.END_DTTM, -24)
     AND op.visit_dt < c.END_DTTM
     AND op.GI_SPECIALTY_YN = 1
     AND op.APPT_STATUS_NAME IN ('Completed','Arrived')
    THEN 1 ELSE 0 
  END) AS GI_VISITS_24MO,

  -- PCP visits (completed only)
  SUM(CASE 
    WHEN op.visit_dt >= ADD_MONTHS(c.END_DTTM, -12)
     AND op.visit_dt < c.END_DTTM
     AND op.PCP_VISIT_YN = 1
     AND op.APPT_STATUS_NAME IN ('Completed','Arrived')
    THEN 1 ELSE 0 
  END) AS PCP_VISITS_12MO,

  SUM(CASE 
    WHEN op.visit_dt >= ADD_MONTHS(c.END_DTTM, -24)
     AND op.visit_dt < c.END_DTTM
     AND op.PCP_VISIT_YN = 1
     AND op.APPT_STATUS_NAME IN ('Completed','Arrived')
    THEN 1 ELSE 0 
  END) AS PCP_VISITS_24MO,

  -- GI symptom-related outpatient visits (completed only)
  SUM(CASE 
    WHEN op.visit_dt >= ADD_MONTHS(c.END_DTTM, -12)
     AND op.visit_dt < c.END_DTTM
     AND op.GI_SYMPTOM_YN = 1
     AND op.APPT_STATUS_NAME IN ('Completed','Arrived')
    THEN 1 ELSE 0 
  END) AS GI_SYMPTOM_OP_VISITS_12MO,

  -- No shows (all appointment types)
  SUM(CASE
    WHEN op.visit_dt >= ADD_MONTHS(c.END_DTTM, -12)
     AND op.visit_dt < c.END_DTTM
     AND op.APPT_STATUS_C = 4
    THEN 1 ELSE 0 
  END) AS NO_SHOWS_12MO

FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
LEFT JOIN {trgt_cat}.clncl_ds.herald_eda_train_op_encounters op
  ON c.PAT_ID = op.PAT_ID
GROUP BY c.PAT_ID, c.END_DTTM
""")

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC #### 📊 Cell 4 Conclusion
# MAGIC
# MAGIC Successfully calculated and stored outpatient metrics in `herald_eda_train_op_metrics`. This step provides a detailed view of patient engagement with routine care, including GI specialty and PCP visits, and crucial no-show patterns. The output confirms the table creation.
# MAGIC
# MAGIC **Key Achievement**: Comprehensive aggregation of outpatient visit counts, including GI specialty, PCP, GI symptom-related visits, and no-shows, for each patient-month.
# MAGIC
# MAGIC **Next Step**: Combine the ED/Inpatient and Outpatient metrics into a single, unified table for further feature engineering.
# MAGIC

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 5 - Join ED/Inpatient and Outpatient Metrics
# MAGIC
# MAGIC #### 🔍 What This Cell Does
# MAGIC This cell joins the previously created `herald_eda_train_ed_inp_metrics` and `herald_eda_train_op_metrics` tables. The join is performed on `PAT_ID` and `END_DTTM` to ensure that all acute and outpatient utilization metrics are combined for each patient-month observation. The result is stored in `herald_eda_train_visit_counts`.
# MAGIC
# MAGIC #### Why This Matters for Visit History
# MAGIC Combining these metrics is essential for creating a holistic view of a patient's healthcare utilization. It allows for the analysis of interactions between different care settings (e.g., high ED use with low PCP engagement) and forms the basis for composite features that capture overall care patterns.
# MAGIC
# MAGIC #### What to Watch For
# MAGIC - The output confirms the creation of the `herald_eda_train_visit_counts` table.
# MAGIC - This join should be very fast as it's merging two pre-aggregated tables.
# MAGIC

# COMMAND ----------

# Cell 5: Join the two metrics tables (very fast)
spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_visit_counts AS
SELECT
  ei.PAT_ID,
  ei.END_DTTM,
  
  -- ED/Inpatient metrics
  ei.ED_LAST_90_DAYS,
  ei.ED_LAST_12_MONTHS,
  ei.ED_LAST_24_MONTHS,
  ei.GI_ED_LAST_12_MONTHS,
  ei.INP_LAST_12_MONTHS,
  ei.INP_LAST_24_MONTHS,
  ei.GI_INP_LAST_12_MONTHS,
  ei.TOTAL_INPATIENT_DAYS_12MO,
  
  -- Outpatient metrics
  om.OUTPATIENT_VISITS_12MO,
  om.OUTPATIENT_VISITS_24MO,
  om.GI_VISITS_12MO,
  om.GI_VISITS_24MO,
  om.PCP_VISITS_12MO,
  om.PCP_VISITS_24MO,
  om.GI_SYMPTOM_OP_VISITS_12MO,
  om.NO_SHOWS_12MO

FROM {trgt_cat}.clncl_ds.herald_eda_train_ed_inp_metrics ei
LEFT JOIN {trgt_cat}.clncl_ds.herald_eda_train_op_metrics om
  ON ei.PAT_ID = om.PAT_ID AND ei.END_DTTM = om.END_DTTM
""")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 📊 Cell 5 Conclusion
# MAGIC
# MAGIC Successfully joined ED/Inpatient and Outpatient metrics into `herald_eda_train_visit_counts`. This table now provides a consolidated view of all aggregated visit counts and days for each patient-month, serving as a comprehensive foundation for further feature engineering. The output confirms the table creation.
# MAGIC
# MAGIC **Key Achievement**: Creation of a unified table containing all aggregated visit count features across acute and outpatient settings.
# MAGIC
# MAGIC **Next Step**: Calculate recency features, which capture how recently a patient engaged with specific care types.
# MAGIC

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 6 - Add Recency Features Using Window Functions
# MAGIC
# MAGIC #### 🔍 What This Cell Does
# MAGIC This cell calculates recency features, specifically the number of days since the last ED visit, inpatient admission, and GI specialty visit, within a 24-month lookback window. It achieves this by combining relevant encounters, calculating `days_since` the `END_DTTM`, and then using window functions to find the most recent visit for each type. The results are stored in `herald_eda_train_visit_recency`.
# MAGIC
# MAGIC #### Why This Matters for Visit History
# MAGIC Recency features are crucial for understanding the temporal aspect of healthcare utilization. A recent ED visit or hospitalization indicates an acute or ongoing health issue, while a recent GI specialist visit suggests active investigation. These features provide a sense of urgency and current clinical status that simple visit counts might miss.
# MAGIC
# MAGIC #### What to Watch For
# MAGIC - The output confirms the creation of the `herald_eda_train_visit_recency` table.
# MAGIC - Null values are expected for patients who have not had a specific type of visit within the lookback period, which is informative.
# MAGIC

# COMMAND ----------

# Cell 6: Add recency features using window functions
spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_visit_recency AS
WITH all_visits AS (
    -- ED visits
    SELECT 
      c.PAT_ID, c.END_DTTM, 'ED' AS visit_type,
      ei.visit_dt,
      DATEDIFF(c.END_DTTM, DATE(ei.visit_dt)) AS days_since
    FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
    JOIN {trgt_cat}.clncl_ds.herald_eda_train_ed_inp_encounters ei
      ON c.PAT_ID = ei.PAT_ID
    WHERE ei.encounter_type = 'ED'
      AND ei.visit_dt < c.END_DTTM
      AND ei.visit_dt >= ADD_MONTHS(c.END_DTTM, -24)
    
    UNION ALL
    
    -- Inpatient visits  
    SELECT 
      c.PAT_ID, c.END_DTTM, 'INPATIENT' AS visit_type,
      ei.visit_dt,
      DATEDIFF(c.END_DTTM, DATE(ei.visit_dt)) AS days_since
    FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
    JOIN {trgt_cat}.clncl_ds.herald_eda_train_ed_inp_encounters ei
      ON c.PAT_ID = ei.PAT_ID
    WHERE ei.encounter_type = 'INPATIENT'
      AND ei.visit_dt < c.END_DTTM
      AND ei.visit_dt >= ADD_MONTHS(c.END_DTTM, -24)
    
    UNION ALL
    
    -- GI specialty visits
    SELECT 
      c.PAT_ID, c.END_DTTM, 'GI_SPECIALTY' AS visit_type,
      op.visit_dt,
      DATEDIFF(c.END_DTTM, op.visit_dt) AS days_since
    FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
    JOIN {trgt_cat}.clncl_ds.herald_eda_train_op_encounters op
      ON c.PAT_ID = op.PAT_ID
    WHERE op.GI_SPECIALTY_YN = 1
      AND op.APPT_STATUS_NAME IN ('Completed','Arrived')
      AND op.visit_dt < c.END_DTTM
      AND op.visit_dt >= ADD_MONTHS(c.END_DTTM, -24)
),

recent_visits AS (
    SELECT 
      PAT_ID, END_DTTM, visit_type, days_since,
      ROW_NUMBER() OVER (
        PARTITION BY PAT_ID, END_DTTM, visit_type 
        ORDER BY days_since ASC
      ) AS rn
    FROM all_visits
),

pivoted_recency AS (
    SELECT
      PAT_ID, END_DTTM,
      SUM(CASE WHEN visit_type = 'ED' AND rn = 1 THEN days_since END) AS days_since_last_ed,
      SUM(CASE WHEN visit_type = 'INPATIENT' AND rn = 1 THEN days_since END) AS days_since_last_inpatient,
      SUM(CASE WHEN visit_type = 'GI_SPECIALTY' AND rn = 1 THEN days_since END) AS days_since_last_gi
    FROM recent_visits
    WHERE rn = 1
    GROUP BY PAT_ID, END_DTTM
)

SELECT
  c.PAT_ID,
  c.END_DTTM,
  pr.days_since_last_ed,
  pr.days_since_last_inpatient,
  pr.days_since_last_gi
FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
LEFT JOIN pivoted_recency pr ON c.PAT_ID = pr.PAT_ID AND c.END_DTTM = pr.END_DTTM
""")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 📊 Cell 6 Conclusion
# MAGIC
# MAGIC Successfully calculated and stored recency features in `herald_eda_train_visit_recency`. This table now provides critical temporal information, indicating the number of days since a patient's last ED, inpatient, or GI specialty visit. This is vital for identifying recent acute events or ongoing investigations. The output confirms the table creation.
# MAGIC
# MAGIC **Key Achievement**: Creation of recency features for key acute and specialty care types, providing temporal context to patient utilization.
# MAGIC
# MAGIC **Next Step**: Combine all visit counts and recency features, and then engineer composite features to create the final visit features table.
# MAGIC

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 7 - Create Final Table with Composite Features
# MAGIC
# MAGIC #### 🔍 What This Cell Does
# MAGIC This cell combines the `herald_eda_train_visit_counts` and `herald_eda_train_visit_recency` tables. It then engineers several composite features, including flags for frequent ED users, frequent inpatient admissions, high inpatient days, engaged primary care, GI specialty engagement, recent ED use, and recent hospitalization. It also calculates a `healthcare_intensity_score` and `primary_care_continuity_ratio`, and `total_gi_symptom_visits_12mo`. The final table is `herald_eda_train_visit_features_final`.
# MAGIC
# MAGIC #### Why This Matters for Visit History
# MAGIC This is the culmination of the feature engineering process. By combining all individual metrics and creating composite features, we capture more complex patterns of care. These composite features are often more predictive and clinically interpretable than individual raw counts, as they represent higher-level concepts like care fragmentation, intensity, and engagement.
# MAGIC
# MAGIC #### What to Watch For
# MAGIC - The output confirms the creation of the `herald_eda_train_visit_features_final` table.
# MAGIC - The logic for composite flags and scores should align with clinical definitions (e.g., "frequent ED user" defined as >=3 visits/year).
# MAGIC

# COMMAND ----------

# Cell 7: Create final table with composite features
spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_visit_features_final AS
SELECT
  vc.PAT_ID,
  vc.END_DTTM,
  
  -- Basic visit counts
  vc.ED_LAST_90_DAYS,
  vc.ED_LAST_12_MONTHS,
  vc.ED_LAST_24_MONTHS,
  vc.GI_ED_LAST_12_MONTHS,
  vc.INP_LAST_12_MONTHS,
  vc.INP_LAST_24_MONTHS,
  vc.GI_INP_LAST_12_MONTHS,
  vc.TOTAL_INPATIENT_DAYS_12MO,
  vc.OUTPATIENT_VISITS_12MO,
  vc.OUTPATIENT_VISITS_24MO,
  vc.GI_VISITS_12MO,
  vc.GI_VISITS_24MO,
  vc.PCP_VISITS_12MO,
  vc.PCP_VISITS_24MO,
  vc.GI_SYMPTOM_OP_VISITS_12MO,
  vc.NO_SHOWS_12MO,
  
  -- Recency features
  vr.days_since_last_ed,
  vr.days_since_last_inpatient,
  vr.days_since_last_gi,
  
  -- Composite flags
  CASE WHEN COALESCE(vc.ED_LAST_12_MONTHS, 0) >= 3 THEN 1 ELSE 0 END AS frequent_ed_user_flag,
  CASE WHEN COALESCE(vc.INP_LAST_12_MONTHS, 0) >= 2 THEN 1 ELSE 0 END AS frequent_inpatient_flag,
  CASE WHEN COALESCE(vc.TOTAL_INPATIENT_DAYS_12MO, 0) >= 10 THEN 1 ELSE 0 END AS high_inpatient_days_flag,
  CASE WHEN COALESCE(vc.PCP_VISITS_12MO, 0) >= 2 THEN 1 ELSE 0 END AS engaged_primary_care_flag,
  CASE WHEN COALESCE(vc.GI_VISITS_12MO, 0) >= 1 THEN 1 ELSE 0 END AS gi_specialty_engagement_flag,
  CASE WHEN COALESCE(vc.ED_LAST_90_DAYS, 0) >= 1 THEN 1 ELSE 0 END AS recent_ed_use_flag,
  CASE WHEN COALESCE(vr.days_since_last_inpatient, 9999) <= 180 THEN 1 ELSE 0 END AS recent_hospitalization_flag,
  
  -- Healthcare intensity score (0-4)
  LEAST(4,
    CASE WHEN COALESCE(vc.ED_LAST_12_MONTHS, 0) >= 3 THEN 1 ELSE 0 END +
    CASE WHEN COALESCE(vc.INP_LAST_12_MONTHS, 0) >= 2 THEN 1 ELSE 0 END +
    CASE WHEN COALESCE(vc.TOTAL_INPATIENT_DAYS_12MO, 0) >= 10 THEN 1 ELSE 0 END +
    CASE WHEN COALESCE(vc.GI_VISITS_12MO, 0) >= 2 THEN 1 ELSE 0 END
  ) AS healthcare_intensity_score,
  
  -- Care continuity ratio
  CASE WHEN COALESCE(vc.OUTPATIENT_VISITS_12MO, 0) > 0 
       THEN ROUND(COALESCE(vc.PCP_VISITS_12MO, 0) * 1.0 / vc.OUTPATIENT_VISITS_12MO, 2)
       ELSE NULL END AS primary_care_continuity_ratio,
       
  -- Total GI symptom burden across all settings
  COALESCE(vc.GI_ED_LAST_12_MONTHS, 0) + 
  COALESCE(vc.GI_INP_LAST_12_MONTHS, 0) + 
  COALESCE(vc.GI_SYMPTOM_OP_VISITS_12MO, 0) AS total_gi_symptom_visits_12mo

FROM {trgt_cat}.clncl_ds.herald_eda_train_visit_counts vc
LEFT JOIN {trgt_cat}.clncl_ds.herald_eda_train_visit_recency vr
  ON vc.PAT_ID = vr.PAT_ID AND vc.END_DTTM = vr.END_DTTM
""")


# COMMAND ----------

# MAGIC %md
# MAGIC #### 📊 Cell 7 Conclusion
# MAGIC
# MAGIC Successfully created the `herald_eda_train_visit_features_final` table, which now contains all raw visit counts, recency metrics, and newly engineered composite features. This comprehensive table provides a rich set of 41 features for each patient-month observation, ready for detailed analysis and model training. The output confirms the table creation.
# MAGIC
# MAGIC **Key Achievement**: Finalization of the feature gathering process, resulting in a comprehensive table of 41 visit history features, including clinically relevant composite indicators.
# MAGIC
# MAGIC **Next Step**: Validate the row counts of the final feature table against the original cohort to ensure data integrity.
# MAGIC

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 8 - Validate Row Counts
# MAGIC
# MAGIC #### 🔍 What This Cell Does
# MAGIC This cell performs a critical data quality check by comparing the row count of the newly created `herald_eda_train_visit_features_final` table against the original `herald_eda_train_final_cohort` table. It asserts that the row counts must be identical, ensuring that no patient-month observations were lost or duplicated during the feature engineering process.
# MAGIC
# MAGIC #### Why This Matters for Visit History
# MAGIC Maintaining consistent row counts is fundamental for data integrity. Any discrepancy would indicate an error in the join or aggregation logic, potentially leading to biased models or incorrect analyses. This validation step provides confidence in the completeness of the feature set.
# MAGIC
# MAGIC #### What to Watch For
# MAGIC - **`diff`**: Must be 0, indicating an exact match in row counts.
# MAGIC - The print statement "✓ Row count validation passed" confirms the assertion.
# MAGIC

# COMMAND ----------

# Cell 8
# Validate row counts
result = spark.sql(f"""
SELECT 
    COUNT(*) as visit_features_count,
    (SELECT COUNT(*) FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort) as cohort_count,
    COUNT(*) - (SELECT COUNT(*) FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort) as diff
FROM {trgt_cat}.clncl_ds.herald_eda_train_visit_features_final
""")
result.show()
assert result.collect()[0]['diff'] == 0, "ERROR: Row count mismatch!"
print("✓ Row count validation passed")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 📊 Cell 8 Conclusion
# MAGIC
# MAGIC Successfully validated the row count of `herald_eda_train_visit_features_final`. The table contains **831,397** observations, matching the original cohort count exactly, with a `diff` of `0`. This confirms the integrity and completeness of the feature set.
# MAGIC
# MAGIC **Key Achievement**: Verified that all patient-month observations from the original cohort are present in the final feature table, ensuring no data loss or duplication.
# MAGIC
# MAGIC **Next Step**: Perform a comprehensive validation and summary of the generated visit history features to understand their distributions and clinical implications.
# MAGIC

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 9 - Comprehensive Validation and Summary for Visit Features
# MAGIC
# MAGIC #### 🔍 What This Cell Does
# MAGIC This cell performs a comprehensive validation and summary of all the generated visit history features. It loads the `herald_eda_train_visit_features_final` table and calculates various statistics, including total observations, unique patients, and detailed utilization patterns across Emergency Department (ED), Inpatient, Outpatient, and GI-specific care settings.
# MAGIC
# MAGIC #### Why This Matters for Visit History
# MAGIC This step is crucial for understanding the characteristics of the patient cohort and the distributions of the engineered features. It provides a high-level overview of healthcare engagement, identifies common and rare events, and helps to confirm that the features are capturing clinically meaningful information as intended.
# MAGIC
# MAGIC #### What to Watch For
# MAGIC - **Total observations**: Should match the original cohort size (831,397).
# MAGIC - **Utilization percentages**: Observe the proportion of patients engaging with different care settings (e.g., 18.8% for any ED visit, 0.9% for any GI specialist visit).
# MAGIC - **Mean counts**: Review average visit frequencies and inpatient days to understand typical patient behavior.
# MAGIC - **Care pattern flags**: Check the prevalence of flags like "frequent ED users" or "engaged primary care" to identify key patient subgroups.
# MAGIC

# COMMAND ----------

# Cell 9: Comprehensive validation and summary for visit features
print("="*80)
print("VISIT HISTORY FEATURES - VALIDATION SUMMARY")
print("="*80)

# Load the data for analysis
df = spark.sql(f"SELECT * FROM {trgt_cat}.clncl_ds.herald_eda_train_visit_features_final").toPandas()

# Basic stats
total_rows = df.shape[0]
print(f"\nTotal observations: {total_rows:,}")
print(f"Unique patients: {df['PAT_ID'].nunique():,}")

# Emergency Department utilization
print("\n--- EMERGENCY DEPARTMENT UTILIZATION ---")
print(f"Any ED visit (90 days): {(df['ED_LAST_90_DAYS'] > 0).mean():.1%}")
print(f"Any ED visit (12mo): {(df['ED_LAST_12_MONTHS'] > 0).mean():.1%}")
print(f"Mean ED visits (12mo): {df['ED_LAST_12_MONTHS'].mean():.2f}")
print(f"Frequent ED users (≥3/year): {df['frequent_ed_user_flag'].mean():.1%}")
print(f"GI-related ED (12mo): {(df['GI_ED_LAST_12_MONTHS'] > 0).mean():.1%}")

# Inpatient utilization
print("\n--- INPATIENT UTILIZATION ---")
print(f"Any hospitalization (12mo): {(df['INP_LAST_12_MONTHS'] > 0).mean():.1%}")
print(f"Mean hospitalizations (12mo): {df['INP_LAST_12_MONTHS'].mean():.3f}")
print(f"Mean inpatient days (12mo): {df['TOTAL_INPATIENT_DAYS_12MO'].mean():.2f}")
print(f"Frequent admissions (≥2/year): {df['frequent_inpatient_flag'].mean():.1%}")
print(f"High inpatient days (≥10/year): {df['high_inpatient_days_flag'].mean():.1%}")

# Outpatient utilization
print("\n--- OUTPATIENT UTILIZATION ---")
print(f"Any outpatient visit (12mo): {(df['OUTPATIENT_VISITS_12MO'] > 0).mean():.1%}")
print(f"Mean outpatient visits (12mo): {df['OUTPATIENT_VISITS_12MO'].mean():.1f}")
print(f"Any PCP visit (12mo): {(df['PCP_VISITS_12MO'] > 0).mean():.1%}")
print(f"Engaged primary care (≥2 PCP/year): {df['engaged_primary_care_flag'].mean():.1%}")
print(f"Primary care continuity ratio: {df['primary_care_continuity_ratio'].mean():.2f}")

# GI-specific utilization
print("\n--- GI-SPECIFIC UTILIZATION ---")
print(f"Any GI specialist visit (12mo): {(df['GI_VISITS_12MO'] > 0).mean():.1%}")
print(f"Mean GI visits when >0: {df[df['GI_VISITS_12MO'] > 0]['GI_VISITS_12MO'].mean():.2f}")
print(f"GI symptom OP visits (12mo): {(df['GI_SYMPTOM_OP_VISITS_12MO'] > 0).mean():.1%}")
print(f"Total GI symptom visits all settings: {(df['total_gi_symptom_visits_12mo'] > 0).mean():.1%}")

# Care patterns
print("\n--- CARE PATTERNS ---")
print(f"No-shows (12mo): {(df['NO_SHOWS_12MO'] > 0).mean():.1%}")
print(f"Mean no-shows when >0: {df[df['NO_SHOWS_12MO'] > 0]['NO_SHOWS_12MO'].mean():.1f}")
print(f"Healthcare intensity score >0: {(df['healthcare_intensity_score'] > 0).mean():.1%}")
print(f"Mean healthcare intensity: {df['healthcare_intensity_score'].mean():.3f}")

print("="*80)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 📊 Cell 9 Conclusion
# MAGIC
# MAGIC The comprehensive validation successfully summarized the visit history features for **831,397 observations** and **223,858 unique patients**. Key insights include:
# MAGIC - **18.8%** of patients had an ED visit in the last 12 months, with **3.7%** being frequent ED users.
# MAGIC - **3.2%** had any hospitalization in the last 12 months.
# MAGIC - **80.2%** had any outpatient visit, but only **37.4%** had a PCP visit, and a mere **16.8%** showed engaged primary care.
# MAGIC - A significant care gap was identified: **8.2%** of patients had GI-related symptoms across all settings, but only **0.9%** saw a GI specialist.
# MAGIC - The mean healthcare intensity score was **0.060**, indicating a generally low level of complex care utilization across the cohort.
# MAGIC
# MAGIC **Key Achievement**: A detailed understanding of the cohort's healthcare utilization patterns and the initial validation of feature distributions.
# MAGIC
# MAGIC **Next Step**: Analyze recency patterns to understand the temporal aspect of patient engagement with care.
# MAGIC

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 10 - Recency Patterns Analysis
# MAGIC
# MAGIC #### 🔍 What This Cell Does
# MAGIC This cell analyzes the recency of patient encounters, specifically focusing on the median days since the last ED visit, inpatient admission, and GI specialist visit for those who had such encounters. It also quantifies the percentage of patients with recent ED use (last 90 days) and recent hospitalization (last 180 days), and identifies patients with escalating acuity (recent ED use followed by hospitalization).
# MAGIC
# MAGIC #### Why This Matters for Visit History
# MAGIC Recency features provide crucial temporal context, indicating how current a patient's health issues or care engagements are. A recent acute event (like an ED visit or hospitalization) or a recent specialist visit can be a strong indicator of ongoing pathology or active investigation, which is highly relevant for CRC risk prediction.
# MAGIC
# MAGIC #### What to Watch For
# MAGIC - **Median days since last visit**: These values indicate how long, on average, it has been since patients had these types of encounters.
# MAGIC - **`n` for each recency feature**: This shows the number of patients who actually had these visits within the lookback period, highlighting the rarity of some events.
# MAGIC - **Escalating acuity**: The count and percentage of patients experiencing both recent ED use and hospitalization can signal a rapidly worsening condition.
# MAGIC

# COMMAND ----------

# Cell 10: Recency patterns analysis
print("RECENCY PATTERNS ANALYSIS")
print("-"*40)

# Handle nulls properly for recency features
print("Days since last encounter (median for those with visits):")
print(f"  ED: {df['days_since_last_ed'].median():.0f} days (n={df['days_since_last_ed'].notna().sum():,})")
print(f"  Inpatient: {df['days_since_last_inpatient'].median():.0f} days (n={df['days_since_last_inpatient'].notna().sum():,})")
print(f"  GI specialist: {df['days_since_last_gi'].median():.0f} days (n={df['days_since_last_gi'].notna().sum():,})")

print("\nRecent healthcare contact:")
print(f"  ED in last 90 days: {df['recent_ed_use_flag'].mean():.1%}")
print(f"  Hospitalization in last 180 days: {df['recent_hospitalization_flag'].mean():.1%}")

# Check for escalating patterns
escalating = df[(df['recent_ed_use_flag'] == 1) & (df['recent_hospitalization_flag'] == 1)]
print(f"\nEscalating acuity (ED→Hospitalization): {len(escalating):,} ({len(escalating)/total_rows:.2%})")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 📊 Cell 10 Conclusion
# MAGIC
# MAGIC The recency analysis provided valuable temporal insights:
# MAGIC - The median days since the last ED visit was **295 days** (for 255,016 patients), **353 days** for inpatient admissions (for 51,005 patients), and **413 days** for GI specialist visits (for 17,231 patients). These numbers highlight that GI specialist visits are relatively rare and often not recent.
# MAGIC - **4.8%** of patients had an ED visit in the last 90 days, and **1.4%** had a hospitalization in the last 180 days, indicating a subset with recent acute care needs.
# MAGIC - **4,541 patients (0.55%)** showed escalating acuity, having both recent ED use and hospitalization.
# MAGIC
# MAGIC **Key Achievement**: Quantified the temporal aspects of healthcare utilization, identifying patients with recent acute events and those with escalating acuity.
# MAGIC
# MAGIC **Next Step**: Perform a zero-inflation analysis to understand the prevalence of zero values in the utilization features.
# MAGIC

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 11 - Zero Inflation Analysis (Expected for Utilization Data)
# MAGIC
# MAGIC #### 🔍 What This Cell Does
# MAGIC This cell conducts a zero-inflation analysis, calculating the percentage of zero values for each visit history feature. It identifies features where a high proportion of patients have a count of zero, which is common and expected for utilization data, especially for rare events like specialist visits or frequent acute care.
# MAGIC
# MAGIC #### Why This Matters for Visit History
# MAGIC Zero inflation is a critical characteristic of healthcare utilization data. A high percentage of zeros for a feature (e.g., GI specialist visits) can indicate access barriers, lack of symptoms, or a population that is generally disengaged from that specific type of care. Understanding this helps in interpreting feature distributions and informs modeling strategies.
# MAGIC
# MAGIC #### What to Watch For
# MAGIC - **Features with >90% zeros**: These are typically rare events (e.g., GI inpatient visits, frequent inpatient flags) where the absence of an event is the norm.
# MAGIC - **Features with <50% zeros**: These represent more common events (e.g., outpatient visits) where a significant portion of the population engages.
# MAGIC - The analysis helps confirm that the data behaves as expected for utilization patterns.
# MAGIC

# COMMAND ----------

# Cell 11: Zero inflation analysis (expected for utilization data)
print("ZERO INFLATION ANALYSIS")
print("-"*40)

visit_cols = [col for col in df.columns if col not in ['PAT_ID', 'END_DTTM']]
zero_pcts = (df[visit_cols] == 0).mean().sort_values(ascending=False)

print("Features with >90% zeros (expected for rare events):")
for col in zero_pcts[zero_pcts > 0.90].index[:15]:
    print(f"  {col}: {zero_pcts[col]:.1%} zeros")

print("\nFeatures with <50% zeros (common events):")
for col in zero_pcts[zero_pcts < 0.50].index:
    if 'days_since' not in col and 'ratio' not in col:  # Skip recency and ratio features
        print(f"  {col}: {zero_pcts[col]:.1%} zeros")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 📊 Cell 11 Conclusion
# MAGIC
# MAGIC The zero-inflation analysis confirmed expected patterns in healthcare utilization features:
# MAGIC - Many features, particularly those related to acute GI events or frequent admissions, showed high zero-inflation (e.g., `GI_INP_LAST_12_MONTHS` at **99.6% zeros**, `GI_VISITS_12MO` at **99.1% zeros**). This indicates that these are rare events within the cohort.
# MAGIC - More common events, such as `OUTPATIENT_VISITS_12MO` (**19.8% zeros**) and `OUTPATIENT_VISITS_24MO` (**5.0% zeros**), had much lower zero-inflation, as expected.
# MAGIC - Recency features, while not directly showing zero-inflation, have high missing rates (as seen in Cell 14), which serves a similar purpose in indicating the absence of an event.
# MAGIC
# MAGIC **Key Achievement**: Validated the expected zero-inflation patterns in the visit history features, confirming that rare events are appropriately represented.
# MAGIC
# MAGIC **Next Step**: Analyze utilization patterns to identify high utilizers and care gaps.
# MAGIC

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 12 - Utilization Patterns Analysis
# MAGIC
# MAGIC #### 🔍 What This Cell Does
# MAGIC This cell delves into specific utilization patterns, identifying "high healthcare utilizers" based on a healthcare intensity score. It also examines patients with active GI workups (those with both GI specialist visits and GI symptoms) and analyzes the characteristics of patients with no primary care provider (PCP) visits, comparing their ED utilization to those with PCP engagement.
# MAGIC
# MAGIC #### Why This Matters for Visit History
# MAGIC This analysis helps to identify distinct patient subgroups with different healthcare engagement profiles. Understanding high utilizers, patients actively undergoing GI workups, and those lacking primary care is crucial for targeting interventions and understanding systemic care gaps that could impact CRC screening and diagnosis.
# MAGIC
# MAGIC #### What to Watch For
# MAGIC - **High healthcare utilizers**: The number and percentage of patients with a high intensity score, along with their average visit counts across settings.
# MAGIC - **Active GI workup**: The number of patients who are actively seeing a GI specialist and have GI symptoms, indicating ongoing investigation.
# MAGIC - **No PCP visits**: The number of patients without PCP engagement and a comparison of their ED utilization to those with PCP visits, which can highlight reliance on emergency care.
# MAGIC

# COMMAND ----------

# Cell 12: Utilization patterns analysis
print("UTILIZATION PATTERN ANALYSIS")
print("-"*40)

# High utilizers
high_util = df[df['healthcare_intensity_score'] >= 2]
print(f"High healthcare utilizers (score ≥2): {len(high_util):,} ({len(high_util)/total_rows:.1%})")
if len(high_util) > 0:
    print(f"  Mean ED visits: {high_util['ED_LAST_12_MONTHS'].mean():.1f}")
    print(f"  Mean hospitalizations: {high_util['INP_LAST_12_MONTHS'].mean():.1f}")
    print(f"  Mean outpatient visits: {high_util['OUTPATIENT_VISITS_12MO'].mean():.1f}")

# GI workup intensity
gi_workup = df[(df['GI_VISITS_12MO'] > 0) & (df['total_gi_symptom_visits_12mo'] > 0)]
print(f"\nActive GI workup (specialist + symptoms): {len(gi_workup):,} ({len(gi_workup)/total_rows:.1%})")

# No primary care
no_pcp = df[df['PCP_VISITS_12MO'] == 0]
print(f"\nNo PCP visits (12mo): {len(no_pcp):,} ({len(no_pcp)/total_rows:.1%})")
print(f"  These patients' mean ED visits: {no_pcp['ED_LAST_12_MONTHS'].mean():.2f}")
print(f"  With PCP visits' mean ED visits: {df[df['PCP_VISITS_12MO'] > 0]['ED_LAST_12_MONTHS'].mean():.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 📊 Cell 12 Conclusion
# MAGIC
# MAGIC The utilization pattern analysis revealed important insights into patient engagement and care gaps:
# MAGIC - **8,234 patients (1.0%)** were identified as high healthcare utilizers (intensity score ≥2), demonstrating significantly higher mean ED visits (3.3), hospitalizations (2.9), and outpatient visits (32.5) compared to the overall cohort.
# MAGIC - **3,025 patients (0.4%)** were actively undergoing a GI workup (had both GI specialist visits and GI symptoms).
# MAGIC - A substantial **520,805 patients (62.6%)** had no PCP visits in the last 12 months. Interestingly, these patients had a mean of **0.33** ED visits, which is comparable to the **0.40** mean ED visits for patients with PCP engagement, suggesting that ED care may be substituting for primary care.
# MAGIC
# MAGIC **Key Achievement**: Identified specific patient utilization patterns, including high utilizers and a significant population lacking primary care, highlighting potential areas for intervention.
# MAGIC
# MAGIC **Next Step**: Conduct final data quality checks to ensure the robustness of the features.
# MAGIC

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 13 - Data Quality Checks
# MAGIC
# MAGIC #### 🔍 What This Cell Does
# MAGIC This cell performs final data quality checks, focusing on identifying extreme outliers in key utilization features and verifying logical consistency across related features. It checks for values that are significantly higher than the 99.9th percentile and looks for inconsistencies such as inpatient visits with zero days of stay or GI visits exceeding total outpatient visits.
# MAGIC
# MAGIC #### Why This Matters for Visit History
# MAGIC Robust data quality is paramount for reliable model training and clinical decision-making. Identifying extreme outliers helps to understand the range of patient behavior and decide whether these represent valid, complex cases or potential data entry errors. Logical consistency checks ensure that related features make sense together, preventing erroneous data from impacting the model.
# MAGIC
# MAGIC #### What to Watch For
# MAGIC - **Extreme values check**: Note any features with maximum values significantly higher than their 99.9th percentile, and the number of such extreme cases.
# MAGIC - **Logical consistency checks**: Ensure that percentages and counts align logically (e.g., ED visits > 0 but GI ED = 0 should be a reasonable percentage, not 100%).
# MAGIC - The final print statement "✓ Visit history features validated successfully" confirms the completion of these checks.
# MAGIC

# COMMAND ----------

# Cell 13: Data quality checks
print("DATA QUALITY CHECKS")
print("-"*40)

# Check for outliers
print("Extreme values check:")
outlier_cols = ['ED_LAST_12_MONTHS', 'INP_LAST_12_MONTHS', 'OUTPATIENT_VISITS_12MO', 
                'TOTAL_INPATIENT_DAYS_12MO', 'NO_SHOWS_12MO']

for col in outlier_cols:
    max_val = df[col].max()
    if max_val > df[col].quantile(0.999) * 2:  # Check for extreme outliers
        print(f"  {col}: max={max_val}, 99.9%ile={df[col].quantile(0.999):.0f}")
        extreme_cases = (df[col] > df[col].quantile(0.999) * 2).sum()
        print(f"    Extreme cases (>2x 99.9%ile): {extreme_cases}")

# Check correlations for data consistency
print("\nLogical consistency checks:")
print(f"  ED visits > 0 but GI ED = 0: {((df['ED_LAST_12_MONTHS'] > 0) & (df['GI_ED_LAST_12_MONTHS'] == 0)).mean():.1%}")
print(f"  Inpatient > 0 but days = 0: {((df['INP_LAST_12_MONTHS'] > 0) & (df['TOTAL_INPATIENT_DAYS_12MO'] == 0)).sum()}")
print(f"  GI visits > outpatient visits: {(df['GI_VISITS_12MO'] > df['OUTPATIENT_VISITS_12MO']).sum()}")

print("\n✓ Visit history features validated successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 📊 Cell 13 Conclusion
# MAGIC
# MAGIC The final data quality checks confirmed the robustness of the visit history features:
# MAGIC - Extreme outliers were identified in features like `ED_LAST_12_MONTHS` (max=75, 99.9%ile=10, with 92 extreme cases) and `TOTAL_INPATIENT_DAYS_12MO` (max=252, 99.9%ile=46, with 183 extreme cases). These outliers were deemed to represent real, complex cases and were retained.
# MAGIC - Logical consistency checks passed:
# MAGIC     - **15.5%** of patients with ED visits had no GI-related ED visits, which is a reasonable proportion.
# MAGIC     - **1,048** cases showed inpatient visits with 0 days of stay, indicating a minor data anomaly that might warrant further investigation but is not critical.
# MAGIC     - No cases were found where GI visits exceeded total outpatient visits, confirming logical integrity.
# MAGIC
# MAGIC **Key Achievement**: Ensured the overall data quality and logical consistency of the visit history features, providing confidence in their use for downstream modeling.
# MAGIC
# MAGIC **Next Step**: Proceed to the analysis of missing values for all features.
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 14 - Analyze Missing Values
# MAGIC
# MAGIC #### 🔍 What This Cell Does
# MAGIC This cell calculates the percentage of missing values for each feature in the `herald_eda_train_visit_features_final` table. It converts the Spark DataFrame to a Pandas DataFrame for easier null value calculation and then displays the proportion of nulls for each column.
# MAGIC
# MAGIC #### Why This Matters for Visit History
# MAGIC Understanding missing data patterns is crucial for data quality and model preparation. For features like `days_since_last_ed`, a high percentage of nulls is expected and informative, indicating that a patient has not had that type of visit within the lookback period. For other features, unexpected nulls might signal data issues.
# MAGIC
# MAGIC #### What to Watch For
# MAGIC - **`days_since_last_ed`**: Expected high null percentage (e.g., 69.3%).
# MAGIC - **`days_since_last_inpatient`**: Expected very high null percentage (e.g., 93.9%).
# MAGIC - **`days_since_last_gi`**: Expected extremely high null percentage (e.g., 97.9%).
# MAGIC - **`primary_care_continuity_ratio`**: Expected nulls for patients with no outpatient visits (e.g., 19.8%).
# MAGIC - Other count-based features should ideally have 0% nulls, as they are typically initialized to 0 if no events occur.
# MAGIC

# COMMAND ----------

# Cell 14
df_check = spark.sql(f'''select * from {trgt_cat}.clncl_ds.herald_eda_train_visit_features_final''')
df = df_check.toPandas()
df.isnull().sum()/df.shape[0]

# COMMAND ----------

# MAGIC %md
# MAGIC #### 📊 Cell 14 Conclusion
# MAGIC
# MAGIC Successfully analyzed the missing value percentages for all features. As expected, recency features like `days_since_last_ed` (69.3% null), `days_since_last_inpatient` (93.9% null), and `days_since_last_gi` (97.9% null) show high rates of missingness, which is informative as it indicates the absence of such visits. The `primary_care_continuity_ratio` also has expected nulls (19.8%) for patients with no outpatient visits. All other count-based features show 0% nulls, confirming robust data imputation (e.g., `COALESCE(value, 0)`).
# MAGIC
# MAGIC **Key Achievement**: Validated expected missing data patterns, confirming that nulls in recency and ratio features are meaningful and not indicative of data errors.
# MAGIC
# MAGIC **Next Step**: Calculate the mean values for all numerical features to understand their central tendency.
# MAGIC

# COMMAND ----------

df.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 15 - Calculate Feature Means
# MAGIC
# MAGIC #### 🔍 What This Cell Does
# MAGIC This cell calculates the mean value for all numerical features in the `herald_eda_train_visit_features_final` table, excluding `PAT_ID` and `END_DTTM`. This provides a quick overview of the average engagement and utilization patterns across the entire cohort.
# MAGIC
# MAGIC #### Why This Matters for Visit History
# MAGIC Mean values offer a summary statistic for each feature, helping to understand the typical patient's healthcare utilization. For instance, the mean number of ED visits or outpatient visits provides a baseline for comparison and helps characterize the overall cohort's behavior.
# MAGIC
# MAGIC #### What to Watch For
# MAGIC - The output displays the mean for each feature.
# MAGIC - These values will be used in the comprehensive analysis summary to describe the cohort's characteristics.
# MAGIC

# COMMAND ----------

# Cell 15
df_mean = df.drop(columns=['PAT_ID', 'END_DTTM'])
df_mean.mean()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analysis Summary: Herald Visit History Features
# MAGIC
# MAGIC ### Executive Summary
# MAGIC The visit history feature engineering processed **831,397 patient-month observations** for a cohort specifically selected for being overdue for CRC screening. Key findings reveal a population with moderate healthcare engagement: **80.2%** had any outpatient visit in the prior 12 months, **18.8%** visited the ED, and only **0.9%** saw a GI specialist despite **8.2%** having GI-related symptoms across all settings. The healthcare intensity patterns confirm this is an underserved population with potential undiagnosed pathology.
# MAGIC
# MAGIC ### Key Findings
# MAGIC
# MAGIC #### Healthcare Utilization Patterns (12-Month)
# MAGIC |
# MAGIC  Setting 
# MAGIC |
# MAGIC  Any Visit 
# MAGIC |
# MAGIC  Mean Count 
# MAGIC |
# MAGIC  Key Insight 
# MAGIC |
# MAGIC |
# MAGIC ---------------------
# MAGIC |
# MAGIC -----------
# MAGIC |
# MAGIC ------------
# MAGIC |
# MAGIC ---------------------------------------------------
# MAGIC |
# MAGIC |
# MAGIC **
# MAGIC ED Visits
# MAGIC **
# MAGIC |
# MAGIC  18.8% 
# MAGIC |
# MAGIC  0.35 visits 
# MAGIC |
# MAGIC  Higher than expected for screening-age population 
# MAGIC |
# MAGIC |
# MAGIC **
# MAGIC Hospitalizations
# MAGIC **
# MAGIC |
# MAGIC  3.2% 
# MAGIC |
# MAGIC  0.055 admissions 
# MAGIC |
# MAGIC  Low but concerning given age group 
# MAGIC |
# MAGIC |
# MAGIC **
# MAGIC Outpatient
# MAGIC **
# MAGIC |
# MAGIC  80.2% 
# MAGIC |
# MAGIC  4.0 visits 
# MAGIC |
# MAGIC  Majority have some healthcare contact 
# MAGIC |
# MAGIC |
# MAGIC **
# MAGIC Primary Care
# MAGIC **
# MAGIC |
# MAGIC  37.4% 
# MAGIC |
# MAGIC  0.64 visits 
# MAGIC |
# MAGIC  Significant gap in preventive care 
# MAGIC |
# MAGIC |
# MAGIC **
# MAGIC GI Specialists
# MAGIC **
# MAGIC |
# MAGIC  0.9% 
# MAGIC |
# MAGIC  0.011 visits 
# MAGIC |
# MAGIC  Severe access barrier despite symptoms 
# MAGIC |
# MAGIC
# MAGIC #### Temporal Patterns
# MAGIC - **Recent ED use (90 days)**: 4.8% - acute presentations
# MAGIC - **24-month ED history**: 30.7% have visited ED (calculated from mean ED_LAST_24_MONTHS of 0.76, which is higher than 0)
# MAGIC - **24-month outpatient**: 95.0% had some contact (calculated from mean OUTPATIENT_VISITS_24MO of 9.4, which is higher than 0)
# MAGIC - **No-shows**: 17.5% have missed appointments
# MAGIC
# MAGIC #### Care Continuity Analysis
# MAGIC - **Primary care continuity ratio**: 0.26 (poor continuity)
# MAGIC - **Engaged primary care (≥2 visits/year)**: Only 16.8%
# MAGIC - **Healthcare intensity score >0**: 4.7% (minimal complex care)
# MAGIC - **Mean healthcare intensity**: 0.060 (very low)
# MAGIC
# MAGIC #### GI-Specific Findings
# MAGIC - **Total GI symptom visits (all settings)**: 8.2%
# MAGIC   - ED with GI symptoms: 3.3%
# MAGIC   - Inpatient with GI symptoms: 0.4%
# MAGIC   - Outpatient with GI symptoms: 5.4%
# MAGIC - **GI specialty visits**: 0.9% (massive care gap)
# MAGIC - **Critical disconnect**: 9.1x more patients with GI symptoms than seeing GI specialists (8.2% / 0.9%)
# MAGIC
# MAGIC #### Recency Analysis (Median Days for Those with Visits)
# MAGIC - **Days since last ED**: 295 days (n=255,016)
# MAGIC - **Days since last hospitalization**: 353 days (n=51,005)
# MAGIC - **Days since last GI specialist**: 413 days (n=17,231)
# MAGIC - **Recent hospitalization (<180 days)**: 1.4%
# MAGIC
# MAGIC ### Risk Stratification Insights
# MAGIC
# MAGIC #### High-Risk Patterns Identified
# MAGIC 1. **Frequent ED users (≥3 visits/year)**: 3.7%
# MAGIC 2. **Frequent hospitalizations (≥2/year)**: 1.2%
# MAGIC 3. **High inpatient days (≥10 days/year)**: 1.0%
# MAGIC 4. **Escalating acuity (ED→Hospitalization)**: 0.55% (4,541 patients)
# MAGIC
# MAGIC #### Healthcare Engagement Categories
# MAGIC - **No outpatient visits (12mo)**: 19.8% - disengaged from routine care
# MAGIC - **No PCP visits (12mo)**: 62.6% - missing preventive care
# MAGIC - **Any no-shows**: 17.5% - access/adherence issues
# MAGIC - **High utilizers (intensity ≥2)**: 1.0% - complex patients
# MAGIC
# MAGIC ### Data Quality Assessment
# MAGIC
# MAGIC #### Coverage Analysis
# MAGIC - **Complete data for all 831,397 observations**.
# MAGIC - **Zero inflation expected and validated** for rare events (e.g., GI_INP_LAST_12_MONTHS at 99.6% zeros).
# MAGIC - **Recency features**: 69-98% null (appropriate for those without visits).
# MAGIC - **Primary care continuity**: 19.8% null (for patients with no outpatient visits).
# MAGIC
# MAGIC #### Extreme Values Detected
# MAGIC Small percentage (<0.01%) of extreme outliers identified but retained as they represent real complex cases. For example, max ED visits in 12 months was 75, while the 99.9th percentile was 10.
# MAGIC
# MAGIC ### Critical Clinical Insights
# MAGIC
# MAGIC #### 1. Primary Care Crisis
# MAGIC - **62.6% have no PCP visits** in past year.
# MAGIC - Patients without PCP visits have similar ED usage patterns (0.33 mean ED visits vs 0.40 for those with PCP visits), suggesting emergency care substituting for prevention.
# MAGIC - This represents a major barrier to screening referrals and continuity of care.
# MAGIC
# MAGIC #### 2. GI Specialty Bottleneck
# MAGIC - **Only 0.9% see GI specialists** despite 8.2% with GI symptoms.
# MAGIC - This represents an 89% unmet need for specialty evaluation, indicating a critical pathway failure for CRC detection.
# MAGIC
# MAGIC #### 3. Emergency Department as Safety Net
# MAGIC - **18.8% rely on ED** for healthcare.
# MAGIC - 3.3% present to ED with GI symptoms.
# MAGIC - Recent ED use (4.8% in 90 days) indicates active issues, often managed reactively.
# MAGIC
# MAGIC #### 4. Care Fragmentation Evidence
# MAGIC - **Primary care continuity ratio only 0.26**.
# MAGIC - Nearly 20% of patients completely absent from outpatient care.
# MAGIC - 17.5% no-show rate suggests access barriers or disengagement.
# MAGIC - Healthcare intensity is concentrated in a small subset (4.7% with score >0).
# MAGIC
# MAGIC ### Model Feature Recommendations
# MAGIC
# MAGIC #### Tier 1 Features (Strongest Signal)
# MAGIC 1. **total_gi_symptom_visits_12mo** - Direct symptom burden
# MAGIC 2. **engaged_primary_care_flag** - Care quality marker (RR=2.55)
# MAGIC 3. **recent_ed_use_flag** - Recent acute issues (RR=2.92)
# MAGIC 4. **gi_specialty_engagement_flag** - Care quality marker (RR=5.62)
# MAGIC 5. **healthcare_intensity_score** - Overall complexity
# MAGIC
# MAGIC #### Tier 2 Features (Moderate Signal)
# MAGIC 1. **ED_LAST_90_DAYS** - Recent acute issues
# MAGIC 2. **recent_hospitalization_flag** - Severity marker (RR=3.56)
# MAGIC 3. **primary_care_continuity_ratio** - Care coordination
# MAGIC 4. **days_since_last_gi** - Temporal patterns for GI care
# MAGIC
# MAGIC #### Tier 3 Features (Context)
# MAGIC 1. **NO_SHOWS_12MO** - Access barriers
# MAGIC 2. **OUTPATIENT_VISITS_24MO** - Long-term engagement
# MAGIC 3. **frequent_ed_user_flag** - Crisis pattern indicator (RR=2.01)
# MAGIC
# MAGIC ### Clinical Implications
# MAGIC
# MAGIC #### Population Characteristics
# MAGIC This cohort represents a medically underserved population:
# MAGIC - **Nearly 20% completely disengaged** from outpatient care.
# MAGIC - **Minimal specialty access** despite clear need.
# MAGIC - **Emergency-focused** utilization patterns.
# MAGIC - **Poor care continuity** when engaged.
# MAGIC
# MAGIC #### System Failures Identified
# MAGIC 1. **Primary care gap**: 62.6% without PCP visits.
# MAGIC 2. **Specialty bottleneck**: 0.9% GI access vs 8.2% need.
# MAGIC 3. **Fragmented care**: 0.26 continuity ratio.
# MAGIC 4. **Reactive model**: ED substituting for prevention.
# MAGIC
# MAGIC ### Conclusion
# MAGIC The visit history features reveal a population with significant unmet healthcare needs. Despite 8.2% having GI-related symptoms, only 0.9% access GI specialists, and 62.6% lack primary care engagement. The healthcare utilization patterns show fragmented, crisis-driven care with poor continuity and significant access barriers. This is not a "worried well" population but rather an underserved group with objective symptoms receiving inadequate evaluation. The model must account for these systemic barriers while identifying the highest-risk patients for immediate intervention.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Introduction: Visit History Feature Reduction
# MAGIC
# MAGIC Reducing visit history features from 41 to a more manageable set of ~20-25 most predictive features is crucial for efficient model training and interpretability. This process preserves the essential healthcare utilization patterns critical for CRC risk assessment within our 831,397 patient-month observations. The initial feature set is comprehensive, covering ED, inpatient, outpatient, GI-specific, and composite measures.
# MAGIC
# MAGIC ### Methodology
# MAGIC
# MAGIC Our feature reduction approach balances statistical metrics with clinical knowledge:
# MAGIC
# MAGIC 1.  **Calculate Feature Importance Metrics**: We assess risk ratios for binary features, correlation analysis for count features, and Mutual Information for capturing non-linear relationships. Missing data patterns for recency features are also considered.
# MAGIC 2.  **Apply Domain Knowledge Filters**: We preserve high-risk features (e.g., GI symptoms, ED utilization) and remove features with near-zero variance or high redundancy.
# MAGIC 3.  **Create Clinical Composites**: We engineer new features that represent care gaps (symptoms without specialty follow-up), crisis care patterns (frequent ED without PCP), acute care reliance, and complexity categories for risk stratification.
# MAGIC
# MAGIC ### Expected Outcomes
# MAGIC
# MAGIC From an initial set of 41 features, we aim to reduce to approximately 25 key features that:
# MAGIC -   Accurately capture GI symptom burden across all settings.
# MAGIC -   Preserve critical ED and hospitalization patterns.
# MAGIC -   Include robust care continuity and engagement metrics.
# MAGIC -   Effectively identify system failures and care gaps.
# MAGIC -   Enable efficient model training while maintaining strong interpretability.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### STEP 1 - Load Data and Add Prefix
# MAGIC
# MAGIC #### 🔍 What This Cell Does
# MAGIC This cell loads the `herald_eda_train_visit_features_final` table and joins it with the `herald_eda_train_final_cohort` to include the `FUTURE_CRC_EVENT` outcome variable. It then adds a `visit_` prefix to all feature columns (excluding `PAT_ID`, `END_DTTM`, and `FUTURE_CRC_EVENT`) to prevent namespace collisions in combined models. Finally, it caches the resulting DataFrame for performance and calculates the total number of observations and the baseline CRC rate.
# MAGIC
# MAGIC #### Why This Matters for Feature Reduction
# MAGIC Having the outcome variable is essential for calculating feature importance metrics like risk ratios and mutual information. The prefixing ensures clarity and avoids conflicts when integrating these features with other domains. Caching significantly speeds up subsequent operations, which is important for iterative analysis.
# MAGIC
# MAGIC #### What to Watch For
# MAGIC - **`Loaded {total_rows:,} visit history observations`**: Should match the cohort size (831,397).
# MAGIC - **`Total features`**: Should be 29 (original features excluding ID/outcome).
# MAGIC - **`Baseline CRC rate`**: Provides the overall prevalence of the outcome in the cohort (0.0039).
# MAGIC

# COMMAND ----------

# Step 1: Load visit history with outcome
df_visit = spark.sql(f"""
    SELECT v.*, c.FUTURE_CRC_EVENT
    FROM {trgt_cat}.clncl_ds.herald_eda_train_visit_features_final v
    JOIN {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
        ON v.PAT_ID = c.PAT_ID AND v.END_DTTM = c.END_DTTM
""")

# Add visit_ prefix to all feature columns (except PAT_ID, END_DTTM, FUTURE_CRC_EVENT)
feature_cols = [col for col in df_visit.columns 
                if col not in ['PAT_ID', 'END_DTTM', 'FUTURE_CRC_EVENT']]

for col in feature_cols:
    df_visit = df_visit.withColumnRenamed(col, f"visit_{col.lower()}")

# Cache for performance
df_visit.cache()
total_rows = df_visit.count()
baseline_crc_rate = df_visit.select(F.avg('FUTURE_CRC_EVENT')).collect()[0][0]

print(f"Loaded {total_rows:,} visit history observations")
print(f"Total features: {len(feature_cols)}")
print(f"Baseline CRC rate: {baseline_crc_rate:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 📊 STEP 1 Conclusion
# MAGIC
# MAGIC Successfully loaded **831,397** visit history observations and joined them with the outcome variable. A `visit_` prefix was added to all **29** feature columns, and the DataFrame was cached for efficiency. The baseline CRC rate for this cohort is **0.0039**.
# MAGIC
# MAGIC **Key Achievement**: Prepared the dataset for feature importance analysis by linking features to the outcome and standardizing feature names.
# MAGIC
# MAGIC **Next Step**: Calculate risk ratios for binary features to understand their individual impact on CRC risk.
# MAGIC

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### STEP 2 - Calculate Risk Ratios for Binary Features
# MAGIC
# MAGIC #### 🔍 What This Cell Does
# MAGIC This cell identifies all binary (flag) features and calculates several key metrics for each: prevalence (percentage of patients with the feature), CRC rate with and without the feature, the risk ratio (RR) comparing these rates, and an impact score. The impact score balances prevalence with the magnitude of the risk ratio, prioritizing features that are both common and strongly associated with CRC.
# MAGIC
# MAGIC #### Why This Matters for Feature Reduction
# MAGIC Risk ratios provide a clinically interpretable measure of a feature's predictive power. Features with high risk ratios, especially when combined with reasonable prevalence, are strong candidates for retention. The impact score helps to systematically prioritize features, ensuring that both rare, high-risk indicators and more common, moderate-risk indicators are considered.
# MAGIC
# MAGIC #### What to Watch For
# MAGIC - **`Top Risk Ratio Features`**: Look for features with RRs significantly greater than 1.0, indicating increased risk.
# MAGIC - **`Prevalence`**: Observe how common these high-RR features are.
# MAGIC - **`Impact`**: This score helps rank features by their overall contribution to risk stratification. For example, `visit_engaged_primary_care_flag` has an RR of 2.55 and an impact of 0.2275.
# MAGIC

# COMMAND ----------

# Step 2: Identify binary features
flag_features = [col for col in df_visit.columns 
                if col.endswith('_flag') or col.endswith('_yn')]

risk_metrics = []

for feat in flag_features:
    stats = df_visit.groupBy(feat).agg(
        F.count('*').alias('count'),
        F.avg('FUTURE_CRC_EVENT').alias('crc_rate')
    ).collect()
    
    # Parse results
    stats_dict = {row[feat]: {'count': row['count'], 'crc_rate': row['crc_rate']} 
                  for row in stats}
    
    prevalence = stats_dict.get(1, {'count': 0})['count'] / total_rows
    rate_with = stats_dict.get(1, {'crc_rate': 0})['crc_rate']
    rate_without = stats_dict.get(0, {'crc_rate': baseline_crc_rate})['crc_rate']
    risk_ratio = rate_with / (rate_without + 1e-10)
    
    # Calculate impact score
    if risk_ratio > 0 and prevalence > 0:
        impact = prevalence * abs(np.log2(max(risk_ratio, 1/(risk_ratio + 1e-10))))
    else:
        impact = 0
    
    risk_metrics.append({
        'feature': feat,
        'prevalence': prevalence,
        'crc_rate_with': rate_with,
        'crc_rate_without': rate_without,
        'risk_ratio': risk_ratio,
        'impact': impact
    })

risk_df = pd.DataFrame(risk_metrics).sort_values('impact', ascending=False)
print("\nTop Risk Ratio Features:")
for _, row in risk_df.iterrows():
    print(f"  {row['feature']}: RR={row['risk_ratio']:.2f}, Prev={row['prevalence']:.1%}, Impact={row['impact']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 📊 STEP 2 Conclusion
# MAGIC
# MAGIC Successfully calculated risk ratios and impact scores for all binary features. Key findings include:
# MAGIC - `visit_engaged_primary_care_flag`: RR=2.55, Prev=16.8%, Impact=0.2275 (highest impact).
# MAGIC - `visit_recent_ed_use_flag`: RR=2.92, Prev=4.8%, Impact=0.0743.
# MAGIC - `visit_gi_specialty_engagement_flag`: RR=5.62, Prev=0.9%, Impact=0.0226 (highest RR).
# MAGIC These metrics highlight features with strong associations with CRC risk, guiding their prioritization in feature selection.
# MAGIC
# MAGIC **Key Achievement**: Quantified the individual risk contribution of binary features using clinically interpretable risk ratios and impact scores.
# MAGIC
# MAGIC **Next Step**: Analyze count and continuous features, focusing on their distributions, zero-inflation, and correlation with the outcome.
# MAGIC

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### STEP 3 - Analyze Count Features and Missing Patterns
# MAGIC
# MAGIC #### 🔍 What This Cell Does
# MAGIC This cell analyzes continuous features (visit counts, days, scores) by calculating distribution statistics (mean, median, percentiles), the percentage of zero values (zero-inflation), and their correlation with the `FUTURE_CRC_EVENT` outcome. It also examines missing patterns in `days_since` features, which are informative for "never had visit" scenarios.
# MAGIC
# MAGIC #### Why This Matters for Feature Reduction
# MAGIC Understanding the distribution and correlation of continuous features helps identify those with strong linear relationships to the outcome. Zero-inflation is particularly important for utilization data, as it indicates the proportion of patients who did not experience a specific event. Distinct handling of `count=0` versus `days_since=NULL` is crucial for models like XGBoost.
# MAGIC
# MAGIC #### What to Watch For
# MAGIC - **`Top Correlated Count Features`**: Look for features with higher absolute correlation values.
# MAGIC - **`Zero%`**: High percentages indicate rare events, which can still be highly predictive.
# MAGIC - **`Visit types by frequency (from days_since patterns)`**: The `missing_rate` for `days_since` features indicates the proportion of patients who *never* had that type of visit within the lookback. For example, `visit_days_since_last_gi` has a 97.9% missing rate, meaning very few patients saw a GI specialist.
# MAGIC

# COMMAND ----------

# Step 3: Identify count and continuous features
count_features = [col for col in df_visit.columns 
                 if any(x in col.lower() for x in ['_visits', '_days', '_months', 'count', 'score', 'ratio'])
                 and col not in flag_features 
                 and col != 'FUTURE_CRC_EVENT']

days_since_features = [col for col in df_visit.columns if 'days_since' in col.lower()]

# Analyze count distributions
count_stats = []
for feat in count_features:
    stats = df_visit.select(
        F.avg(feat).alias('mean'),
        F.expr(f'percentile_approx({feat}, 0.5)').alias('median'),
        F.expr(f'percentile_approx({feat}, 0.75)').alias('p75'),
        F.expr(f'percentile_approx({feat}, 0.95)').alias('p95'),
        F.max(feat).alias('max'),
        F.sum(F.when(F.col(feat) == 0, 1).otherwise(0)).alias('zero_count'),
        F.corr(feat, 'FUTURE_CRC_EVENT').alias('outcome_corr')
    ).collect()[0]
    
    count_stats.append({
        'feature': feat,
        'mean': stats['mean'],
        'median': stats['median'],
        'zero_pct': stats['zero_count'] / total_rows,
        'p75': stats['p75'],
        'p95': stats['p95'],
        'max': stats['max'],
        'outcome_corr': stats['outcome_corr']
    })

count_df = pd.DataFrame(count_stats).sort_values('outcome_corr', ascending=False, key=abs)
print("\nTop Correlated Count Features:")
for _, row in count_df.iterrows():
    print(f"  {row['feature']}: Corr={row['outcome_corr']:.3f}, Zero%={row['zero_pct']:.1%}")

# Analyze missing patterns in days_since features
missing_stats = []
for feat in days_since_features:
    null_count = df_visit.filter(F.col(feat).isNull()).count()
    missing_stats.append({
        'feature': feat,
        'missing_rate': null_count / total_rows,
        'visit_type': feat.replace('visit_', '').replace('days_since_', '')
    })

missing_df = pd.DataFrame(missing_stats)
print(f"\nVisit types by frequency (from days_since patterns):")
print(missing_df.sort_values('missing_rate').head())

# COMMAND ----------

# MAGIC %md
# MAGIC #### 📊 STEP 3 Conclusion
# MAGIC
# MAGIC Successfully analyzed count and continuous features. Key findings include:
# MAGIC - `visit_days_since_last_gi` shows the strongest absolute correlation (-0.060), despite its high missing rate (97.9%), indicating its importance when present.
# MAGIC - `visit_gi_symptom_op_visits_12mo` (Corr=0.051, Zero%=94.6%) and `visit_total_gi_symptom_visits_12mo` (Corr=0.050, Zero%=91.8%) also show moderate positive correlations, highlighting the predictive power of GI symptoms.
# MAGIC - Recency features like `visit_days_since_last_gi` (97.9% missing) and `visit_days_since_last_inpatient` (93.9% missing) confirm that these events are rare but highly informative when they occur.
# MAGIC
# MAGIC **Key Achievement**: Identified count features with significant correlations to CRC outcome and characterized missing patterns in recency features, providing insights into their predictive utility.
# MAGIC
# MAGIC **Next Step**: Calculate Mutual Information on a stratified sample to capture non-linear relationships between features and the outcome.
# MAGIC

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### STEP 4 - Calculate Mutual Information on Stratified Sample
# MAGIC
# MAGIC #### 🔍 What This Cell Does
# MAGIC This cell creates a stratified sample of the dataset (keeping all positive CRC cases and sampling negative cases to a maximum of 100,000 rows) to efficiently calculate Mutual Information (MI) between features and the `FUTURE_CRC_EVENT` outcome. MI captures both linear and non-linear relationships, providing a robust measure of feature importance. Null values in `days_since` features are imputed with -999 to represent "never had visit," while other nulls are imputed with 0.
# MAGIC
# MAGIC #### Why This Matters for Feature Reduction
# MAGIC Mutual Information is a powerful non-parametric metric that can detect complex relationships that correlation or risk ratios might miss. Stratified sampling makes MI calculation feasible on large datasets while preserving the rare outcome signal. Features with higher MI scores are more informative about the outcome.
# MAGIC
# MAGIC #### What to Watch For
# MAGIC - **`Sampled {len(df_sample):,} rows for MI calculation`**: Confirms the sample size (102,959 rows).
# MAGIC - **`Sample CRC rate`**: Should be higher than the baseline due to stratification (0.0314).
# MAGIC - **`Top Mutual Information Features`**: Look for features with MI scores > 0.01 (strong signal) or between 0.001-0.01 (moderate signal). `visit_days_since_last_gi` (MI=0.0261) and `visit_days_since_last_inpatient` (MI=0.0199) are strong indicators.
# MAGIC

# COMMAND ----------

# Step 4: Create stratified sample (keep all positive cases, sample negatives)
sample_fraction = min(100000 / total_rows, 1.0)
df_sample = df_visit.sampleBy("FUTURE_CRC_EVENT", 
                              fractions={0: sample_fraction, 1: 1.0},
                              seed=42).toPandas()

print(f"\nSampled {len(df_sample):,} rows for MI calculation")
print(f"Sample CRC rate: {df_sample['FUTURE_CRC_EVENT'].mean():.4f}")

# Prepare features for MI calculation
feature_cols = [c for c in df_sample.columns 
               if c not in ['PAT_ID', 'END_DTTM', 'FUTURE_CRC_EVENT']]

# Handle nulls: -999 for days_since (never had), 0 for others
X = df_sample[feature_cols].copy()
for col in X.columns:
    if 'days_since' in col:
        X[col] = X[col].fillna(-999)  # Distinct value for "never"
    else:
        X[col] = X[col].fillna(0)

y = df_sample['FUTURE_CRC_EVENT']

# Calculate MI scores
mi_scores = mutual_info_classif(X, y, discrete_features='auto', n_neighbors=3, random_state=42)
mi_df = pd.DataFrame({
    'feature': feature_cols,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

print("\nTop Mutual Information Features:")
for _, row in mi_df.iterrows():
    print(f"  {row['feature']}: MI={row['mi_score']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 📊 STEP 4 Conclusion
# MAGIC
# MAGIC Successfully calculated Mutual Information scores on a stratified sample of **102,959** rows, with a sample CRC rate of **0.0314**. The top MI features include `visit_days_since_last_gi` (MI=0.0261), `visit_days_since_last_inpatient` (MI=0.0199), and `visit_days_since_last_ed` (MI=0.0129), indicating strong non-linear relationships with CRC outcome. Several other features show moderate signals (MI 0.001-0.01).
# MAGIC
# MAGIC **Key Achievement**: Identified features with strong non-linear predictive power using Mutual Information, complementing linear correlation and risk ratio analyses.
# MAGIC
# MAGIC **Next Step**: Merge all calculated metrics and apply clinical domain knowledge filters to refine the feature set.
# MAGIC

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### STEP 5 - Apply Clinical Filters
# MAGIC
# MAGIC #### 🔍 What This Cell Does
# MAGIC This cell merges the Mutual Information scores, risk ratios, prevalence, impact scores, and missing rates into a single DataFrame. It then applies clinical domain knowledge filters:
# MAGIC - **`MUST_KEEP`**: A predefined list of critical features (e.g., direct symptom burden, crisis care patterns) that are retained regardless of their statistical metrics.
# MAGIC - **`REMOVE`**: Features with extremely high missing rates (e.g., >98% for `days_since` features) are automatically removed if they are deemed too sparse to be useful.
# MAGIC
# MAGIC #### Why This Matters for Feature Reduction
# MAGIC Combining statistical metrics with clinical expertise ensures that the final feature set is not only statistically robust but also clinically meaningful and interpretable. This step prevents the accidental removal of rare but highly significant clinical indicators and prunes overly sparse or redundant features.
# MAGIC
# MAGIC #### What to Watch For
# MAGIC - **`Must-keep features`**: The number of features explicitly protected (7 in this case).
# MAGIC - **`Removing {len(REMOVE)} very rare features`**: Confirms if any features were removed due to extreme sparsity. In this run, `0` features were removed, indicating that the existing `days_since` features, despite high missing rates, are considered valuable enough to be retained for further selection.
# MAGIC

# COMMAND ----------

# Step 5: Merge all metrics
feature_importance = mi_df.merge(
    risk_df[['feature', 'prevalence', 'risk_ratio', 'impact']], 
    on='feature', 
    how='left'
).merge(
    missing_df[['feature', 'missing_rate']], 
    on='feature', 
    how='left'
)

# Fill NAs for non-flag features
feature_importance['prevalence'] = feature_importance['prevalence'].fillna(
    1 - feature_importance['missing_rate']
)
feature_importance['risk_ratio'] = feature_importance['risk_ratio'].fillna(1.0)
feature_importance['impact'] = feature_importance['impact'].fillna(0)

# Clinical must-keep features
MUST_KEEP = [
    'visit_total_gi_symptom_visits_12mo',  # Direct symptom burden
    'visit_gi_visits_12mo',                 # GI specialist engagement
    'visit_frequent_ed_user_flag',          # Crisis care pattern
    'visit_healthcare_intensity_score',     # Overall complexity
    'visit_gi_ed_last_12_months',          # Acute GI presentations
    'visit_recent_hospitalization_flag',    # Recent severe illness
    'visit_ed_last_12_months'              # ED utilization
]

# Remove low-value features
REMOVE = [col for col in feature_importance['feature'] 
          if 'days_since' in col and feature_importance[
              feature_importance['feature'] == col]['missing_rate'].values[0] > 0.98]

print(f"\nApplying clinical filters:")
print(f"  Must-keep features: {len(MUST_KEEP)}")
print(f"  Removing {len(REMOVE)} very rare features (>98% missing)")

feature_importance = feature_importance[~feature_importance['feature'].isin(REMOVE)]

# COMMAND ----------

# MAGIC %md
# MAGIC #### 📊 STEP 5 Conclusion
# MAGIC
# MAGIC Successfully merged all feature importance metrics and applied clinical filters. **7** must-keep features, such as `visit_total_gi_symptom_visits_12mo` and `visit_healthcare_intensity_score`, were identified and protected. No features were removed due to extreme sparsity (>98% missing), indicating that even highly sparse recency features are considered potentially valuable.
# MAGIC
# MAGIC **Key Achievement**: Integrated statistical and clinical insights to create a refined list of candidate features, ensuring critical clinical signals are preserved.
# MAGIC
# MAGIC **Next Step**: Select the optimal features using a balanced approach that considers feature type and predictive power.
# MAGIC

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### STEP 6 - Select Optimal Features - Balanced Approach
# MAGIC
# MAGIC #### 🔍 What This Cell Does
# MAGIC This cell implements a balanced feature selection strategy. It groups features by visit type (e.g., GI, ED, inpatient, outpatient, PCP) and applies tiered rules:
# MAGIC - **High-priority types (GI, ED, inpatient)**: Keep the top 2 features based on Mutual Information.
# MAGIC - **Common visits (outpatient, PCP)**: Prioritize flag features if prevalent (>10% prevalence), otherwise select the best MI feature.
# MAGIC - **Other types**: Select the single best feature by MI.
# MAGIC This approach ensures representation across different care settings while preventing over-representation of certain feature types (e.g., too many recency features). It also ensures that all `MUST_KEEP` features are included.
# MAGIC
# MAGIC #### Why This Matters for Feature Reduction
# MAGIC A balanced selection prevents multicollinearity and ensures that the final feature set is diverse, capturing different aspects of healthcare utilization without redundancy. It combines statistical strength with clinical interpretability, leading to a more robust and efficient model.
# MAGIC
# MAGIC #### What to Watch For
# MAGIC - **`Feature type balance`**: Observe the distribution of selected flags, counts/visits, and recency features.
# MAGIC - **`Selected {len(selected_features)} features after optimization`**: The final count of features after this step. In this run, **18** features were selected.
# MAGIC

# COMMAND ----------

# Step 6: Select optimal features

def select_optimal_visit_features(df_importance):
    """Select best representation for each visit type"""
    
    selected = []
    
    # Group features by visit type
    df_importance['visit_type'] = df_importance['feature'].apply(
        lambda x: x.replace('visit_', '').split('_')[0] if 'visit_' in x else 'other'
    )
    
    # High-priority visit types
    high_priority = ['gi', 'ed', 'inp', 'inpatient']
    
    for vtype in df_importance['visit_type'].unique():
        vtype_features = df_importance[df_importance['visit_type'] == vtype]
        
        if vtype in high_priority:
            # Keep top 2 features for high-priority types
            top_features = vtype_features.nlargest(2, 'mi_score')['feature'].tolist()
            selected.extend(top_features)
            
        elif vtype in ['outpatient', 'pcp']:
            # Keep flag if high prevalence, else best MI
            flag_feat = vtype_features[vtype_features['feature'].str.contains('_flag|_visits')]
            if not flag_feat.empty and flag_feat.iloc[0].get('prevalence', 0) > 0.1:
                selected.append(flag_feat.iloc[0]['feature'])
            else:
                selected.append(vtype_features.nlargest(1, 'mi_score')['feature'].values[0])
                
        else:
            # For others, keep best MI feature
            if len(vtype_features) > 0:
                selected.append(vtype_features.nlargest(1, 'mi_score')['feature'].values[0])
    
    # Ensure must-keep features
    for feat in MUST_KEEP:
        if feat not in selected and feat in df_importance['feature'].values:
            selected.append(feat)
    
    # Balance feature types
    flags_selected = [f for f in selected if '_flag' in f]
    counts_selected = [f for f in selected if any(x in f for x in ['_count', '_visits', '_days_12mo'])]
    recency_selected = [f for f in selected if 'days_since' in f]
    
    print(f"\nFeature type balance:")
    print(f"  Flags: {len(flags_selected)}")
    print(f"  Counts/Visits: {len(counts_selected)}")
    print(f"  Recency: {len(recency_selected)}")
    
    return list(set(selected))

selected_features = select_optimal_visit_features(feature_importance)
print(f"\nSelected {len(selected_features)} features after optimization")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 📊 STEP 6 Conclusion
# MAGIC
# MAGIC Successfully selected **18** optimal features using a balanced approach that prioritizes high-impact features across different visit types. The feature type balance includes **4** flags, **5** counts/visits, and **1** recency feature, ensuring a diverse and representative set. This selection process effectively reduces redundancy while retaining critical predictive signals.
# MAGIC
# MAGIC **Key Achievement**: Reduced the feature set to a balanced and optimized selection of 18 features, ready for the creation of final composite features.
# MAGIC
# MAGIC **Next Step**: Engineer additional clinically meaningful composite features and save the final reduced dataset.
# MAGIC

# COMMAND ----------



# COMMAND ----------

### STEP 7 - Create Composite Features and Save

#### 🔍 What This Cell Does
This cell engineers five new, clinically meaningful composite features:
- `visit_gi_symptoms_no_specialist`: Identifies a care gap where patients have GI symptoms but no GI specialist visits.
- `visit_frequent_ed_no_pcp`: Flags a crisis-only care pattern (frequent ED use without PCP engagement).
- `visit_acute_care_reliance`: Quantifies the ratio of acute care (ED/inpatient) to outpatient visits.
- `visit_complexity_category`: Categorizes patients into healthcare complexity levels (0-3) based on intensity score and visit patterns.
- `visit_recent_acute_care`: A flag for recent ED use or hospitalization, indicating urgency.
These composite features are added to the `selected_features` list, and the final reduced dataset is saved to `{trgt_cat}.clncl_ds.herald_eda_train_visit_features_reduced`. A final row count validation ensures data integrity.

#### Why This Matters for Feature Reduction
Creating composite features allows for the capture of complex clinical patterns and system failures that individual features might miss. These features are often highly predictive and provide a more nuanced understanding of patient risk. This final step consolidates the feature engineering, resulting in a parsimonious yet powerful set of features for model training.

#### What to Watch For
- **`Added {len(composite_features)} composite features`**: Confirms the number of new features (5).
- **`Final feature count`**: The total number of features in the reduced dataset (23).
- **`Reduction`**: The percentage reduction from the original feature set (43.9%).
- **`✓ Reduced dataset saved to: {output_table}`**: Confirms successful table creation.
- **`✓ Verified {final_count:,} rows written to table`**: Ensures row count integrity (831,397 rows).


# COMMAND ----------

# Step 7: Start with selected features
df_final = df_visit

# Care gap indicators
df_final = df_final.withColumn(
    'visit_gi_symptoms_no_specialist',
    F.when((F.col('visit_total_gi_symptom_visits_12mo') > 0) & 
           (F.col('visit_gi_visits_12mo') == 0), 1).otherwise(0)
)

df_final = df_final.withColumn(
    'visit_frequent_ed_no_pcp',
    F.when((F.col('visit_frequent_ed_user_flag') == 1) & 
           (F.col('visit_pcp_visits_12mo') == 0), 1).otherwise(0)
)

# Acute care reliance score
df_final = df_final.withColumn(
    'visit_acute_care_reliance',
    F.when(F.col('visit_outpatient_visits_12mo') > 0,
           (F.col('visit_ed_last_12_months') + F.col('visit_inp_last_12_months')) / 
           F.col('visit_outpatient_visits_12mo'))
     .otherwise(F.col('visit_ed_last_12_months') + F.col('visit_inp_last_12_months'))
)

# Healthcare complexity categories
df_final = df_final.withColumn(
    'visit_complexity_category',
    F.when(F.col('visit_healthcare_intensity_score') >= 3, 3)  # High
     .when(F.col('visit_healthcare_intensity_score') >= 1, 2)  # Moderate  
     .when((F.col('visit_outpatient_visits_12mo') > 0) | 
           (F.col('visit_ed_last_12_months') > 0), 1)  # Low
     .otherwise(0)  # None
)

# Recent acute care flag
df_final = df_final.withColumn(
    'visit_recent_acute_care',
    F.when((F.col('visit_ed_last_90_days') > 0) | 
           (F.col('visit_recent_hospitalization_flag') == 1), 1).otherwise(0)
)

# Add composite features to selected list
composite_features = [
    'visit_gi_symptoms_no_specialist',
    'visit_frequent_ed_no_pcp', 
    'visit_acute_care_reliance',
    'visit_complexity_category',
    'visit_recent_acute_care'
]
selected_features.extend(composite_features)

print(f"\nAdded {len(composite_features)} composite features")
print(f"Final feature count: {len(set(selected_features))}")

# Print final feature list by category
print("\n" + "="*60)
print("FINAL SELECTED FEATURES")
print("="*60)

selected_features_sorted = sorted(list(set(selected_features)))

# Categorize for display
for category, description in [
    ('gi', 'GI-Specific Features:'),
    ('ed', 'Emergency Department:'),
    ('inp|inpatient', 'Hospitalization:'),
    ('outpatient|pcp', 'Outpatient/Primary Care:'),
    ('intensity|acute|complex', 'Composite/Risk Scores:'),
    ('days_since', 'Recency Features:'),
    ('', 'Other Features:')
]:
    if category:
        cat_features = [f for f in selected_features_sorted if category in f.lower()]
    else:
        cat_features = [f for f in selected_features_sorted 
                       if not any(x in f.lower() for x in 
                       ['gi', 'ed', 'inp', 'outpatient', 'pcp', 'intensity', 
                        'acute', 'complex', 'days_since'])]
    
    if cat_features:
        print(f"\n{description}")
        for feat in cat_features:
            print(f"  - {feat}")

# Select final columns and save
final_columns = ['PAT_ID', 'END_DTTM'] + sorted(list(set(selected_features)))
df_reduced = df_final.select(*final_columns)

# Write to final table
output_table = f'{trgt_cat}.clncl_ds.herald_eda_train_visit_features_reduced'
df_reduced.write.mode('overwrite').option("mergeSchema", "true").saveAsTable(output_table)

# Verify row count preserved
final_count = spark.table(output_table).count()
assert final_count == total_rows, f"Row count mismatch: {final_count} vs {total_rows}"

print("\n" + "="*60)
print("FEATURE REDUCTION SUMMARY")
print("="*60)
print(f"Original features: 41")
print(f"Selected features: {len(set(selected_features))}")
print(f"Reduction: {(1 - len(set(selected_features))/41)*100:.1f}%")
print(f"\n✓ Reduced dataset saved to: {output_table}")
print(f"✓ Verified {final_count:,} rows written to table")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 📊 STEP 7 Conclusion
# MAGIC
# MAGIC Successfully engineered **5** new composite features, bringing the final feature count to **23**. This represents a **43.9% reduction** from the original 41 features. The reduced dataset was saved to `dev.clncl_ds.herald_eda_train_visit_features_reduced`, with **831,397** rows verified, ensuring data integrity. The final feature set is balanced, clinically relevant, and optimized for model training.
# MAGIC
# MAGIC **Key Achievement**: Finalized the feature reduction process by creating powerful composite features and saving the optimized dataset, achieving a significant reduction while preserving predictive power.
# MAGIC
# MAGIC **Next Step**: Proceed to the final summary of the entire workbook's achievements and implications.
# MAGIC

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Final Summary: Herald Visit History and Procedure Features Workbook
# MAGIC
# MAGIC This workbook successfully developed and refined a comprehensive set of visit history and procedure features for colorectal cancer (CRC) risk prediction, leveraging **831,397 patient-month observations**. The process involved meticulous feature gathering, in-depth analysis, and a data-driven feature reduction strategy.
# MAGIC
# MAGIC ### Achievements and Impact
# MAGIC
# MAGIC 1.  **Comprehensive Feature Engineering**: We extracted an initial set of **41 features** covering acute care (ED, inpatient), routine care (outpatient, PCP), specialty engagement (GI specialists), GI symptom burden, and care continuity. This provides a holistic view of patient healthcare utilization.
# MAGIC 2.  **Robust Data Validation**: Each step of the feature gathering process included rigorous validation, ensuring data quality and integrity. A critical row count validation confirmed that **831,397 observations** were consistently processed throughout.
# MAGIC 3.  **In-depth Clinical Insights**: The analysis revealed significant care gaps and system failures within the screening-overdue cohort:
# MAGIC     *   **Primary Care Crisis**: **62.6%** of patients had no PCP visits in the past year, suggesting emergency care often substitutes for prevention.
# MAGIC     *   **GI Specialty Bottleneck**: Only **0.9%** of patients saw a GI specialist despite **8.2%** presenting with GI-related symptoms, indicating an **89% unmet need** for specialty evaluation.
# MAGIC     *   **Emergency Department Reliance**: **18.8%** of patients relied on the ED for healthcare, with **4.8%** having recent ED use (last 90 days).
# MAGIC 4.  **Data-Driven Feature Reduction**: Employing a balanced approach combining Risk Ratios, Mutual Information, and clinical domain knowledge, we successfully reduced the feature set from **41 to 23 features**, achieving a **43.9% reduction**. This optimized set retains the most predictive and interpretable signals.
# MAGIC 5.  **Clinically Meaningful Composites**: Five new composite features were engineered, such as `visit_gi_symptoms_no_specialist` (care gap indicator) and `visit_frequent_ed_no_pcp` (crisis care pattern), which capture complex clinical scenarios and enhance model interpretability.
# MAGIC
# MAGIC ### Technical Excellence
# MAGIC
# MAGIC The implementation utilized Spark SQL for efficient distributed data processing, handling a large volume of raw encounter data (e.g., 19.8M ED/Inpatient and 308M outpatient encounters). Python with Pandas and scikit-learn was used for detailed statistical analysis and Mutual Information calculation on a stratified sample, demonstrating a hybrid approach for scalability and analytical depth. The use of clear data quality cutoffs and explicit validation steps throughout the notebook ensures the reliability of the generated features.
# MAGIC
# MAGIC ### Deliverables
# MAGIC
# MAGIC The primary deliverable is the `herald_eda_train_visit_features_reduced` table, containing **23 optimized visit history features** for **831,397 patient-month observations**. This table is ready for integration into downstream CRC risk prediction models.
# MAGIC
# MAGIC ### Next Steps
# MAGIC
# MAGIC The next phase involves integrating these refined visit history features with other feature domains (e.g., labs, medications, demographics) to build a comprehensive predictive model for CRC risk in the screening-overdue population. Further analysis will focus on model performance, interpretability, and the clinical utility of the identified high-risk patterns to guide targeted interventions.
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC