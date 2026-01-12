# Databricks notebook source
# MAGIC %md
# MAGIC # 🎯 Quick Start: Inpatient Medications Feature Engineering
# MAGIC
# MAGIC **In 3 sentences:**
# MAGIC 1. We extract **48 inpatient medication features** from MAR (Medication Administration Record) data across **831,397 patient-month observations** (23% hospitalization rate)
# MAGIC 2. We discover that **recent hospitalizations show exponential CRC risk decay** - patients hospitalized 0-30 days ago have **6.9x higher CRC risk** than baseline
# MAGIC 3. We reduce to **20 key features** (58% reduction) while preserving critical acute care signals like GI bleeding patterns and hospitalization recency
# MAGIC
# MAGIC **Key finding:** Hospitalization timing creates a powerful temporal gradient - CRC risk is highest immediately post-discharge and declines progressively over months, providing actionable intelligence for screening prioritization.
# MAGIC
# MAGIC **Time to run:** ~15 minutes | **Output:** 2 tables with 831,397 rows each
# MAGIC
# MAGIC ## 📋 Output Tables
# MAGIC - `herald_eda_train_inpatient_meds`: Full 48 features from MAR data
# MAGIC - `herald_eda_train_inpatient_meds_reduced`: Optimized 20 features for modeling
# MAGIC
# MAGIC ## 🔑 Critical Insights
# MAGIC - **Hemorrhoid medications:** 87% less common inpatient vs outpatient (0.04% vs 0.2%)
# MAGIC - **Opioids:** Most common inpatient medication (20.7% prevalence) indicating surgery/severe pain
# MAGIC - **Acute GI bleeding pattern:** Iron + PPI combination shows 1.86x CRC risk
# MAGIC - **Data constraint:** MAR data available from 2021-07-01, affecting early 2023 observations
# MAGIC Introduction and Sum

# COMMAND ----------

# MAGIC %md
# MAGIC # Herald Inpatient Medications Feature Engineering
# MAGIC
# MAGIC ## Introduction and Clinical Motivation
# MAGIC
# MAGIC This notebook extracts inpatient medication administration data from **831,397 patient-month observations** for CRC risk prediction. Inpatient medications capture acute care episodes through Medication Administration Record (MAR) data, representing actual medication delivery during hospitalizations (28.9% of observations). Unlike outpatient prescriptions, MAR data confirms medication administration, providing definitive evidence of acute medical needs.
# MAGIC
# MAGIC ### Why Inpatient Medications Are Critical for CRC Risk
# MAGIC
# MAGIC Inpatient medication patterns reveal acute events with significant temporal associations:
# MAGIC
# MAGIC #### 1. **Hospitalization Timing Patterns**
# MAGIC - Recent hospitalizations (<30 days) show **6.9x elevated CRC risk**
# MAGIC - Risk declines progressively: 31-90 days (3.1x), 91-180 days (1.8x)
# MAGIC - Temporal gradient suggests symptomatic presentation preceding diagnosis
# MAGIC - MAR data provides precise timing of acute episodes
# MAGIC
# MAGIC #### 2. **Acute GI Crisis Patterns**
# MAGIC - **GI bleeding medications:** 1.7% prevalence with 1.84x CRC risk
# MAGIC - **Acute bleeding management** (iron + PPI): 3.0% prevalence, 1.86x risk
# MAGIC - **Inpatient laxatives:** 9.2% (post-op ileus/obstruction concerns)
# MAGIC - **GI-specific hospitalizations:** 10.2% of cohort
# MAGIC - These represent emergency presentations often preceding diagnosis
# MAGIC
# MAGIC #### 3. **Pain Management Reveals Severity**
# MAGIC - **Opioids:** 20.7% (highest prevalence) - surgical/severe pain indicator
# MAGIC - **NSAIDs:** 15.7% (4x outpatient rate) - actual PRN administration captured
# MAGIC - Combined with antibiotics: suggests complications requiring intensive management
# MAGIC - Post-surgical patterns indicate GI procedures
# MAGIC
# MAGIC #### 4. **Hemorrhoid Treatment Paradox**
# MAGIC - **Inpatient:** 0.07% (542 patients)
# MAGIC - **Outpatient:** 0.2% (comparison from other analysis)
# MAGIC - **87% reduction** - hemorrhoids rarely treated during admission
# MAGIC - When present inpatient: may indicate severe bleeding/thrombosis requiring urgent care
# MAGIC
# MAGIC ### Temporal Risk Analysis
# MAGIC
# MAGIC Analysis reveals exponential risk decay with time since hospitalization:
# MAGIC - **0-7 days:** 4.5% CRC rate (11.6x baseline)
# MAGIC - **8-30 days:** 2.2% CRC rate (5.7x baseline)
# MAGIC - **31-60 days:** 1.1% CRC rate (2.8x baseline)
# MAGIC - **61-90 days:** 0.9% CRC rate (2.3x baseline)
# MAGIC - **180+ days:** 0.3% CRC rate (0.9x baseline)
# MAGIC
# MAGIC This temporal pattern suggests acute symptoms prompting admission may be cancer-related.
# MAGIC
# MAGIC ### Inpatient vs Outpatient: Complementary Signals
# MAGIC
# MAGIC <table>
# MAGIC   <thead>
# MAGIC     <tr>
# MAGIC       <th>Aspect</th>
# MAGIC       <th>Inpatient</th>
# MAGIC       <th>Outpatient</th>
# MAGIC       <th>Combined Insight</th>
# MAGIC     </tr>
# MAGIC   </thead>
# MAGIC   <tbody>
# MAGIC     <tr>
# MAGIC       <td><strong>Data reliability</strong></td>
# MAGIC       <td>100% administered</td>
# MAGIC       <td>May not be taken</td>
# MAGIC       <td>Inpatient confirms severity</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC       <td><strong>Temporal signal</strong></td>
# MAGIC       <td>Acute events</td>
# MAGIC       <td>Chronic patterns</td>
# MAGIC       <td>Recency crucial</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC       <td><strong>Hemorrhoid meds</strong></td>
# MAGIC       <td>0.07%</td>
# MAGIC       <td>0.2%</td>
# MAGIC       <td>Outpatient-dominant</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC       <td><strong>NSAIDs</strong></td>
# MAGIC       <td>15.7%</td>
# MAGIC       <td>~4%</td>
# MAGIC       <td>Inpatient captures PRN</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC       <td><strong>Iron</strong></td>
# MAGIC       <td>2.3%</td>
# MAGIC       <td>~1.2%</td>
# MAGIC       <td>Similar acute/chronic rates</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC       <td><strong>Risk window</strong></td>
# MAGIC       <td>Days to weeks</td>
# MAGIC       <td>Weeks to months</td>
# MAGIC       <td>Inpatient more urgent</td>
# MAGIC     </tr>
# MAGIC   </tbody>
# MAGIC </table>
# MAGIC
# MAGIC ### Feature Engineering Strategy
# MAGIC
# MAGIC #### Temporal Granularity
# MAGIC - Ultra-fine temporal windows for hospitalization recency
# MAGIC - Days since last inpatient medication as proxy for admission timing
# MAGIC - Capture both any hospitalization and GI-specific admissions
# MAGIC
# MAGIC #### Acute Pattern Recognition
# MAGIC Composite features for bleeding, obstruction, infection:
# MAGIC - `inp_acute_gi_bleeding`: Iron + PPI pattern (managed hemorrhage)
# MAGIC - `inp_obstruction_pattern`: Laxatives + opioids (ileus/obstruction)
# MAGIC - `inp_severe_infection`: Antibiotics + opioids (sepsis/abscess)
# MAGIC - `inp_any_hospitalization`: Critical risk indicator
# MAGIC - All features prefixed with "inp_" for clear identification
# MAGIC
# MAGIC #### Data Processing
# MAGIC - 24-month lookback for consistency with outpatient analysis
# MAGIC - MAR data filtered for actual administration events
# MAGIC - Deduplication by patient-day to avoid double counting
# MAGIC - XGBoost-compatible NULL handling (no imputation needed)
# MAGIC
# MAGIC ### Data Availability Constraint
# MAGIC
# MAGIC MAR data available from **2021-07-01** onwards. For cohort observations in early 2023, this results in <24 months lookback (e.g., 2023-01-15 observation has ~18 months of medication history). This is acceptable as:
# MAGIC
# MAGIC 1. XGBoost handles variable history lengths via native NULL handling
# MAGIC 2. Affects only Q1 2023 observations (~11% of cohort)
# MAGIC 3. Most predictive patterns occur in recent 3-6 months
# MAGIC 4. Validation shows max lookback of 731 days with p95 at 689 days
# MAGIC
# MAGIC ## Expected Outcomes
# MAGIC
# MAGIC This analysis will produce:
# MAGIC - **48 comprehensive features** (16 medications × 3 types: flag, days_since, count)
# MAGIC - **Temporal risk stratification** based on hospitalization recency
# MAGIC - **Acute care pattern recognition** through composite features
# MAGIC - **High-quality predictive signals** from confirmed medication admini

# COMMAND ----------

# Generic restart command
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

# Initialize a Spark session for distributed data processing
spark = SparkSession.builder.getOrCreate()

# Ensure date/time comparisons use Central Time
spark.conf.set("spark.sql.session.timeZone", "America/Chicago")

# Define target catalog for SQL based on the environment variable
trgt_cat = os.environ.get('trgt_cat')

# Use appropriate Spark catalog based on the target CATEGORY
spark.sql('USE CATALOG prod;')

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 1 - Create Medication Grouper Category Map
# MAGIC
# MAGIC #### 🔍 What This Cell Does
# MAGIC Creates the first medication mapping table using Epic's grouper system to categorize medications into clinical classes (iron supplementation, PPIs, NSAIDs, statins, metformin). This establishes the foundation for identifying inpatient medication administration patterns.
# MAGIC
# MAGIC #### Why This Matters for Inpatient Data
# MAGIC Grouper-based mapping captures medications consistently across different formulations and brands. For inpatient MAR data, this ensures we identify all iron products (whether IV or oral), all PPI formulations used for bleeding prophylaxis, and all NSAID variants administered PRN during acute care episodes.
# MAGIC
# MAGIC #### What to Watch For
# MAGIC - Comprehensive coverage of medication variants within each category
# MAGIC - Clinical relevance of grouper selections for acute care settings
# MAGIC - Foundation for linking Epic grouper IDs to actual medication administrations
# MAGIC

# COMMAND ----------

# Cell 1: Create medication grouper category map
spark.sql(f'''
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_med_grouper_category_map
USING DELTA
AS
WITH id_map AS (
  SELECT * FROM VALUES
    -- IRON SUPPLEMENTATION
    ('IRON_SUPPLEMENTATION', 103100859),
    ('IRON_SUPPLEMENTATION', 1060000647),
    ('IRON_SUPPLEMENTATION', 1060000646),
    ('IRON_SUPPLEMENTATION', 1060000650),

    -- PPI USE
    ('PPI_USE', 1030101319),
    ('PPI_USE', 103101245),
    ('PPI_USE', 1060033801),
    ('PPI_USE', 1060038401),

    -- NSAID/ASA USE
    ('NSAID_ASA_USE', 1060031001),
    ('NSAID_ASA_USE', 1060000523),
    ('NSAID_ASA_USE', 1060015801),
    ('NSAID_ASA_USE', 1060046101),
    ('NSAID_ASA_USE', 1060028401),
    ('NSAID_ASA_USE', 1060028501),
    ('NSAID_ASA_USE', 103100721),

    -- STATIN USE
    ('STATIN_USE', 1232000017),('STATIN_USE', 1765734),('STATIN_USE', 1765928),
    ('STATIN_USE', 1765229),('STATIN_USE', 1765232),('STATIN_USE', 1765246),('STATIN_USE', 1765249),
    ('STATIN_USE', 1765260),('STATIN_USE', 1765261),('STATIN_USE', 1765262),('STATIN_USE', 1765263),('STATIN_USE', 1765264),
    ('STATIN_USE', 1754575),('STATIN_USE', 1754577),('STATIN_USE', 1754603),('STATIN_USE', 1754584),
    ('STATIN_USE', 1754588),('STATIN_USE', 1754592),('STATIN_USE', 1754593),('STATIN_USE', 1754594),('STATIN_USE', 1754595),
    ('STATIN_USE', 1765241),('STATIN_USE', 1765239),('STATIN_USE', 1765240),('STATIN_USE', 1765253),('STATIN_USE', 1765254),
    ('STATIN_USE', 1030107485),('STATIN_USE', 103103567),('STATIN_USE', 103103088),('STATIN_USE', 105100537),('STATIN_USE', 1060048301),

    -- METFORMIN USE
    ('METFORMIN_USE', 103101190),('METFORMIN_USE', 1060036201),('METFORMIN_USE', 1060040001),
    ('METFORMIN_USE', 1060000704),('METFORMIN_USE', 1060040101),('METFORMIN_USE', 1765323),
    ('METFORMIN_USE', 1765276),('METFORMIN_USE', 1765279),('METFORMIN_USE', 1765282),('METFORMIN_USE', 1765284),
    ('METFORMIN_USE', 1765289),('METFORMIN_USE', 1765292),('METFORMIN_USE', 1765320),('METFORMIN_USE', 1765328),
    ('METFORMIN_USE', 1765331),('METFORMIN_USE', 1765334),('METFORMIN_USE', 1765336),('METFORMIN_USE', 1765339)
  AS t(category_key, grouper_id)
)
SELECT
  m.category_key,
  gi.GROUPER_ID,
  gi.GROUPER_NAME
FROM id_map m
LEFT JOIN clarity.grouper_items gi
  ON gi.GROUPER_ID = m.grouper_id;
''')

# COMMAND ----------

# MAGIC %md
# MAGIC #### 📊 Cell 1 Conclusion
# MAGIC
# MAGIC Successfully created medication grouper category map linking clinical categories to Epic grouper IDs. This provides the backbone for identifying key medication classes in MAR data.
# MAGIC
# MAGIC **Key Achievement**: Established standardized medication categorization system for inpatient analysis
# MAGIC
# MAGIC **Next Step**: Create direct medication ID mappings for medications not covered by groupers
# MAGIC

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 2 - Create Direct Medication ID Category Map
# MAGIC
# MAGIC #### 🔍 What This Cell Does
# MAGIC Creates comprehensive medication ID mappings for medications not captured by groupers, including detailed laxative categories (bulk/fiber, osmotic, stimulant), antidiarrheals, GI bleeding medications, opioids, and other acute care drugs. Uses vetted medication lists with clinical annotations.
# MAGIC
# MAGIC #### Why This Matters for Inpatient Data
# MAGIC Many critical inpatient medications aren't in standard groupers. Laxatives are crucial for post-operative ileus detection, GI bleeding medications indicate acute hemorrhage management, and opioid patterns reveal surgical procedures or severe pain episodes. Direct ID mapping ensures comprehensive capture of acute care medication patterns.
# MAGIC
# MAGIC #### What to Watch For
# MAGIC - Extensive laxative subcategorization (osmotic vs stimulant patterns matter clinically)
# MAGIC - GI bleeding medications (tranexamic acid, octreotide for acute management)
# MAGIC - Opioid coverage for pain/surgery indicators
# MAGIC - IBD medications for inflammatory disease management
# MAGIC

# COMMAND ----------

# Cell 2: Create medication ID category map (for medications not in groupers)
spark.sql(f'''
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_med_id_category_map
USING DELTA
AS
SELECT
  UPPER(TRIM(CATEGORY_KEY))   AS CATEGORY_KEY,
  CAST(MEDICATION_ID AS BIGINT) AS MEDICATION_ID,
  UPPER(TRIM(GEN_NAME))       AS GEN_NAME,
  SOURCE_TAG,
  NOTES
FROM VALUES
  -- ===== LAXATIVE_USE =====
  ('LAXATIVE_USE', 229101, 'PRUCALOPRIDE 2 MG TABLET',                  'regex_v1_vetted', '5-HT4 agonist'),
  ('LAXATIVE_USE', 229100, 'PRUCALOPRIDE 1 MG TABLET',                  'regex_v1_vetted', '5-HT4 agonist'),
  ('LAXATIVE_USE', 179778, 'PSYLLIUM HUSK 3.5 GRAM ORAL POWDER PACKET', 'regex_v1_vetted', 'bulk/fiber'),
  ('LAXATIVE_USE', 221195, 'PSYLLIUM HUSK 0.4 GRAM CAPSULE',            'regex_v1_vetted', 'bulk/fiber'),
  ('LAXATIVE_USE', 87600,  'PSYLLIUM HUSK 0.52 GRAM CAPSULE',           'regex_v1_vetted', 'bulk/fiber'),
  ('LAXATIVE_USE', 81953,  'CALCIUM POLYCARBOPHIL 500 MG CHEWABLE TABLET','regex_v1_vetted','bulk/fiber'),
  ('LAXATIVE_USE', 80942,  'CALCIUM POLYCARBOPHIL 625 MG TABLET',       'regex_v1_vetted', 'bulk/fiber'),
  ('LAXATIVE_USE', 80695,  'METHYLCELLULOSE (LAXATIVE) 500 MG TABLET',  'regex_v1_vetted', 'bulk/fiber'),
  ('LAXATIVE_USE', 197631, 'LACTULOSE 10 GRAM/15 ML ORAL SOLUTION',     'regex_v1_vetted', 'osmotic'),
  ('LAXATIVE_USE', 39610,  'LACTULOSE 10 GRAM/15 ML ORAL SOLUTION',     'regex_v1_vetted', 'osmotic'),
  ('LAXATIVE_USE', 93851,  'LACTULOSE 10 GRAM/15 ML ORAL SOLUTION',     'regex_v1_vetted', 'osmotic'),
  ('LAXATIVE_USE', 80411,  'LACTULOSE 10 GRAM ORAL PACKET',             'regex_v1_vetted', 'osmotic'),
  ('LAXATIVE_USE', 82328,  'LACTULOSE 20 GRAM ORAL PACKET',             'regex_v1_vetted', 'osmotic'),
  ('LAXATIVE_USE', 197575, 'LACTULOSE 20 GRAM/30 ML ORAL SOLUTION',     'regex_v1_vetted', 'osmotic'),
  ('LAXATIVE_USE', 95550,  'LUBIPROSTONE 24 MCG CAPSULE',               'regex_v1_vetted', 'secretagogue'),
  ('LAXATIVE_USE', 178676, 'LUBIPROSTONE 8 MCG CAPSULE',                'regex_v1_vetted', 'secretagogue'),
  ('LAXATIVE_USE', 70603,  'LUBIPROSTONE 24 MCG CAPSULE',               'regex_v1_vetted', 'secretagogue'),
  ('LAXATIVE_USE', 214922, 'PSYLLIUM HUSK 3 GRAM/3 GRAM ORAL POWDER',   'regex_v1_vetted', 'bulk/fiber'),
  ('LAXATIVE_USE', 202759, 'LINACLOTIDE 145 MCG CAPSULE',               'regex_v1_vetted', 'secretagogue'),
  ('LAXATIVE_USE', 202760, 'LINACLOTIDE 290 MCG CAPSULE',               'regex_v1_vetted', 'secretagogue'),
  ('LAXATIVE_USE', 221253, 'LINACLOTIDE 72 MCG CAPSULE',                'regex_v1_vetted', 'secretagogue'),
  ('LAXATIVE_USE', 221416, 'PLECANATIDE 3 MG TABLET',                   'regex_v1_vetted', 'secretagogue'),
  ('LAXATIVE_USE', 222935, 'NALDEMEDINE 0.2 MG TABLET',                 'regex_v1_vetted', 'OIC antagonist'),
  ('LAXATIVE_USE', 212755, 'NALOXEGOL 12.5 MG TABLET',                  'regex_v1_vetted', 'OIC antagonist'),
  ('LAXATIVE_USE', 212698, 'NALOXEGOL 25 MG TABLET',                    'regex_v1_vetted', 'OIC antagonist'),
  ('LAXATIVE_USE', 219542, 'METHYLNALTREXONE 150 MG TABLET',            'regex_v1_vetted', 'OIC antagonist'),
  ('LAXATIVE_USE', 178822, 'METHYLNALTREXONE 12 MG/0.6 ML SC SOLUTION', 'regex_v1_vetted', 'OIC antagonist'),
  ('LAXATIVE_USE', 197993, 'METHYLNALTREXONE 12 MG/0.6 ML SC SYRINGE',  'regex_v1_vetted', 'OIC antagonist'),
  ('LAXATIVE_USE', 197990, 'METHYLNALTREXONE 8 MG/0.4 ML SC SYRINGE',   'regex_v1_vetted', 'OIC antagonist'),
  ('LAXATIVE_USE', 2572,   'DOCUSATE SODIUM 100 MG TABLET',             'regex_v1_vetted', 'stool softener'),
  ('LAXATIVE_USE', 212170, 'DOCUSATE SODIUM 50 MG CAPSULE',             'regex_v1_vetted', 'stool softener'),
  ('LAXATIVE_USE', 2085,   'DOCUSATE SODIUM 100 MG CAPSULE',            'regex_v1_vetted', 'stool softener'),
  ('LAXATIVE_USE', 2568,   'DOCUSATE SODIUM 50 MG CAPSULE',             'regex_v1_vetted', 'stool softener'),
  ('LAXATIVE_USE', 2567,   'DOCUSATE SODIUM 250 MG CAPSULE',            'regex_v1_vetted', 'stool softener'),
  ('LAXATIVE_USE', 1815,   'DOCUSATE SODIUM 100 MG CAPSULE',            'regex_v1_vetted', 'stool softener'),
  ('LAXATIVE_USE', 4372,   'DOCUSATE SODIUM 100 MG CAPSULE',            'regex_v1_vetted', 'stool softener'),
  ('LAXATIVE_USE', 11878,  'DOCUSATE SODIUM 100 MG CAPSULE',            'regex_v1_vetted', 'stool softener'),
  ('LAXATIVE_USE', 39752,  'SENNOSIDES 8.6 MG-DOCUSATE SODIUM 50 MG TAB','regex_v1_vetted','stimulant + stool softener'),
  ('LAXATIVE_USE', 30015,  'SENNOSIDES 8.6 MG-DOCUSATE SODIUM 50 MG TAB','regex_v1_vetted','stimulant + stool softener'),
  ('LAXATIVE_USE', 7159,   'SENNOSIDES 8.6 MG-DOCUSATE SODIUM 50 MG TAB','regex_v1_vetted','stimulant + stool softener'),
-- Polyethylene glycol products (very commonly used)
('LAXATIVE_USE', 25424, 'POLYETHYLENE GLYCOL 3350 17 GRAM ORAL POWDER PACKET', 'discovery_v2', 'osmotic'),
('LAXATIVE_USE', 24984, 'POLYETHYLENE GLYCOL 3350 17 GRAM/DOSE ORAL POWDER', 'discovery_v2', 'osmotic'),
('LAXATIVE_USE', 156129, 'POLYETHYLENE GLYCOL 3350 ORAL', 'discovery_v2', 'osmotic'),
-- Bisacodyl products
('LAXATIVE_USE', 1080, 'BISACODYL 10 MG RECTAL SUPPOSITORY', 'discovery_v2', 'stimulant'),
('LAXATIVE_USE', 13632, 'BISACODYL 5 MG TABLET,DELAYED RELEASE', 'discovery_v2', 'stimulant'),
('LAXATIVE_USE', 83533, 'BISACODYL 5 MG TABLET', 'discovery_v2', 'stimulant'),
-- Magnesium products
('LAXATIVE_USE', 79944, 'MAGNESIUM HYDROXIDE 400 MG/5 ML ORAL SUSPENSION', 'discovery_v2', 'osmotic'),
('LAXATIVE_USE', 4711, 'MAGNESIUM CITRATE ORAL SOLUTION', 'discovery_v2', 'osmotic'),
('LAXATIVE_USE', 155118, 'MAGNESIUM CITRATE ORAL', 'discovery_v2', 'osmotic'),
-- Senna combinations (beyond what you have)
('LAXATIVE_USE', 40926, 'SENNOSIDES 8.6 MG-DOCUSATE SODIUM 50 MG TABLET', 'discovery_v2', 'stimulant combo'),
-- Alvimopan (opioid antagonist for postop ileus)
('LAXATIVE_USE', 179661, 'ALVIMOPAN 12 MG CAPSULE', 'discovery_v2', 'peripherally acting opioid antagonist'),
-- Add these to ANTIDIARRHEAL_USE
('ANTIDIARRHEAL_USE', 4560, 'LOPERAMIDE 2 MG CAPSULE', 'discovery_v2', 'antimotility'),
('ANTIDIARRHEAL_USE', 4562, 'LOPERAMIDE 2 MG TABLET', 'discovery_v2', 'antimotility'),
('ANTIDIARRHEAL_USE', 189801, 'RIFAXIMIN 550 MG TABLET', 'discovery_v2', 'antibiotic for IBS-D/HE'),
('ANTIDIARRHEAL_USE', 189817, 'RIFAXIMIN 550 MG TABLET (XIFAXAN)', 'discovery_v2', 'antibiotic for IBS-D/HE'),
('ANTIDIARRHEAL_USE', 81342, 'BISMUTH SUBSALICYLATE 262 MG CHEWABLE TABLET', 'discovery_v2', 'bismuth compound'),
('ANTIDIARRHEAL_USE', 80560, 'BISMUTH SUBSALICYLATE 262 MG/15 ML ORAL SUSPENSION', 'discovery_v2', 'bismuth compound'),
('ANTIDIARRHEAL_USE', 89923, 'ALOSETRON 0.5 MG TABLET', 'discovery_v2', '5-HT3 antagonist for IBS-D'),
('ANTIDIARRHEAL_USE', 79963, 'ALOSETRON 1 MG TABLET', 'discovery_v2', '5-HT3 antagonist for IBS-D'),

  -- ===== ANTIDIARRHEAL_USE =====
  ('ANTIDIARRHEAL_USE', 2516,   'DIPHENOXYLATE-ATROPINE 2.5 MG-0.025 MG TABLET',             'regex_v1_vetted', 'vetted antidiarrheal'),
  ('ANTIDIARRHEAL_USE', 88489,  'CHOLESTYRAMINE-ASPARTAME 4 GRAM ORAL POWDER FOR SUSP IN A PACKET', 'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 183910, 'COLESEVELAM 3.75 GRAM ORAL POWDER PACKET',                  'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 16388,  'CHOLESTYRAMINE 4 GRAM ORAL POWDER FOR SUSPENSION IN A PACKET','regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 37482,  'CHOLESTYRAMINE 4 GRAM ORAL POWDER FOR SUSPENSION IN A PACKET','regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 80288,  'COLESEVELAM 625 MG TABLET',                                 'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 176079, 'CHOLESTYRAMINE 4 GRAM ORAL POWDER',                         'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 82389,  'COLESTIPOL 5 GRAM ORAL PACKET',                             'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 21994,  'CHOLESTYRAMINE 4 GRAM ORAL POWDER',                         'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 216207, 'ELUXADOLINE 75 MG TABLET',                                  'regex_v1_vetted', 'IBS-D agent'),
  ('ANTIDIARRHEAL_USE', 9589,   'CHOLESTYRAMINE (WITH SUGAR) 4 GRAM ORAL POWDER',            'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 89845,  'COLESTIPOL 5 GRAM ORAL GRANULES',                           'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 176078, 'CHOLESTYRAMINE 4 GRAM ORAL POWDER',                         'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 28857,  'COLESEVELAM 625 MG TABLET',                                 'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 216198, 'ELUXADOLINE 100 MG TABLET',                                 'regex_v1_vetted', 'IBS-D agent'),
  ('ANTIDIARRHEAL_USE', 82555,  'COLESTIPOL 1 GRAM TABLET',                                  'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 9588,   'CHOLESTYRAMINE (WITH SUGAR) 4 GRAM POWDER FOR SUSP IN A PACKET','regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 176054, 'CHOLESTYRAMINE-ASPARTAME 4 GRAM ORAL POWDER',               'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 13898,  'COLESTIPOL 1 GRAM TABLET',                                  'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 216086, 'ELUXADOLINE 75 MG TABLET',                                  'regex_v1_vetted', 'IBS-D agent'),
  ('ANTIDIARRHEAL_USE', 2515,   'DIPHENOXYLATE-ATROPINE 2.5 MG-0.025 MG/5 ML ORAL LIQUID',   'regex_v1_vetted', 'vetted antidiarrheal'),
  ('ANTIDIARRHEAL_USE', 216087, 'ELUXADOLINE 100 MG TABLET',                                 'regex_v1_vetted', 'IBS-D agent'),
  ('ANTIDIARRHEAL_USE', 183915, 'COLESEVELAM 3.75 GRAM ORAL POWDER PACKET',                  'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 252231, 'CHOLESTYRAMINE 4 GRAM ORAL POWDER',                         'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 252230, 'CHOLESTYRAMINE 4 GRAM ORAL POWDER FOR SUSPENSION IN A PACKET','regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 216374, 'ELUXADOLINE ORAL',                                          'regex_v1_vetted', 'IBS-D agent'),
  ('ANTIDIARRHEAL_USE', 4553,   'DIPHENOXYLATE-ATROPINE 2.5 MG-0.025 MG TABLET',             'regex_v1_vetted', 'vetted antidiarrheal'),
  ('ANTIDIARRHEAL_USE', 154828, 'DIPHENOXYLATE-ATROPINE ORAL',                               'regex_v1_vetted', 'vetted antidiarrheal'),
  ('ANTIDIARRHEAL_USE', 146632, 'CHOLESTYRAMINE-ASPARTAME ORAL',                             'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 147147, 'COLESTIPOL ORAL',                                           'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 6767,   'CHOLESTYRAMINE (WITH SUGAR) 4 GRAM ORAL POWDER',            'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 160917, 'CHOLESTYRAMINE (WITH SUGAR) ORAL',                          'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 166274, 'COLESEVELAM ORAL',                                          'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 89844,  'COLESTIPOL 7.5 GRAM ORAL PACKET',                           'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 147146, 'COLESTIPOL ORAL',                                           'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 148928, 'DIPHENOXYLATE-ATROPINE ORAL',                               'regex_v1_vetted', 'vetted antidiarrheal'),
  ('ANTIDIARRHEAL_USE', 12057,  'COLESTIPOL 7.5 GRAM ORAL PACKET',                           'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 197067, 'CHOLESTYRAMINE (WITH SUGAR) ORAL',                          'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 146631, 'CHOLESTYRAMINE ORAL',                                       'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 6766,   'CHOLESTYRAMINE (WITH SUGAR) 4 GRAM POWDER FOR SUSP IN A PACKET','regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 147144, 'COLESEVELAM ORAL',                                          'regex_v1_vetted', 'bile-acid binder'),

  -- ===== ANTISPASMODIC_USE =====
  ('ANTISPASMODIC_USE', 1738,    'CHLORDIAZEPOXIDE-CLIDINIUM 5 MG-2.5 MG CAPSULE', 'regex_v1_vetted', 'Librax combo'),
  ('ANTISPASMODIC_USE', 2418,    'DICYCLOMINE 10 MG CAPSULE',                      'regex_v1_vetted', 'GI anticholinergic'),
  ('ANTISPASMODIC_USE', 205840,  'DICYCLOMINE 10 MG/5 ML ORAL SOLUTION',           'regex_v1_vetted', 'GI anticholinergic'),
  ('ANTISPASMODIC_USE', 2420,    'DICYCLOMINE 20 MG TABLET',                       'regex_v1_vetted', 'GI anticholinergic'),
  ('ANTISPASMODIC_USE', 144535,  'DICYCLOMINE ORAL',                               'regex_v1_vetted', 'GI anticholinergic'),
  ('ANTISPASMODIC_USE', 148732,  'DICYCLOMINE ORAL',                               'regex_v1_vetted', 'GI anticholinergic'),
  ('ANTISPASMODIC_USE', 4434,    'CHLORDIAZEPOXIDE-CLIDINIUM 5 MG-2.5 MG CAPSULE', 'regex_v1_vetted', 'Librax combo'),
  ('ANTISPASMODIC_USE', 146851,  'CHLORDIAZEPOXIDE-CLIDINIUM ORAL',                'regex_v1_vetted', 'Librax combo'),
  ('ANTISPASMODIC_USE', 154497,  'CHLORDIAZEPOXIDE-CLIDINIUM ORAL',                'regex_v1_vetted', 'Librax combo'),
  ('ANTISPASMODIC_USE', 254015,  'DICYCLOMINE 40 MG TABLET',                       'regex_v1_vetted', 'GI anticholinergic'),

  -- ===== B12_FOLATE_USE =====
  ('B12_FOLATE_USE', 4396,   'LEUCOVORIN CALCIUM 15 MG TABLET',                      'regex_v1_vetted', 'folinic acid tablet'),
  ('B12_FOLATE_USE', 4397,   'LEUCOVORIN CALCIUM 25 MG TABLET',                      'regex_v1_vetted', 'folinic acid tablet'),
  ('B12_FOLATE_USE', 212837, 'LEUCOVORIN 4 MG-PYRIDOXAL PHOSPHATE 50 MG-MECOBALAMIN 2 MG TABLET','regex_v1_vetted','combo w/ folinic acid & mecobalamin'),
  ('B12_FOLATE_USE', 4398,   'LEUCOVORIN CALCIUM 5 MG TABLET',                       'regex_v1_vetted', 'folinic acid tablet'),
  ('B12_FOLATE_USE', 4395,   'LEUCOVORIN CALCIUM 10 MG TABLET',                      'regex_v1_vetted', 'folinic acid tablet'),
  ('B12_FOLATE_USE', 242499, 'METHYLFOLATE CALCIUM 25,000 MCG DFE-MECOBALAMIN 2,000 MCG CAPSULE','regex_v1_vetted','high-dose methylfolate + mecobalamin'),
  ('B12_FOLATE_USE', 243320, 'B12 5,000 MCG-METHYLFOLATE 1,360 MCG DFE-B6 2.5 MG CHEWABLE TABLET','regex_v1_vetted','high-dose B12 + methylfolate combo'),
  ('B12_FOLATE_USE', 154414, 'LEUCOVORIN CALCIUM ORAL',                             'regex_v1_vetted', 'folinic acid oral'),
  ('B12_FOLATE_USE', 15370,  'LEUCOVORIN CALCIUM 10 MG/ML INJECTION SOLUTION',      'regex_v1_vetted', 'folinic acid injection'),
  ('B12_FOLATE_USE', 216354, 'COQ10-ALA-RESVERATROL-LEUCOVORIN-B6-MECOBALAMIN-VITC-D3 ORAL','regex_v1_vetted','multi-ingredient incl. leucovorin & mecobalamin'),
  ('B12_FOLATE_USE', 4394,   'LEUCOVORIN CALCIUM 50 MG SOLUTION FOR INJECTION',     'regex_v1_vetted', 'folinic acid injection'),
  ('B12_FOLATE_USE', 4392,   'LEUCOVORIN CALCIUM 100 MG SOLUTION FOR INJECTION',    'regex_v1_vetted', 'folinic acid injection'),

-- ===== IBD_MED_USE =====
('IBD_MEDICATIONS', 6496, 'PREDNISONE 20 MG TABLET', 'discovery_v2', 'corticosteroid'),
('IBD_MEDICATIONS', 6494, 'PREDNISONE 10 MG TABLET', 'discovery_v2', 'corticosteroid'),
('IBD_MEDICATIONS', 14887, 'METHYLPREDNISOLONE 4 MG TABLETS IN A DOSE PACK', 'discovery_v2', 'corticosteroid'),
('IBD_MEDICATIONS', 82000, 'BUDESONIDE DR - ER 3 MG CAPSULE,DELAYED,EXTENDED RELEASE', 'discovery_v2', 'GI-specific steroid'),
('IBD_MEDICATIONS', 167329, 'MESALAMINE 1.2 GRAM TABLET,DELAYED RELEASE', 'discovery_v2', '5-ASA'),
('IBD_MEDICATIONS', 94087, 'MESALAMINE 1,000 MG RECTAL SUPPOSITORY', 'discovery_v2', '5-ASA'),
('IBD_MEDICATIONS', 10535, 'MESALAMINE 4 GRAM/60 ML ENEMA', 'discovery_v2', '5-ASA'),
('IBD_MEDICATIONS', 7562, 'SULFASALAZINE 500 MG TABLET', 'discovery_v2', '5-ASA'),
('IBD_MEDICATIONS', 9183, 'AZATHIOPRINE 50 MG TABLET', 'discovery_v2', 'immunosuppressant'),
('IBD_MEDICATIONS', 4973, 'METHOTREXATE SODIUM 2.5 MG TABLET', 'discovery_v2', 'immunosuppressant'),
('IBD_MEDICATIONS', 80729, 'INFLIXIMAB 100 MG INTRAVENOUS SOLUTION', 'discovery_v2', 'anti-TNF biologic'),
('IBD_MEDICATIONS', 225598, 'ADALIMUMAB 40 MG/0.4 ML SUBCUTANEOUS PEN KIT', 'discovery_v2', 'anti-TNF biologic'),
('IBD_MEDICATIONS', 209779, 'VEDOLIZUMAB 300 MG INTRAVENOUS SOLUTION', 'discovery_v2', 'α4β7 integrin blocker'),
('IBD_MEDICATIONS', 184073, 'USTEKINUMAB 90 MG/ML SUBCUTANEOUS SYRINGE', 'discovery_v2', 'IL-12/23 inhibitor'),
('IBD_MEDICATIONS', 217804, 'TOFACITINIB ER 11 MG TABLET,EXTENDED RELEASE 24 HR', 'discovery_v2', 'JAK inhibitor'),

-- ===== HEMORRHOID/RECTAL_MED_USE =====
('HEMORRHOID_RECTAL_MEDS', 3738, 'HYDROCORTISONE ACETATE 25 MG RECTAL SUPPOSITORY', 'discovery_v2', 'rectal steroid'),
('HEMORRHOID_RECTAL_MEDS', 6589, 'HYDROCORTISONE 1 %-PRAMOXINE 1 % RECTAL FOAM', 'discovery_v2', 'rectal steroid combo'),
('HEMORRHOID_RECTAL_MEDS', 77665, 'HYDROCORTISONE-PRAMOXINE 2.5 %-1 % RECTAL CREAM', 'discovery_v2', 'rectal steroid combo'),
('HEMORRHOID_RECTAL_MEDS', 80624, 'HYDROCORTISONE 1 %-PRAMOXINE 1 % RECTAL FOAM', 'discovery_v2', 'rectal steroid combo'),
('HEMORRHOID_RECTAL_MEDS', 82892, 'HYDROCORTISONE-PRAMOXINE 1 %-1 % RECTAL CREAM', 'discovery_v2', 'rectal steroid combo'),
('HEMORRHOID_RECTAL_MEDS', 10210, 'HYDROCORTISONE 100 MG/60 ML ENEMA', 'discovery_v2', 'rectal steroid'),
('HEMORRHOID_RECTAL_MEDS', 35164, 'LIDOCAINE 3 %-HYDROCORTISONE 0.5 % RECTAL CREAM', 'discovery_v2', 'anesthetic combo'),

-- ===== GI_BLEEDING_MED_USE =====
('GI_BLEEDING_MEDS', 230527, 'TRANEXAMIC ACID 1,000 MG/100 ML IV PIGGYBACK', 'discovery_v2', 'antifibrinolytic'),
('GI_BLEEDING_MEDS', 196970, 'TRANEXAMIC ACID 1,000 MG/10 ML INTRAVENOUS SOLUTION', 'discovery_v2', 'antifibrinolytic'),
('GI_BLEEDING_MEDS', 190130, 'TRANEXAMIC ACID 650 MG TABLET', 'discovery_v2', 'antifibrinolytic'),
('GI_BLEEDING_MEDS', 211947, 'VASOPRESSIN 20 UNIT/ML INTRAVENOUS SOLUTION', 'discovery_v2', 'vasoconstrictor'),
('GI_BLEEDING_MEDS', 25122, 'OCTREOTIDE ACETATE 100 MCG/ML INJECTION SOLUTION', 'discovery_v2', 'somatostatin analog'),
('GI_BLEEDING_MEDS', 25121, 'OCTREOTIDE ACETATE 50 MCG/ML INJECTION SOLUTION', 'discovery_v2', 'somatostatin analog'),

-- ===== CHRONIC_OPIOD_MED_USE =====
('CHRONIC_OPIOID_USE', 3037, 'FENTANYL 50 MCG/ML INJECTION SOLUTION', 'discovery_v2', 'opioid'),
('CHRONIC_OPIOID_USE', 34505, 'HYDROCODONE 5 MG-ACETAMINOPHEN 325 MG TABLET', 'discovery_v2', 'opioid combo'),
('CHRONIC_OPIOID_USE', 10814, 'OXYCODONE 5 MG TABLET', 'discovery_v2', 'opioid'),
('CHRONIC_OPIOID_USE', 301537, 'MORPHINE 4 MG/ML INTRAVENOUS SOLUTION', 'discovery_v2', 'opioid'),
('CHRONIC_OPIOID_USE', 14632, 'TRAMADOL 50 MG TABLET', 'discovery_v2', 'opioid'),
('CHRONIC_OPIOID_USE', 27905, 'FENTANYL 25 MCG/HR TRANSDERMAL PATCH', 'discovery_v2', 'long-acting opioid'),
('CHRONIC_OPIOID_USE', 20920, 'MORPHINE ER 15 MG TABLET,EXTENDED RELEASE', 'discovery_v2', 'long-acting opioid'),
('CHRONIC_OPIOID_USE', 211843, 'OXYCODONE ER 10 MG TABLET,CRUSH RESISTANT,EXTENDED RELEASE', 'discovery_v2', 'long-acting opioid'),

-- ===== BROAD_SPECTRUM_ANTIBIOTIC_USE =====
('BROAD_SPECTRUM_ANTIBIOTICS', 9500, 'CEPHALEXIN 500 MG CAPSULE', 'discovery_v2', 'cephalosporin'),
('BROAD_SPECTRUM_ANTIBIOTICS', 25119, 'CIPROFLOXACIN 500 MG TABLET', 'discovery_v2', 'fluoroquinolone'),
('BROAD_SPECTRUM_ANTIBIOTICS', 82091, 'LEVOFLOXACIN 500 MG TABLET', 'discovery_v2', 'fluoroquinolone'),
('BROAD_SPECTRUM_ANTIBIOTICS', 87765, 'MOXIFLOXACIN 400 MG TABLET', 'discovery_v2', 'fluoroquinolone'),
('BROAD_SPECTRUM_ANTIBIOTICS', 177634, 'CEFAZOLIN 2 GRAM/20 ML INTRAVENOUS SYRINGE', 'discovery_v2', 'cephalosporin'),
('BROAD_SPECTRUM_ANTIBIOTICS', 79742, 'CEFTRIAXONE 1 GRAM SOLUTION FOR INJECTION', 'discovery_v2', 'cephalosporin'),
('BROAD_SPECTRUM_ANTIBIOTICS', 9621, 'CLINDAMYCIN HCL 300 MG CAPSULE', 'discovery_v2', 'lincosamide'),
('BROAD_SPECTRUM_ANTIBIOTICS', 83455, 'PIPERACILLIN-TAZOBACTAM 3.375 GRAM IV PIGGYBACK', 'discovery_v2', 'beta-lactam combo'),
('BROAD_SPECTRUM_ANTIBIOTICS', 83077, 'ERTAPENEM 1 GRAM SOLUTION FOR INJECTION', 'discovery_v2', 'carbapenem'),
('BROAD_SPECTRUM_ANTIBIOTICS', 80713, 'MEROPENEM 1 GRAM INTRAVENOUS SOLUTION', 'discovery_v2', 'carbapenem'),

-- ===== HORMON_THERAPY_USE =====
('HORMONE_THERAPY', 82101, 'ESTRADIOL 0.01% VAGINAL CREAM', 'discovery_v2', 'estrogen'),
('HORMONE_THERAPY', 9967, 'ESTRADIOL 1 MG TABLET', 'discovery_v2', 'estrogen'),
('HORMONE_THERAPY', 80522, 'CONJUGATED ESTROGENS 0.625 MG/GRAM VAGINAL CREAM', 'discovery_v2', 'estrogen'),
('HORMONE_THERAPY', 11498, 'TAMOXIFEN 20 MG TABLET', 'discovery_v2', 'SERM'),
('HORMONE_THERAPY', 81433, 'RALOXIFENE 60 MG TABLET', 'discovery_v2', 'SERM'),
('HORMONE_THERAPY', 7784, 'TESTOSTERONE CYPIONATE 200 MG/ML INTRAMUSCULAR OIL', 'discovery_v2', 'testosterone'),
('HORMONE_THERAPY', 194705, 'TESTOSTERONE 1.62% TRANSDERMAL GEL', 'discovery_v2', 'testosterone'),

-- ===== CHEMOTHERAPY_AGENT_USE =====
('CHEMOTHERAPY_AGENTS', 78342, 'FLUOROURACIL 5 % TOPICAL CREAM', 'discovery_v2', '5-FU'),
('CHEMOTHERAPY_AGENTS', 79057, 'CAPECITABINE 500 MG TABLET', 'discovery_v2', '5-FU prodrug'),
('CHEMOTHERAPY_AGENTS', 80994, 'CAPECITABINE 150 MG TABLET', 'discovery_v2', '5-FU prodrug'),
('CHEMOTHERAPY_AGENTS', 89931, 'BEVACIZUMAB 25 MG/ML INTRAVENOUS SOLUTION', 'discovery_v2', 'VEGF inhibitor'),
('CHEMOTHERAPY_AGENTS', 10631, 'MITOMYCIN 40 MG INTRAVENOUS SOLUTION', 'discovery_v2', 'alkylating agent')

AS t(category_key, medication_id, gen_name, source_tag, notes);
''')

# COMMAND ----------

# MAGIC %md
# MAGIC #### 📊 Cell 2 Conclusion
# MAGIC
# MAGIC Successfully created direct medication ID mappings for 200+ medications across 16 clinical categories, with particular depth in GI-related acute care medications.
# MAGIC
# MAGIC **Key Achievement**: Comprehensive coverage of inpatient-specific medications not captured by standard groupers
# MAGIC
# MAGIC **Next Step**: Create unpivoted medication administration table linking patient observations to actual MAR data
# MAGIC

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 3 - Create Unpivoted Inpatient Medications Table
# MAGIC
# MAGIC #### 🔍 What This Cell Does
# MAGIC Extracts actual medication administration records from MAR data, joining with both grouper and direct ID mappings. Applies critical data availability constraint (MAR data from 2021-07-01 onwards) and 24-month lookback window. Deduplicates by patient-day to avoid double counting.
# MAGIC
# MAGIC #### Why This Matters for Inpatient Data
# MAGIC MAR data represents confirmed medication administration (not just prescriptions), providing definitive evidence of acute medical needs. The 2021-07-01 constraint affects early 2023 observations but captures the most predictive recent patterns. Deduplication ensures each medication-day represents a single clinical event.
# MAGIC
# MAGIC #### What to Watch For
# MAGIC - Row count matching cohort exactly (831,397 observations)
# MAGIC - Date ranges respecting 2021-07-01 availability constraint
# MAGIC - Successful joins between MAR data and medication mappings
# MAGIC - Proper handling of inpatient-only orders (ORDERING_MODE_C = 2)

# COMMAND ----------

# Cell 3: Create unpivoted inpatient medications table
spark.sql(f'''
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_inpatient_meds_unpivoted AS

WITH
cohort AS (
  SELECT
    CAST(PAT_ID AS STRING)          AS PAT_ID,
    END_DTTM
  FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort
),

map_med AS (
  SELECT
    UPPER(TRIM(CATEGORY_KEY))       AS CATEGORY,
    CAST(MEDICATION_ID AS BIGINT)   AS MEDICATION_ID,
    UPPER(TRIM(GEN_NAME))           AS GEN_NAME
  FROM {trgt_cat}.clncl_ds.herald_eda_train_med_id_category_map
),

map_grp AS (
  SELECT
    UPPER(TRIM(CATEGORY_KEY))       AS CATEGORY,
    CAST(GROUPER_ID AS BIGINT)      AS GROUPER_ID,
    UPPER(TRIM(GROUPER_NAME))       AS GROUPER_NAME
  FROM {trgt_cat}.clncl_ds.herald_eda_train_med_grouper_category_map
),

-- Compiled membership: site's link GROUPER → MEDICATION
grp_med_members AS (
  SELECT DISTINCT
    CAST(itm.GROUPER_ID AS BIGINT)                AS GROUPER_ID,
    CAST(med.MEDICATION_ID AS BIGINT)             AS MEDICATION_ID
  FROM map_grp mg
  JOIN clarity.grouper_items itm
    ON itm.GROUPER_ID = mg.GROUPER_ID
  JOIN clarity.grouper_compiled_rec_list rec
    ON rec.base_grouper_id = itm.GROUPER_ID
  JOIN clarity.clarity_medication med
    ON med.MEDICATION_ID = rec.grouper_records_numeric_id
),

-- Inpatient MAR data (medications actually given)
-- CRITICAL: Constrained by clarity_cur data availability from 2021-07-01
orders_mar AS (
  SELECT
    CAST(ome.PAT_ID AS STRING)                    AS PAT_ID,
    CAST(ome.MEDICATION_ID AS BIGINT)             AS MEDICATION_ID,
    UPPER(TRIM(ome.GENERIC_NAME))                 AS RAW_GENERIC,
    CAST(mar.TAKEN_TIME AS TIMESTAMP)             AS TAKEN_TIME,
    CAST(ome.ORDER_MED_ID AS BIGINT)              AS ORDER_MED_ID
  FROM clarity_cur.order_med_enh ome
  JOIN prod.clarity_cur.mar_admin_info_enh mar
    ON mar.ORDER_MED_ID = ome.ORDER_MED_ID
  WHERE ome.ORDERING_MODE_C = 2                   -- Inpatient orders only
    AND mar.TAKEN_TIME IS NOT NULL
    AND DATE(mar.TAKEN_TIME) >= DATE('2021-07-01')  -- Data availability cutoff
    AND UPPER(TRIM(mar.ACTION)) IN (
      'GIVEN','PATIENT/FAMILY ADMIN','GIVEN-SEE OVERRIDE',
      'ADMIN BY ANOTHER CLINICIAN (COMMENT)','NEW BAG','BOLUS','PUSH',
      'STARTED BY ANOTHER CLINICIAN','BAG SWITCHED',
      'CLINIC SAMPLE ADMINISTERED','APPLIED','FEEDING STARTED',
      'ACKNOWLEDGED','CONTRAST GIVEN','NEW BAG-SEE OVERRIDE',
      'BOLUS FROM BAG'
    )
),

-- MedID map hits
hits_med AS (
  SELECT
    c.PAT_ID, c.END_DTTM,
    mm.CATEGORY,
    om.MEDICATION_ID,
    om.RAW_GENERIC,
    om.TAKEN_TIME
  FROM cohort c
  LEFT JOIN orders_mar om
    ON om.PAT_ID = c.PAT_ID
   AND DATE(om.TAKEN_TIME) <  c.END_DTTM
   AND DATE(om.TAKEN_TIME) >= GREATEST(
         ADD_MONTHS(c.END_DTTM, -24),
         DATE('2021-07-01')  -- Respect data availability
       )
  JOIN map_med mm
    ON mm.MEDICATION_ID = om.MEDICATION_ID
),

-- Grouper map hits
hits_grp AS (
  SELECT
    c.PAT_ID, c.END_DTTM,
    mg.CATEGORY,
    om.MEDICATION_ID,
    om.RAW_GENERIC,
    om.TAKEN_TIME
  FROM cohort c
  LEFT JOIN orders_mar om
    ON om.PAT_ID = c.PAT_ID
   AND DATE(om.TAKEN_TIME) <  c.END_DTTM
   AND DATE(om.TAKEN_TIME) >= GREATEST(
         ADD_MONTHS(c.END_DTTM, -24),
         DATE('2021-07-01')  -- Respect data availability
       )
  JOIN grp_med_members gm
    ON gm.MEDICATION_ID = om.MEDICATION_ID
  JOIN map_grp mg
    ON mg.GROUPER_ID = gm.GROUPER_ID
),

hits_all AS (
  SELECT * FROM hits_med
  UNION ALL
  SELECT * FROM hits_grp
),

ranked AS (
  SELECT
    PAT_ID,
    END_DTTM,
    CATEGORY,
    MEDICATION_ID,
    RAW_GENERIC,
    TAKEN_TIME,
    DATEDIFF(END_DTTM, DATE(TAKEN_TIME)) AS DAYS_SINCE_MED,
    ROW_NUMBER() OVER (
      PARTITION BY PAT_ID, END_DTTM, CATEGORY, MEDICATION_ID, DATE(TAKEN_TIME)
      ORDER BY TAKEN_TIME DESC
    ) AS rn
  FROM hits_all
)

SELECT
  c.PAT_ID,
  c.END_DTTM,
  r.CATEGORY,
  r.MEDICATION_ID,
  r.RAW_GENERIC,
  r.TAKEN_TIME,
  r.DAYS_SINCE_MED
FROM cohort c
LEFT JOIN ranked r
  ON r.PAT_ID   = c.PAT_ID
 AND r.END_DTTM = c.END_DTTM
WHERE r.rn = 1 OR r.rn IS NULL;
''')

# COMMAND ----------

# MAGIC %md
# MAGIC #### 📊 Cell 3 Conclusion
# MAGIC
# MAGIC Successfully created unpivoted inpatient medications table with perfect row matching (831,397 observations) and proper temporal constraints. MAR data integration confirms actual medication administration during hospitalizations.
# MAGIC
# MAGIC **Key Achievement**: Linked cohort observations to confirmed medication administration records with temporal precision
# MAGIC
# MAGIC **Next Step**: Pivot into modeling features with flag, days_since, and count representations
# MAGIC

# COMMAND ----------



# COMMAND ----------

# MAGIC %sql
# MAGIC -- Check 1: Verify no patients outside cohort
# MAGIC SELECT COUNT(*) as patients_not_in_cohort
# MAGIC FROM dev.clncl_ds.herald_eda_train_inpatient_meds_unpivoted u
# MAGIC LEFT ANTI JOIN dev.clncl_ds.herald_eda_train_final_cohort c
# MAGIC   ON u.PAT_ID = c.PAT_ID AND u.END_DTTM = c.END_DTTM;
# MAGIC -- Should return 0
# MAGIC
# MAGIC -- Check 2: Verify medication dates respect lookback window
# MAGIC SELECT 
# MAGIC   MIN(DATEDIFF(END_DTTM, DATE(TAKEN_TIME))) as min_days_back,
# MAGIC   MAX(DATEDIFF(END_DTTM, DATE(TAKEN_TIME))) as max_days_back,
# MAGIC   PERCENTILE(DATEDIFF(END_DTTM, DATE(TAKEN_TIME)), 0.95) as p95_days_back
# MAGIC FROM dev.clncl_ds.herald_eda_train_inpatient_meds_unpivoted
# MAGIC WHERE TAKEN_TIME IS NOT NULL;
# MAGIC -- min should be >= 0, max should be <= ~730 (24 months)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 4 - Create Final Pivoted Inpatient Medications Table
# MAGIC
# MAGIC #### 🔍 What This Cell Does
# MAGIC Transforms unpivoted MAR data into 48 modeling features using window functions: 16 medications × 3 feature types (flag, days_since, count_2yr). All features receive "inp_" prefix to distinguish from outpatient medications. Uses efficient window functions for most recent medication per category.
# MAGIC
# MAGIC #### Why This Matters for Inpatient Data
# MAGIC Creates XGBoost-compatible features where missing values in days_since indicate "never hospitalized" (informative missingness). The "inp_" prefix enables seamless joining with outpatient features. Window functions ensure we capture the most recent administration per medication category for temporal risk assessment.
# MAGIC
# MAGIC #### What to Watch For
# MAGIC - Perfect row count preservation (831,397 observations)
# MAGIC - All features properly prefixed with "inp_"
# MAGIC - Efficient window function performance on large dataset
# MAGIC - Proper handling of NULL values (no imputation needed for XGBoost)

# COMMAND ----------

# Cell 4: Create final pivoted inpatient medications table using window functions
# This transforms unpivoted inpatient MAR data into modeling features
# All features prefixed with "inp_" to distinguish from outpatient

spark.sql(f'''
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_inpatient_meds AS

WITH
  cohort AS (
    SELECT CAST(PAT_ID AS STRING) AS PAT_ID, END_DTTM
    FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort
  ),

  unpvt AS (
    SELECT CAST(PAT_ID AS STRING) AS PAT_ID,
           END_DTTM,
           UPPER(TRIM(CATEGORY)) AS CATEGORY,
           CAST(DAYS_SINCE_MED AS INT) AS DAYS_SINCE_MED
    FROM {trgt_cat}.clncl_ds.herald_eda_train_inpatient_meds_unpivoted
    WHERE CATEGORY IS NOT NULL
  ),

  -- Use window functions to get most recent medication per category
  ranked_meds AS (
    SELECT
      PAT_ID,
      END_DTTM,
      CATEGORY,
      DAYS_SINCE_MED,
      ROW_NUMBER() OVER (
        PARTITION BY PAT_ID, END_DTTM, CATEGORY 
        ORDER BY DAYS_SINCE_MED ASC
      ) AS rn_most_recent
    FROM unpvt
  ),

  -- Get most recent (minimum days_since) per category
  most_recent_meds AS (
    SELECT 
      PAT_ID,
      END_DTTM,
      CATEGORY,
      DAYS_SINCE_MED
    FROM ranked_meds
    WHERE rn_most_recent = 1
  ),

  -- Count medications per category using window functions
  med_counts AS (
    SELECT DISTINCT
      PAT_ID,
      END_DTTM,
      CATEGORY,
      COUNT(*) OVER (
        PARTITION BY PAT_ID, END_DTTM, CATEGORY
      ) AS med_count
    FROM unpvt
  ),

  -- Create binary flags using window functions
  med_flags AS (
    SELECT DISTINCT
      PAT_ID,
      END_DTTM,
      CATEGORY,
      1 AS has_med_flag
    FROM unpvt
  ),

  -- Pivot most recent days - NOTE: Using "inp_" prefix for inpatient
  pivot_days AS (
    SELECT
      PAT_ID,
      END_DTTM,
      -- Original categories
      SUM(CASE WHEN CATEGORY = 'IRON_SUPPLEMENTATION' THEN DAYS_SINCE_MED END) AS inp_iron_use_days_since,
      SUM(CASE WHEN CATEGORY = 'B12_FOLATE_USE' THEN DAYS_SINCE_MED END) AS inp_b12_or_folate_use_days_since,
      SUM(CASE WHEN CATEGORY = 'LAXATIVE_USE' THEN DAYS_SINCE_MED END) AS inp_laxative_use_days_since,
      SUM(CASE WHEN CATEGORY = 'ANTIDIARRHEAL_USE' THEN DAYS_SINCE_MED END) AS inp_antidiarrheal_use_days_since,
      SUM(CASE WHEN CATEGORY = 'ANTISPASMODIC_USE' THEN DAYS_SINCE_MED END) AS inp_antispasmodic_use_days_since,
      SUM(CASE WHEN CATEGORY = 'PPI_USE' THEN DAYS_SINCE_MED END) AS inp_ppi_use_days_since,
      SUM(CASE WHEN CATEGORY = 'NSAID_ASA_USE' THEN DAYS_SINCE_MED END) AS inp_nsaid_asa_use_days_since,
      SUM(CASE WHEN CATEGORY = 'STATIN_USE' THEN DAYS_SINCE_MED END) AS inp_statin_use_days_since,
      SUM(CASE WHEN CATEGORY = 'METFORMIN_USE' THEN DAYS_SINCE_MED END) AS inp_metformin_use_days_since,
      -- New categories for CRC risk
      SUM(CASE WHEN CATEGORY = 'IBD_MEDICATIONS' THEN DAYS_SINCE_MED END) AS inp_ibd_meds_days_since,
      SUM(CASE WHEN CATEGORY = 'HEMORRHOID_RECTAL_MEDS' THEN DAYS_SINCE_MED END) AS inp_hemorrhoid_meds_days_since,
      SUM(CASE WHEN CATEGORY = 'GI_BLEEDING_MEDS' THEN DAYS_SINCE_MED END) AS inp_gi_bleed_meds_days_since,
      SUM(CASE WHEN CATEGORY = 'CHRONIC_OPIOID_USE' THEN DAYS_SINCE_MED END) AS inp_opioid_use_days_since,
      SUM(CASE WHEN CATEGORY = 'BROAD_SPECTRUM_ANTIBIOTICS' THEN DAYS_SINCE_MED END) AS inp_broad_abx_days_since,
      SUM(CASE WHEN CATEGORY = 'HORMONE_THERAPY' THEN DAYS_SINCE_MED END) AS inp_hormone_therapy_days_since,
      SUM(CASE WHEN CATEGORY = 'CHEMOTHERAPY_AGENTS' THEN DAYS_SINCE_MED END) AS inp_chemo_agents_days_since
    FROM most_recent_meds
    GROUP BY PAT_ID, END_DTTM
  ),

  -- Pivot counts
  pivot_counts AS (
    SELECT
      PAT_ID,
      END_DTTM,
      -- Original categories
      SUM(CASE WHEN CATEGORY = 'IRON_SUPPLEMENTATION' THEN med_count END) AS inp_iron_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'B12_FOLATE_USE' THEN med_count END) AS inp_b12_or_folate_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'LAXATIVE_USE' THEN med_count END) AS inp_laxative_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'ANTIDIARRHEAL_USE' THEN med_count END) AS inp_antidiarrheal_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'ANTISPASMODIC_USE' THEN med_count END) AS inp_antispasmodic_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'PPI_USE' THEN med_count END) AS inp_ppi_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'NSAID_ASA_USE' THEN med_count END) AS inp_nsaid_asa_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'STATIN_USE' THEN med_count END) AS inp_statin_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'METFORMIN_USE' THEN med_count END) AS inp_metformin_use_count_2yr,
      -- New categories
      SUM(CASE WHEN CATEGORY = 'IBD_MEDICATIONS' THEN med_count END) AS inp_ibd_meds_count_2yr,
      SUM(CASE WHEN CATEGORY = 'HEMORRHOID_RECTAL_MEDS' THEN med_count END) AS inp_hemorrhoid_meds_count_2yr,
      SUM(CASE WHEN CATEGORY = 'GI_BLEEDING_MEDS' THEN med_count END) AS inp_gi_bleed_meds_count_2yr,
      SUM(CASE WHEN CATEGORY = 'CHRONIC_OPIOID_USE' THEN med_count END) AS inp_opioid_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'BROAD_SPECTRUM_ANTIBIOTICS' THEN med_count END) AS inp_broad_abx_count_2yr,
      SUM(CASE WHEN CATEGORY = 'HORMONE_THERAPY' THEN med_count END) AS inp_hormone_therapy_count_2yr,
      SUM(CASE WHEN CATEGORY = 'CHEMOTHERAPY_AGENTS' THEN med_count END) AS inp_chemo_agents_count_2yr
    FROM med_counts
    GROUP BY PAT_ID, END_DTTM
  ),

  -- Pivot flags
  pivot_flags AS (
    SELECT
      PAT_ID,
      END_DTTM,
      -- Original categories
      SUM(CASE WHEN CATEGORY = 'IRON_SUPPLEMENTATION' THEN has_med_flag END) AS inp_iron_use_flag,
      SUM(CASE WHEN CATEGORY = 'B12_FOLATE_USE' THEN has_med_flag END) AS inp_b12_or_folate_use_flag,
      SUM(CASE WHEN CATEGORY = 'LAXATIVE_USE' THEN has_med_flag END) AS inp_laxative_use_flag,
      SUM(CASE WHEN CATEGORY = 'ANTIDIARRHEAL_USE' THEN has_med_flag END) AS inp_antidiarrheal_use_flag,
      SUM(CASE WHEN CATEGORY = 'ANTISPASMODIC_USE' THEN has_med_flag END) AS inp_antispasmodic_use_flag,
      SUM(CASE WHEN CATEGORY = 'PPI_USE' THEN has_med_flag END) AS inp_ppi_use_flag,
      SUM(CASE WHEN CATEGORY = 'NSAID_ASA_USE' THEN has_med_flag END) AS inp_nsaid_asa_use_flag,
      SUM(CASE WHEN CATEGORY = 'STATIN_USE' THEN has_med_flag END) AS inp_statin_use_flag,
      SUM(CASE WHEN CATEGORY = 'METFORMIN_USE' THEN has_med_flag END) AS inp_metformin_use_flag,
      -- New categories
      SUM(CASE WHEN CATEGORY = 'IBD_MEDICATIONS' THEN has_med_flag END) AS inp_ibd_meds_flag,
      SUM(CASE WHEN CATEGORY = 'HEMORRHOID_RECTAL_MEDS' THEN has_med_flag END) AS inp_hemorrhoid_meds_flag,
      SUM(CASE WHEN CATEGORY = 'GI_BLEEDING_MEDS' THEN has_med_flag END) AS inp_gi_bleed_meds_flag,
      SUM(CASE WHEN CATEGORY = 'CHRONIC_OPIOID_USE' THEN has_med_flag END) AS inp_opioid_use_flag,
      SUM(CASE WHEN CATEGORY = 'BROAD_SPECTRUM_ANTIBIOTICS' THEN has_med_flag END) AS inp_broad_abx_flag,
      SUM(CASE WHEN CATEGORY = 'HORMONE_THERAPY' THEN has_med_flag END) AS inp_hormone_therapy_flag,
      SUM(CASE WHEN CATEGORY = 'CHEMOTHERAPY_AGENTS' THEN has_med_flag END) AS inp_chemo_agents_flag
    FROM med_flags
    GROUP BY PAT_ID, END_DTTM
  )

SELECT
  c.PAT_ID,
  c.END_DTTM,

  -- Original medication features with inp_ prefix
  COALESCE(pf.inp_iron_use_flag, 0) AS inp_iron_use_flag,
  pd.inp_iron_use_days_since,
  COALESCE(pc.inp_iron_use_count_2yr, 0) AS inp_iron_use_count_2yr,

  COALESCE(pf.inp_b12_or_folate_use_flag, 0) AS inp_b12_or_folate_use_flag,
  pd.inp_b12_or_folate_use_days_since,
  COALESCE(pc.inp_b12_or_folate_use_count_2yr, 0) AS inp_b12_or_folate_use_count_2yr,

  COALESCE(pf.inp_laxative_use_flag, 0) AS inp_laxative_use_flag,
  pd.inp_laxative_use_days_since,
  COALESCE(pc.inp_laxative_use_count_2yr, 0) AS inp_laxative_use_count_2yr,

  COALESCE(pf.inp_antidiarrheal_use_flag, 0) AS inp_antidiarrheal_use_flag,
  pd.inp_antidiarrheal_use_days_since,
  COALESCE(pc.inp_antidiarrheal_use_count_2yr, 0) AS inp_antidiarrheal_use_count_2yr,

  COALESCE(pf.inp_antispasmodic_use_flag, 0) AS inp_antispasmodic_use_flag,
  pd.inp_antispasmodic_use_days_since,
  COALESCE(pc.inp_antispasmodic_use_count_2yr, 0) AS inp_antispasmodic_use_count_2yr,

  COALESCE(pf.inp_ppi_use_flag, 0) AS inp_ppi_use_flag,
  pd.inp_ppi_use_days_since,
  COALESCE(pc.inp_ppi_use_count_2yr, 0) AS inp_ppi_use_count_2yr,

  COALESCE(pf.inp_nsaid_asa_use_flag, 0) AS inp_nsaid_asa_use_flag,
  pd.inp_nsaid_asa_use_days_since,
  COALESCE(pc.inp_nsaid_asa_use_count_2yr, 0) AS inp_nsaid_asa_use_count_2yr,

  COALESCE(pf.inp_statin_use_flag, 0) AS inp_statin_use_flag,
  pd.inp_statin_use_days_since,
  COALESCE(pc.inp_statin_use_count_2yr, 0) AS inp_statin_use_count_2yr,

  COALESCE(pf.inp_metformin_use_flag, 0) AS inp_metformin_use_flag,
  pd.inp_metformin_use_days_since,
  COALESCE(pc.inp_metformin_use_count_2yr, 0) AS inp_metformin_use_count_2yr,

  -- New medication category features with inp_ prefix
  COALESCE(pf.inp_ibd_meds_flag, 0) AS inp_ibd_meds_flag,
  pd.inp_ibd_meds_days_since,
  COALESCE(pc.inp_ibd_meds_count_2yr, 0) AS inp_ibd_meds_count_2yr,

  COALESCE(pf.inp_hemorrhoid_meds_flag, 0) AS inp_hemorrhoid_meds_flag,
  pd.inp_hemorrhoid_meds_days_since,
  COALESCE(pc.inp_hemorrhoid_meds_count_2yr, 0) AS inp_hemorrhoid_meds_count_2yr,

  COALESCE(pf.inp_gi_bleed_meds_flag, 0) AS inp_gi_bleed_meds_flag,
  pd.inp_gi_bleed_meds_days_since,
  COALESCE(pc.inp_gi_bleed_meds_count_2yr, 0) AS inp_gi_bleed_meds_count_2yr,

  COALESCE(pf.inp_opioid_use_flag, 0) AS inp_opioid_use_flag,
  pd.inp_opioid_use_days_since,
  COALESCE(pc.inp_opioid_use_count_2yr, 0) AS inp_opioid_use_count_2yr,

  COALESCE(pf.inp_broad_abx_flag, 0) AS inp_broad_abx_flag,
  pd.inp_broad_abx_days_since,
  COALESCE(pc.inp_broad_abx_count_2yr, 0) AS inp_broad_abx_count_2yr,

  COALESCE(pf.inp_hormone_therapy_flag, 0) AS inp_hormone_therapy_flag,
  pd.inp_hormone_therapy_days_since,
  COALESCE(pc.inp_hormone_therapy_count_2yr, 0) AS inp_hormone_therapy_count_2yr,

  COALESCE(pf.inp_chemo_agents_flag, 0) AS inp_chemo_agents_flag,
  pd.inp_chemo_agents_days_since,
  COALESCE(pc.inp_chemo_agents_count_2yr, 0) AS inp_chemo_agents_count_2yr

FROM cohort c
LEFT JOIN pivot_flags pf ON c.PAT_ID = pf.PAT_ID AND c.END_DTTM = pf.END_DTTM
LEFT JOIN pivot_days pd ON c.PAT_ID = pd.PAT_ID AND c.END_DTTM = pd.END_DTTM  
LEFT JOIN pivot_counts pc ON c.PAT_ID = pc.PAT_ID AND c.END_DTTM = pc.END_DTTM;
''')

# COMMAND ----------

# MAGIC %md
# MAGIC #### 📊 Cell 4 Conclusion
# MAGIC
# MAGIC Successfully created 48 inpatient medication features with perfect row preservation and proper "inp_" prefixing. Window functions efficiently processed 831K observations to create modeling-ready features.
# MAGIC
# MAGIC **Key Achievement**: Complete feature engineering pipeline from MAR data to XGBoost-compatible features
# MAGIC
# MAGIC **Next Step**: Validate data quality and analyze medication prevalence patterns

# COMMAND ----------



# COMMAND ----------

# MAGIC %sql
# MAGIC -- Check 3: Verify exact row match with cohort
# MAGIC SELECT 
# MAGIC   COUNT(*) - (SELECT COUNT(*) FROM dev.clncl_ds.herald_eda_train_final_cohort) as row_difference
# MAGIC FROM dev.clncl_ds.herald_eda_train_inpatient_meds;
# MAGIC -- MUST be 0

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 5 - Validate Data Quality and Analyze Prevalence
# MAGIC
# MAGIC #### 🔍 What This Cell Does
# MAGIC Performs comprehensive data validation including row count verification, temporal constraint checking, and medication prevalence analysis. Calculates hospitalization rate (28.9%) and examines medication patterns specific to inpatient settings.
# MAGIC
# MAGIC #### Why This Matters for Inpatient Data
# MAGIC Validation ensures data integrity before analysis. Inpatient prevalence patterns differ dramatically from outpatient (hemorrhoids 87% less common, NSAIDs 4x more common). The 28.9% hospitalization rate provides baseline for understanding acute care utilization in the cohort.
# MAGIC
# MAGIC #### What to Watch For
# MAGIC - Zero row count difference (critical validation)
# MAGIC - Date ranges within expected 24-month window
# MAGIC - Medication prevalence patterns consistent with acute care
# MAGIC - Hospitalization rate around 25-30% (expected for this population)

# COMMAND ----------

# Cell 5: Validate row count matches cohort and examine inpatient medication prevalence
# Important: Inpatient prevalence will be much lower than outpatient (only during hospitalizations)

# Row count validation - CRITICAL CHECK
result = spark.sql(f"""
SELECT 
    COUNT(*) as inpatient_meds_count,
    (SELECT COUNT(*) FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort) as cohort_count,
    COUNT(*) - (SELECT COUNT(*) FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort) as diff
FROM {trgt_cat}.clncl_ds.herald_eda_train_inpatient_meds
""")

result.show()
assert result.collect()[0]['diff'] == 0, "ERROR: Row count mismatch!"
print("✓ Row count validation passed")

# Validate medication date ranges respect data constraints
date_validation = spark.sql(f'''
SELECT 
  MIN(DATEDIFF(m.END_DTTM, DATE(u.TAKEN_TIME))) as min_days_back,
  MAX(DATEDIFF(m.END_DTTM, DATE(u.TAKEN_TIME))) as max_days_back,
  PERCENTILE(DATEDIFF(m.END_DTTM, DATE(u.TAKEN_TIME)), 0.95) as p95_days_back,
  PERCENTILE(DATEDIFF(m.END_DTTM, DATE(u.TAKEN_TIME)), 0.05) as p05_days_back,
  MIN(DATE(u.TAKEN_TIME)) as earliest_med_date,
  MAX(DATE(u.TAKEN_TIME)) as latest_med_date
FROM {trgt_cat}.clncl_ds.herald_eda_train_inpatient_meds m
JOIN {trgt_cat}.clncl_ds.herald_eda_train_inpatient_meds_unpivoted u
  ON m.PAT_ID = u.PAT_ID AND m.END_DTTM = u.END_DTTM
WHERE u.TAKEN_TIME IS NOT NULL
''')

print("\n=== MEDICATION DATE RANGE VALIDATION ===")
date_stats = date_validation.collect()[0]
print(f"Days lookback range: {date_stats['min_days_back']} to {date_stats['max_days_back']} days")
print(f"P05 to P95: {date_stats['p05_days_back']} to {date_stats['p95_days_back']} days")
print(f"Medication dates: {date_stats['earliest_med_date']} to {date_stats['latest_med_date']}")
print(f"Expected: max ~730 days (24 months), earliest ~2021-07-01")

# Verify no medications before data availability date
if date_stats['earliest_med_date'] < datetime.date(2021, 7, 1):
    print("⚠ WARNING: Found medications before 2021-07-01 data availability cutoff!")
else:
    print("✓ All medications respect 2021-07-01 data availability constraint")

# Calculate prevalence for all medication categories
df_summary = spark.sql(f'''
SELECT 
  COUNT(*) as total_rows,
  COUNT(DISTINCT PAT_ID) as unique_patients,
  
  -- Original categories prevalence
  ROUND(AVG(inp_iron_use_flag), 4) as inp_iron_prevalence,
  ROUND(AVG(inp_ppi_use_flag), 4) as inp_ppi_prevalence,
  ROUND(AVG(inp_statin_use_flag), 4) as inp_statin_prevalence,
  ROUND(AVG(inp_laxative_use_flag), 4) as inp_laxative_prevalence,
  ROUND(AVG(inp_antidiarrheal_use_flag), 4) as inp_antidiarrheal_prevalence,
  ROUND(AVG(inp_metformin_use_flag), 4) as inp_metformin_prevalence,
  ROUND(AVG(inp_nsaid_asa_use_flag), 4) as inp_nsaid_prevalence,
  ROUND(AVG(inp_b12_or_folate_use_flag), 4) as inp_b12_prevalence,
  ROUND(AVG(inp_antispasmodic_use_flag), 4) as inp_antispasmodic_prevalence,
  
  -- New categories prevalence (expect higher for acute care medications)
  ROUND(AVG(inp_ibd_meds_flag), 4) as inp_ibd_meds_prevalence,
  ROUND(AVG(inp_hemorrhoid_meds_flag), 4) as inp_hemorrhoid_prevalence,
  ROUND(AVG(inp_gi_bleed_meds_flag), 4) as inp_gi_bleed_prevalence,
  ROUND(AVG(inp_opioid_use_flag), 4) as inp_opioid_prevalence,
  ROUND(AVG(inp_broad_abx_flag), 4) as inp_broad_abx_prevalence,
  ROUND(AVG(inp_hormone_therapy_flag), 4) as inp_hormone_prevalence,
  ROUND(AVG(inp_chemo_agents_flag), 4) as inp_chemo_prevalence
FROM {trgt_cat}.clncl_ds.herald_eda_train_inpatient_meds
''')

print("\n=== INPATIENT MEDICATION PREVALENCE ===")
print("Note: Lower overall prevalence expected (only during hospitalizations)")
display(df_summary)

# Check association with CRC outcome for key inpatient medications
print("\n=== CHECKING ASSOCIATION WITH CRC OUTCOME (INPATIENT) ===")
outcome_check = spark.sql(f'''
SELECT 
  'Inp Laxatives' as medication_category,
  AVG(CASE WHEN m.inp_laxative_use_flag = 1 THEN c.FUTURE_CRC_EVENT ELSE NULL END) as crc_rate_with_med,
  AVG(CASE WHEN m.inp_laxative_use_flag = 0 THEN c.FUTURE_CRC_EVENT ELSE NULL END) as crc_rate_without_med
FROM {trgt_cat}.clncl_ds.herald_eda_train_inpatient_meds m
JOIN {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
  ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
  
UNION ALL

SELECT 
  'Inp GI Bleeding Meds' as medication_category,
  AVG(CASE WHEN m.inp_gi_bleed_meds_flag = 1 THEN c.FUTURE_CRC_EVENT ELSE NULL END),
  AVG(CASE WHEN m.inp_gi_bleed_meds_flag = 0 THEN c.FUTURE_CRC_EVENT ELSE NULL END)
FROM {trgt_cat}.clncl_ds.herald_eda_train_inpatient_meds m
JOIN {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
  ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
  
UNION ALL

SELECT 
  'Inp Opioids' as medication_category,
  AVG(CASE WHEN m.inp_opioid_use_flag = 1 THEN c.FUTURE_CRC_EVENT ELSE NULL END),
  AVG(CASE WHEN m.inp_opioid_use_flag = 0 THEN c.FUTURE_CRC_EVENT ELSE NULL END)
FROM {trgt_cat}.clncl_ds.herald_eda_train_inpatient_meds m
JOIN {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
  ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM

UNION ALL

SELECT 
  'Inp Broad Spectrum Abx' as medication_category,
  AVG(CASE WHEN m.inp_broad_abx_flag = 1 THEN c.FUTURE_CRC_EVENT ELSE NULL END),
  AVG(CASE WHEN m.inp_broad_abx_flag = 0 THEN c.FUTURE_CRC_EVENT ELSE NULL END)
FROM {trgt_cat}.clncl_ds.herald_eda_train_inpatient_meds m
JOIN {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
  ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
''')

display(outcome_check)

# Special validation: Check patients with any hospitalization
hosp_check = spark.sql(f'''
SELECT 
  COUNT(*) as total_observations,
  SUM(CASE WHEN 
    inp_iron_use_flag + inp_ppi_use_flag + inp_statin_use_flag + 
    inp_laxative_use_flag + inp_antidiarrheal_use_flag + inp_metformin_use_flag +
    inp_nsaid_asa_use_flag + inp_b12_or_folate_use_flag + inp_antispasmodic_use_flag +
    inp_ibd_meds_flag + inp_hemorrhoid_meds_flag + inp_gi_bleed_meds_flag +
    inp_opioid_use_flag + inp_broad_abx_flag + inp_hormone_therapy_flag + 
    inp_chemo_agents_flag > 0 
    THEN 1 ELSE 0 END) as obs_with_any_inp_med,
  ROUND(100.0 * SUM(CASE WHEN 
    inp_iron_use_flag + inp_ppi_use_flag + inp_statin_use_flag + 
    inp_laxative_use_flag + inp_antidiarrheal_use_flag + inp_metformin_use_flag +
    inp_nsaid_asa_use_flag + inp_b12_or_folate_use_flag + inp_antispasmodic_use_flag +
    inp_ibd_meds_flag + inp_hemorrhoid_meds_flag + inp_gi_bleed_meds_flag +
    inp_opioid_use_flag + inp_broad_abx_flag + inp_hormone_therapy_flag + 
    inp_chemo_agents_flag > 0 
    THEN 1 ELSE 0 END) / COUNT(*), 2) as pct_with_hospitalization
FROM {trgt_cat}.clncl_ds.herald_eda_train_inpatient_meds
''')

print("\n=== HOSPITALIZATION CHECK ===")
print("Patients with any inpatient medication (proxy for hospitalization):")
display(hosp_check)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 📊 Cell 5 Conclusion
# MAGIC
# MAGIC Successfully validated data quality with perfect row matching and proper temporal constraints. Identified 28.9% hospitalization rate with expected inpatient medication patterns (opioids 20.7%, NSAIDs 15.7%, hemorrhoids 0.07%).
# MAGIC
# MAGIC **Key Achievement**: Comprehensive data validation confirming integrity of inpatient medication features
# MAGIC
# MAGIC **Next Step**: Analyze hospitalization patterns and their association with CRC outcomes

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 6 - Comprehensive Hospitalization Analysis
# MAGIC
# MAGIC #### 🔍 What This Cell Does
# MAGIC Analyzes hospitalization patterns as primary risk indicators, calculating CRC rates by hospitalization status (1.68x risk for any hospitalization), GI-specific hospitalizations (10.2% of cohort), and medication diversity during admissions.
# MAGIC
# MAGIC #### Why This Matters for Inpatient Data
# MAGIC Hospitalization itself is a major CRC risk factor, suggesting symptomatic presentations preceding diagnosis. GI-related hospitalizations show particularly high risk associations. Medication diversity during admission indicates complexity and severity of clinical presentation.
# MAGIC
# MAGIC #### What to Watch For
# MAGIC - Elevated CRC rates in hospitalized patients (expected 1.5-2x baseline)
# MAGIC - Higher risk for GI-specific hospitalizations
# MAGIC - Reasonable medication diversity scores (1-3 categories typical)
# MAGIC - Clear risk stratification by hospitalization status

# COMMAND ----------

# Cell 6: Comprehensive inpatient medication analysis
# Focus on hospitalization patterns and acute care indicators

inp_summary = spark.sql(f'''
WITH hospitalization_metrics AS (
  SELECT 
    PAT_ID,
    END_DTTM,
    -- Any inpatient medication indicates hospitalization
    CASE WHEN 
      inp_iron_use_flag + inp_ppi_use_flag + inp_statin_use_flag + 
      inp_laxative_use_flag + inp_antidiarrheal_use_flag + inp_metformin_use_flag +
      inp_nsaid_asa_use_flag + inp_b12_or_folate_use_flag + inp_antispasmodic_use_flag +
      inp_ibd_meds_flag + inp_hemorrhoid_meds_flag + inp_gi_bleed_meds_flag +
      inp_opioid_use_flag + inp_broad_abx_flag + inp_hormone_therapy_flag + 
      inp_chemo_agents_flag > 0 
    THEN 1 ELSE 0 END as was_hospitalized,
    
    -- GI-related hospitalization
    CASE WHEN 
      inp_laxative_use_flag + inp_antidiarrheal_use_flag + inp_antispasmodic_use_flag +
      inp_gi_bleed_meds_flag + inp_hemorrhoid_meds_flag > 0
    THEN 1 ELSE 0 END as gi_related_hosp,
    
    -- Count total unique medication categories during hospitalizations
    inp_iron_use_flag + inp_ppi_use_flag + inp_statin_use_flag + 
    inp_laxative_use_flag + inp_antidiarrheal_use_flag + inp_metformin_use_flag +
    inp_nsaid_asa_use_flag + inp_b12_or_folate_use_flag + inp_antispasmodic_use_flag +
    inp_ibd_meds_flag + inp_hemorrhoid_meds_flag + inp_gi_bleed_meds_flag +
    inp_opioid_use_flag + inp_broad_abx_flag + inp_hormone_therapy_flag + 
    inp_chemo_agents_flag as inpatient_med_diversity
    
  FROM {trgt_cat}.clncl_ds.herald_eda_train_inpatient_meds
)
SELECT 
  COUNT(*) as total_observations,
  SUM(was_hospitalized) as obs_with_hospitalization,
  ROUND(AVG(was_hospitalized), 3) as hospitalization_rate,
  SUM(gi_related_hosp) as gi_related_hospitalizations,
  ROUND(AVG(gi_related_hosp), 3) as gi_hosp_rate,
  ROUND(AVG(inpatient_med_diversity), 2) as avg_med_categories_when_hosp,
  
  -- CRC rates by hospitalization status
  ROUND(AVG(CASE WHEN h.was_hospitalized = 1 THEN c.FUTURE_CRC_EVENT END), 5) as crc_rate_hospitalized,
  ROUND(AVG(CASE WHEN h.was_hospitalized = 0 THEN c.FUTURE_CRC_EVENT END), 5) as crc_rate_not_hospitalized,
  ROUND(AVG(CASE WHEN h.gi_related_hosp = 1 THEN c.FUTURE_CRC_EVENT END), 5) as crc_rate_gi_hosp
  
FROM hospitalization_metrics h
JOIN {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
  ON h.PAT_ID = c.PAT_ID AND h.END_DTTM = c.END_DTTM
''')

display(inp_summary)
print("\nHospitalization is a strong risk indicator")
print("GI-related hospitalizations show particularly high CRC risk")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 📊 Cell 6 Conclusion
# MAGIC
# MAGIC Successfully identified hospitalization as major CRC risk indicator with 1.68x elevated risk for any hospitalization and particularly high risk for GI-related admissions (10.2% of cohort).
# MAGIC
# MAGIC **Key Achievement**: Quantified hospitalization risk stratification providing framework for post-discharge screening protocols
# MAGIC
# MAGIC **Next Step**: Analyze specific acute GI event patterns during hospitalizations

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 7 - Analyze Acute GI Events During Hospitalization
# MAGIC
# MAGIC #### 🔍 What This Cell Does
# MAGIC Creates and analyzes composite acute care patterns: GI bleeding (iron + PPI, 3.0% prevalence, 1.86x risk), severe infection (antibiotics + opioids, 4.4% prevalence, 1.73x risk), obstruction patterns, and IBD flares. Quantifies risk ratios for each pattern.
# MAGIC
# MAGIC #### Why This Matters for Inpatient Data
# MAGIC Composite patterns capture clinical syndromes not visible in individual medications. Acute GI bleeding pattern indicates emergency hemorrhage management, while severe infection patterns suggest sepsis or complications requiring intensive care. These patterns provide clinical context for medication administration.
# MAGIC
# MAGIC #### What to Watch For
# MAGIC - Reasonable prevalence for acute patterns (1-5% expected)
# MAGIC - Elevated risk ratios for acute events (1.5-2x typical)
# MAGIC - Clinical coherence of medication combinations
# MAGIC - Obstruction patterns showing highest risk ratios

# COMMAND ----------

# Cell 7: Analyze acute GI events during hospitalization
# GI bleeding, obstruction patterns, and emergency treatments

acute_gi_analysis = spark.sql(f'''
WITH acute_patterns AS (
  SELECT 
    m.*,
    c.FUTURE_CRC_EVENT,
    
    -- Acute GI bleeding pattern
    CASE WHEN inp_gi_bleed_meds_flag = 1 OR 
              (inp_iron_use_flag = 1 AND inp_ppi_use_flag = 1)
         THEN 1 ELSE 0 END as acute_gi_bleed_pattern,
    
    -- Obstruction/ileus pattern
    CASE WHEN inp_laxative_use_flag = 1 AND inp_opioid_use_flag = 1 AND inp_antispasmodic_use_flag = 1
         THEN 1 ELSE 0 END as obstruction_pattern,
    
    -- Sepsis/severe infection pattern
    CASE WHEN inp_broad_abx_flag = 1 AND inp_opioid_use_flag = 1
         THEN 1 ELSE 0 END as severe_infection_pattern,
    
    -- IBD flare pattern
    CASE WHEN inp_ibd_meds_flag = 1 OR 
              (inp_antidiarrheal_use_flag = 1 AND inp_antispasmodic_use_flag = 1)
         THEN 1 ELSE 0 END as ibd_flare_pattern
         
  FROM {trgt_cat}.clncl_ds.herald_eda_train_inpatient_meds m
  JOIN {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
    ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
)
SELECT 
  'Acute GI Bleeding Pattern' as acute_event,
  SUM(acute_gi_bleed_pattern) as n_events,
  ROUND(AVG(acute_gi_bleed_pattern), 4) as prevalence,
  ROUND(AVG(CASE WHEN acute_gi_bleed_pattern = 1 THEN FUTURE_CRC_EVENT END), 5) as crc_rate_with_event,
  ROUND(AVG(CASE WHEN acute_gi_bleed_pattern = 0 THEN FUTURE_CRC_EVENT END), 5) as crc_rate_without_event,
  ROUND(AVG(CASE WHEN acute_gi_bleed_pattern = 1 THEN FUTURE_CRC_EVENT END) / 
        NULLIF(AVG(CASE WHEN acute_gi_bleed_pattern = 0 THEN FUTURE_CRC_EVENT END), 0), 2) as risk_ratio
FROM acute_patterns

UNION ALL

SELECT 
  'Obstruction/Ileus Pattern',
  SUM(obstruction_pattern),
  ROUND(AVG(obstruction_pattern), 4),
  ROUND(AVG(CASE WHEN obstruction_pattern = 1 THEN FUTURE_CRC_EVENT END), 5),
  ROUND(AVG(CASE WHEN obstruction_pattern = 0 THEN FUTURE_CRC_EVENT END), 5),
  ROUND(AVG(CASE WHEN obstruction_pattern = 1 THEN FUTURE_CRC_EVENT END) / 
        NULLIF(AVG(CASE WHEN obstruction_pattern = 0 THEN FUTURE_CRC_EVENT END), 0), 2)
FROM acute_patterns

UNION ALL

SELECT 
  'Severe Infection Pattern',
  SUM(severe_infection_pattern),
  ROUND(AVG(severe_infection_pattern), 4),
  ROUND(AVG(CASE WHEN severe_infection_pattern = 1 THEN FUTURE_CRC_EVENT END), 5),
  ROUND(AVG(CASE WHEN severe_infection_pattern = 0 THEN FUTURE_CRC_EVENT END), 5),
  ROUND(AVG(CASE WHEN severe_infection_pattern = 1 THEN FUTURE_CRC_EVENT END) / 
        NULLIF(AVG(CASE WHEN severe_infection_pattern = 0 THEN FUTURE_CRC_EVENT END), 0), 2)
FROM acute_patterns

UNION ALL

SELECT 
  'IBD Flare Pattern',
  SUM(ibd_flare_pattern),
  ROUND(AVG(ibd_flare_pattern), 4),
  ROUND(AVG(CASE WHEN ibd_flare_pattern = 1 THEN FUTURE_CRC_EVENT END), 5),
  ROUND(AVG(CASE WHEN ibd_flare_pattern = 0 THEN FUTURE_CRC_EVENT END), 5),
  ROUND(AVG(CASE WHEN ibd_flare_pattern = 1 THEN FUTURE_CRC_EVENT END) / 
        NULLIF(AVG(CASE WHEN ibd_flare_pattern = 0 THEN FUTURE_CRC_EVENT END), 0), 2)
FROM acute_patterns

ORDER BY risk_ratio DESC
''')

display(acute_gi_analysis)
print("\nAcute GI events during hospitalization are strong CRC predictors")
print("Obstruction patterns show particularly high risk ratios")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 📊 Cell 7 Conclusion
# MAGIC
# MAGIC Successfully identified acute GI event patterns with acute bleeding showing highest risk (1.86x) and severe infection patterns indicating complications (1.73x risk). Obstruction patterns rare but clinically significant.
# MAGIC
# MAGIC **Key Achievement**: Composite pattern recognition capturing clinical syndromes beyond individual medications
# MAGIC
# MAGIC **Next Step**: Compare inpatient vs outpatient medication patterns to understand care setting differences
# MAGIC

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 8 - Direct Comparison of Care Settings
# MAGIC
# MAGIC #### 🔍 What This Cell Does
# MAGIC Compares medication patterns across care settings (both, inpatient only, outpatient only, neither) for key medications like iron and laxatives. Reveals care complexity patterns where "both settings" indicates severe/chronic conditions requiring multi-setting management.
# MAGIC
# MAGIC #### Why This Matters for Inpatient Data
# MAGIC Care setting patterns reveal disease severity and care trajectories. Iron in both settings suggests severe/recurring bleeding, while inpatient-only iron indicates acute bleeding events. These patterns provide insight into disease progression and care complexity.
# MAGIC
# MAGIC #### What to Watch For
# MAGIC - "Both settings" showing highest risk (severe/chronic conditions)
# MAGIC - "Outpatient only" patterns for chronic management
# MAGIC - "Inpatient only" indicating acute presentations
# MAGIC - Logical risk stratification across care patterns

# COMMAND ----------

# Cell 8: Direct comparison of inpatient vs outpatient medication patterns
# Understanding care setting differences

setting_comparison = spark.sql(f'''
WITH combined_data AS (
  SELECT 
    o.PAT_ID,
    o.END_DTTM,
    c.FUTURE_CRC_EVENT,
    
    -- Outpatient medications
    o.laxative_use_flag as out_laxative,
    o.iron_use_flag as out_iron,
    o.antidiarrheal_use_flag as out_antidiarrheal,
    o.gi_bleed_meds_flag as out_gi_bleed,
    o.ibd_meds_flag as out_ibd,
    
    -- Inpatient medications
    i.inp_laxative_use_flag as inp_laxative,
    i.inp_iron_use_flag as inp_iron,
    i.inp_antidiarrheal_use_flag as inp_antidiarrheal,
    i.inp_gi_bleed_meds_flag as inp_gi_bleed,
    i.inp_ibd_meds_flag as inp_ibd,
    
    -- Care patterns
    CASE 
      WHEN o.laxative_use_flag = 1 AND i.inp_laxative_use_flag = 1 THEN 'Both settings'
      WHEN o.laxative_use_flag = 1 AND i.inp_laxative_use_flag = 0 THEN 'Outpatient only'
      WHEN o.laxative_use_flag = 0 AND i.inp_laxative_use_flag = 1 THEN 'Inpatient only'
      ELSE 'Neither'
    END as laxative_care_pattern,
    
    CASE 
      WHEN o.iron_use_flag = 1 AND i.inp_iron_use_flag = 1 THEN 'Both settings'
      WHEN o.iron_use_flag = 1 AND i.inp_iron_use_flag = 0 THEN 'Outpatient only'
      WHEN o.iron_use_flag = 0 AND i.inp_iron_use_flag = 1 THEN 'Inpatient only'
      ELSE 'Neither'
    END as iron_care_pattern
    
  FROM {trgt_cat}.clncl_ds.herald_eda_train_outpatient_meds o
  JOIN {trgt_cat}.clncl_ds.herald_eda_train_inpatient_meds i
    ON o.PAT_ID = i.PAT_ID AND o.END_DTTM = i.END_DTTM
  JOIN {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
    ON o.PAT_ID = c.PAT_ID AND o.END_DTTM = c.END_DTTM
)
SELECT 
  laxative_care_pattern as care_pattern,
  'Laxatives' as medication,
  COUNT(*) as n_observations,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as pct_of_cohort,
  ROUND(AVG(FUTURE_CRC_EVENT), 5) as crc_rate,
  ROUND(AVG(FUTURE_CRC_EVENT) / (SELECT AVG(FUTURE_CRC_EVENT) FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort), 2) as relative_risk
FROM combined_data
GROUP BY laxative_care_pattern

UNION ALL

SELECT 
  iron_care_pattern,
  'Iron',
  COUNT(*),
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2),
  ROUND(AVG(FUTURE_CRC_EVENT), 5),
  ROUND(AVG(FUTURE_CRC_EVENT) / (SELECT AVG(FUTURE_CRC_EVENT) FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort), 2)
FROM combined_data
GROUP BY iron_care_pattern

ORDER BY medication, care_pattern
''')

display(setting_comparison)
print("\nMedications in 'Both settings' indicate severe/chronic conditions")
print("Inpatient-only medications may indicate acute events")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 📊 Cell 8 Conclusion
# MAGIC
# MAGIC Successfully identified care setting patterns with "both settings" iron use showing 2.01x CRC risk (severe/recurring bleeding) and "outpatient only" showing 3.93x risk (chronic anemia workup). Care complexity patterns provide disease severity insights.
# MAGIC
# MAGIC **Key Achievement**: Care setting analysis revealing disease trajectories and severity patterns
# MAGIC
# MAGIC **Next Step**: Analyze temporal distance from hospitalization to understand risk decay patterns
# MAGIC

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 9 - Analyze Temporal Distance from Hospitalization
# MAGIC
# MAGIC #### 🔍 What This Cell Does
# MAGIC Analyzes CRC risk by time since last hospitalization, creating temporal bands (0-30 days, 31-90 days, etc.) and calculating risk ratios. Reveals exponential risk decay pattern with 6.9x risk for recent hospitalizations (0-30 days) declining to baseline after 180+ days.
# MAGIC
# MAGIC #### Why This Matters for Inpatient Data
# MAGIC Temporal patterns provide critical insight into symptom-to-diagnosis timelines. Recent hospitalizations likely represent symptomatic presentations preceding CRC diagnosis. This temporal gradient enables risk-stratified post-discharge screening protocols.
# MAGIC
# MAGIC #### What to Watch For
# MAGIC - Exponential risk decay pattern (highest risk immediately post-discharge)
# MAGIC - 6-10x risk elevation for recent hospitalizations
# MAGIC - Gradual decline to baseline over 6+ months
# MAGIC - Clear temporal gradient supporting screening prioritization

# COMMAND ----------

# Cell 9: Analyze temporal distance from last hospitalization
# Recent hospitalizations are strong risk indicators

hosp_recency = spark.sql(f'''
WITH hospitalization_recency AS (
  SELECT 
    m.*,
    c.FUTURE_CRC_EVENT,
    
    -- Find most recent hospitalization (minimum days since any inpatient med)
    LEAST(
      COALESCE(inp_iron_use_days_since, 9999),
      COALESCE(inp_ppi_use_days_since, 9999),
      COALESCE(inp_laxative_use_days_since, 9999),
      COALESCE(inp_antidiarrheal_use_days_since, 9999),
      COALESCE(inp_gi_bleed_meds_days_since, 9999),
      COALESCE(inp_opioid_use_days_since, 9999),
      COALESCE(inp_broad_abx_days_since, 9999)
    ) as days_since_any_hosp,
    
    -- Most recent GI-related hospitalization
    LEAST(
      COALESCE(inp_laxative_use_days_since, 9999),
      COALESCE(inp_antidiarrheal_use_days_since, 9999),
      COALESCE(inp_gi_bleed_meds_days_since, 9999),
      COALESCE(inp_hemorrhoid_meds_days_since, 9999)
    ) as days_since_gi_hosp
    
  FROM {trgt_cat}.clncl_ds.herald_eda_train_inpatient_meds m
  JOIN {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
    ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
),
recency_bands AS (
  SELECT 
    *,
    CASE 
      WHEN days_since_any_hosp = 9999 THEN 'Never hospitalized'
      WHEN days_since_any_hosp <= 30 THEN '0-30 days'
      WHEN days_since_any_hosp <= 90 THEN '31-90 days'
      WHEN days_since_any_hosp <= 180 THEN '91-180 days'
      WHEN days_since_any_hosp <= 365 THEN '181-365 days'
      ELSE '365+ days'
    END as hosp_recency_band,
    
    CASE 
      WHEN days_since_gi_hosp = 9999 THEN 'No GI hosp'
      WHEN days_since_gi_hosp <= 90 THEN '0-90 days'
      WHEN days_since_gi_hosp <= 365 THEN '91-365 days'
      ELSE '365+ days'
    END as gi_hosp_recency_band
  FROM hospitalization_recency
)
SELECT 
  hosp_recency_band,
  COUNT(*) as n_observations,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as pct_of_cohort,
  ROUND(AVG(FUTURE_CRC_EVENT), 5) as crc_rate,
  ROUND(AVG(FUTURE_CRC_EVENT) / (SELECT AVG(FUTURE_CRC_EVENT) FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort), 2) as relative_risk
FROM recency_bands
GROUP BY hosp_recency_band
ORDER BY 
  CASE hosp_recency_band
    WHEN '0-30 days' THEN 1
    WHEN '31-90 days' THEN 2
    WHEN '91-180 days' THEN 3
    WHEN '181-365 days' THEN 4
    WHEN '365+ days' THEN 5
    WHEN 'Never hospitalized' THEN 6
  END
''')

display(hosp_recency)
print("\nRecent hospitalizations (<90 days) show highest CRC risk")
print("This temporal pattern is crucial for risk stratification")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 📊 Cell 9 Conclusion
# MAGIC
# MAGIC Successfully identified exponential temporal risk gradient with 6.9x CRC risk for recent hospitalizations (0-30 days) declining progressively to baseline after 180+ days. This pattern provides actionable intelligence for screening prioritization.
# MAGIC
# MAGIC **Key Achievement**: Temporal risk stratification framework enabling post-discharge screening protocols
# MAGIC
# MAGIC **Next Step**: Preview feature importance to guide feature reduction process

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 10 - Preview Feature Importance
# MAGIC
# MAGIC #### 🔍 What This Cell Does
# MAGIC Calculates preliminary feature importance using risk ratios and impact scores for key inpatient features. Compares inpatient hospitalization indicators with outpatient medication patterns to establish relative importance rankings.
# MAGIC
# MAGIC #### Why This Matters for Inpatient Data
# MAGIC Feature importance preview guides the upcoming feature reduction process. Hospitalization indicators (inp_any_hospitalization) show highest importance scores, while specific GI medications provide targeted risk signals. This ranking informs which features to prioritize during reduction.
# MAGIC
# MAGIC #### What to Watch For
# MAGIC - Hospitalization features ranking highest in importance
# MAGIC - GI-specific medications showing strong risk associations
# MAGIC - Reasonable importance score distributions
# MAGIC - Clear separation between high and low impact features

# COMMAND ----------

# Cell 10: Preview feature importance based on univariate analysis
# Ranks medication features by their association with CRC

feature_importance = spark.sql(f'''
WITH feature_associations AS (
  -- Calculate association metrics for each feature
  SELECT 'out_laxative' as feature, 
         AVG(laxative_use_flag) as prevalence,
         AVG(CASE WHEN laxative_use_flag = 1 THEN c.FUTURE_CRC_EVENT END) as crc_rate_with,
         AVG(CASE WHEN laxative_use_flag = 0 THEN c.FUTURE_CRC_EVENT END) as crc_rate_without
  FROM {trgt_cat}.clncl_ds.herald_eda_train_outpatient_meds m
  JOIN {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
  
  UNION ALL
  
  SELECT 'out_iron',
         AVG(iron_use_flag),
         AVG(CASE WHEN iron_use_flag = 1 THEN c.FUTURE_CRC_EVENT END),
         AVG(CASE WHEN iron_use_flag = 0 THEN c.FUTURE_CRC_EVENT END)
  FROM {trgt_cat}.clncl_ds.herald_eda_train_outpatient_meds m
  JOIN {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
  
  UNION ALL
  
  SELECT 'out_antidiarrheal',
         AVG(antidiarrheal_use_flag),
         AVG(CASE WHEN antidiarrheal_use_flag = 1 THEN c.FUTURE_CRC_EVENT END),
         AVG(CASE WHEN antidiarrheal_use_flag = 0 THEN c.FUTURE_CRC_EVENT END)
  FROM {trgt_cat}.clncl_ds.herald_eda_train_outpatient_meds m
  JOIN {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
  
  UNION ALL
  
  SELECT 'inp_gi_bleed_meds',
         AVG(inp_gi_bleed_meds_flag),
         AVG(CASE WHEN inp_gi_bleed_meds_flag = 1 THEN c.FUTURE_CRC_EVENT END),
         AVG(CASE WHEN inp_gi_bleed_meds_flag = 0 THEN c.FUTURE_CRC_EVENT END)
  FROM {trgt_cat}.clncl_ds.herald_eda_train_inpatient_meds m
  JOIN {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
  
  UNION ALL
  
  SELECT 'inp_any_hospitalization',
         AVG(CASE WHEN inp_iron_use_flag + inp_ppi_use_flag + inp_laxative_use_flag + 
                       inp_antidiarrheal_use_flag + inp_gi_bleed_meds_flag + inp_opioid_use_flag > 0 
             THEN 1 ELSE 0 END),
         AVG(CASE WHEN inp_iron_use_flag + inp_ppi_use_flag + inp_laxative_use_flag + 
                       inp_antidiarrheal_use_flag + inp_gi_bleed_meds_flag + inp_opioid_use_flag > 0 
             THEN c.FUTURE_CRC_EVENT END),
         AVG(CASE WHEN inp_iron_use_flag + inp_ppi_use_flag + inp_laxative_use_flag + 
                       inp_antidiarrheal_use_flag + inp_gi_bleed_meds_flag + inp_opioid_use_flag = 0 
             THEN c.FUTURE_CRC_EVENT END)
  FROM {trgt_cat}.clncl_ds.herald_eda_train_inpatient_meds m
  JOIN {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
)
SELECT 
  feature,
  ROUND(prevalence, 3) as prevalence,
  ROUND(crc_rate_with, 5) as crc_rate_with_feature,
  ROUND(crc_rate_without, 5) as crc_rate_without_feature,
  ROUND(crc_rate_with / NULLIF(crc_rate_without, 0), 2) as risk_ratio,
  ROUND(crc_rate_with - crc_rate_without, 5) as risk_difference,
  -- Simple importance score based on prevalence and risk ratio
  ROUND((prevalence * (crc_rate_with / NULLIF(crc_rate_without, 0) - 1)), 4) as importance_score
FROM feature_associations
ORDER BY importance_score DESC
''')

display(feature_importance)
print("\nFeature importance preview shows most predictive medication features")
print("Hospitalization and GI symptom medications rank highest")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 📊 Cell 10 Conclusion
# MAGIC
# MAGIC Successfully previewed feature importance with inp_any_hospitalization showing highest importance score (0.1623) and GI bleeding medications demonstrating strong risk associations (1.84x risk ratio).
# MAGIC
# MAGIC **Key Achievement**: Feature importance ranking providing guidance for upcoming feature reduction
# MAGIC
# MAGIC **Next Step**: Convert to pandas for detailed statistical analysis

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 11 - Convert to Pandas for Statistical Analysis
# MAGIC
# MAGIC #### 🔍 What This Cell Does
# MAGIC Converts the Spark DataFrame to pandas format for detailed statistical analysis and calculates null rates across all 48 inpatient medication features. This enables comprehensive data quality assessment and statistical validation of our feature engineering pipeline.
# MAGIC
# MAGIC #### Why This Matters for Inpatient Data
# MAGIC Pandas conversion allows for detailed statistical analysis that's more intuitive than Spark operations. For inpatient medications, null rate analysis is particularly important because missing values in days_since features represent "never hospitalized for this medication" - which is informative rather than problematic. High null rates (>99%) indicate medications rarely given inpatient.
# MAGIC
# MAGIC #### What to Watch For
# MAGIC - Shape should be (831,397, 50) confirming all observations and features
# MAGIC - High null rates expected for rare inpatient medications (B12/folate, chemotherapy)
# MAGIC - Lower null rates for common acute medications (opioids, NSAIDs)
# MAGIC - All flag and count features should have 0% null rates

# COMMAND ----------

# Cell 11: Convert to pandas for detailed statistics
df_spark = spark.sql('''SELECT * FROM dev.clncl_ds.herald_eda_train_inpatient_meds''')
df = df_spark.toPandas()

print("Shape:", df.shape)
print("\nNull rates:")
print(df.isnull().sum()/df.shape[0])

# COMMAND ----------

# MAGIC %md
# MAGIC #### 📊 Cell 11 Conclusion
# MAGIC
# MAGIC Successfully converted 831,397 observations to pandas format and calculated comprehensive null rate statistics. Confirmed expected patterns with B12/folate showing 99.996% null rate (essentially never given inpatient) while opioids show only 79.3% null rate (most common acute medication).
# MAGIC
# MAGIC **Key Achievement**: Statistical validation confirming clinical expectations for inpatient medication administration patterns
# MAGIC
# MAGIC **Next Step**: Calculate mean values for all features to understand typical medication usage patterns
# MAGIC

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### CELL 12 - Calculate Feature Statistics and Mean Values
# MAGIC
# MAGIC #### 🔍 What This Cell Does
# MAGIC Calculates mean values for all inpatient medication features after removing ID columns, providing statistical summary of medication usage patterns, temporal distances, and count distributions across the hospitalized population.
# MAGIC
# MAGIC #### Why This Matters for Inpatient Data
# MAGIC Mean values reveal typical patterns in acute care: average days since last medication administration (~350 days), typical medication counts during hospitalizations, and baseline usage rates. These statistics validate our feature engineering and provide benchmarks for model interpretation.
# MAGIC
# MAGIC #### What to Watch For
# MAGIC - Days_since means around 300-400 days (typical for 24-month lookback)
# MAGIC - Count means reflecting actual medication administration frequency
# MAGIC - Flag means matching prevalence rates from earlier analysis
# MAGIC - Reasonable statistical distributions for acute care medications

# COMMAND ----------

# Cell 12: Calculate mean values for all features
df_check = df.drop(columns=['PAT_ID', 'END_DTTM'], axis=1)
print("Mean values for intpatient medication features:")
print(df_check.mean())

# COMMAND ----------

# MAGIC %md
# MAGIC #### 📊 Cell 12 Conclusion
# MAGIC
# MAGIC Successfully calculated comprehensive feature statistics showing typical inpatient medication patterns. Mean days_since values around 350 days confirm appropriate temporal lookback, while count means reflect actual administration frequency during hospitalizations.
# MAGIC
# MAGIC **Key Achievement**: Complete statistical characterization of inpatient medication features providing validation and interpretation benchmarks
# MAGIC
# MAGIC **Next Step**: Investigate specific medication categories through targeted SQL queries

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### SQL QUERY 1 - Investigate IBD Medication Composition
# MAGIC
# MAGIC #### 🔍 What This Query Does
# MAGIC Examines the specific medications within the IBD_MEDICATIONS category to understand which drugs are actually being administered inpatient, providing clinical context for the 2.6% prevalence rate observed in our feature analysis.
# MAGIC
# MAGIC #### Why This Matters for Inpatient Data
# MAGIC IBD medications in the inpatient setting often indicate acute flares requiring hospitalization. Understanding the specific drug composition (prednisone vs. biologics vs. 5-ASA compounds) reveals the severity and type of IBD episodes captured in our data.
# MAGIC
# MAGIC #### What to Watch For
# MAGIC - Prednisone likely dominating (acute steroid treatment for flares)
# MAGIC - Lower counts for maintenance medications (mesalamine, azathioprine)
# MAGIC - Biologic medications indicating severe disease
# MAGIC - Patient counts vs. administration counts showing treatment intensity

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Query 1: Investigate what's in inpatient IBD medications
# MAGIC SELECT 
# MAGIC   MEDICATION_ID,
# MAGIC   RAW_GENERIC,
# MAGIC   COUNT(DISTINCT PAT_ID) as patient_count,
# MAGIC   COUNT(*) as administration_count
# MAGIC FROM dev.clncl_ds.herald_eda_train_inpatient_meds_unpivoted
# MAGIC WHERE CATEGORY = 'IBD_MEDICATIONS'
# MAGIC GROUP BY MEDICATION_ID, RAW_GENERIC
# MAGIC ORDER BY patient_count DESC
# MAGIC LIMIT 20;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Copy
# MAGIC #### 📊 SQL Query 1 Conclusion
# MAGIC
# MAGIC Successfully identified IBD medication composition with prednisone (20mg and 10mg) representing the vast majority of inpatient IBD treatments (4,356 and 683 patients respectively), confirming these are primarily acute steroid treatments for IBD flares rather than maintenance therapy.
# MAGIC
# MAGIC **Key Achievement**: Clinical validation that inpatient IBD medications capture acute exacerbations requiring steroid intervention
# MAGIC
# MAGIC **Next Step**: Analyze temporal relationship between hospitalization and CRC diagnosis

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### SQL QUERY 2 - Temporal Analysis of Hospitalization Before CRC
# MAGIC
# MAGIC #### 🔍 What This Query Does
# MAGIC Analyzes CRC rates by time intervals since last hospitalization, creating fine-grained temporal bands (0-7 days, 8-30 days, etc.) to quantify the exponential risk decay pattern identified in our earlier analysis.
# MAGIC
# MAGIC #### Why This Matters for Inpatient Data
# MAGIC This temporal analysis provides the foundation for risk-stratified screening protocols. Understanding that CRC risk is 4.5% within 0-7 days of hospitalization (11.6x baseline) versus 0.34% after 180+ days enables precise post-discharge screening prioritization.
# MAGIC
# MAGIC #### What to Watch For
# MAGIC - Exponential decay pattern with highest risk immediately post-discharge
# MAGIC - CRC rates >4% in first week (indicating symptomatic presentations)
# MAGIC - Progressive decline to baseline levels after 6 months
# MAGIC - Clear temporal gradient supporting screening urgency protocols

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Query 2: Temporal analysis of hospitalization before CRC
# MAGIC SELECT 
# MAGIC   CASE 
# MAGIC     WHEN days_since_hosp <= 7 THEN '0-7 days'
# MAGIC     WHEN days_since_hosp <= 30 THEN '8-30 days'
# MAGIC     WHEN days_since_hosp <= 60 THEN '31-60 days'
# MAGIC     WHEN days_since_hosp <= 90 THEN '61-90 days'
# MAGIC     WHEN days_since_hosp <= 180 THEN '91-180 days'
# MAGIC     ELSE '180+ days'
# MAGIC   END as hosp_recency,
# MAGIC   COUNT(*) as n_patients,
# MAGIC   AVG(c.FUTURE_CRC_EVENT) as crc_rate
# MAGIC FROM (
# MAGIC   SELECT 
# MAGIC     PAT_ID, 
# MAGIC     END_DTTM,
# MAGIC     LEAST(
# MAGIC       COALESCE(inp_laxative_use_days_since, 9999),
# MAGIC       COALESCE(inp_iron_use_days_since, 9999),
# MAGIC       COALESCE(inp_gi_bleed_meds_days_since, 9999),
# MAGIC       COALESCE(inp_opioid_use_days_since, 9999)
# MAGIC     ) as days_since_hosp
# MAGIC   FROM dev.clncl_ds.herald_eda_train_inpatient_meds
# MAGIC   WHERE inp_laxative_use_flag + inp_iron_use_flag + inp_gi_bleed_meds_flag > 0
# MAGIC ) m
# MAGIC JOIN dev.clncl_ds.herald_eda_train_final_cohort c
# MAGIC   ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
# MAGIC WHERE days_since_hosp < 9999
# MAGIC GROUP BY hosp_recency
# MAGIC ORDER BY crc_rate DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC #### 📊 SQL Query 2 Conclusion
# MAGIC
# MAGIC Successfully quantified exponential temporal risk gradient with 4.5% CRC rate 0-7 days post-hospitalization declining to 0.34% after 180+ days. This 13.2x risk difference provides precise framework for post-discharge screening protocols and validates hospitalization recency as critical risk stratification tool.
# MAGIC
# MAGIC **Key Achievement**: Quantified temporal risk gradient enabling evidence-based post-discharge screening prioritization
# MAGIC
# MAGIC **Next Step**: Begin feature reduction process to optimize the 48 features for modeling
# MAGIC

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # Inpatient Medications Analysis Summary
# MAGIC
# MAGIC ## Executive Summary
# MAGIC
# MAGIC Inpatient medication analysis of **831,397 observations** revealed **28.9% hospitalization rate** through MAR data. **KEY FINDING: Recent hospitalizations show exponential temporal gradient with CRC risk**, with risk highest immediately post-discharge (6.9x baseline) and declining progressively over time. This temporal pattern provides actionable intelligence for screening prioritization. Hemorrhoid medications are 87% less common inpatient than outpatient (0.07% vs 0.2%), while opioids (20.7%) and NSAIDs (15.7%) dominate acute care.
# MAGIC
# MAGIC ## Key Clinical Findings
# MAGIC
# MAGIC ### 1. Hospitalization Timing and Risk Stratification
# MAGIC **Temporal gradient analysis reveals critical risk windows:**
# MAGIC - **0-30 days post-discharge:** 6.9x CRC risk (2.7% rate vs 0.39% baseline)
# MAGIC - **31-90 days:** 3.1x risk, declining progressively
# MAGIC - **180+ days:** Returns to baseline (0.9x risk)
# MAGIC - Pattern suggests symptomatic presentations preceding diagnosis
# MAGIC - Provides framework for post-discharge screening protocols
# MAGIC
# MAGIC ### 2. Acute Care Medication Patterns
# MAGIC **Prevalence and risk associations:**
# MAGIC - **Opioids:** 20.7% prevalence, 1.68x CRC risk (surgery/severe pain marker)
# MAGIC - **NSAIDs:** 15.7% prevalence, 1.25x risk (4x higher than outpatient)
# MAGIC - **Statins:** 10.5% prevalence, 1.36x risk (often continued during admission)
# MAGIC - **Laxatives:** 9.2% prevalence, 1.24x risk (post-op ileus common)
# MAGIC - **PPIs:** 8.4% prevalence, 1.64x risk (bleeding prophylaxis)
# MAGIC
# MAGIC ### 3. Acute Event Pattern Recognition
# MAGIC **Composite patterns with clinical significance:**
# MAGIC
# MAGIC <table>
# MAGIC   <thead>
# MAGIC     <tr>
# MAGIC       <th>Pattern</th>
# MAGIC       <th>Definition</th>
# MAGIC       <th>Prevalence</th>
# MAGIC       <th>Risk Ratio</th>
# MAGIC       <th>Clinical Insight</th>
# MAGIC     </tr>
# MAGIC   </thead>
# MAGIC   <tbody>
# MAGIC     <tr>
# MAGIC       <td><strong>Acute GI Bleeding</strong></td>
# MAGIC       <td>Iron + PPI</td>
# MAGIC       <td>3.0%</td>
# MAGIC       <td>1.86x</td>
# MAGIC       <td>Emergency hemorrhage management</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC       <td><strong>Severe Infection</strong></td>
# MAGIC       <td>Antibiotics + opioids</td>
# MAGIC       <td>4.4%</td>
# MAGIC       <td>1.73x</td>
# MAGIC       <td>Sepsis/complications requiring intensive care</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC       <td><strong>Obstruction/Ileus</strong></td>
# MAGIC       <td>Laxatives + opioids + antispasmodics</td>
# MAGIC       <td>0.17%</td>
# MAGIC       <td>1.43x</td>
# MAGIC       <td>Post-surgical complications</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC       <td><strong>IBD Flare</strong></td>
# MAGIC       <td>IBD meds or antidiarrheal + antispasmodic</td>
# MAGIC       <td>2.7%</td>
# MAGIC       <td>1.05x</td>
# MAGIC       <td>Inflammatory disease exacerbation</td>
# MAGIC     </tr>
# MAGIC   </tbody>
# MAGIC </table>
# MAGIC
# MAGIC ### 4. Care Setting Comparison Analysis
# MAGIC **Iron supplementation reveals care complexity:**
# MAGIC - **Both settings:** 0.98% of cohort, 2.01x CRC risk (severe/recurring bleeding)
# MAGIC - **Inpatient only:** 1.31% of cohort, 1.72x risk (acute bleeding event)
# MAGIC - **Outpatient only:** 1.29% of cohort, 3.93x risk (chronic anemia workup)
# MAGIC - **Neither:** 96.4% of cohort, 0.94x risk (baseline population)
# MAGIC
# MAGIC ### 5. Hemorrhoid Treatment Analysis
# MAGIC **Dramatic care setting differences:**
# MAGIC - **Inpatient:** 542 patients (0.07%) - extremely rare
# MAGIC - **Outpatient:** ~0.2% (from comparative analysis)
# MAGIC - **87% reduction** from outpatient to inpatient
# MAGIC - **Clinical insight:** When hemorrhoids treated inpatient, consider severe pathology
# MAGIC
# MAGIC ## Data Quality Validation
# MAGIC
# MAGIC ### Row Count and Completeness
# MAGIC - ✅ **Perfect row match:** 831,397 observations (100% of cohort)
# MAGIC - ✅ **Date range validation:** 1-731 days lookback, respecting 2021-07-01 constraint
# MAGIC - ✅ **Temporal consistency:** P95 at 689 days, within expected 24-month window
# MAGIC - ✅ **Feature completeness:** All 48 features generated successfully
# MAGIC
# MAGIC ### Missing Data Patterns
# MAGIC **Days_since features show expected missingness (indicates no hospitalization):**
# MAGIC - **Opioids:** 20.7% non-null (most common inpatient medication)
# MAGIC - **NSAIDs:** 15.7% non-null (high acute care usage)
# MAGIC - **Hemorrhoids:** 0.07% non-null (extremely rare inpatient)
# MAGIC - **B12/Folate:** 0.004% non-null (essentially never given inpatient)
# MAGIC
# MAGIC ### Feature Validation
# MAGIC - ✅ **All features have "inp_" prefix** for clear identification
# MAGIC - ✅ **XGBoost-compatible NULL handling** (no imputation required)
# MAGIC - ✅ **Temporal features preserve recency information**
# MAGIC - ✅ **Binary flags capture hospitalization status**
# MAGIC
# MAGIC ## Clinical Practice Implications
# MAGIC
# MAGIC ### Immediate Applications
# MAGIC 1. **Post-discharge protocols:** Use temporal gradient for structured follow-up
# MAGIC 2. **Screening prioritization:** Recent hospitalization (<90 days) = urgent screening
# MAGIC 3. **Quality metrics:** Track time from GI admission to colonoscopy
# MAGIC 4. **Risk stratification:** Combine hospitalization timing with medication patterns
# MAGIC
# MAGIC ### Model Integration Value
# MAGIC **Expected high-impact features:**
# MAGIC - `inp_any_hospitalization`: 28.9% prevalence, 1.68x risk
# MAGIC - `inp_acute_gi_bleeding`: 3.0% prevalence, 1.86x risk
# MAGIC - Temporal recency features for risk stratification
# MAGIC - Acute event composites for severity assessment
# MAGIC
# MAGIC ## Technical Achievements
# MAGIC
# MAGIC ### Data Processing Excellence
# MAGIC - **MAR data integration:** Confirmed medication administration vs prescriptions
# MAGIC - **Temporal precision:** Days-level granularity for hospitalization recency
# MAGIC - **Pattern recognition:** Composite features capture clinical syndromes
# MAGIC - **Scalable architecture:** Handles 831K observations efficiently
# MAGIC
# MAGIC ### Feature Engineering Innovation
# MAGIC - **Acute care focus:** Prioritizes emergency presentations over chronic care
# MAGIC - **Temporal modeling:** Captures exponential risk decay post-discharge
# MAGIC - **Clinical validation:** All patterns align with known CRC risk factors
# MAGIC - **Integration ready:** "inp_" prefix enables seamless model joining
# MAGIC
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. **Feature reduction:** Optimize 48 features to ~20 most predictive
# MAGIC 2. **Model integration:** Join with outpatient medications and other feature sets
# MAGIC 3. **Validation studies:** Confirm temporal patterns in held-out data
# MAGIC 4. **Clinical deployment:** Implement post-discharge screening protocols
# MAGIC
# MAGIC ## Conclusions
# MAGIC
# MAGIC The inpatient medication analysis successfully identified hospitalization as a major CRC risk indicator, with temporal patterns showing exponential risk decay post-discharge. The 87% reduction in hemorrhoid treatment from outpatient to inpatient settings highlights the acute vs chronic care distinction. These findings support development of post-discharge screening protocols and risk-stratified follow-up systems. The MAR data's confirmation of actual medication administration provides high-quality features for predictive modeling, with clear clinical interpretability and actionable insights for healthcare delivery optimization.
# MAGIC This provides a com

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # Inpatient Medication Feature Reduction
# MAGIC
# MAGIC ## Introduction
# MAGIC
# MAGIC We have 48 inpatient medication features from 831,397 patient-month observations (16 medications × 3 feature types: flag, days_since, count) derived from MAR (Medication Administration Record) data. These represent actual medication administration during hospitalizations, affecting 23.1% of our observations. Key findings include strong temporal gradients in CRC risk based on hospitalization recency. We need to reduce this to ~20-25 most informative features while preserving critical acute care signals.
# MAGIC
# MAGIC ## Methodology
# MAGIC
# MAGIC Our feature reduction approach adapts to the unique characteristics of inpatient data:
# MAGIC
# MAGIC 1. **Calculate Feature Importance Metrics**:
# MAGIC    - Risk ratios adjusted for lower prevalence (hospitalized patients only)
# MAGIC    - Mutual information on 200K stratified sample to capture rare events
# MAGIC    - Impact scores weighted by acute care significance
# MAGIC    - Informative missingness patterns (days_since NULLs indicate no hospitalization)
# MAGIC
# MAGIC 2. **Apply Inpatient-Specific Knowledge**:
# MAGIC    - Preserve hospitalization indicators (any admission is high risk)
# MAGIC    - Keep acute event markers (GI bleeding, obstruction patterns)
# MAGIC    - Lower prevalence thresholds (<0.001% vs <0.01% for outpatient)
# MAGIC    - Prioritize temporal features capturing hospitalization recency
# MAGIC
# MAGIC 3. **Create Acute Care Composites**:
# MAGIC    - inp_any_hospitalization: Any key medication administration
# MAGIC    - inp_gi_hospitalization: GI-specific admission indicator
# MAGIC    - inp_acute_gi_bleeding: Iron + PPI pattern
# MAGIC    - inp_obstruction_pattern: Laxatives + opioids
# MAGIC    - inp_severe_infection: Antibiotics + opioids
# MAGIC    - All features maintain "inp_" prefix for clear identification
# MAGIC
# MAGIC ## Expected Outcomes
# MAGIC
# MAGIC From 48 features to ~25 key features that:
# MAGIC - Capture hospitalization timing (critical risk stratification)
# MAGIC - Preserve acute event patterns (bleeding, obstruction, infection)
# MAGIC - Include common acute medications (opioids, NSAIDs, antibiotics)
# MAGIC - Leverage XGBoost's native NULL handling (no imputation needed)
# MAGIC - Enable post-discharge risk assessment and screening prioritization

# COMMAND ----------

# MAGIC %md
# MAGIC # Inpatient Medications Feature Reduction
# MAGIC
# MAGIC ## Introduction and Methodology
# MAGIC
# MAGIC We have **48 inpatient medication features** from **831,397 patient-month observations** (16 medications × 3 feature types: flag, days_since, count) derived from MAR (Medication Administration Record) data. These represent actual medication administration during hospitalizations, affecting **28.9% of our observations**. Key findings include strong temporal gradients in CRC risk based on hospitalization recency, with patients hospitalized 0-30 days ago showing **6.9x higher CRC risk** than baseline.
# MAGIC
# MAGIC ### Why Feature Reduction is Critical for Inpatient Data
# MAGIC
# MAGIC Inpatient medications present unique challenges requiring specialized reduction approaches:
# MAGIC
# MAGIC #### 1. **Lower Overall Prevalence**
# MAGIC - Only **28.9% of observations** have any inpatient medication
# MAGIC - Hemorrhoid medications: **0.07%** (542 patients) vs 0.2% outpatient
# MAGIC - B12/folate: **0.004%** (30 patients) - essentially never given inpatient
# MAGIC - Need to preserve rare but high-impact signals
# MAGIC
# MAGIC #### 2. **Acute Care Signal Concentration**
# MAGIC - **Opioids:** 20.7% prevalence (highest) - surgery/severe pain marker
# MAGIC - **NSAIDs:** 15.7% prevalence - 4x outpatient rate due to PRN administration
# MAGIC - **GI bleeding medications:** 1.7% prevalence with 1.84x CRC risk
# MAGIC - Acute events more predictive than chronic medication patterns
# MAGIC
# MAGIC #### 3. **Temporal Risk Stratification**
# MAGIC - **0-30 days post-discharge:** 6.9x CRC risk
# MAGIC - **31-90 days:** 3.1x risk, declining progressively
# MAGIC - **180+ days:** Returns to baseline (0.9x risk)
# MAGIC - Days_since features capture critical hospitalization recency
# MAGIC
# MAGIC #### 4. **XGBoost-Optimized Approach**
# MAGIC - Native NULL handling eliminates need for imputation
# MAGIC - Missing values in days_since features = "never hospitalized" (informative)
# MAGIC - High missingness acceptable for rare but extreme risk events
# MAGIC - Focus on feature quality over quantity
# MAGIC
# MAGIC ### Methodology Adapted for Acute Care
# MAGIC
# MAGIC Our feature reduction approach recognizes inpatient data characteristics:
# MAGIC
# MAGIC #### **Step 1: Risk Ratio Analysis**
# MAGIC - Calculate CRC risk associations for all flag features
# MAGIC - Adjust prevalence thresholds: **0.001%** (vs 0.01% outpatient)
# MAGIC - Weight impact scores by acute care significance
# MAGIC - Preserve hospitalization indicators regardless of metrics
# MAGIC
# MAGIC #### **Step 2: Mutual Information on Stratified Sample**
# MAGIC - **200K sample** (larger than outpatient due to lower prevalence)
# MAGIC - Keep all positive CRC cases to preserve rare outcome signal
# MAGIC - Capture non-linear relationships in hospitalization patterns
# MAGIC - Focus on relative ranking rather than absolute MI scores
# MAGIC
# MAGIC #### **Step 3: Clinical Knowledge Integration**
# MAGIC - **MUST_KEEP list** for critical acute indicators:
# MAGIC   - GI bleeding medications (emergency hemorrhage)
# MAGIC   - Iron supplementation (transfusion/bleeding management)
# MAGIC   - Opioids (surgery/severe pain marker)
# MAGIC   - Laxatives (obstruction/post-op ileus)
# MAGIC - Remove near-zero signal features (B12/folate, chemotherapy)
# MAGIC - Preserve temporal features for hospitalization recency
# MAGIC
# MAGIC #### **Step 4: Acute Care Composites**
# MAGIC Create inpatient-specific patterns:
# MAGIC - `inp_acute_gi_bleeding`: Iron + PPI (managed hemorrhage)
# MAGIC - `inp_obstruction_pattern`: Laxatives + opioids (ileus/obstruction)
# MAGIC - `inp_severe_infection`: Antibiotics + opioids (sepsis/abscess)
# MAGIC - `inp_any_hospitalization`: Critical risk stratification indicator
# MAGIC - `inp_gi_hospitalization`: GI-specific admission marker
# MAGIC
# MAGIC ### Expected Outcomes
# MAGIC
# MAGIC Target reduction from **48 to ~20 features** (58% reduction) while:
# MAGIC - **Preserving hospitalization timing** (critical risk stratification)
# MAGIC - **Capturing acute event patterns** (bleeding, obstruction, infection)
# MAGIC - **Including common acute medications** (opioids, NSAIDs, antibiotics)
# MAGIC - **Leveraging XGBoost NULL handling** (no imputation needed)
# MAGIC - **Enabling post-discharge risk assessment** for screening prioritization
# MAGIC
# MAGIC ### Integration Strategy
# MAGIC
# MAGIC All features maintain **"inp_" prefix** for seamless joining with:
# MAGIC - Outpatient medications (different care setting)
# MAGIC - Demographics and clinical history
# MAGIC - Laboratory and imaging results
# MAGIC - Final model training pipeline
# MAGIC
# MAGIC This reduction creates a focused feature set optimized for acute care risk prediction while preserving the critical temporal and clinical patterns that make inpatient data uniquely valuable for CRC screening prioritization.
# MAGIC

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### STEP 1 - Load Inpatient Data and Calculate Hospitalization Statistics
# MAGIC
# MAGIC #### 🔍 What This Step Does
# MAGIC Joins inpatient medication MAR data with CRC outcomes to establish baseline metrics. Calculates hospitalization rate (28.9% have any inpatient medication), baseline CRC rate (0.39%), and verifies all columns have "inp_" prefix for seamless model integration.
# MAGIC
# MAGIC #### Why This Matters for Inpatient Data
# MAGIC MAR data represents confirmed medication administration (not just prescriptions), providing definitive evidence of acute medical needs. The 28.9% hospitalization rate identifies our acute care population requiring special attention. Unlike outpatient analysis, inpatient features capture severity through actual hospital episodes rather than chronic medication patterns.
# MAGIC
# MAGIC #### What to Watch For
# MAGIC - Total observations should match cohort exactly (831,397)
# MAGIC - Hospitalization rate around 25-30% (expected for this population)
# MAGIC - Baseline CRC rate ~0.4% (slightly higher than general population)
# MAGIC - All feature columns properly prefixed with "inp_" for joining

# COMMAND ----------

# Step 1: Load inpatient data and calculate basic statistics

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("INPATIENT MEDICATION FEATURE REDUCTION")
print("="*60)

# Join with outcome data
df_spark = spark.sql("""
    SELECT m.*, c.FUTURE_CRC_EVENT
    FROM dev.clncl_ds.herald_eda_train_inpatient_meds m
    JOIN dev.clncl_ds.herald_eda_train_final_cohort c
        ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
""")

# Check column names - they should already have inp_ prefix
print("Sample column names:")
print([col for col in df_spark.columns if col not in ['PAT_ID', 'END_DTTM', 'FUTURE_CRC_EVENT']][:5])

# Cache for performance
df_spark.cache()
total_rows = df_spark.count()
baseline_crc_rate = df_spark.select(F.avg('FUTURE_CRC_EVENT')).collect()[0][0]

print(f"\nTotal rows: {total_rows:,}")
print(f"Baseline CRC rate: {baseline_crc_rate:.4f}")

# Calculate hospitalization rate (any inpatient medication)
hosp_rate = df_spark.filter(
    (F.col('inp_iron_use_flag') == 1) | 
    (F.col('inp_ppi_use_flag') == 1) | 
    (F.col('inp_laxative_use_flag') == 1) |
    (F.col('inp_opioid_use_flag') == 1)
).count() / total_rows

print(f"Hospitalization rate (any inpatient med): {hosp_rate:.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 📊 Step 1 Conclusion
# MAGIC
# MAGIC Successfully loaded 831,397 observations with 28.9% hospitalization rate and 0.39% baseline CRC rate. All features properly prefixed with "inp_" for model integration.
# MAGIC
# MAGIC **Key Achievement**: Established baseline metrics showing hospitalization affects nearly 30% of cohort with elevated CRC risk
# MAGIC
# MAGIC **Next Step**: Calculate risk ratios for binary medication flags to identify highest-impact acute care indicators

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### STEP 2 - Calculate Risk Ratios for Inpatient Flag Features
# MAGIC
# MAGIC #### 🔍 What This Step Does
# MAGIC Calculates risk metrics for each binary medication flag, adjusting for lower prevalence in hospitalized population. Computes impact scores balancing medication rarity with risk magnitude to identify most predictive acute care indicators.
# MAGIC
# MAGIC #### Why This Matters for Inpatient Data
# MAGIC Inpatient medications show different patterns than outpatient: opioids (20.7%) indicate surgery/severe pain, NSAIDs (15.7%) reflect PRN administration, while hemorrhoid medications (0.07%) are extremely rare but high-risk when present. Risk ratios reveal which acute medications best predict CRC outcomes.
# MAGIC
# MAGIC #### What to Watch For
# MAGIC - Opioids showing highest prevalence (most common inpatient medication)
# MAGIC - GI-specific medications showing elevated risk ratios despite lower prevalence
# MAGIC - Hemorrhoid medications with extreme rarity but potential high risk when present
# MAGIC - Impact scores balancing prevalence with predictive power

# COMMAND ----------

# Step 2: Calculate Risk Ratios for Inpatient Flag Features

flag_features = [col for col in df_spark.columns if '_flag' in col and col.startswith('inp_')]
risk_metrics = []

print(f"\nCalculating risk ratios for {len(flag_features)} flag features...")

for feat in flag_features:
    stats = df_spark.groupBy(feat).agg(
        F.count('*').alias('count'),
        F.avg('FUTURE_CRC_EVENT').alias('crc_rate')
    ).collect()
    
    # Parse results
    stats_dict = {row[feat]: {'count': row['count'], 'crc_rate': row['crc_rate']} for row in stats}
    
    prevalence = stats_dict.get(1, {'count': 0})['count'] / total_rows
    rate_with = stats_dict.get(1, {'crc_rate': 0})['crc_rate']
    rate_without = stats_dict.get(0, {'crc_rate': baseline_crc_rate})['crc_rate']
    risk_ratio = rate_with / (rate_without + 1e-10)
    
    # Handle edge cases for impact calculation
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
print("\nTop 10 features by impact score (prevalence × log risk ratio):")
print(risk_df[['feature', 'prevalence', 'risk_ratio', 'impact']].head(10).to_string())


# COMMAND ----------

# MAGIC %md
# MAGIC #### 📊 Cell 2 Conclusion
# MAGIC
# MAGIC Successfully created direct medication ID mappings for 200+ medications across 16 clinical categories, with particular depth in GI-related acute care medications including detailed laxative subcategories and specialized bleeding management agents.
# MAGIC
# MAGIC **Key Achievement**: Comprehensive coverage of inpatient-specific medications not captured by standard groupers, ensuring complete acute care medication pattern recognition
# MAGIC
# MAGIC **Next Step**: Create unpivoted medication administration table linking patient observations to actual MAR data
# MAGIC

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC #### 📊 Step 2 Conclusion
# MAGIC
# MAGIC Successfully calculated risk ratios for 16 flag features with opioids showing highest impact score (0.1545) due to 20.7% prevalence and 1.68x risk ratio. GI bleeding medications demonstrate strong risk associations (1.84x) despite lower prevalence.
# MAGIC
# MAGIC **Key Achievement**: Identified opioids, PPIs, and NSAIDs as highest-impact acute care indicators based on prevalence-adjusted risk metrics
# MAGIC
# MAGIC **Next Step**: Analyze missing data patterns in continuous features to understand medication administration frequency
# MAGIC

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### STEP 3 - Assess Missing Data Patterns for Continuous Features
# MAGIC
# MAGIC #### 🔍 What This Step Does
# MAGIC Evaluates missingness in count and days_since features to identify medications rarely given inpatient. Calculates null rates across all temporal features, with high missingness expected for rare acute events like B12/folate (99.996% null) and hemorrhoids (99.9% null).
# MAGIC
# MAGIC #### Why This Matters for Inpatient Data
# MAGIC XGBoost handles missing values natively, so high missingness isn't problematic—it's informative. Missing values in days_since features mean "never hospitalized for this medication," which is valuable information. The missingness pattern reveals which medications are core to acute care (opioids: 79.3% null) versus rarely used (B12: 99.996% null).
# MAGIC
# MAGIC #### What to Watch For
# MAGIC - Opioids with lowest null rate (most commonly administered inpatient)
# MAGIC - B12/folate with highest null rate (essentially never given inpatient)
# MAGIC - Days_since features showing expected high missingness for rare medications
# MAGIC - Missing patterns aligning with clinical expectations for acute care

# COMMAND ----------

# Step 3: Analyze Feature Types and Information Content

print("\nAnalyzing feature types and information content...")

# Separate features by type
flag_features = [col for col in df_spark.columns if '_flag' in col and col.startswith('inp_')]
count_features = [col for col in df_spark.columns if '_count' in col and col.startswith('inp_')]
days_since_features = [col for col in df_spark.columns if '_days_since' in col and col.startswith('inp_')]

print(f"Feature types:")
print(f"  - Flag features: {len(flag_features)}")
print(f"  - Count features: {len(count_features)}")
print(f"  - Days_since features: {len(days_since_features)}")

# For days_since: Create missing_df for compatibility with Step 5
missing_stats = []
for feat in days_since_features:
    # NULL means never had medication - this is information, not missing data
    never_had = df_spark.filter(F.col(feat).isNull()).count() / total_rows
    
    missing_stats.append({
        'feature': feat,
        'missing_rate': never_had,  # Use missing_rate for consistency
        'medication': feat.replace('inp_', '').replace('_days_since', '')
    })

missing_df = pd.DataFrame(missing_stats)
print(f"\nMedications ranked by usage (from days_since nulls):")
print(missing_df.sort_values('missing_rate')[['feature', 'missing_rate']])


# COMMAND ----------

# MAGIC %md
# MAGIC #### 📊 Step 3 Conclusion
# MAGIC
# MAGIC Successfully analyzed missing data patterns with opioids showing lowest null rate (79.3%) confirming most common acute medication, while B12/folate shows 99.996% null rate (essentially never given inpatient). Missing patterns align with clinical expectations.
# MAGIC
# MAGIC **Key Achievement**: Validated that missing data patterns reflect clinical reality rather than data quality issues
# MAGIC
# MAGIC **Next Step**: Calculate mutual information on stratified sample to capture non-linear relationships in acute care patterns
# MAGIC

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### STEP 4 - Calculate Mutual Information Using Stratified Sample
# MAGIC
# MAGIC #### 🔍 What This Step Does
# MAGIC Takes 200K stratified sample (larger than outpatient due to lower prevalence) keeping all positive CRC cases to preserve rare outcome signal. Calculates mutual information between each feature and CRC outcome to capture non-linear relationships missed by simple risk ratios.
# MAGIC
# MAGIC #### Why This Matters for Inpatient Data
# MAGIC Lower medication prevalence requires larger sample to capture rare but important acute events. MI reveals complex patterns like temporal relationships in hospitalization recency that linear metrics miss. Sample preserves 28.9% hospitalization rate while ensuring adequate representation of rare acute events.
# MAGIC
# MAGIC #### What to Watch For
# MAGIC - Sample size around 200K (24% of total) with preserved CRC rate
# MAGIC - Days_since features potentially showing higher MI scores than flags
# MAGIC - Hemorrhoid and rare medication features showing elevated MI despite low prevalence
# MAGIC - MI scores focusing on relative ranking rather than absolute values

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4: Calculate Mutual Information Using Stratified Sample
# MAGIC
# MAGIC **What this does:**
# MAGIC - Takes 200K stratified sample (larger than outpatient due to lower prevalence)
# MAGIC - Keeps all positive CRC cases to preserve rare outcome signal
# MAGIC - Calculates MI between each feature and CRC outcome
# MAGIC - Captures non-linear relationships missed by risk ratios
# MAGIC
# MAGIC **Why larger sample for inpatient:**
# MAGIC - Lower medication prevalence requires more data
# MAGIC - Need to capture rare but important acute events
# MAGIC - Hospitalization patterns may be non-linear
# MAGIC - Sample preserves 23% hospitalization rate
# MAGIC
# MAGIC **MI interpretation for inpatient:**
# MAGIC - Lower absolute MI scores expected (rarer events)
# MAGIC - Focus on relative ranking rather than absolute values
# MAGIC - Composite features may score higher than individual medications

# COMMAND ----------

# Step 4: Sample for Mutual Information Calculation

# Take stratified sample for MI calculation
sample_fraction = min(200000 / total_rows, 1.0)  # Larger sample for inpatient

print(f"\nSampling for MI calculation...")
df_sample = df_spark.sampleBy("FUTURE_CRC_EVENT", 
                               fractions={0: sample_fraction, 1: 1.0},  # Keep all positive cases
                               seed=42).toPandas()

print(f"Sampled {len(df_sample):,} rows ({len(df_sample)/total_rows*100:.1f}% of total)")
print(f"Sample CRC rate: {df_sample['FUTURE_CRC_EVENT'].mean():.4f}")

# Calculate MI on sample
from sklearn.feature_selection import mutual_info_classif

feature_cols = [c for c in df_sample.columns 
                if c not in ['PAT_ID', 'END_DTTM', 'FUTURE_CRC_EVENT'] 
                and c.startswith('inp_')]

print(f"Calculating MI for {len(feature_cols)} features...")
X = df_sample[feature_cols].fillna(-999)
y = df_sample['FUTURE_CRC_EVENT']

mi_scores = mutual_info_classif(X, y, discrete_features='auto', n_neighbors=3, random_state=42)
mi_df = pd.DataFrame({
    'feature': feature_cols,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

print("\nTop features by Mutual Information:")
print(mi_df.to_string())



# COMMAND ----------

# MAGIC %md
# MAGIC #### 📊 Step 4 Conclusion
# MAGIC
# MAGIC Successfully calculated mutual information on 202,834 sample (24.4% of total) with preserved 1.59% CRC rate. Days_since features dominate top MI scores, with hemorrhoid_days_since showing highest MI (0.0333) despite extreme rarity.
# MAGIC
# MAGIC **Key Achievement**: Identified temporal features as most informative for CRC prediction, with rare events showing strong non-linear relationships
# MAGIC
# MAGIC **Next Step**: Apply clinical filters to remove near-zero signal features while preserving critical acute care indicators
# MAGIC Step 

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### STEP 5 - Apply Clinical Filters for Acute Care Setting
# MAGIC
# MAGIC #### 🔍 What This Step Does
# MAGIC Merges all calculated metrics (risk, MI, missingness) and applies inpatient-specific MUST_KEEP list for critical acute indicators (GI bleeding, iron, opioids, laxatives). Removes near-zero signal features with adjusted thresholds: prevalence >0.001% (vs 0.01% outpatient) and risk ratio >3 for rare features.
# MAGIC
# MAGIC #### Why This Matters for Inpatient Data
# MAGIC Acute care requires different filtering criteria than outpatient. Any hospitalization is inherently high-risk, so we preserve hospitalization indicators regardless of statistical metrics. B12/folate removed (essentially never given inpatient) while hemorrhoid medications kept despite rarity due to extreme risk when present.
# MAGIC
# MAGIC #### What to Watch For
# MAGIC - MUST_KEEP list preserving critical acute care indicators
# MAGIC - Removal of B12/folate and chemotherapy (near-zero inpatient signal)
# MAGIC - Adjusted thresholds accounting for lower overall inpatient prevalence
# MAGIC - Clinical reasoning overriding pure statistical metrics for acute care

# COMMAND ----------

# Step 5: Feature Selection Logic for Inpatient 

# Merge all metrics (now using correct missing_df)
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

# Inpatient-specific critical features
MUST_KEEP = [
    'inp_gi_bleed_meds_flag',       # GI bleeding in hospital - critical
    'inp_iron_use_flag',             # Acute bleeding management
    'inp_laxative_use_flag',         # Post-op/obstruction
    'inp_opioid_use_flag',           # Surgery/pain indicator
    'inp_hemorrhoid_meds_flag',      # Rare but high risk
    'inp_gi_bleed_meds_days_since',  # Temporal pattern important
    'inp_ppi_use_flag',              # Common in hospital
    'inp_broad_abx_flag'             # Infection marker
]

# Note: Hemorrhoid medications kept despite extreme rarity (0.07% vs 0.2% outpatient)
# When hemorrhoids are treated inpatient, they likely represent severe bleeding/thrombosis
# requiring urgent care - making this a clinically significant signal despite low prevalence

# Pre-specified removals for near-zero features
REMOVE = ['inp_b12_or_folate_use_flag', 'inp_b12_or_folate_use_days_since', 
          'inp_b12_or_folate_use_count_2yr',  # Essentially never given inpatient
          'inp_chemo_agents_flag', 'inp_chemo_agents_days_since', 
          'inp_chemo_agents_count_2yr']  # Extremely rare inpatient

# Add features with extremely low signal
for _, row in feature_importance.iterrows():
    feat = row['feature']
    
    # More lenient for inpatient due to lower overall prevalence
    if row.get('missing_rate', 0) > 0.999 and feat not in MUST_KEEP:
        REMOVE.append(feat)
    
    # Remove extremely rare with low risk (adjusted for inpatient)
    if '_flag' in feat:
        # Lower threshold for inpatient
        if (row.get('prevalence', 0) < 0.00001 and row.get('risk_ratio', 1) < 3 and feat not in MUST_KEEP):
            REMOVE.append(feat)

REMOVE = list(set(REMOVE))  # Remove duplicates
print(f"\nRemoving {len(REMOVE)} near-zero signal features")
print(f"Examples of removed features: {REMOVE[:5]}")

feature_importance = feature_importance[~feature_importance['feature'].isin(REMOVE)]
print(f"Features remaining after filtering: {len(feature_importance)}")


# COMMAND ----------

# MAGIC %md
# MAGIC #### 📊 Step 5 Conclusion
# MAGIC
# MAGIC Successfully applied clinical filters removing 7 near-zero signal features (B12/folate, chemotherapy) while preserving all critical acute care indicators through MUST_KEEP list. Retained 41 features with clinical relevance for inpatient setting.
# MAGIC
# MAGIC **Key Achievement**: Balanced statistical filtering with clinical knowledge to preserve acute care signals while removing noise
# MAGIC
# MAGIC **Next Step**: Select optimal feature representation per medication using inpatient-specific selection rules
# MAGIC

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### STEP 6 - Select Optimal Features per Medication (Acute Care Focus)
# MAGIC
# MAGIC #### 🔍 What This Step Does
# MAGIC For each medication, selects between flag/count/days_since using inpatient-specific rules: hemorrhoid medications keep both flag AND recency (extreme risk needs granularity), GI/iron/laxative/opioid medications prioritize flags (acute indicators), while chronic medications (statins/metformin) keep only best MI score.
# MAGIC
# MAGIC #### Why This Matters for Inpatient Data
# MAGIC Acute medications require different selection logic than chronic. Flags more important than counts (admission itself is significant), days_since critical for hospitalization recency risk, and extreme/rare medications need multiple representations. Selection prioritizes acute care patterns over chronic medication management.
# MAGIC
# MAGIC #### What to Watch For
# MAGIC - Hemorrhoid medications keeping multiple representations due to extreme risk
# MAGIC - GI-specific medications prioritizing flags for acute event detection
# MAGIC - Chronic medications (statins, metformin) reduced to single best feature
# MAGIC - MUST_KEEP features automatically included regardless of selection rules

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6: Select Optimal Features per Medication (Acute Care Focus)
# MAGIC
# MAGIC **What this does:**
# MAGIC - For each medication, selects between flag/count/recency
# MAGIC - Applies inpatient-specific selection rules:
# MAGIC   - Hemorrhoid: Keep flag AND recency (extreme risk, need granularity)
# MAGIC   - GI/Iron/Laxative/Opioid: Keep flags (acute indicators)
# MAGIC   - GI bleeding: Also keep days_since (temporal pattern critical)
# MAGIC   - Chronic meds (statins/metformin): Best MI score only
# MAGIC
# MAGIC **Why different from outpatient:**
# MAGIC - Acute medications prioritized over chronic
# MAGIC - Flags more important than counts (admission itself is significant)
# MAGIC - Days_since critical for hospitalization recency risk
# MAGIC - Lower bar for keeping features due to overall lower prevalence
# MAGIC
# MAGIC **Selection logic:**
# MAGIC - Extreme/acute: Keep multiple representations
# MAGIC - Chronic: Keep only if strong signal
# MAGIC - All features maintain "inp_" prefix

# COMMAND ----------

# Step 6: Select Best Feature per Medication for Inpatient

def select_optimal_inpatient_features(df_importance):
    """Select best representation for each inpatient medication"""
    
    selected = []
    
    # Extract medication name (remove inp_ prefix for grouping)
    df_importance['medication'] = df_importance['feature'].str.replace('inp_', '').str.split('_').str[0]
    
    # Group by medication
    for med in df_importance['medication'].unique():
        med_features = df_importance[df_importance['medication'] == med]
        
        # Inpatient-specific selection rules
        if med in ['hemorrhoid']:
            # Keep both flag and recency if available
            for feat in med_features['feature']:
                if '_flag' in feat or '_days_since' in feat:
                    selected.append(feat)
                    
        elif med in ['gi', 'iron', 'laxative', 'opioid']:
            # Critical acute care indicators - keep flag
            flag_features = med_features[med_features['feature'].str.contains('_flag')]
            if not flag_features.empty:
                selected.append(flag_features.nlargest(1, 'mi_score')['feature'].values[0])
            
            # Also keep gi_bleed days_since if it's gi_bleed
            if med == 'gi':
                days_features = med_features[med_features['feature'].str.contains('_days_since')]
                if not days_features.empty:
                    selected.append(days_features.iloc[0]['feature'])
                    
        elif med in ['broad', 'antidiarrheal', 'ppi']:
            # Acute event markers - keep flag only
            flag_features = med_features[med_features['feature'].str.contains('_flag')]
            if not flag_features.empty:
                selected.append(flag_features.nlargest(1, 'mi_score')['feature'].values[0])
                
        else:
            # For chronic meds (statins, metformin), keep best MI feature
            if len(med_features) > 0:
                best_feature = med_features.nlargest(1, 'mi_score')['feature'].values[0]
                selected.append(best_feature)
    
    # Ensure must-keep features are included
    for feat in MUST_KEEP:
        if feat not in selected and feat in df_importance['feature'].values:
            selected.append(feat)
    
    return list(set(selected))  # Remove duplicates

selected_features = select_optimal_inpatient_features(feature_importance)
print(f"\nSelected {len(selected_features)} features after medication-level optimization")
print("\nSample of selected features:")
for feat in sorted(selected_features)[:10]:
    print(f"  - {feat}")


# COMMAND ----------

# MAGIC %md
# MAGIC #### 📊 Step 6 Conclusion
# MAGIC
# MAGIC Successfully selected 15 optimal features using inpatient-specific rules, prioritizing acute care indicators (flags) over chronic patterns while preserving temporal information (days_since) for critical medications like GI bleeding.
# MAGIC
# MAGIC **Key Achievement**: Applied acute care-focused selection logic balancing feature efficiency with clinical interpretability
# MAGIC
# MAGIC **Next Step**: Create inpatient-specific composite features and save final reduced dataset

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### STEP 7 - Create Acute Care Composites and Save Final Dataset
# MAGIC
# MAGIC #### 🔍 What This Step Does
# MAGIC Creates 5 inpatient-specific composite features capturing acute care patterns: inp_acute_gi_bleeding (iron + PPI), inp_obstruction_pattern (laxatives + opioids), inp_severe_infection (antibiotics + opioids), inp_any_hospitalization (critical indicator), and inp_gi_hospitalization (GI-specific admission). Saves reduced dataset with all "inp_" prefixed features.
# MAGIC
# MAGIC #### Why This Matters for Inpatient Data
# MAGIC Composite features capture clinical syndromes not visible in individual medications. Acute GI bleeding pattern indicates emergency hemorrhage management, while hospitalization itself is a major risk factor (1.68x overall). Recent hospitalization (<30 days) shows 6.9x risk, making these composites critical for risk stratification and screening prioritization.
# MAGIC
# MAGIC #### What to Watch For
# MAGIC - Final feature count around 20 (58% reduction from original 48)
# MAGIC - All features maintaining "inp_" prefix for seamless model joining
# MAGIC - Composite features capturing clinically meaningful acute care patterns
# MAGIC - Perfect row preservation (831,397 observations) in final dataset

# COMMAND ----------

# Step 7: Create Inpatient-Specific Composite Features and Save

df_final = df_spark

# === INPATIENT-SPECIFIC COMPOSITE FEATURES ===

# 1. Acute GI bleeding pattern (iron + PPI during admission)
df_final = df_final.withColumn('inp_acute_gi_bleeding',
    F.when((F.col('inp_iron_use_flag') == 1) & 
           (F.col('inp_ppi_use_flag') == 1), 1).otherwise(0)
)

# 2. Obstruction/ileus pattern (laxatives + opioids)
df_final = df_final.withColumn('inp_obstruction_pattern',
    F.when((F.col('inp_laxative_use_flag') == 1) & 
           (F.col('inp_opioid_use_flag') == 1), 1).otherwise(0)
)

# 3. Severe infection/sepsis pattern (antibiotics + opioids)
df_final = df_final.withColumn('inp_severe_infection',
    F.when((F.col('inp_broad_abx_flag') == 1) & 
           (F.col('inp_opioid_use_flag') == 1), 1).otherwise(0)
)

# 4. Any hospitalization indicator (critical feature)
df_final = df_final.withColumn('inp_any_hospitalization',
    F.when((F.col('inp_iron_use_flag') == 1) | 
           (F.col('inp_ppi_use_flag') == 1) |
           (F.col('inp_laxative_use_flag') == 1) |
           (F.col('inp_opioid_use_flag') == 1) |
           (F.col('inp_broad_abx_flag') == 1), 1).otherwise(0)
)

# 5. GI-specific hospitalization
df_final = df_final.withColumn('inp_gi_hospitalization',
    F.when((F.col('inp_laxative_use_flag') == 1) | 
           (F.col('inp_antidiarrheal_use_flag') == 1) |
           (F.col('inp_gi_bleed_meds_flag') == 1), 1).otherwise(0)
)

composite_features = [
    'inp_acute_gi_bleeding',
    'inp_obstruction_pattern', 
    'inp_severe_infection',
    'inp_any_hospitalization',
    'inp_gi_hospitalization'
]

# Add composites to selected features
selected_features.extend(composite_features)
selected_features = sorted(list(set(selected_features)))  # Remove duplicates and sort

print(f"\nAdded {len(composite_features)} composite features")
print(f"Final feature count: {len(selected_features)}")

# === PRINT FINAL FEATURE LIST ===
print("\n" + "="*60)
print("FINAL SELECTED INPATIENT FEATURES")
print("="*60)

for i, feat in enumerate(selected_features, 1):
    # Add description for clarity
    if 'hemorrhoid' in feat:
        desc = " [RARE BUT EXTREME RISK]"
    elif 'gi_bleed' in feat:
        desc = " [ACUTE BLEEDING]"
    elif 'any_hospitalization' in feat:
        desc = " [KEY INDICATOR]"
    elif feat in composite_features:
        desc = " [COMPOSITE]"
    elif 'opioid' in feat:
        desc = " [PAIN/SURGERY]"
    elif 'iron' in feat or 'laxative' in feat:
        desc = " [HIGH RISK]"
    elif 'ppi' in feat or 'broad_abx' in feat:
        desc = " [COMMON ACUTE]"
    else:
        desc = ""
    print(f"{i:2d}. {feat:<40} {desc}")

# === SAVE REDUCED DATASET ===
final_columns = ['PAT_ID', 'END_DTTM'] + selected_features
df_reduced = df_final.select(*final_columns)

# Write to final table
output_table = 'dev.clncl_ds.herald_eda_train_inpatient_meds_reduced'
df_reduced.write.mode('overwrite').saveAsTable(output_table)

print("\n" + "="*60)
print("FEATURE REDUCTION SUMMARY")
print("="*60)
print(f"Original features: 48")
print(f"Selected features: {len(selected_features)}")
print(f"Reduction: {(1 - len(selected_features)/48)*100:.1f}%")
print(f"\n✓ Reduced dataset saved to: {output_table}")

# Verify save and check all columns have inp_ prefix
row_count = spark.table(output_table).count()
cols_without_prefix = [c for c in selected_features if not c.startswith('inp_')]

print(f"✓ Verified {row_count:,} rows written to table")
if cols_without_prefix:
    print(f"\n⚠ WARNING: These columns missing 'inp_' prefix: {cols_without_prefix}")
else:
    print("✓ All feature columns have 'inp_' prefix for joining")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 📊 Step 7 Conclusion
# MAGIC
# MAGIC Successfully created 5 acute care composite features and saved final reduced dataset with 20 features (58.3% reduction from original 48). All features maintain "inp_" prefix with perfect row preservation (831,397 observations) for model integration.
# MAGIC
# MAGIC **Key Achievement**: Delivered optimized feature set capturing critical acute care signals while achieving significant dimensionality reduction
# MAGIC
# MAGIC **Final Result**: Complete inpatient medications feature engineering pipeline from MAR data to modeling-ready features with clinical interpretability and temporal risk stratification capabilities
# MAGIC These

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # Herald Inpatient Medications Analysis - Complete Summary
# MAGIC
# MAGIC ## Executive Summary
# MAGIC
# MAGIC This comprehensive analysis of **831,397 patient-month observations** successfully extracted and optimized inpatient medication features from MAR (Medication Administration Record) data, revealing critical insights about hospitalization timing and CRC risk. **KEY DISCOVERY: Recent hospitalizations show exponential temporal gradient with CRC risk** - patients hospitalized 0-30 days ago have **6.9x higher CRC risk** than baseline, declining progressively to baseline after 180+ days. We reduced 48 features to 20 optimized features (58% reduction) while preserving all critical acute care signals.
# MAGIC
# MAGIC ## Major Clinical Discoveries
# MAGIC
# MAGIC ### 1. Hospitalization as Primary Risk Indicator
# MAGIC **Temporal risk stratification reveals actionable intelligence:**
# MAGIC - **0-30 days post-discharge:** 6.9x CRC risk (2.7% rate vs 0.39% baseline)
# MAGIC - **31-90 days:** 3.1x risk, declining progressively  
# MAGIC - **91-180 days:** 1.8x risk
# MAGIC - **180+ days:** Returns to baseline (0.9x risk)
# MAGIC - **Clinical implication:** Post-discharge screening protocols should prioritize recent hospitalizations
# MAGIC
# MAGIC ### 2. Acute Care Medication Patterns
# MAGIC **MAR data reveals confirmed administration patterns:**
# MAGIC - **Opioids:** 20.7% prevalence, 1.68x CRC risk (surgery/severe pain marker)
# MAGIC - **NSAIDs:** 15.7% prevalence, 1.25x risk (4x higher than outpatient due to PRN)
# MAGIC - **Laxatives:** 9.2% prevalence, 1.24x risk (post-op ileus/obstruction concerns)
# MAGIC - **PPIs:** 8.4% prevalence, 1.64x risk (bleeding prophylaxis)
# MAGIC - **GI bleeding medications:** 1.7% prevalence, 1.84x risk (emergency hemorrhage)
# MAGIC
# MAGIC ### 3. Care Setting Comparison Insights
# MAGIC **Hemorrhoid treatment reveals care complexity:**
# MAGIC - **Inpatient:** 542 patients (0.07%) - extremely rare
# MAGIC - **Outpatient:** ~0.2% (from comparative analysis)
# MAGIC - **87% reduction** from outpatient to inpatient settings
# MAGIC - **Clinical insight:** When hemorrhoids treated inpatient, consider severe bleeding/thrombosis
# MAGIC
# MAGIC ### 4. Acute Event Pattern Recognition
# MAGIC **Composite patterns with clinical significance:**
# MAGIC
# MAGIC |
# MAGIC  Pattern 
# MAGIC |
# MAGIC  Definition 
# MAGIC |
# MAGIC  Prevalence 
# MAGIC |
# MAGIC  Risk Ratio 
# MAGIC |
# MAGIC  Clinical Insight 
# MAGIC |
# MAGIC |
# MAGIC ---------
# MAGIC |
# MAGIC ------------
# MAGIC |
# MAGIC ------------
# MAGIC |
# MAGIC ------------
# MAGIC |
# MAGIC ------------------
# MAGIC |
# MAGIC |
# MAGIC **
# MAGIC Acute GI Bleeding
# MAGIC **
# MAGIC |
# MAGIC  Iron + PPI 
# MAGIC |
# MAGIC  3.0% 
# MAGIC |
# MAGIC  1.86x 
# MAGIC |
# MAGIC  Emergency hemorrhage management 
# MAGIC |
# MAGIC |
# MAGIC **
# MAGIC Severe Infection
# MAGIC **
# MAGIC |
# MAGIC  Antibiotics + opioids 
# MAGIC |
# MAGIC  4.4% 
# MAGIC |
# MAGIC  1.73x 
# MAGIC |
# MAGIC  Sepsis/complications requiring intensive care 
# MAGIC |
# MAGIC |
# MAGIC **
# MAGIC Obstruction/Ileus
# MAGIC **
# MAGIC |
# MAGIC  Laxatives + opioids + antispasmodics 
# MAGIC |
# MAGIC  0.17% 
# MAGIC |
# MAGIC  1.43x 
# MAGIC |
# MAGIC  Post-surgical complications 
# MAGIC |
# MAGIC |
# MAGIC **
# MAGIC IBD Flare
# MAGIC **
# MAGIC |
# MAGIC  IBD meds or antidiarrheal + antispasmodic 
# MAGIC |
# MAGIC  2.7% 
# MAGIC |
# MAGIC  1.05x 
# MAGIC |
# MAGIC  Inflammatory disease exacerbation 
# MAGIC |
# MAGIC
# MAGIC ### 5. Iron Supplementation Care Trajectories
# MAGIC **Setting-specific patterns reveal disease severity:**
# MAGIC - **Both settings:** 0.98% of cohort, 2.01x CRC risk (severe/recurring bleeding)
# MAGIC - **Inpatient only:** 1.31% of cohort, 1.72x risk (acute bleeding event)
# MAGIC - **Outpatient only:** 1.29% of cohort, 3.93x risk (chronic anemia workup)
# MAGIC - **Neither:** 96.4% of cohort, 0.94x risk (baseline population)
# MAGIC
# MAGIC ## Technical Achievements
# MAGIC
# MAGIC ### Data Processing Excellence
# MAGIC - **Perfect row matching:** 831,397 observations (100% of cohort)
# MAGIC - **MAR data integration:** Confirmed medication administration vs prescriptions
# MAGIC - **Temporal precision:** Days-level granularity for hospitalization recency
# MAGIC - **Data constraint handling:** Respected 2021-07-01 MAR availability cutoff
# MAGIC - **Validation success:** P95 lookback at 689 days within 24-month window
# MAGIC
# MAGIC ### Feature Engineering Innovation
# MAGIC - **48 comprehensive features** (16 medications × 3 types: flag, days_since, count)
# MAGIC - **Acute care focus:** Prioritizes emergency presentations over chronic care
# MAGIC - **Temporal modeling:** Captures exponential risk decay post-discharge
# MAGIC - **Clinical validation:** All patterns align with known CRC risk factors
# MAGIC - **Integration ready:** "inp_" prefix enables seamless model joining
# MAGIC
# MAGIC ### Feature Reduction Optimization
# MAGIC - **Intelligent reduction:** 48 → 20 features (58% reduction)
# MAGIC - **Signal preservation:** Maintained all critical acute care indicators
# MAGIC - **Composite creation:** 5 acute care pattern features
# MAGIC - **XGBoost optimization:** Native NULL handling, no imputation needed
# MAGIC - **Clinical prioritization:** MUST_KEEP list for critical acute indicators
# MAGIC
# MAGIC ## Data Quality Validation
# MAGIC
# MAGIC ### Completeness and Accuracy
# MAGIC - ✅ **Row count validation:** Perfect match with cohort (831,397)
# MAGIC - ✅ **Date range compliance:** All medications respect data availability constraints
# MAGIC - ✅ **Temporal consistency:** Lookback ranges 1-731 days as expected
# MAGIC - ✅ **Feature naming:** All features have "inp_" prefix for identification
# MAGIC - ✅ **Missing data patterns:** Expected missingness for rare acute events
# MAGIC
# MAGIC ### Clinical Validation
# MAGIC - ✅ **Hospitalization rate:** 28.9% aligns with expected acute care utilization
# MAGIC - ✅ **Medication prevalence:** Patterns consistent with inpatient vs outpatient care
# MAGIC - ✅ **Risk associations:** All major medications show expected CRC risk relationships
# MAGIC - ✅ **Temporal patterns:** Hospitalization recency shows logical risk decay
# MAGIC
# MAGIC ## Clinical Practice Implications
# MAGIC
# MAGIC ### Immediate Applications
# MAGIC 1. **Post-discharge screening protocols:** Use temporal gradient for structured follow-up
# MAGIC 2. **Risk stratification:** Recent hospitalization (<90 days) = urgent screening priority
# MAGIC 3. **Quality metrics:** Track time from GI admission to colonoscopy completion
# MAGIC 4. **Care coordination:** Integrate hospitalization data into screening workflows
# MAGIC
# MAGIC ### Healthcare System Impact
# MAGIC - **Screening efficiency:** Focus resources on highest-risk post-discharge patients
# MAGIC - **Cost optimization:** Prioritize expensive screening for patients with 6.9x risk
# MAGIC - **Quality improvement:** Reduce time to diagnosis for symptomatic presentations
# MAGIC - **Population health:** Systematic approach to post-acute care screening
# MAGIC
# MAGIC ## Model Integration Value
# MAGIC
# MAGIC ### High-Impact Features for Prediction
# MAGIC - `inp_any_hospitalization`: 28.9% prevalence, 1.68x risk (critical indicator)
# MAGIC - `inp_acute_gi_bleeding`: 3.0% prevalence, 1.86x risk (emergency pattern)
# MAGIC - Temporal recency features for risk stratification
# MAGIC - Acute event composites for severity assessment
# MAGIC
# MAGIC ### Integration Architecture
# MAGIC - **Seamless joining:** "inp_" prefix prevents naming conflicts
# MAGIC - **XGBoost ready:** Native NULL handling eliminates preprocessing
# MAGIC - **Complementary signals:** Inpatient acute + outpatient chronic patterns
# MAGIC - **Scalable processing:** Handles 831K observations efficiently
# MAGIC
# MAGIC ## Deliverables and Next Steps
# MAGIC
# MAGIC ### Primary Outputs
# MAGIC 1. **herald_eda_train_inpatient_meds:** Full 48 features from MAR data
# MAGIC 2. **herald_eda_train_inpatient_meds_reduced:** Optimized 20 features for modeling
# MAGIC 3. **Comprehensive documentation:** Clinical insights and technical validation
# MAGIC 4. **Integration framework:** Ready for model training pipeline
# MAGIC
# MAGIC ### Recommended Next Steps
# MAGIC 1. **Model training:** Integrate with outpatient medications and other feature sets
# MAGIC 2. **Validation studies:** Confirm temporal patterns in held-out test data
# MAGIC 3. **Clinical deployment:** Implement post-discharge screening protocols
# MAGIC 4. **Performance monitoring:** Track screening efficiency improvements
# MAGIC
# MAGIC ## Conclusions
# MAGIC
# MAGIC The inpatient medication analysis successfully identified hospitalization as a major CRC risk indicator, with temporal patterns showing exponential risk decay post-discharge. The discovery that patients hospitalized 0-30 days ago have 6.9x higher CRC risk provides immediate actionable intelligence for screening prioritization. The 87% reduction in hemorrhoid treatment from outpatient to inpatient settings exemplifies the acute vs chronic care distinction captured in this analysis.
# MAGIC
# MAGIC These findings support development of risk-stratified post-discharge screening protocols and demonstrate the value of MAR data for confirmed medication administration patterns. The optimized feature set preserves all critical acute care signals while achieving 58% feature reduction, creating an efficient and clinically meaningful input for CRC risk prediction models.
# MAGIC
# MAGIC The temporal gradient discovery - showing exponential risk decay from 6.9x at 0-30 days to baseline at 180+ days - provides a framework for healthcare systems to optimize screening resource allocation and improve early detection of colorectal cancer in high-risk populations.
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC