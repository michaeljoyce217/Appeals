# data/featurization.py
# Rebuttal Engine v2 - Featurization (Data Gathering)
#
# This notebook handles TWO data pipelines:
# 1. Gold Standard Letters: Ingest past winning rebuttals for vector search
# 2. New Denials: Pull clinical data for accounts needing rebuttals
#
# WHY TWO PIPELINES?
# - Gold letters are ingested ONCE (or when new ones arrive)
# - New denials are processed regularly as they come from the workqueue
#
# Run on Databricks Runtime 15.4 LTS ML

# =============================================================================
# CELL 1: Install Dependencies (run once per cluster)
# =============================================================================
# These packages are not pre-installed on Databricks ML runtime:
# - azure-ai-documentintelligence: PDF text extraction via Azure AI
# - openai: Azure OpenAI API client
#
# %pip install azure-ai-documentintelligence==1.0.2 openai
# dbutils.library.restartPython()

# =============================================================================
# CELL 2: Imports and Configuration
# =============================================================================
import os
import json
import uuid
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, lit
from pyspark.sql.types import (
    StructType, StructField, StringType, ArrayType, FloatType,
    TimestampType, MapType
)

# Get or create Spark session (already exists in Databricks notebooks)
spark = SparkSession.builder.getOrCreate()

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# SCOPE_FILTER controls which cases we process:
# - "sepsis": Only process sepsis-related denials (DRG 870/871/872)
# - "all": Process all denial types (future expansion)
SCOPE_FILTER = "sepsis"

# DRG codes for sepsis:
# - 870: Sepsis with MV >96 hours (most severe)
# - 871: Sepsis without MV >96 hours (severe)
# - 872: Sepsis without MCC (less severe)
SEPSIS_DRG_CODES = ["870", "871", "872"]

# =============================================================================
# CELL 3: Environment Setup
# =============================================================================
# Determine which Unity Catalog to use based on environment variable.
# Default to 'dev' if not set, but USE 'prod' catalog for dev
# (this is intentional - dev environment reads from prod data)
trgt_cat = os.environ.get('trgt_cat', 'dev')
if trgt_cat == 'dev':
    spark.sql('USE CATALOG prod;')
else:
    spark.sql(f'USE CATALOG {trgt_cat};')

# Table names follow convention: {catalog}.fin_ds.fudgesicle_{purpose}
# "fudgesicle" is the internal codename for this project
GOLD_LETTERS_TABLE = f"{trgt_cat}.fin_ds.fudgesicle_gold_letters"
INFERENCE_TABLE = f"{trgt_cat}.fin_ds.fudgesicle_inference"

# Paths to source files - UPDATE THESE for your Databricks workspace
# These should point to your Repos folder or DBFS location
GOLD_LETTERS_PATH = "/Workspace/Repos/your_user/fudgsicle/utils/gold_standard_rebuttals"
DENIAL_LETTERS_PATH = "/Workspace/Repos/your_user/fudgsicle/utils/Sample_Denial_Letters"

print(f"Catalog: {trgt_cat}")
print(f"Gold letters path: {GOLD_LETTERS_PATH}")

# =============================================================================
# CELL 4: Azure Credentials
# =============================================================================
# All secrets are stored in Databricks secret scope 'idp_etl'.
# These credentials are managed by the platform team.
#
# UPDATE THESE KEY NAMES if different in your environment:
# To list available secrets: dbutils.secrets.list(scope='idp_etl')
AZURE_OPENAI_KEY = dbutils.secrets.get(scope='idp_etl', key='az-openai-key1')
AZURE_OPENAI_ENDPOINT = dbutils.secrets.get(scope='idp_etl', key='az-openai-base')
AZURE_DOC_INTEL_KEY = dbutils.secrets.get(scope='idp_etl', key='az-aidcmntintel-key1')
AZURE_DOC_INTEL_ENDPOINT = dbutils.secrets.get(scope='idp_etl', key='az-aidcmntintel-endpoint')

print("Credentials loaded")

# =============================================================================
# CELL 5: Initialize Clients and Document Reading Functions
# =============================================================================
from openai import AzureOpenAI
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential

# Azure OpenAI client - used for:
# - Splitting combined PDF into rebuttal + denial sections
# - Generating embeddings for vector search
openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2024-10-21"  # Latest stable API version
)

# Document Intelligence client - used for:
# - Extracting text from PDF files (denial letters, gold standard letters)
# - Uses "prebuilt-read" model which handles various document layouts
doc_intel_client = DocumentIntelligenceClient(
    endpoint=AZURE_DOC_INTEL_ENDPOINT,
    credential=AzureKeyCredential(AZURE_DOC_INTEL_KEY)
)

def read_pdf(file_path):
    """Read text from a PDF file using Azure Document Intelligence"""
    try:
        with open(file_path, 'rb') as f:
            document_bytes = f.read()

        # Submit document for analysis
        poller = doc_intel_client.begin_analyze_document(
            model_id="prebuilt-read",
            body=AnalyzeDocumentRequest(bytes_source=document_bytes),
        )
        result = poller.result()

        # Extract text line by line, preserving document order
        text_parts = []
        for page in result.pages:
            for line in page.lines:
                text_parts.append(line.content)
        return "\n".join(text_parts)
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
        return f"Error reading PDF: {e}"

def read_document(file_path):
    """Read text from PDF or TXT based on extension"""
    if file_path.lower().endswith('.pdf'):
        return read_pdf(file_path)
    elif file_path.lower().endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return f"Unsupported file format: {file_path}"

print("Azure OpenAI and Document Intelligence clients initialized")

# =============================================================================
# CELL 6: Create Gold Letters Table (run once)
# =============================================================================
# This table stores past winning rebuttal letters with their embeddings.
# The embeddings enable vector search to find similar denials.
#
# Schema:
# - letter_id: UUID for each gold letter
# - source_file: Original PDF filename for traceability
# - payor: Insurance company (extracted during ingestion)
# - denial_text: The original denial letter that was rebutted
# - rebuttal_text: The winning rebuttal letter (what we learn from)
# - denial_embedding: 1536-dim vector from text-embedding-ada-002
# - created_at: When this record was ingested
# - metadata: Additional info like split confidence
create_gold_table_sql = f"""
CREATE TABLE IF NOT EXISTS {GOLD_LETTERS_TABLE} (
    letter_id STRING NOT NULL,
    source_file STRING NOT NULL,
    payor STRING,
    denial_text STRING,
    rebuttal_text STRING,
    denial_embedding ARRAY<FLOAT>,
    created_at TIMESTAMP,
    metadata MAP<STRING, STRING>
)
USING DELTA
COMMENT 'Gold standard rebuttal letters with embeddings for vector search'
"""

spark.sql(create_gold_table_sql)
print(f"Table {GOLD_LETTERS_TABLE} ready")

# =============================================================================
# CELL 7: Create Inference Table (run once)
# =============================================================================
# This table holds the input data for rebuttal generation.
# Each row represents one denial case that needs a rebuttal letter.
#
# Data comes from two sources:
# 1. Clarity (Epic EMR): Patient demographics, clinical notes, diagnosis codes
# 2. Denial letter files: The actual denial letter text
#
# The inference.py script reads from this table to generate rebuttals.
create_inference_table_sql = f"""
CREATE TABLE IF NOT EXISTS {INFERENCE_TABLE} (
    hsp_account_id STRING,
    pat_id STRING,
    pat_mrn_id STRING,
    formatted_name STRING,
    formatted_birthdate STRING,
    facility_name STRING,
    number_of_midnights INT,
    formatted_date_of_service STRING,
    claim_number STRING,
    tax_id STRING,
    npi STRING,
    code STRING,
    dx_name STRING,
    discharge_summary_note_id STRING,
    discharge_note_csn_id STRING,
    discharge_summary_text STRING,
    hp_note_id STRING,
    hp_note_csn_id STRING,
    hp_note_text STRING,
    denial_letter_text STRING,
    denial_letter_filename STRING,
    scope_filter STRING,
    featurization_timestamp STRING,
    insert_tsp TIMESTAMP
)
USING DELTA
COMMENT 'Input data for rebuttal letter generation'
"""

spark.sql(create_inference_table_sql)
print(f"Table {INFERENCE_TABLE} ready")

# #############################################################################
# PART 1: GOLD STANDARD LETTER INGESTION
# Run this section when you have new gold standard letters to ingest.
#
# BACKGROUND:
# The gold standard letters are past rebuttals that WON their appeals.
# Each PDF contains BOTH the winning rebuttal AND the original denial
# (the denial is tacked onto the end of the document).
#
# PROCESS:
# 1. Read PDF with PyPDF2
# 2. Use LLM to split into rebuttal vs denial sections
# 3. Generate embedding of the denial (for matching new denials later)
# 4. Store everything in Delta table
# #############################################################################

# =============================================================================
# CELL 8: LLM Prompt for Splitting Rebuttal from Denial
# =============================================================================
# This prompt instructs GPT-4.1 to separate the combined document.
# The model identifies the "split point" between the two letters
# based on typical business letter markers (letterhead, dates, tone).
#
# Output is JSON with:
# - rebuttal_text: The winning appeal letter
# - denial_text: The original denial being rebutted
# - payor: Insurance company name (for filtering/reporting)
# - split_confidence: How confident the model is (0.0-1.0)
# - split_reasoning: Explanation of how it found the split
SPLIT_PROMPT = """You are analyzing a document that contains TWO business letters combined into one file.

The document structure is:
1. FIRST: A successful rebuttal/appeal letter (from a hospital to an insurance company)
2. SECOND: The original denial letter that was being rebutted (tacked on at the end)

Each letter has typical business letter formatting: logo/letterhead, date, addresses, salutation, body, signature.

Your tasks:
1. Identify where the rebuttal letter ENDS and the original denial letter BEGINS
2. Extract the PAYOR (insurance company) name from the denial letter

Here is the document text:
---
{document_text}
---

Return a JSON object with exactly this structure:
{{
    "rebuttal_text": "The complete text of the rebuttal/appeal letter (the first letter)",
    "denial_text": "The complete text of the original denial letter (the second letter, tacked on at end)",
    "payor": "The insurance company/payor name (e.g., 'UnitedHealthcare', 'Aetna', 'Blue Cross Blue Shield', 'Humana', 'Cigna')",
    "split_confidence": 0.0 to 1.0,
    "split_reasoning": "Brief explanation of how you identified the split point"
}}

Look for indicators like:
- Change in letterhead/sender (hospital vs insurance company)
- Change in date
- Change in tone (appealing vs denying)
- Phrases like "Dear Medical Review Team" (rebuttal) vs "Dear Provider" (denial)
- References to "your denial" (rebuttal) vs "we have determined" (denial)
- The payor name is usually in the denial letter's letterhead, signature block, or referenced in the rebuttal

Return ONLY valid JSON, no markdown."""

# =============================================================================
# CELL 9: Process Gold Standard Letters
# =============================================================================
# Set RUN_GOLD_INGESTION = True when you have new gold letters to ingest.
# This is typically run once initially, then again only when new letters arrive.
RUN_GOLD_INGESTION = False

if RUN_GOLD_INGESTION:
    print("="*60)
    print("GOLD STANDARD LETTER INGESTION")
    print("="*60)

    # Find all PDF files in the gold letters directory
    pdf_files = [f for f in os.listdir(GOLD_LETTERS_PATH) if f.lower().endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDF files")

    # Will collect all successfully processed records here
    gold_records = []

    # Process each PDF file one at a time
    for i, pdf_file in enumerate(pdf_files):
        print(f"\nProcessing {i+1}/{len(pdf_files)}: {pdf_file}")
        file_path = os.path.join(GOLD_LETTERS_PATH, pdf_file)

        # ---------------------------------------------------------------------
        # Step 1: Extract text from PDF using PyPDF2
        # ---------------------------------------------------------------------
        print("  Reading PDF...")
        full_text = read_pdf(file_path)
        print(f"  Extracted {len(full_text)} characters")

        # ---------------------------------------------------------------------
        # Step 2: Use LLM to split document into rebuttal and denial
        # ---------------------------------------------------------------------
        # The combined PDF has the winning rebuttal first, then the original
        # denial tacked on at the end. We need to separate them.
        print("  Splitting rebuttal and denial...")
        response = openai_client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "Extract the two letters accurately. Return only valid JSON."},
                {"role": "user", "content": SPLIT_PROMPT.format(document_text=full_text)}
            ],
            temperature=0.1,  # Low temp for consistent extraction
            max_tokens=8000   # Letters can be long
        )

        # Parse LLM response - handle potential markdown code blocks
        raw_response = response.choices[0].message.content.strip()
        json_str = raw_response
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()

        # Extract fields from JSON response
        split_result = json.loads(json_str)
        rebuttal_text = split_result.get("rebuttal_text", "")
        denial_text = split_result.get("denial_text", "")
        payor = split_result.get("payor", "Unknown")
        split_confidence = str(split_result.get("split_confidence", 0.0))
        split_reasoning = split_result.get("split_reasoning", "")

        print(f"  Rebuttal: {len(rebuttal_text)} chars, Denial: {len(denial_text)} chars")
        print(f"  Payor: {payor}")
        print(f"  Split confidence: {split_confidence}")

        # ---------------------------------------------------------------------
        # Step 3: Generate embedding for the denial text
        # ---------------------------------------------------------------------
        # We embed the DENIAL (not rebuttal) because when a new denial comes in,
        # we want to find the most similar PAST denial to match against.
        # The corresponding rebuttal is what we'll use as a template.
        print("  Generating embedding...")

        # Truncate if too long (ada-002 has 8191 token limit, ~30k chars safe)
        embed_text = denial_text[:30000] if len(denial_text) > 30000 else denial_text
        embed_response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=embed_text
        )
        denial_embedding = embed_response.data[0].embedding  # 1536 dimensions
        print(f"  Embedding dimension: {len(denial_embedding)}")

        # ---------------------------------------------------------------------
        # Step 4: Create record for this gold letter
        # ---------------------------------------------------------------------
        record = {
            "letter_id": str(uuid.uuid4()),  # Unique ID for this letter
            "source_file": pdf_file,          # Original filename for traceability
            "payor": payor,                   # Insurance company
            "denial_text": denial_text,       # Original denial (for reference)
            "rebuttal_text": rebuttal_text,   # Winning rebuttal (what we learn from)
            "denial_embedding": denial_embedding,  # Vector for similarity search
            "created_at": datetime.now(),
            "metadata": {
                "split_confidence": split_confidence,
                "split_reasoning": split_reasoning
            },
        }
        gold_records.append(record)
        print(f"  Record created: {record['letter_id']}")

    print(f"\nProcessed {len(gold_records)} gold standard letters")

    # -------------------------------------------------------------------------
    # Write all records to Delta table
    # -------------------------------------------------------------------------
    if gold_records:
        # Define explicit schema to ensure correct data types
        schema = StructType([
            StructField("letter_id", StringType(), False),
            StructField("source_file", StringType(), False),
            StructField("payor", StringType(), True),
            StructField("denial_text", StringType(), True),
            StructField("rebuttal_text", StringType(), True),
            StructField("denial_embedding", ArrayType(FloatType()), True),
            StructField("created_at", TimestampType(), True),
            StructField("metadata", MapType(StringType(), StringType()), True),
        ])

        # Create DataFrame and append to table
        gold_df = spark.createDataFrame(gold_records, schema)
        gold_df.write.format("delta").mode("append").saveAsTable(GOLD_LETTERS_TABLE)
        print(f"Wrote {len(gold_records)} records to {GOLD_LETTERS_TABLE}")

    # Verify final count
    count = spark.sql(f"SELECT COUNT(*) as cnt FROM {GOLD_LETTERS_TABLE}").collect()[0]["cnt"]
    print(f"Total records in gold letters table: {count}")

else:
    print("Gold ingestion skipped (set RUN_GOLD_INGESTION = True to run)")

# #############################################################################
# PART 2: NEW DENIAL FEATURIZATION
# Run this section to prepare new denial cases for inference.
#
# BACKGROUND:
# When a new denial arrives, we need to gather all the data needed to
# generate a rebuttal. This includes:
# - Patient demographics (name, DOB)
# - Clinical notes (discharge summary, H&P)
# - Diagnosis codes
# - The denial letter text itself
#
# OUTPUT:
# One row per denial case in the inference table, ready for inference.py
# #############################################################################

# =============================================================================
# CELL 10: Target Accounts Configuration
# =============================================================================
# For POC: Manually specify which accounts to process.
# In production: This would come from a workqueue table.
#
# Format: (HSP_ACCOUNT_ID, denial_letter_filename)
# - HSP_ACCOUNT_ID: Hospital account ID from Epic (links to Clarity data)
# - denial_letter_filename: Name of PDF file in DENIAL_LETTERS_PATH
TARGET_ACCOUNTS = [
    # UPDATE THESE with your test cases:
    # ("123456789", "denial_letter_1.pdf"),
    # ("987654321", "denial_letter_2.pdf"),
]

print(f"\nTarget accounts configured: {len(TARGET_ACCOUNTS)}")

# =============================================================================
# CELL 11: Clinical Data Query and Denial Processing
# =============================================================================
# Set RUN_DENIAL_FEATURIZATION = True when you have accounts to process.
RUN_DENIAL_FEATURIZATION = False

if RUN_DENIAL_FEATURIZATION and len(TARGET_ACCOUNTS) > 0:
    print("="*60)
    print("NEW DENIAL FEATURIZATION")
    print("="*60)

    # Build list of account IDs for SQL query
    account_ids = [a[0] for a in TARGET_ACCOUNTS]
    account_list = ",".join(f"'{a}'" for a in account_ids)

    # -------------------------------------------------------------------------
    # Query clinical data from Clarity (Epic's reporting database)
    # -------------------------------------------------------------------------
    # This is a large CTE-based query that joins multiple Clarity tables.
    # It retrieves everything we need to generate a rebuttal letter.
    print("\nQuerying clinical data from Clarity...")

    clinical_query = f"""
    -- Create temporary list of target accounts
    WITH target_accounts AS (
        SELECT explode(array({account_list})) AS hsp_account_id
    ),

    -- Find the most recent encounter for each account
    -- (some accounts may have multiple encounters)
    encounters AS (
        SELECT
            pe.pat_enc_csn_id,
            pe.pat_id,
            pe.hsp_account_id,
            pe.hosp_admsn_time,
            pe.hosp_disch_time,
            ROW_NUMBER() OVER (PARTITION BY pe.hsp_account_id ORDER BY pe.hosp_admsn_time DESC) AS rn
        FROM prod.clarity_cur.pat_enc_hsp_har_enh pe
        INNER JOIN target_accounts ta ON pe.hsp_account_id = ta.hsp_account_id
        WHERE pe.hosp_admsn_time IS NOT NULL
    ),
    latest_encounters AS (
        SELECT * FROM encounters WHERE rn = 1
    ),

    -- Get discharge summary notes (note_type_c = 2)
    -- These summarize the entire hospital stay
    discharge_notes AS (
        SELECT
            e.hsp_account_id,
            hno.note_id AS discharge_note_id,
            hno.pat_enc_csn_id AS discharge_note_csn_id,
            CONCAT_WS(' ', COLLECT_LIST(hnt.note_text)) AS discharge_summary_text
        FROM prod.clarity_cur.hno_info_enh hno
        INNER JOIN latest_encounters e ON hno.pat_enc_csn_id = e.pat_enc_csn_id
        INNER JOIN prod.clarity_cur.hno_note_text_enh hnt ON hno.note_id = hnt.note_id
        WHERE hno.note_type_c = 2
        GROUP BY e.hsp_account_id, hno.note_id, hno.pat_enc_csn_id
    ),

    -- Get H&P (History & Physical) notes (note_type_c = 1)
    -- These document initial presentation and findings
    hp_notes AS (
        SELECT
            e.hsp_account_id,
            hno.note_id AS hp_note_id,
            hno.pat_enc_csn_id AS hp_note_csn_id,
            CONCAT_WS(' ', COLLECT_LIST(hnt.note_text)) AS hp_note_text
        FROM prod.clarity_cur.hno_info_enh hno
        INNER JOIN latest_encounters e ON hno.pat_enc_csn_id = e.pat_enc_csn_id
        INNER JOIN prod.clarity_cur.hno_note_text_enh hnt ON hno.note_id = hnt.note_id
        WHERE hno.note_type_c = 1
        GROUP BY e.hsp_account_id, hno.note_id, hno.pat_enc_csn_id
    ),

    -- Get patient demographics
    patient_info AS (
        SELECT
            e.hsp_account_id,
            p.pat_id,
            p.pat_mrn_id,
            CONCAT(p.pat_last_name, ', ', p.pat_first_name) AS formatted_name,
            DATE_FORMAT(p.birth_date, 'MM/dd/yyyy') AS formatted_birthdate
        FROM prod.clarity_cur.patient_enh p
        INNER JOIN latest_encounters e ON p.pat_id = e.pat_id
    ),

    -- Get account/admission info
    account_info AS (
        SELECT
            ha.hsp_account_id,
            ha.prim_enc_csn_id,
            f.facility_name,
            DATEDIFF(ha.disch_date_time, ha.adm_date_time) AS number_of_midnights,
            CONCAT(DATE_FORMAT(ha.adm_date_time, 'MM/dd/yyyy'), ' - ',
                   DATE_FORMAT(ha.disch_date_time, 'MM/dd/yyyy')) AS formatted_date_of_service
        FROM prod.clarity_cur.hsp_account_enh ha
        INNER JOIN target_accounts ta ON ha.hsp_account_id = ta.hsp_account_id
        LEFT JOIN prod.clarity.zc_loc_facility f ON ha.loc_id = f.facility_id
    ),

    -- Get claim info (for reference numbers)
    claim_info AS (
        SELECT
            hcd.hsp_account_id,
            FIRST(hcd.claim_number) AS claim_number,
            FIRST(hcd.tax_id) AS tax_id,
            FIRST(hcd.npi) AS npi
        FROM prod.clarity.hsp_claim_detail hcd
        INNER JOIN target_accounts ta ON hcd.hsp_account_id = ta.hsp_account_id
        GROUP BY hcd.hsp_account_id
    ),

    -- Get primary diagnosis (line 1)
    diagnosis AS (
        SELECT
            hadl.hsp_account_id,
            FIRST(edg.code) AS code,
            FIRST(edg.dx_name) AS dx_name
        FROM prod.clarity_cur.hsp_acct_dx_list_enh hadl
        INNER JOIN target_accounts ta ON hadl.hsp_account_id = ta.hsp_account_id
        INNER JOIN prod.clarity_cur.edg_current_icd10_enh edg ON hadl.dx_id = edg.dx_id
        WHERE hadl.line = 1  -- Primary diagnosis only
        GROUP BY hadl.hsp_account_id
    )

    -- Final SELECT: Join all CTEs together
    SELECT
        ta.hsp_account_id,
        pi.pat_id,
        pi.pat_mrn_id,
        pi.formatted_name,
        pi.formatted_birthdate,
        ai.facility_name,
        ai.number_of_midnights,
        ai.formatted_date_of_service,
        ci.claim_number,
        ci.tax_id,
        ci.npi,
        d.code,
        d.dx_name,
        CAST(dn.discharge_note_id AS STRING) AS discharge_summary_note_id,
        CAST(dn.discharge_note_csn_id AS STRING) AS discharge_note_csn_id,
        dn.discharge_summary_text,
        CAST(hp.hp_note_id AS STRING) AS hp_note_id,
        CAST(hp.hp_note_csn_id AS STRING) AS hp_note_csn_id,
        hp.hp_note_text
    FROM target_accounts ta
    LEFT JOIN patient_info pi ON ta.hsp_account_id = pi.hsp_account_id
    LEFT JOIN account_info ai ON ta.hsp_account_id = ai.hsp_account_id
    LEFT JOIN claim_info ci ON ta.hsp_account_id = ci.hsp_account_id
    LEFT JOIN diagnosis d ON ta.hsp_account_id = d.hsp_account_id
    LEFT JOIN discharge_notes dn ON ta.hsp_account_id = dn.hsp_account_id
    LEFT JOIN hp_notes hp ON ta.hsp_account_id = hp.hsp_account_id
    """

    # Execute query and convert to Pandas for easier manipulation
    clinical_df = spark.sql(clinical_query).toPandas()
    print(f"Retrieved clinical data for {len(clinical_df)} accounts")

    # -------------------------------------------------------------------------
    # Read denial letter files
    # -------------------------------------------------------------------------
    # For each account, read the associated denial letter PDF (or txt file).
    # Store the extracted text in a dictionary keyed by account ID.
    print("\nReading denial letters...")
    denial_texts = {}      # Maps account_id -> extracted text
    filename_map = {}      # Maps account_id -> filename (for traceability)

    for hsp_account_id, filename in TARGET_ACCOUNTS:
        file_path = os.path.join(DENIAL_LETTERS_PATH, filename)
        filename_map[hsp_account_id] = filename

        print(f"  Reading {filename}...")
        denial_texts[hsp_account_id] = read_document(file_path)
        print(f"    Extracted {len(denial_texts[hsp_account_id])} chars")

    # -------------------------------------------------------------------------
    # Add denial text and metadata to clinical data
    # -------------------------------------------------------------------------
    clinical_df['denial_letter_text'] = clinical_df['hsp_account_id'].map(denial_texts)
    clinical_df['denial_letter_filename'] = clinical_df['hsp_account_id'].map(filename_map)
    clinical_df['scope_filter'] = SCOPE_FILTER
    clinical_df['featurization_timestamp'] = datetime.now().isoformat()

    print(f"\nFeaturized dataset: {len(clinical_df)} rows")
    print(f"Columns: {list(clinical_df.columns)}")

    # Show preview of processed data
    print("\nPREVIEW:")
    print(clinical_df[['hsp_account_id', 'formatted_name', 'denial_letter_filename']].to_string())

    # -------------------------------------------------------------------------
    # Write to Delta table
    # -------------------------------------------------------------------------
    # Set WRITE_TO_TABLE = True when ready to persist data
    WRITE_TO_TABLE = False

    if WRITE_TO_TABLE:
        spark_df = spark.createDataFrame(clinical_df)
        spark_df = spark_df.withColumn("insert_tsp", current_timestamp())
        spark_df.write.mode("append").saveAsTable(INFERENCE_TABLE)
        print(f"\nWrote {len(clinical_df)} rows to {INFERENCE_TABLE}")
    else:
        print("\nTo write to table, set WRITE_TO_TABLE = True")

else:
    if len(TARGET_ACCOUNTS) == 0:
        print("\nNo target accounts configured. Update TARGET_ACCOUNTS list.")
    else:
        print("\nDenial featurization skipped (set RUN_DENIAL_FEATURIZATION = True)")

# =============================================================================
# CELL 12: Verify Tables
# =============================================================================
# Quick check of table status - useful for debugging
print("\n" + "="*60)
print("TABLE STATUS")
print("="*60)

try:
    gold_count = spark.sql(f"SELECT COUNT(*) as cnt FROM {GOLD_LETTERS_TABLE}").collect()[0]["cnt"]
    print(f"Gold letters table: {gold_count} records")
except Exception as e:
    print(f"Gold letters table: Error - {e}")

try:
    inference_count = spark.sql(f"SELECT COUNT(*) as cnt FROM {INFERENCE_TABLE}").collect()[0]["cnt"]
    print(f"Inference table: {inference_count} records")
except Exception as e:
    print(f"Inference table: Error - {e}")

print("\nFeaturization complete.")
