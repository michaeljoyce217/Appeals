# data/featurization.py
# Rebuttal Engine v2 - Featurization
#
# PARSER AGENT lives here - same parsing logic for:
# 1. Gold Standard Letters: Extract denial text + embedding
# 2. New Denials: Extract denial text + embedding
#
# This ensures apples-to-apples comparison in inference.py
#
# Run on Databricks Runtime 15.4 LTS ML

# =============================================================================
# CELL 1: Install Dependencies (run once per cluster)
# =============================================================================
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
    TimestampType, MapType, IntegerType
)

spark = SparkSession.builder.getOrCreate()

# Configuration
SCOPE_FILTER = "sepsis"
SEPSIS_DRG_CODES = ["870", "871", "872"]
EMBEDDING_MODEL = "text-embedding-ada-002"  # 1536 dimensions

# =============================================================================
# CELL 3: Environment Setup
# =============================================================================
trgt_cat = os.environ.get('trgt_cat', 'dev')
if trgt_cat == 'dev':
    spark.sql('USE CATALOG prod;')
else:
    spark.sql(f'USE CATALOG {trgt_cat};')

# Table names
GOLD_LETTERS_TABLE = f"{trgt_cat}.fin_ds.fudgesicle_gold_letters"
INFERENCE_TABLE = f"{trgt_cat}.fin_ds.fudgesicle_inference"

# Paths
GOLD_LETTERS_PATH = "/Workspace/Repos/mijo8881@mercy.net/fudgesicle/utils/gold_standard_rebuttals"
DENIAL_LETTERS_PATH = "/Workspace/Repos/mijo8881@mercy.net/fudgesicle/utils/Sample_Denial_Letters"

print(f"Catalog: {trgt_cat}")

# =============================================================================
# CELL 4: Azure Credentials
# =============================================================================
AZURE_OPENAI_KEY = dbutils.secrets.get(scope='idp_etl', key='az-openai-key1')
AZURE_OPENAI_ENDPOINT = dbutils.secrets.get(scope='idp_etl', key='az-openai-base')
AZURE_DOC_INTEL_KEY = dbutils.secrets.get(scope='idp_etl', key='az-aidcmntintel-key1')
AZURE_DOC_INTEL_ENDPOINT = dbutils.secrets.get(scope='idp_etl', key='az-aidcmntintel-endpoint')

print("Credentials loaded")

# =============================================================================
# CELL 5: Initialize Clients
# =============================================================================
from openai import AzureOpenAI
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential

openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2024-10-21"
)

doc_intel_client = DocumentIntelligenceClient(
    endpoint=AZURE_DOC_INTEL_ENDPOINT,
    credential=AzureKeyCredential(AZURE_DOC_INTEL_KEY)
)

print("Clients initialized")

# =============================================================================
# CELL 6: PARSER AGENT - Core Functions
# =============================================================================
# These functions are used for BOTH gold letters and new denials
# to ensure consistent parsing and embedding

def extract_text_from_pdf(file_path):
    """
    PARSER: Extract text from PDF using Document Intelligence layout model.
    Returns list of strings, one per page.
    """
    with open(file_path, 'rb') as f:
        document_bytes = f.read()

    poller = doc_intel_client.begin_analyze_document(
        model_id="prebuilt-layout",
        body=AnalyzeDocumentRequest(bytes_source=document_bytes),
    )
    result = poller.result()

    pages_text = []
    for page in result.pages:
        page_lines = [line.content for line in page.lines]
        pages_text.append("\n".join(page_lines))

    return pages_text


def identify_denial_start(pages_text):
    """
    PARSER: Identify which page the denial letter starts on.

    Detection strategy:
    1. Skip page 1 (that's always the rebuttal)
    2. Look for: insurer name + address pattern + date in first 10-15 lines
    3. The denial is typically near the MIDDLE of the PDF (attached after rebuttal)

    Returns (denial_start_page, payor_name) - 1-indexed page number.
    """
    import re

    # Insurance company names
    payor_patterns = [
        ("unitedhealth", "UnitedHealthcare"),
        ("united health", "UnitedHealthcare"),
        ("uhc", "UnitedHealthcare"),
        ("optum", "UnitedHealthcare"),
        ("aetna", "Aetna"),
        ("cigna", "Cigna"),
        ("evernorth", "Cigna"),
        ("humana", "Humana"),
        ("anthem", "Anthem"),
        ("elevance", "Anthem"),
        ("blue cross", "Blue Cross Blue Shield"),
        ("blue shield", "Blue Cross Blue Shield"),
        ("bcbs", "Blue Cross Blue Shield"),
        ("wellpoint", "Anthem"),
        ("kaiser", "Kaiser Permanente"),
        ("molina", "Molina Healthcare"),
        ("centene", "Centene"),
        ("ambetter", "Centene"),
        ("wellcare", "Centene"),
        ("healthnet", "Health Net"),
        ("medicaid", "Medicaid"),
        ("medicare", "Medicare"),
    ]

    # Patterns for address and date (signs of a formal business letter header)
    # Address: look for state abbreviations + zip codes
    address_pattern = re.compile(
        r'\b(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)\s*\d{5}',
        re.IGNORECASE
    )
    # Date: MM/DD/YYYY or Month DD, YYYY
    date_pattern = re.compile(
        r'(\d{1,2}/\d{1,2}/\d{4})|((January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})',
        re.IGNORECASE
    )

    # Skip page 1 - that's the rebuttal. Start from page 2.
    for i, page_text in enumerate(pages_text):
        if i == 0:
            continue  # Skip first page (rebuttal starts there)

        # Check first 15 lines (letterhead + address block area)
        first_lines = page_text.split("\n")[:15]
        header_text = "\n".join(first_lines)
        header_lower = header_text.lower()

        # Look for insurer name in header
        found_payor = None
        for pattern, payor_name in payor_patterns:
            if pattern in header_lower:
                found_payor = payor_name
                break

        if found_payor:
            # Verify it looks like a letter header (has address OR date)
            has_address = bool(address_pattern.search(header_text))
            has_date = bool(date_pattern.search(header_text))

            if has_address or has_date:
                print(f"    Found denial: '{found_payor}' + {'address' if has_address else 'date'} on page {i+1}")
                return i + 1, found_payor  # 1-indexed
            else:
                # Found payor name but no address/date - might still be it
                print(f"    Found '{found_payor}' on page {i+1} (no address/date pattern)")
                return i + 1, found_payor

    # Fallback: look for typical denial letter subject lines
    denial_subject_phrases = [
        "claim review determination",
        "medical necessity review",
        "utilization review",
        "peer review determination",
        "clinical review",
        "prior authorization",
        "re: claim",
        "re: member",
        "re: patient",
    ]

    for i, page_text in enumerate(pages_text):
        if i == 0:
            continue  # Skip first page

        first_lines = page_text.split("\n")[:15]
        header_lower = "\n".join(first_lines).lower()

        for phrase in denial_subject_phrases:
            if phrase in header_lower:
                print(f"    Found denial phrase '{phrase}' on page {i+1}")
                return i + 1, "Unknown"

    # Default: no denial found - all pages are rebuttal
    print("    WARNING: Could not identify denial start page")
    return len(pages_text) + 1, "Unknown"


def generate_embedding(text):
    """
    PARSER: Generate embedding vector for text using Azure OpenAI.
    Returns 1536-dimensional vector.
    """
    # Truncate if too long (ada-002 limit ~8k tokens, ~30k chars safe)
    if len(text) > 30000:
        text = text[:30000]

    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def parse_denial_pdf(file_path):
    """
    PARSER: Full parsing pipeline for a denial letter PDF.
    Returns dict with: denial_text, denial_embedding, payor
    """
    # Step 1: Extract text by page
    pages = extract_text_from_pdf(file_path)
    full_text = "\n\n".join(pages)

    # Step 2: Generate embedding of the denial
    embedding = generate_embedding(full_text)

    # Step 3: Try to identify payor from text
    _, payor = identify_denial_start(pages)

    return {
        "denial_text": full_text,
        "denial_embedding": embedding,
        "payor": payor,
        "page_count": len(pages)
    }


def parse_gold_letter_pdf(file_path):
    """
    PARSER: Full parsing pipeline for a gold standard letter PDF.
    Gold letters contain BOTH rebuttal (first) and denial (end).
    Returns dict with: rebuttal_text, denial_text, denial_embedding, payor
    """
    # Step 1: Extract text by page
    pages = extract_text_from_pdf(file_path)

    # Step 2: Identify where denial starts (payor letterhead is giveaway)
    denial_start_page, payor = identify_denial_start(pages)

    # Step 3: Split into rebuttal and denial
    rebuttal_pages = pages[:denial_start_page - 1]
    denial_pages = pages[denial_start_page - 1:]

    rebuttal_text = "\n\n".join(rebuttal_pages) if rebuttal_pages else ""
    denial_text = "\n\n".join(denial_pages) if denial_pages else ""

    # Step 4: Generate embedding of the DENIAL (for matching)
    denial_embedding = generate_embedding(denial_text) if denial_text else None

    return {
        "rebuttal_text": rebuttal_text,
        "denial_text": denial_text,
        "denial_embedding": denial_embedding,
        "payor": payor,
        "denial_start_page": denial_start_page,
        "total_pages": len(pages)
    }


print("Parser Agent functions loaded")

# =============================================================================
# CELL 7: Create Tables
# =============================================================================
# Gold Letters Table - stores past winning rebuttals with denial embeddings
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
"""
spark.sql(create_gold_table_sql)
print(f"Table {GOLD_LETTERS_TABLE} ready")

# Inference Table - includes denial_embedding for rigorous comparison
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
    denial_embedding ARRAY<FLOAT>,
    payor STRING,
    scope_filter STRING,
    featurization_timestamp STRING,
    insert_tsp TIMESTAMP
)
USING DELTA
"""
spark.sql(create_inference_table_sql)
print(f"Table {INFERENCE_TABLE} ready")

# #############################################################################
# PART 1: GOLD STANDARD LETTER INGESTION
# Parser extracts: rebuttal_text, denial_text, denial_embedding, payor
# #############################################################################

# =============================================================================
# CELL 8: Process Gold Standard Letters
# =============================================================================
RUN_GOLD_INGESTION = False

if RUN_GOLD_INGESTION:
    print("="*60)
    print("GOLD STANDARD LETTER INGESTION (Parser Agent)")
    print("="*60)

    pdf_files = [f for f in os.listdir(GOLD_LETTERS_PATH) if f.lower().endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDF files")

    gold_records = []

    for i, pdf_file in enumerate(pdf_files):
        print(f"\nProcessing {i+1}/{len(pdf_files)}: {pdf_file}")
        file_path = os.path.join(GOLD_LETTERS_PATH, pdf_file)

        try:
            # Use Parser Agent to extract everything
            parsed = parse_gold_letter_pdf(file_path)

            print(f"  Pages: {parsed['total_pages']}, Denial starts: page {parsed['denial_start_page']}")
            print(f"  Rebuttal: {len(parsed['rebuttal_text'])} chars")
            print(f"  Denial: {len(parsed['denial_text'])} chars")
            print(f"  Payor: {parsed['payor']}")
            print(f"  Embedding: {len(parsed['denial_embedding'])} dims")

            record = {
                "letter_id": str(uuid.uuid4()),
                "source_file": pdf_file,
                "payor": parsed["payor"],
                "denial_text": parsed["denial_text"],
                "rebuttal_text": parsed["rebuttal_text"],
                "denial_embedding": parsed["denial_embedding"],
                "created_at": datetime.now(),
                "metadata": {
                    "denial_start_page": str(parsed["denial_start_page"]),
                    "total_pages": str(parsed["total_pages"])
                },
            }
            gold_records.append(record)
            print(f"  Record created: {record['letter_id']}")

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    print(f"\nProcessed {len(gold_records)} gold standard letters")

    if gold_records:
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

        gold_df = spark.createDataFrame(gold_records, schema)
        gold_df.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable(GOLD_LETTERS_TABLE)
        print(f"Wrote {len(gold_records)} records to {GOLD_LETTERS_TABLE}")

    count = spark.sql(f"SELECT COUNT(*) as cnt FROM {GOLD_LETTERS_TABLE}").collect()[0]["cnt"]
    print(f"Total records in gold letters table: {count}")

else:
    print("Gold ingestion skipped (set RUN_GOLD_INGESTION = True)")

# #############################################################################
# PART 2: NEW DENIAL FEATURIZATION
# Parser extracts: denial_text, denial_embedding, payor
# Combined with clinical data from Clarity
# #############################################################################

# =============================================================================
# CELL 9: Parse Denial Letters (LLM extracts HSP_ACCOUNT_ID + Payor)
# =============================================================================
DENIAL_PARSER_PROMPT = '''Extract the hospital account ID and insurance payor from this denial letter.

# Denial Letter Text (first pages)
{denial_text}

# Instructions
Find the HOSPITAL ACCOUNT ID (may be labeled as Account #, Hospital Account, HSP Account, Acct, etc.)
Find the INSURANCE PAYOR (the insurance company that sent this denial)

Return ONLY valid JSON:
{{
    "hsp_account_id": "the account number or null if not found",
    "payor": "insurance company name or Unknown"
}}'''


def transform_hsp_account_id(raw_id):
    """
    Transform HSP_ACCOUNT_ID from denial letter format to Clarity format.
    Denial letters often have: H1234567890
    Clarity needs: 12345678 (remove H prefix, remove last 2 digits)
    """
    if not raw_id:
        return None

    # Remove any whitespace
    cleaned = str(raw_id).strip()

    # Remove H prefix if present
    if cleaned.upper().startswith('H'):
        cleaned = cleaned[1:]

    # Remove last 2 digits to match Clarity
    if len(cleaned) > 2:
        cleaned = cleaned[:-2]

    # Return only if we have digits left
    if cleaned and cleaned.isdigit():
        return cleaned

    return None


def extract_denial_info_llm(text):
    """
    Use LLM to extract HSP_ACCOUNT_ID and payor from denial letter.
    More flexible than regex - handles various formats.
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "Extract information accurately. Return only valid JSON."},
                {"role": "user", "content": DENIAL_PARSER_PROMPT.format(denial_text=text[:15000])}
            ],
            temperature=0,
            max_tokens=200
        )

        raw = response.choices[0].message.content.strip()
        # Handle markdown code blocks
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        result = json.loads(raw)
        raw_account_id = result.get("hsp_account_id")
        payor = result.get("payor", "Unknown")

        # Transform account ID to match Clarity format
        transformed_id = transform_hsp_account_id(raw_account_id)

        if raw_account_id and transformed_id:
            print(f"    Account ID: {raw_account_id} → {transformed_id} (transformed)")

        return transformed_id, payor

    except Exception as e:
        print(f"    LLM extraction error: {e}")
        return None, "Unknown"


def process_sample_denial_letters():
    """
    Process all denial letters in Sample_Denial_Letters folder:
    1. Extract text using Document Intelligence
    2. Use LLM to extract HSP_ACCOUNT_ID and payor
    3. Generate embedding
    Returns list of dicts with all extracted info.
    """
    results = []

    if not os.path.exists(DENIAL_LETTERS_PATH):
        print(f"WARNING: Denial letters path not found: {DENIAL_LETTERS_PATH}")
        return results

    pdf_files = [f for f in os.listdir(DENIAL_LETTERS_PATH) if f.lower().endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDF files in Sample_Denial_Letters")

    for i, pdf_file in enumerate(pdf_files):
        print(f"\n[{i+1}/{len(pdf_files)}] Processing {pdf_file}")
        file_path = os.path.join(DENIAL_LETTERS_PATH, pdf_file)

        try:
            # Step 1: Extract text
            pages = extract_text_from_pdf(file_path)
            full_text = "\n\n".join(pages)
            print(f"  Extracted {len(pages)} pages, {len(full_text)} chars")

            # Step 2: LLM extracts account ID and payor
            search_text = "\n".join(pages[:3]) if len(pages) >= 3 else full_text
            hsp_account_id, payor = extract_denial_info_llm(search_text)
            print(f"  Account ID: {hsp_account_id or 'NOT FOUND'}")
            print(f"  Payor: {payor}")

            # Step 3: Generate embedding
            embedding = generate_embedding(full_text)
            print(f"  Embedding: {len(embedding)} dims")

            results.append({
                "filename": pdf_file,
                "hsp_account_id": hsp_account_id,
                "payor": payor,
                "denial_text": full_text,
                "denial_embedding": embedding,
                "page_count": len(pages),
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Summary
    found = sum(1 for r in results if r["hsp_account_id"])
    print(f"\n{'='*60}")
    print(f"SUMMARY: {found}/{len(pdf_files)} letters have HSP_ACCOUNT_ID")
    print(f"{'='*60}")

    return results


# Process denial letters
RUN_DENIAL_PROCESSING = False

if RUN_DENIAL_PROCESSING:
    print("\n" + "="*60)
    print("PROCESSING SAMPLE DENIAL LETTERS")
    print("="*60)
    DENIAL_RESULTS = process_sample_denial_letters()
else:
    DENIAL_RESULTS = []
    print("\nDenial processing skipped (set RUN_DENIAL_PROCESSING = True)")

# Build TARGET_ACCOUNTS from results (only those with account IDs)
TARGET_ACCOUNTS = [
    (r["hsp_account_id"], r["filename"])
    for r in DENIAL_RESULTS
    if r["hsp_account_id"]
]

print(f"\nTarget accounts for Clarity join: {len(TARGET_ACCOUNTS)}")

# =============================================================================
# CELL 10: Process New Denials
# =============================================================================
RUN_DENIAL_FEATURIZATION = False

if RUN_DENIAL_FEATURIZATION and len(TARGET_ACCOUNTS) > 0:
    print("="*60)
    print("NEW DENIAL FEATURIZATION (Parser Agent)")
    print("="*60)

    account_ids = [a[0] for a in TARGET_ACCOUNTS]
    account_values = ", ".join([f"('{acc}')" for acc in account_ids])

    # Query clinical data from Clarity
    # Based on original working prototype query structure
    print("\nQuerying clinical data...")
    clinical_query = f"""
    WITH target_accounts AS (
        SELECT hsp_account_id
        FROM (VALUES {account_values}) AS t(hsp_account_id)
    ),

    -- Get notes using ip_note_type (original working approach)
    notes AS (
        SELECT * FROM (
            SELECT peh.pat_id
                  ,peh.hsp_account_id
                  ,nte.ip_note_type
                  ,nte.note_id
                  ,nte.note_csn_id
                  ,nte.contact_date AS note_contact_date
                  ,nte.ent_inst_local_dttm AS entry_datetime
                  ,CONCAT_WS('\\n', SORT_ARRAY(COLLECT_LIST(STRUCT(nte.line, nte.note_text))).note_text) AS note_text
            FROM clarity_cur.pat_enc_hsp_har_enh peh
            INNER JOIN target_accounts ta ON peh.hsp_account_id = ta.hsp_account_id
            INNER JOIN clarity_cur.hno_note_text_enh nte USING(pat_enc_csn_id)
            WHERE nte.ip_note_type IN ('Discharge Summary', 'H&P')
            GROUP BY peh.pat_id, peh.hsp_account_id, nte.ip_note_type, nte.note_id, nte.note_csn_id, nte.contact_date, nte.ent_inst_local_dttm
        )
        QUALIFY ROW_NUMBER() OVER (PARTITION BY hsp_account_id, ip_note_type ORDER BY note_contact_date DESC, entry_datetime DESC) = 1
    ),

    hp_note AS (
        SELECT hsp_account_id, pat_id,
               note_csn_id AS hp_note_csn_id,
               note_id AS hp_note_id,
               note_text AS hp_note_text
        FROM notes WHERE ip_note_type = 'H&P'
    ),

    discharge AS (
        SELECT hsp_account_id, pat_id,
               note_csn_id AS discharge_note_csn_id,
               note_id AS discharge_summary_note_id,
               note_text AS discharge_summary_text
        FROM notes WHERE ip_note_type = 'Discharge Summary'
    ),

    patient_info AS (
        SELECT ha.hsp_account_id, patient.pat_id, patient.pat_mrn_id,
               CONCAT(patient.pat_first_name, ' ', patient.pat_last_name) AS formatted_name,
               DATE_FORMAT(patient.birth_date, 'MM/dd/yyyy') AS formatted_birthdate,
               'Mercy Hospital' AS facility_name,
               DATEDIFF(ha.disch_date_time, DATE(ha.adm_date_time)) AS number_of_midnights,
               CONCAT(DATE_FORMAT(ha.adm_date_time, 'MM/dd/yyyy'), '-', DATE_FORMAT(ha.disch_date_time, 'MM/dd/yyyy')) AS formatted_date_of_service
        FROM clarity_cur.hsp_account_enh ha
        INNER JOIN target_accounts ta ON ha.hsp_account_id = ta.hsp_account_id
        INNER JOIN clarity_cur.patient_enh patient ON ha.pat_id = patient.pat_id
    )

    SELECT pi.hsp_account_id, pi.pat_id, pi.pat_mrn_id,
           pi.formatted_name, pi.formatted_birthdate,
           pi.facility_name, pi.number_of_midnights, pi.formatted_date_of_service,
           COALESCE(d.discharge_summary_note_id, 'no id available') AS discharge_summary_note_id,
           COALESCE(d.discharge_note_csn_id, 'no id available') AS discharge_note_csn_id,
           COALESCE(d.discharge_summary_text, 'No Note Available') AS discharge_summary_text,
           COALESCE(h.hp_note_id, 'no id available') AS hp_note_id,
           COALESCE(h.hp_note_csn_id, 'no id available') AS hp_note_csn_id,
           COALESCE(h.hp_note_text, 'No Note Available') AS hp_note_text
    FROM patient_info pi
    LEFT JOIN discharge d ON pi.hsp_account_id = d.hsp_account_id
    LEFT JOIN hp_note h ON pi.hsp_account_id = h.hsp_account_id
    """

    clinical_df = spark.sql(clinical_query).toPandas()
    print(f"Retrieved {len(clinical_df)} accounts from Clarity")

    # Use pre-parsed denial data from DENIAL_RESULTS (already has text, embedding, payor)
    print("\nJoining with pre-parsed denial data...")
    denial_lookup = {r["hsp_account_id"]: r for r in DENIAL_RESULTS if r["hsp_account_id"]}

    clinical_df['denial_letter_text'] = clinical_df['hsp_account_id'].map(
        lambda x: denial_lookup.get(x, {}).get('denial_text'))
    clinical_df['denial_letter_filename'] = clinical_df['hsp_account_id'].map(
        lambda x: denial_lookup.get(x, {}).get('filename'))
    clinical_df['denial_embedding'] = clinical_df['hsp_account_id'].map(
        lambda x: denial_lookup.get(x, {}).get('denial_embedding'))
    clinical_df['payor'] = clinical_df['hsp_account_id'].map(
        lambda x: denial_lookup.get(x, {}).get('payor'))
    clinical_df['scope_filter'] = SCOPE_FILTER
    clinical_df['featurization_timestamp'] = datetime.now().isoformat()

    print(f"\nFeaturized {len(clinical_df)} rows with embeddings")

    # Write to table
    WRITE_TO_TABLE = False

    if WRITE_TO_TABLE:
        spark_df = spark.createDataFrame(clinical_df)
        spark_df = spark_df.withColumn("insert_tsp", current_timestamp())
        spark_df.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable(INFERENCE_TABLE)
        print(f"Wrote {len(clinical_df)} rows to {INFERENCE_TABLE}")
    else:
        print("To write, set WRITE_TO_TABLE = True")

else:
    print("Denial featurization skipped")

# =============================================================================
# CELL 11: Verify Tables
# =============================================================================
print("\n" + "="*60)
print("TABLE STATUS")
print("="*60)

try:
    gold_count = spark.sql(f"SELECT COUNT(*) as cnt FROM {GOLD_LETTERS_TABLE}").collect()[0]["cnt"]
    print(f"Gold letters: {gold_count} records")
except Exception as e:
    print(f"Gold letters: {e}")

try:
    inf_count = spark.sql(f"SELECT COUNT(*) as cnt FROM {INFERENCE_TABLE}").collect()[0]["cnt"]
    print(f"Inference: {inf_count} records")
except Exception as e:
    print(f"Inference: {e}")

print("\nFeaturization complete.")
