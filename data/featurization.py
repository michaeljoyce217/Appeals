# data/featurization.py
# Appeal Engine v2 - Featurization
#
# PARSER AGENT lives here - same parsing logic for:
# 1. Gold Standard Letters: Extract denial text + embedding
# 2. New Denials: Extract denial text + embedding
#
# This ensures apples-to-apples comparison in inference.py
#
# Run on Databricks Runtime 15.4 LTS ML

# =============================================================================
# CELL 1: Install Dependencies (run this cell FIRST, then restart)
# =============================================================================
# IMPORTANT: Run this cell by itself, then run the rest of the notebook.
# After restart, the packages persist for the cluster session.
#
# Uncomment and run ONCE per cluster session:
# %pip install azure-ai-documentintelligence==1.0.2 openai python-docx
# dbutils.library.restartPython()
#
# After restart completes, run Cell 2 onwards (leave this cell commented)

# =============================================================================
# CELL 2: Imports and Configuration
# =============================================================================
import os
import json
import uuid
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, lit, col
from pyspark.sql.types import (
    StructType, StructField, StringType, ArrayType, FloatType,
    TimestampType, MapType, IntegerType, BooleanType
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
PROPEL_DATA_TABLE = f"{trgt_cat}.fin_ds.fudgesicle_propel_data"

# Paths
GOLD_LETTERS_PATH = "/Workspace/Repos/mijo8881@mercy.net/fudgesicle/utils/gold_standard_appeals"
DENIAL_LETTERS_PATH = "/Workspace/Repos/mijo8881@mercy.net/fudgesicle/utils/sample_denial_letters"
PROPEL_DATA_PATH = "/Workspace/Repos/mijo8881@mercy.net/fudgesicle/utils/propel_data"

print(f"Catalog: {trgt_cat}")

# =============================================================================
# CELL 3B: Validation Checkpoints
# =============================================================================
# These checkpoints verify that each step has behaved as expected.

def checkpoint_paths():
    """
    CHECKPOINT: Verify all required paths exist and contain expected files.
    Call this before running any ingestion.
    """
    print("\n" + "="*60)
    print("CHECKPOINT: Path Validation")
    print("="*60)

    all_valid = True

    # Check gold_standard_appeals
    if os.path.exists(GOLD_LETTERS_PATH):
        pdf_count = len([f for f in os.listdir(GOLD_LETTERS_PATH) if f.lower().endswith('.pdf')])
        docx_count = len([f for f in os.listdir(GOLD_LETTERS_PATH) if f.lower().endswith('.docx')])
        print(f"[OK] gold_standard_appeals: {pdf_count} PDFs, {docx_count} DOCX files")
        if pdf_count == 0:
            print("  [WARN] No PDF files found - gold letter ingestion will have nothing to process")
    else:
        print(f"[FAIL] gold_standard_appeals not found: {GOLD_LETTERS_PATH}")
        all_valid = False

    # Check sample_denial_letters
    if os.path.exists(DENIAL_LETTERS_PATH):
        pdf_count = len([f for f in os.listdir(DENIAL_LETTERS_PATH) if f.lower().endswith('.pdf')])
        print(f"[OK] sample_denial_letters: {pdf_count} PDFs")
        if pdf_count == 0:
            print("  [WARN] No PDF files found - denial processing will have nothing to process")
    else:
        print(f"[FAIL] sample_denial_letters not found: {DENIAL_LETTERS_PATH}")
        all_valid = False

    # Check propel_data
    if os.path.exists(PROPEL_DATA_PATH):
        pdf_count = len([f for f in os.listdir(PROPEL_DATA_PATH) if f.lower().endswith('.pdf')])
        print(f"[OK] propel_data: {pdf_count} PDFs")
        if pdf_count == 0:
            print("  [WARN] No PDF files found - propel ingestion will have nothing to process")
    else:
        print(f"[FAIL] propel_data not found: {PROPEL_DATA_PATH}")
        all_valid = False

    if all_valid:
        print("\n[CHECKPOINT PASSED] All paths valid")
    else:
        print("\n[CHECKPOINT FAILED] Fix path issues before proceeding")

    return all_valid


def checkpoint_table(table_name, expected_columns=None, min_rows=0):
    """
    CHECKPOINT: Verify a table exists and has expected structure.

    Args:
        table_name: Full table name (catalog.schema.table)
        expected_columns: List of column names that must exist
        min_rows: Minimum number of rows expected
    """
    print(f"\nCHECKPOINT: Validating {table_name}")

    try:
        # Check table exists and get count
        count = spark.sql(f"SELECT COUNT(*) as cnt FROM {table_name}").collect()[0]["cnt"]
        print(f"  Rows: {count}")

        if count < min_rows:
            print(f"  [WARN] Expected at least {min_rows} rows, found {count}")
            return False

        # Check columns if specified
        if expected_columns:
            df = spark.sql(f"SELECT * FROM {table_name} LIMIT 1")
            actual_columns = set(df.columns)
            missing = set(expected_columns) - actual_columns
            if missing:
                print(f"  [FAIL] Missing columns: {missing}")
                return False
            print(f"  [OK] All expected columns present")

        print(f"  [CHECKPOINT PASSED]")
        return True

    except Exception as e:
        print(f"  [FAIL] Could not validate table: {e}")
        return False


def checkpoint_gold_letters():
    """CHECKPOINT: Validate gold letters table after ingestion."""
    return checkpoint_table(
        GOLD_LETTERS_TABLE,
        expected_columns=["letter_id", "source_file", "payor", "denial_text", "rebuttal_text", "denial_embedding"],
        min_rows=1
    )


def checkpoint_propel_data():
    """CHECKPOINT: Validate propel data table after ingestion."""
    return checkpoint_table(
        PROPEL_DATA_TABLE,
        expected_columns=["condition_name", "source_file", "definition_text", "definition_summary"],
        min_rows=1
    )


def checkpoint_inference_table():
    """CHECKPOINT: Validate inference table after denial processing."""
    return checkpoint_table(
        INFERENCE_TABLE,
        expected_columns=["hsp_account_id", "denial_letter_text", "denial_embedding", "payor", "is_sepsis"],
        min_rows=1
    )


# Run path checkpoint on load
checkpoint_paths()

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
    1. Skip page 1 (that's always the appeal)
    2. Look for: insurer name + address pattern + date in first 10-15 lines
    3. The denial is typically near the MIDDLE of the PDF (attached after appeal)

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

    # Skip page 1 - that's the appeal. Start from page 2.
    for i, page_text in enumerate(pages_text):
        if i == 0:
            continue  # Skip first page (appeal starts there)

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

    # Default: no denial found - all pages are appeal
    print("    WARNING: Could not identify denial start page")
    return len(pages_text) + 1, "Unknown"


def generate_embedding(text):
    """
    PARSER: Generate embedding vector for text using Azure OpenAI.
    Returns 1536-dimensional vector.
    """
    # Truncate if too long (ada-002 limit is 8192 tokens)
    # Medical text averages ~3-4 chars/token, so 20k chars is safe
    if len(text) > 20000:
        text = text[:20000]

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
    Gold letters contain BOTH appeal (first) and denial (end).
    Returns dict with: rebuttal_text (appeal), denial_text, denial_embedding, payor
    """
    # Step 1: Extract text by page
    pages = extract_text_from_pdf(file_path)

    # Step 2: Identify where denial starts (payor letterhead is giveaway)
    denial_start_page, payor = identify_denial_start(pages)

    # Step 3: Split into appeal and denial
    rebuttal_pages = pages[:denial_start_page - 1]  # rebuttal_text column name kept for schema compatibility
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


def extract_text_from_docx(file_path):
    """
    PARSER: Extract text from DOCX file.
    Returns the full text content.
    """
    from docx import Document
    doc = Document(file_path)
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n\n".join(paragraphs)


print("Parser Agent functions loaded")

# =============================================================================
# CELL 7: Create Tables
# =============================================================================
# Gold Letters Table - stores past winning appeals with denial embeddings
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

# Inference Table - DO NOT pre-create
# Let Spark create it from DataFrame schema to avoid type mismatches
# (Python floats -> ARRAY<DOUBLE>, not ARRAY<FLOAT>)
print(f"Inference table {INFERENCE_TABLE} will be created on first write")

# Propel Data Table - stores official clinical definitions
# definition_text = full original text from DOCX
# definition_summary = LLM-extracted key criteria for prompts (shorter, focused)
create_propel_table_sql = f"""
CREATE TABLE IF NOT EXISTS {PROPEL_DATA_TABLE} (
    condition_name STRING NOT NULL,
    source_file STRING NOT NULL,
    definition_text STRING,
    definition_summary STRING,
    created_at TIMESTAMP
)
USING DELTA
COMMENT 'Official Propel clinical definitions for conditions'
"""
spark.sql(create_propel_table_sql)
print(f"Table {PROPEL_DATA_TABLE} ready")

# #############################################################################
# PART 1: GOLD STANDARD LETTER INGESTION
# Parser extracts: rebuttal_text (appeal), denial_text, denial_embedding, payor
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
            print(f"  Appeal: {len(parsed['rebuttal_text'])} chars")
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
        gold_df.write.format("delta").mode("overwrite").saveAsTable(GOLD_LETTERS_TABLE)
        print(f"Wrote {len(gold_records)} records to {GOLD_LETTERS_TABLE}")

    count = spark.sql(f"SELECT COUNT(*) as cnt FROM {GOLD_LETTERS_TABLE}").collect()[0]["cnt"]
    print(f"Total records in gold letters table: {count}")

    # Validate gold letters table
    checkpoint_gold_letters()

else:
    print("Gold ingestion skipped (set RUN_GOLD_INGESTION = True)")

# #############################################################################
# PART 1B: PROPEL DATA INGESTION
# Official clinical definitions for conditions (sepsis, etc.)
# #############################################################################

# =============================================================================
# CELL 8B-1: Propel Extraction Prompt and Function
# =============================================================================
PROPEL_EXTRACTION_PROMPT = '''You are extracting the key clinical criteria from an official Propel definition document.

Your task: Extract ONLY the essential diagnostic criteria that a physician would use to determine if a patient meets the clinical definition.

OUTPUT FORMAT:
- Use clear, concise bullet points
- Include specific thresholds, values, and timeframes
- Preserve medical terminology exactly as written
- Include ALL required criteria (do not summarize away important details)
- Remove administrative content, references, and background explanations
- Target length: 500-1000 words (enough to capture all criteria, short enough for an LLM prompt)

CONDITION: {condition_name}

FULL PROPEL DOCUMENT:
{definition_text}

---

Extract the key clinical criteria below:'''


def extract_propel_summary(condition_name, definition_text):
    """
    Extract key clinical criteria from a Propel definition using LLM.
    Returns a concise summary suitable for inclusion in prompts.
    """
    if not definition_text or len(definition_text) < 100:
        return definition_text  # Too short to summarize

    prompt = PROPEL_EXTRACTION_PROMPT.format(
        condition_name=condition_name.title(),
        definition_text=definition_text
    )

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2000,
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        print(f"  Warning: Extraction failed ({e}), using full text")
        return definition_text


# =============================================================================
# CELL 8B-2: Process Propel Data Files
# =============================================================================
RUN_PROPEL_INGESTION = False

if RUN_PROPEL_INGESTION:
    print("\n" + "="*60)
    print("PROPEL DATA INGESTION")
    print("="*60)

    if os.path.exists(PROPEL_DATA_PATH):
        # Propel files are PDFs named like "propel_sepsis.pdf"
        pdf_files = [f for f in os.listdir(PROPEL_DATA_PATH) if f.lower().endswith('.pdf')]
        print(f"Found {len(pdf_files)} PDF files in propel_data")

        propel_records = []

        for i, pdf_file in enumerate(pdf_files):
            print(f"\n[{i+1}/{len(pdf_files)}] Processing {pdf_file}")
            file_path = os.path.join(PROPEL_DATA_PATH, pdf_file)

            try:
                # Extract text from PDF using Document Intelligence
                pages_text = extract_text_from_pdf(file_path)
                definition_text = "\n\n".join(pages_text)
                print(f"  Extracted {len(definition_text)} chars from {len(pages_text)} pages")

                # Derive condition name from filename
                # "propel_sepsis.pdf" -> "sepsis"
                # "sepsis.pdf" -> "sepsis"
                base_name = os.path.splitext(pdf_file)[0].lower()
                if base_name.startswith("propel_"):
                    condition_name = base_name[7:]  # Remove "propel_" prefix
                else:
                    condition_name = base_name
                print(f"  Condition: {condition_name}")

                # Extract key criteria summary using LLM
                print(f"  Extracting key criteria via LLM...")
                definition_summary = extract_propel_summary(condition_name, definition_text)
                print(f"  Summary: {len(definition_summary)} chars")

                propel_records.append({
                    "condition_name": condition_name,
                    "source_file": pdf_file,
                    "definition_text": definition_text,
                    "definition_summary": definition_summary,
                    "created_at": datetime.now(),
                })

            except Exception as e:
                print(f"  ERROR: {e}")
                continue

        if propel_records:
            propel_df = spark.createDataFrame(propel_records)
            propel_df.write.format("delta").mode("overwrite").saveAsTable(PROPEL_DATA_TABLE)
            print(f"\nWrote {len(propel_records)} records to {PROPEL_DATA_TABLE}")

        count = spark.sql(f"SELECT COUNT(*) as cnt FROM {PROPEL_DATA_TABLE}").collect()[0]["cnt"]
        print(f"Total records in propel data table: {count}")

        # Validate propel data table
        checkpoint_propel_data()
    else:
        print(f"WARNING: Propel data path not found: {PROPEL_DATA_PATH}")

else:
    print("\nPropel ingestion skipped (set RUN_PROPEL_INGESTION = True)")

# #############################################################################
# PART 2: NEW DENIAL FEATURIZATION
# Parser extracts: denial_text, denial_embedding, payor
# Combined with clinical data from Clarity
# #############################################################################

# =============================================================================
# CELL 9: Parse Denial Letters (LLM extracts HSP_ACCOUNT_ID + Payor)
# =============================================================================
DENIAL_PARSER_PROMPT = '''Extract key information from this denial letter.

# Denial Letter Text
{denial_text}

# Instructions
Find:
1. HOSPITAL ACCOUNT ID - starts with "H" followed by digits (e.g., H1234567890)
2. INSURANCE PAYOR - the company that sent this denial
3. ORIGINAL DRG - the DRG code the hospital billed (e.g., 871)
4. PROPOSED DRG - the DRG the payor wants to change it to (e.g., 872)
5. IS SEPSIS RELATED - does this denial involve sepsis, severe sepsis, or septic shock?

Return ONLY these lines (no JSON):
ACCOUNT_ID: [H-prefixed number or NONE]
PAYOR: [insurance company name]
ORIGINAL_DRG: [3-digit code or NONE]
PROPOSED_DRG: [3-digit code or NONE]
IS_SEPSIS: [YES or NO]'''


def transform_hsp_account_id(raw_id):
    """
    Transform HSP_ACCOUNT_ID from denial letter format to Clarity format.
    Denial letters have: H1234567890
    Clarity needs: 12345678 (remove H prefix, remove last 2 digits)

    Only processes IDs that start with H - skips others.
    """
    if not raw_id:
        return None

    # Remove any whitespace
    cleaned = str(raw_id).strip()

    # MUST start with H - skip if not
    if not cleaned.upper().startswith('H'):
        print(f"    Skipping non-H account ID: {cleaned}")
        return None

    # Remove H prefix
    cleaned = cleaned[1:]

    # Remove last 2 digits to match Clarity
    if len(cleaned) > 2:
        cleaned = cleaned[:-2]
    else:
        print(f"    Account ID too short after removing H: {raw_id}")
        return None

    # Return only if we have digits left
    if cleaned and cleaned.isdigit():
        return cleaned

    return None


def extract_denial_info_llm(text):
    """
    Use LLM to extract all denial info using simple key-value format.
    Returns dict with: hsp_account_id, payor, original_drg, proposed_drg, is_sepsis
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "Extract information accurately. Return only the requested format."},
                {"role": "user", "content": DENIAL_PARSER_PROMPT.format(denial_text=text[:15000])}
            ],
            temperature=0,
            max_tokens=200
        )

        raw = response.choices[0].message.content.strip()

        # Parse key-value format
        result = {
            "hsp_account_id": None,
            "payor": "Unknown",
            "original_drg": None,
            "proposed_drg": None,
            "is_sepsis": False
        }

        for line in raw.split("\n"):
            line = line.strip()
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().upper()
            value = value.strip()

            if key == "ACCOUNT_ID":
                if value and value.upper() != "NONE":
                    result["hsp_account_id"] = transform_hsp_account_id(value)
                    if result["hsp_account_id"]:
                        print(f"    Account ID: {value} → {result['hsp_account_id']}")
            elif key == "PAYOR":
                result["payor"] = value if value else "Unknown"
            elif key == "ORIGINAL_DRG":
                if value and value.upper() != "NONE":
                    result["original_drg"] = value
            elif key == "PROPOSED_DRG":
                if value and value.upper() != "NONE":
                    result["proposed_drg"] = value
            elif key == "IS_SEPSIS":
                result["is_sepsis"] = value.upper() == "YES"

        return result

    except Exception as e:
        print(f"    LLM extraction error: {e}")
        return {
            "hsp_account_id": None,
            "payor": "Unknown",
            "original_drg": None,
            "proposed_drg": None,
            "is_sepsis": False
        }


def process_sample_denial_letters():
    """
    Process all denial letters in Sample_Denial_Letters folder:
    1. Extract text using Document Intelligence
    2. Use LLM to extract all denial info (account ID, payor, DRGs, sepsis flag)
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

            # Step 2: LLM extracts all denial info
            search_text = "\n".join(pages[:3]) if len(pages) >= 3 else full_text
            info = extract_denial_info_llm(search_text)
            print(f"  Account ID: {info['hsp_account_id'] or 'NOT FOUND'}")
            print(f"  Payor: {info['payor']}")
            print(f"  DRG: {info['original_drg']} → {info['proposed_drg']}")
            print(f"  Sepsis: {info['is_sepsis']}")

            # Step 3: Generate embedding
            embedding = generate_embedding(full_text)
            print(f"  Embedding: {len(embedding)} dims")

            results.append({
                "filename": pdf_file,
                "hsp_account_id": info["hsp_account_id"],
                "payor": info["payor"],
                "original_drg": info["original_drg"],
                "proposed_drg": info["proposed_drg"],
                "is_sepsis": info["is_sepsis"],
                "denial_text": full_text,
                "denial_embedding": embedding,
                "page_count": len(pages),
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Summary
    found = sum(1 for r in results if r["hsp_account_id"])
    sepsis_count = sum(1 for r in results if r["is_sepsis"])
    print(f"\n{'='*60}")
    print(f"SUMMARY: {found}/{len(pdf_files)} letters have HSP_ACCOUNT_ID")
    print(f"         {sepsis_count}/{len(pdf_files)} are sepsis-related")
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
    # Note types are configurable - comment/uncomment as needed for different conditions
    print("\nQuerying clinical data...")
    clinical_query = f"""
    WITH target_accounts AS (
        SELECT hsp_account_id
        FROM (VALUES {account_values}) AS t(hsp_account_id)
    ),

    -- Get notes using ip_note_type (original working approach)
    -- All available note types listed below - uncomment as needed for different conditions
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
            WHERE nte.ip_note_type IN (
                -- =================================================================
                -- SEPSIS-RELEVANT NOTE TYPES (14 active)
                -- Codes reference IP_NOTE_TYPE_C from Clarity
                -- =================================================================
                'Progress Notes',           -- Code 1: Daily physician documentation
                'Consults',                 -- Code 2: Specialist consultations (ID, pulm, etc.)
                'H&P',                      -- Code 4: History & Physical - admission assessment
                'Discharge Summary',        -- Code 5: Complete hospitalization summary
                'ED Notes',                 -- Code 6: Emergency department notes
                'Initial Assessments',      -- Code 7: Early clinical picture
                'ED Triage Notes',          -- Code 8: Arrival vitals, chief complaint
                'ED Provider Notes',        -- Code 19: ED physician assessment
                'Addendum Note',            -- Code 29: Updates/corrections to notes
                'Hospital Course',          -- Code 32: Timeline narrative
                'Subjective & Objective',   -- Code 33: Clinical findings (S&O)
                'Assessment & Plan Note',   -- Code 38: Physician reasoning
                'Nursing Note',             -- Code 70: Vital signs, observations
                'Code Documentation'        -- Code 10000: Code events (if applicable)

                -- =================================================================
                -- OTHER NOTE TYPES (commented out - uncomment for other conditions)
                -- =================================================================
                -- 'Procedures',             -- Code 3: Procedure documentation
                -- 'Case Communication',     -- Code 9: Case communication notes
                -- 'OR Nursing',             -- Code 10: OR nursing notes
                -- 'OR Surgeon',             -- Code 11: Surgeon OR notes
                -- 'OR PreOp',               -- Code 12: Pre-operative notes
                -- 'OR PostOp',              -- Code 13: Post-operative notes
                -- 'OR Anesthesia',          -- Code 14: Anesthesia notes
                -- 'Anesthesia Preprocedure Evaluation',  -- Code 24
                -- 'Anesthesia Postprocedure Evaluation', -- Code 25
                -- 'H&P (View-Only)',        -- Code 26: Read-only H&P
                -- 'Interval H&P Note',      -- Code 27: Interval H&P
                -- 'Anesthesia Procedure Notes', -- Code 28
                -- 'L&D Delivery Note',      -- Code 35: Labor & Delivery
                -- 'Pre-Procedure Assessment', -- Code 50
                -- 'BH Treatment Plan',      -- Code 93: Behavioral health
                -- 'Group Note',             -- Code 94: Group therapy
                -- 'Multi-Disciplinary Team Discussion', -- Code 98
                -- 'Dialysis Plan of Care Note', -- Code 100
                -- 'Dialysis Rounding Note', -- Code 105
                -- 'Sedation Documentation', -- Code 10001
                -- 'Brief Op Note',          -- Code 100019
                -- 'Operative Report',       -- Code 100020
                -- 'Therapy Evaluation',     -- Code 100023
                -- 'Therapy Treatment',      -- Code 100024
                -- 'Therapy Discharge',      -- Code 100028
                -- 'Therapy Progress Note',  -- Code 100031
                -- 'Wound Care',             -- Code 100040
                -- 'Query',                  -- Code 108000
                -- 'Strategy of Care'        -- Code 109000
            )
            GROUP BY peh.pat_id, peh.hsp_account_id, nte.ip_note_type, nte.note_id, nte.note_csn_id, nte.contact_date, nte.ent_inst_local_dttm
        )
        QUALIFY ROW_NUMBER() OVER (PARTITION BY hsp_account_id, ip_note_type ORDER BY note_contact_date DESC, entry_datetime DESC) = 1
    ),

    -- =================================================================
    -- SEPSIS-RELEVANT NOTE CTEs (14 active)
    -- =================================================================
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

    progress_note AS (
        SELECT hsp_account_id, pat_id,
               note_csn_id AS progress_note_csn_id,
               note_id AS progress_note_id,
               note_text AS progress_note_text
        FROM notes WHERE ip_note_type = 'Progress Notes'
    ),

    consult_note AS (
        SELECT hsp_account_id, pat_id,
               note_csn_id AS consult_note_csn_id,
               note_id AS consult_note_id,
               note_text AS consult_note_text
        FROM notes WHERE ip_note_type = 'Consults'
    ),

    ed_notes AS (
        SELECT hsp_account_id, pat_id,
               note_csn_id AS ed_notes_csn_id,
               note_id AS ed_notes_id,
               note_text AS ed_notes_text
        FROM notes WHERE ip_note_type = 'ED Notes'
    ),

    initial_assessment AS (
        SELECT hsp_account_id, pat_id,
               note_csn_id AS initial_assessment_csn_id,
               note_id AS initial_assessment_id,
               note_text AS initial_assessment_text
        FROM notes WHERE ip_note_type = 'Initial Assessments'
    ),

    ed_triage AS (
        SELECT hsp_account_id, pat_id,
               note_csn_id AS ed_triage_csn_id,
               note_id AS ed_triage_id,
               note_text AS ed_triage_text
        FROM notes WHERE ip_note_type = 'ED Triage Notes'
    ),

    ed_provider_note AS (
        SELECT hsp_account_id, pat_id,
               note_csn_id AS ed_provider_note_csn_id,
               note_id AS ed_provider_note_id,
               note_text AS ed_provider_note_text
        FROM notes WHERE ip_note_type = 'ED Provider Notes'
    ),

    addendum_note AS (
        SELECT hsp_account_id, pat_id,
               note_csn_id AS addendum_note_csn_id,
               note_id AS addendum_note_id,
               note_text AS addendum_note_text
        FROM notes WHERE ip_note_type = 'Addendum Note'
    ),

    hospital_course AS (
        SELECT hsp_account_id, pat_id,
               note_csn_id AS hospital_course_csn_id,
               note_id AS hospital_course_id,
               note_text AS hospital_course_text
        FROM notes WHERE ip_note_type = 'Hospital Course'
    ),

    subjective_objective AS (
        SELECT hsp_account_id, pat_id,
               note_csn_id AS subjective_objective_csn_id,
               note_id AS subjective_objective_id,
               note_text AS subjective_objective_text
        FROM notes WHERE ip_note_type = 'Subjective & Objective'
    ),

    assessment_plan AS (
        SELECT hsp_account_id, pat_id,
               note_csn_id AS assessment_plan_csn_id,
               note_id AS assessment_plan_id,
               note_text AS assessment_plan_text
        FROM notes WHERE ip_note_type = 'Assessment & Plan Note'
    ),

    nursing_note AS (
        SELECT hsp_account_id, pat_id,
               note_csn_id AS nursing_note_csn_id,
               note_id AS nursing_note_id,
               note_text AS nursing_note_text
        FROM notes WHERE ip_note_type = 'Nursing Note'
    ),

    code_documentation AS (
        SELECT hsp_account_id, pat_id,
               note_csn_id AS code_documentation_csn_id,
               note_id AS code_documentation_id,
               note_text AS code_documentation_text
        FROM notes WHERE ip_note_type = 'Code Documentation'
    ),

    -- =================================================================
    -- OTHER NOTE CTEs (commented out - uncomment for other conditions)
    -- =================================================================
    -- op_note AS (
    --     SELECT hsp_account_id, pat_id,
    --            note_csn_id AS op_note_csn_id,
    --            note_id AS op_note_id,
    --            note_text AS op_note_text
    --     FROM notes WHERE ip_note_type = 'OP Note'
    -- ),

    -- procedure_note AS (
    --     SELECT hsp_account_id, pat_id,
    --            note_csn_id AS procedure_note_csn_id,
    --            note_id AS procedure_note_id,
    --            note_text AS procedure_note_text
    --     FROM notes WHERE ip_note_type = 'Procedure Note'
    -- ),

    -- critical_care_note AS (
    --     SELECT hsp_account_id, pat_id,
    --            note_csn_id AS critical_care_note_csn_id,
    --            note_id AS critical_care_note_id,
    --            note_text AS critical_care_note_text
    --     FROM notes WHERE ip_note_type = 'Critical Care Notes'
    -- ),

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
           -- Discharge Summary (Code 5)
           COALESCE(d.discharge_summary_note_id, 'no id available') AS discharge_summary_note_id,
           COALESCE(d.discharge_note_csn_id, 'no id available') AS discharge_note_csn_id,
           COALESCE(d.discharge_summary_text, 'No Note Available') AS discharge_summary_text,
           -- H&P (Code 4)
           COALESCE(h.hp_note_id, 'no id available') AS hp_note_id,
           COALESCE(h.hp_note_csn_id, 'no id available') AS hp_note_csn_id,
           COALESCE(h.hp_note_text, 'No Note Available') AS hp_note_text,
           -- Progress Notes (Code 1)
           COALESCE(pn.progress_note_id, 'no id available') AS progress_note_id,
           COALESCE(pn.progress_note_csn_id, 'no id available') AS progress_note_csn_id,
           COALESCE(pn.progress_note_text, 'No Note Available') AS progress_note_text,
           -- Consults (Code 2)
           COALESCE(cn.consult_note_id, 'no id available') AS consult_note_id,
           COALESCE(cn.consult_note_csn_id, 'no id available') AS consult_note_csn_id,
           COALESCE(cn.consult_note_text, 'No Note Available') AS consult_note_text,
           -- ED Notes (Code 6)
           COALESCE(edn.ed_notes_id, 'no id available') AS ed_notes_id,
           COALESCE(edn.ed_notes_csn_id, 'no id available') AS ed_notes_csn_id,
           COALESCE(edn.ed_notes_text, 'No Note Available') AS ed_notes_text,
           -- Initial Assessments (Code 7)
           COALESCE(ia.initial_assessment_id, 'no id available') AS initial_assessment_id,
           COALESCE(ia.initial_assessment_csn_id, 'no id available') AS initial_assessment_csn_id,
           COALESCE(ia.initial_assessment_text, 'No Note Available') AS initial_assessment_text,
           -- ED Triage Notes (Code 8)
           COALESCE(edt.ed_triage_id, 'no id available') AS ed_triage_id,
           COALESCE(edt.ed_triage_csn_id, 'no id available') AS ed_triage_csn_id,
           COALESCE(edt.ed_triage_text, 'No Note Available') AS ed_triage_text,
           -- ED Provider Notes (Code 19)
           COALESCE(edp.ed_provider_note_id, 'no id available') AS ed_provider_note_id,
           COALESCE(edp.ed_provider_note_csn_id, 'no id available') AS ed_provider_note_csn_id,
           COALESCE(edp.ed_provider_note_text, 'No Note Available') AS ed_provider_note_text,
           -- Addendum Note (Code 29)
           COALESCE(an.addendum_note_id, 'no id available') AS addendum_note_id,
           COALESCE(an.addendum_note_csn_id, 'no id available') AS addendum_note_csn_id,
           COALESCE(an.addendum_note_text, 'No Note Available') AS addendum_note_text,
           -- Hospital Course (Code 32)
           COALESCE(hc.hospital_course_id, 'no id available') AS hospital_course_id,
           COALESCE(hc.hospital_course_csn_id, 'no id available') AS hospital_course_csn_id,
           COALESCE(hc.hospital_course_text, 'No Note Available') AS hospital_course_text,
           -- Subjective & Objective (Code 33)
           COALESCE(so.subjective_objective_id, 'no id available') AS subjective_objective_id,
           COALESCE(so.subjective_objective_csn_id, 'no id available') AS subjective_objective_csn_id,
           COALESCE(so.subjective_objective_text, 'No Note Available') AS subjective_objective_text,
           -- Assessment & Plan Note (Code 38)
           COALESCE(ap.assessment_plan_id, 'no id available') AS assessment_plan_id,
           COALESCE(ap.assessment_plan_csn_id, 'no id available') AS assessment_plan_csn_id,
           COALESCE(ap.assessment_plan_text, 'No Note Available') AS assessment_plan_text,
           -- Nursing Note (Code 70)
           COALESCE(nn.nursing_note_id, 'no id available') AS nursing_note_id,
           COALESCE(nn.nursing_note_csn_id, 'no id available') AS nursing_note_csn_id,
           COALESCE(nn.nursing_note_text, 'No Note Available') AS nursing_note_text,
           -- Code Documentation (Code 10000)
           COALESCE(cd.code_documentation_id, 'no id available') AS code_documentation_id,
           COALESCE(cd.code_documentation_csn_id, 'no id available') AS code_documentation_csn_id,
           COALESCE(cd.code_documentation_text, 'No Note Available') AS code_documentation_text
    FROM patient_info pi
    LEFT JOIN discharge d ON pi.hsp_account_id = d.hsp_account_id
    LEFT JOIN hp_note h ON pi.hsp_account_id = h.hsp_account_id
    LEFT JOIN progress_note pn ON pi.hsp_account_id = pn.hsp_account_id
    LEFT JOIN consult_note cn ON pi.hsp_account_id = cn.hsp_account_id
    LEFT JOIN ed_notes edn ON pi.hsp_account_id = edn.hsp_account_id
    LEFT JOIN initial_assessment ia ON pi.hsp_account_id = ia.hsp_account_id
    LEFT JOIN ed_triage edt ON pi.hsp_account_id = edt.hsp_account_id
    LEFT JOIN ed_provider_note edp ON pi.hsp_account_id = edp.hsp_account_id
    LEFT JOIN addendum_note an ON pi.hsp_account_id = an.hsp_account_id
    LEFT JOIN hospital_course hc ON pi.hsp_account_id = hc.hsp_account_id
    LEFT JOIN subjective_objective so ON pi.hsp_account_id = so.hsp_account_id
    LEFT JOIN assessment_plan ap ON pi.hsp_account_id = ap.hsp_account_id
    LEFT JOIN nursing_note nn ON pi.hsp_account_id = nn.hsp_account_id
    LEFT JOIN code_documentation cd ON pi.hsp_account_id = cd.hsp_account_id
    """

    # Execute clinical query in Spark (DO NOT collect to driver - memory issue)
    clinical_spark_df = spark.sql(clinical_query)
    clinical_count = clinical_spark_df.count()
    print(f"Retrieved {clinical_count} accounts from Clarity")

    # Use pre-parsed denial data from DENIAL_RESULTS (already has text, embedding, payor)
    # Convert denial results to Spark DataFrame for distributed join
    print("\nJoining with pre-parsed denial data (in Spark, not Pandas)...")

    if DENIAL_RESULTS:
        # Create denial DataFrame from parsed results
        denial_records = [
            {
                "denial_hsp_account_id": str(r["hsp_account_id"]),
                "denial_letter_text": r.get("denial_text"),
                "denial_letter_filename": r.get("filename"),
                "denial_embedding": r.get("denial_embedding"),
                "payor": r.get("payor"),
                "original_drg": r.get("original_drg"),
                "proposed_drg": r.get("proposed_drg"),
                "is_sepsis": r.get("is_sepsis", False),
            }
            for r in DENIAL_RESULTS
            if r["hsp_account_id"]
        ]

        denial_schema = StructType([
            StructField("denial_hsp_account_id", StringType(), True),
            StructField("denial_letter_text", StringType(), True),
            StructField("denial_letter_filename", StringType(), True),
            StructField("denial_embedding", ArrayType(FloatType()), True),
            StructField("payor", StringType(), True),
            StructField("original_drg", StringType(), True),
            StructField("proposed_drg", StringType(), True),
            StructField("is_sepsis", BooleanType(), True),
        ])

        denial_spark_df = spark.createDataFrame(denial_records, denial_schema)
        print(f"  Denial records: {denial_spark_df.count()}")

        # Cast clinical hsp_account_id to string for join
        clinical_spark_df = clinical_spark_df.withColumn(
            "hsp_account_id",
            col("hsp_account_id").cast("string")
        )

        # Join clinical data with denial data in Spark (distributed)
        joined_df = clinical_spark_df.join(
            denial_spark_df,
            clinical_spark_df["hsp_account_id"] == denial_spark_df["denial_hsp_account_id"],
            "left"
        ).drop("denial_hsp_account_id")

        # Add metadata columns
        joined_df = joined_df.withColumn("scope_filter", lit(SCOPE_FILTER))
        joined_df = joined_df.withColumn("featurization_timestamp", lit(datetime.now().isoformat()))

        final_count = joined_df.count()
        print(f"\nFeaturized {final_count} rows with embeddings")
    else:
        print("  No denial results to join")
        joined_df = clinical_spark_df.withColumn("scope_filter", lit(SCOPE_FILTER))
        joined_df = joined_df.withColumn("featurization_timestamp", lit(datetime.now().isoformat()))

    # Write to table (all in Spark - no driver memory issues)
    WRITE_TO_TABLE = False

    if WRITE_TO_TABLE:
        joined_df = joined_df.withColumn("insert_tsp", current_timestamp())
        joined_df.write.format("delta").mode("overwrite").saveAsTable(INFERENCE_TABLE)
        print(f"Wrote to {INFERENCE_TABLE}")
    else:
        print("To write, set WRITE_TO_TABLE = True")

else:
    print("Denial featurization skipped")

# =============================================================================
# CELL 11: Verify Tables
# =============================================================================
# =============================================================================
# CELL FINAL: Validation Summary
# =============================================================================
print("\n" + "="*60)
print("FINAL VALIDATION CHECKPOINTS")
print("="*60)

# Run all checkpoints
checkpoint_results = {}

print("\n--- Gold Letters Table ---")
checkpoint_results["gold_letters"] = checkpoint_gold_letters()

print("\n--- Propel Data Table ---")
checkpoint_results["propel_data"] = checkpoint_propel_data()

print("\n--- Inference Table ---")
checkpoint_results["inference"] = checkpoint_inference_table()

# Summary
print("\n" + "="*60)
print("CHECKPOINT SUMMARY")
print("="*60)
passed = sum(1 for v in checkpoint_results.values() if v)
total = len(checkpoint_results)
print(f"Passed: {passed}/{total}")

for name, result in checkpoint_results.items():
    status = "[OK]" if result else "[FAIL/WARN]"
    print(f"  {status} {name}")

if passed == total:
    print("\n[ALL CHECKPOINTS PASSED] Ready for inference")
else:
    print("\n[CHECKPOINTS INCOMPLETE] Review warnings above before running inference")

print("\nFeaturization complete.")
