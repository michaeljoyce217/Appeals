# data/featurization.py
# Sepsis Appeal Engine - Knowledge Base Ingestion
#
# ONE-TIME SETUP: Ingests gold standard letters and Propel definitions
# into Unity Catalog tables for use by inference.py
#
# This file does NOT process denial letters - that happens in inference.py
# on a per-letter basis (production-like workflow).
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
import uuid
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, ArrayType, FloatType,
    TimestampType, MapType
)

spark = SparkSession.builder.getOrCreate()

# Configuration
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
PROPEL_DATA_TABLE = f"{trgt_cat}.fin_ds.fudgesicle_propel_data"

# Paths
GOLD_LETTERS_PATH = "/Workspace/Repos/mijo8881@mercy.net/fudgesicle/utils/gold_standard_appeals_sepsis_only"
PROPEL_DATA_PATH = "/Workspace/Repos/mijo8881@mercy.net/fudgesicle/utils/propel_data"

print(f"Catalog: {trgt_cat}")

# =============================================================================
# CELL 3B: Validation Checkpoints
# =============================================================================
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
# CELL 6: Core Parser Functions
# =============================================================================

def extract_text_from_pdf(file_path):
    """
    Extract text from PDF using Document Intelligence layout model.
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
    Identify which page the denial letter starts on in a gold standard letter.
    Gold letters contain appeal first, then the original denial attached.

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
    address_pattern = re.compile(
        r'\b(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)\s*\d{5}',
        re.IGNORECASE
    )
    date_pattern = re.compile(
        r'(\d{1,2}/\d{1,2}/\d{4})|((January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})',
        re.IGNORECASE
    )

    # Skip page 1 - that's the appeal. Start from page 2.
    for i, page_text in enumerate(pages_text):
        if i == 0:
            continue

        first_lines = page_text.split("\n")[:15]
        header_text = "\n".join(first_lines)
        header_lower = header_text.lower()

        found_payor = None
        for pattern, payor_name in payor_patterns:
            if pattern in header_lower:
                found_payor = payor_name
                break

        if found_payor:
            has_address = bool(address_pattern.search(header_text))
            has_date = bool(date_pattern.search(header_text))

            if has_address or has_date:
                print(f"    Found denial: '{found_payor}' + {'address' if has_address else 'date'} on page {i+1}")
                return i + 1, found_payor
            else:
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
            continue

        first_lines = page_text.split("\n")[:15]
        header_lower = "\n".join(first_lines).lower()

        for phrase in denial_subject_phrases:
            if phrase in header_lower:
                print(f"    Found denial phrase '{phrase}' on page {i+1}")
                return i + 1, "Unknown"

    print("    WARNING: Could not identify denial start page")
    return len(pages_text) + 1, "Unknown"


def generate_embedding(text):
    """
    Generate embedding vector for text using Azure OpenAI.
    Returns 1536-dimensional vector.
    """
    if len(text) > 20000:
        text = text[:20000]

    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def parse_gold_letter_pdf(file_path):
    """
    Parse a gold standard letter PDF.
    Gold letters contain BOTH appeal (first) and denial (attached at end).
    Returns dict with: rebuttal_text (appeal), denial_text, denial_embedding, payor
    """
    pages = extract_text_from_pdf(file_path)
    denial_start_page, payor = identify_denial_start(pages)

    rebuttal_pages = pages[:denial_start_page - 1]
    denial_pages = pages[denial_start_page - 1:]

    rebuttal_text = "\n\n".join(rebuttal_pages) if rebuttal_pages else ""
    denial_text = "\n\n".join(denial_pages) if denial_pages else ""

    denial_embedding = generate_embedding(denial_text) if denial_text else None

    return {
        "rebuttal_text": rebuttal_text,
        "denial_text": denial_text,
        "denial_embedding": denial_embedding,
        "payor": payor,
        "denial_start_page": denial_start_page,
        "total_pages": len(pages)
    }


print("Parser functions loaded")

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

# Propel Data Table - stores official clinical definitions
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
# #############################################################################

# =============================================================================
# CELL 8: Process Gold Standard Letters
# =============================================================================
RUN_GOLD_INGESTION = False

if RUN_GOLD_INGESTION:
    print("="*60)
    print("GOLD STANDARD LETTER INGESTION")
    print("="*60)

    pdf_files = [f for f in os.listdir(GOLD_LETTERS_PATH) if f.lower().endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDF files")

    gold_records = []

    for i, pdf_file in enumerate(pdf_files):
        print(f"\nProcessing {i+1}/{len(pdf_files)}: {pdf_file}")
        file_path = os.path.join(GOLD_LETTERS_PATH, pdf_file)

        try:
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

    checkpoint_gold_letters()

else:
    print("Gold ingestion skipped (set RUN_GOLD_INGESTION = True)")

# #############################################################################
# PART 2: PROPEL DATA INGESTION
# #############################################################################

# =============================================================================
# CELL 9: Propel Extraction Prompt and Function
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
        return definition_text

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
# CELL 10: Process Propel Data Files
# =============================================================================
RUN_PROPEL_INGESTION = False

if RUN_PROPEL_INGESTION:
    print("\n" + "="*60)
    print("PROPEL DATA INGESTION")
    print("="*60)

    if os.path.exists(PROPEL_DATA_PATH):
        pdf_files = [f for f in os.listdir(PROPEL_DATA_PATH) if f.lower().endswith('.pdf')]
        print(f"Found {len(pdf_files)} PDF files in propel_data")

        propel_records = []

        for i, pdf_file in enumerate(pdf_files):
            print(f"\n[{i+1}/{len(pdf_files)}] Processing {pdf_file}")
            file_path = os.path.join(PROPEL_DATA_PATH, pdf_file)

            try:
                pages_text = extract_text_from_pdf(file_path)
                definition_text = "\n\n".join(pages_text)
                print(f"  Extracted {len(definition_text)} chars from {len(pages_text)} pages")

                # Derive condition name from filename
                base_name = os.path.splitext(pdf_file)[0].lower()
                if base_name.startswith("propel_"):
                    condition_name = base_name[7:]
                else:
                    condition_name = base_name
                print(f"  Condition: {condition_name}")

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

        checkpoint_propel_data()
    else:
        print(f"WARNING: Propel data path not found: {PROPEL_DATA_PATH}")

else:
    print("\nPropel ingestion skipped (set RUN_PROPEL_INGESTION = True)")

# =============================================================================
# CELL FINAL: Validation Summary
# =============================================================================
print("\n" + "="*60)
print("FINAL VALIDATION CHECKPOINTS")
print("="*60)

checkpoint_results = {}

print("\n--- Gold Letters Table ---")
checkpoint_results["gold_letters"] = checkpoint_gold_letters()

print("\n--- Propel Data Table ---")
checkpoint_results["propel_data"] = checkpoint_propel_data()

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
    print("\n[ALL CHECKPOINTS PASSED] Knowledge base ready for inference.py")
else:
    print("\n[CHECKPOINTS INCOMPLETE] Run ingestion steps as needed")

print("\nFeaturization complete.")
