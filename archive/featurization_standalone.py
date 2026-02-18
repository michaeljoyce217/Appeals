# featurization_standalone.py
# ==============================================================================
# COMPLETELY SELF-CONTAINED - NO EXTERNAL IMPORTS FROM poc/
# ==============================================================================
# This script:
# 1. Creates all Delta tables needed for the pipeline
# 2. Reads denial letters from a folder (PDF/DOCX) using Azure Doc Intelligence
# 3. Writes to denial_letters Delta table
# ==============================================================================

# =============================================================================
# CONFIGURATION - EDIT THESE VALUES
# =============================================================================

CATALOG = "dev"  # Change to "test" or "prod"
SCHEMA = "clncl_ds"  # Your schema

# Paths - UPDATE THESE for your environment
DENIAL_LETTERS_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/denial_letters"  # Where PDFs/DOCXs live

# Scan detection thresholds
MIN_CHARS_PER_PAGE = 100  # PDFs with less are likely scans
MIN_TOTAL_CHARS = 50      # Documents with less are empty/failed
SKIP_SCANNED_PDFS = True  # Set False to attempt processing scans (not recommended)

# =============================================================================
# IMPORTS
# =============================================================================

import os
import uuid
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List

# Databricks imports (available in notebook)
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    FloatType, TimestampType, BooleanType, ArrayType, MapType
)

# =============================================================================
# GET SPARK AND DBUTILS
# =============================================================================

spark = SparkSession.builder.getOrCreate()

try:
    # Databricks environment
    dbutils = spark._jvm.com.databricks.service.DBUtils(spark._jsparkSession)
except:
    # Alternative method
    from pyspark.dbutils import DBUtils
    dbutils = DBUtils(spark)

print(f"Catalog: {CATALOG}")
print(f"Schema: {SCHEMA}")

# =============================================================================
# TABLE NAMES
# =============================================================================

DENIAL_LETTERS_TABLE = f"{CATALOG}.{SCHEMA}.fudgesicle_denial_letters"
DENIAL_EXTRACTS_TABLE = f"{CATALOG}.{SCHEMA}.fudgesicle_denial_extracts"
CLINICAL_CRITERIA_TABLE = f"{CATALOG}.{SCHEMA}.fudgesicle_clinical_criteria"
SOFA_THRESHOLDS_TABLE = f"{CATALOG}.{SCHEMA}.fudgesicle_sofa_thresholds"
GOLD_TEMPLATES_TABLE = f"{CATALOG}.{SCHEMA}.fudgesicle_gold_templates"
DRAFT_LETTERS_TABLE = f"{CATALOG}.{SCHEMA}.fudgesicle_draft_letters"

# =============================================================================
# AZURE CREDENTIALS FROM DATABRICKS SECRETS
# =============================================================================

print("\n[STEP 0] Loading Azure credentials from secrets...")

DOC_INTEL_ENDPOINT = dbutils.secrets.get(scope="idp_etl", key="az-doc-intelligence-endpoint")
DOC_INTEL_KEY = dbutils.secrets.get(scope="idp_etl", key="az-doc-intelligence-key")

print("  Azure Document Intelligence credentials loaded")

# =============================================================================
# DOCUMENT RESULT CLASS
# =============================================================================

@dataclass
class DocumentResult:
    """Result from reading a document."""
    text: str
    file_path: str
    file_type: str
    page_count: int
    char_count: int
    is_scanned: bool
    is_usable: bool
    scan_confidence: float
    skip_reason: Optional[str] = None

# =============================================================================
# DOCUMENT READER FUNCTIONS
# =============================================================================

def read_document_with_analysis(file_path: str) -> DocumentResult:
    """
    Read a document with full analysis including scan detection.
    Uses Azure AI Document Intelligence for PDF/DOCX.
    """
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
    from azure.core.credentials import AzureKeyCredential

    _, ext = os.path.splitext(file_path.lower())
    file_type = ext.lstrip(".")

    # Create client
    client = DocumentIntelligenceClient(
        endpoint=DOC_INTEL_ENDPOINT,
        credential=AzureKeyCredential(DOC_INTEL_KEY)
    )

    # Read file bytes
    with open(file_path, "rb") as f:
        document_bytes = f.read()

    # Analyze document
    poller = client.begin_analyze_document(
        model_id="prebuilt-read",
        body=AnalyzeDocumentRequest(bytes_source=document_bytes),
    )
    result = poller.result()

    # Extract text and gather metrics
    text_parts = []
    page_count = len(result.pages) if result.pages else 0
    total_lines = 0
    handwritten_spans = 0
    total_spans = 0

    for page in result.pages:
        if page.lines:
            total_lines += len(page.lines)
            for line in page.lines:
                text_parts.append(line.content)

    # Check for handwritten content
    if hasattr(result, 'styles') and result.styles:
        for style in result.styles:
            if hasattr(style, 'spans'):
                total_spans += len(style.spans)
                if getattr(style, 'is_handwritten', False):
                    handwritten_spans += len(style.spans)

    text = "\n".join(text_parts)
    char_count = len(text)

    # Calculate scan confidence
    scan_confidence = calculate_scan_confidence(
        char_count=char_count,
        page_count=page_count,
        total_lines=total_lines,
        handwritten_spans=handwritten_spans,
        total_spans=total_spans,
        file_type=file_type
    )

    is_scanned = scan_confidence > 0.7
    is_usable = not is_scanned and char_count >= MIN_TOTAL_CHARS

    # Determine skip reason
    skip_reason = None
    if is_scanned:
        skip_reason = f"Likely scanned document (confidence: {scan_confidence:.0%})"
    elif char_count < MIN_TOTAL_CHARS:
        skip_reason = f"Insufficient text extracted ({char_count} chars)"

    return DocumentResult(
        text=text,
        file_path=file_path,
        file_type=file_type,
        page_count=page_count,
        char_count=char_count,
        is_scanned=is_scanned,
        is_usable=is_usable,
        scan_confidence=scan_confidence,
        skip_reason=skip_reason
    )


def calculate_scan_confidence(char_count: int, page_count: int,
                              total_lines: int, handwritten_spans: int,
                              total_spans: int, file_type: str) -> float:
    """Calculate confidence that document is a scan (0-1, higher = more likely scan)."""

    # DOCX files are never scans
    if file_type in ["docx", "doc"]:
        return 0.0

    if page_count == 0:
        return 1.0

    # Metric 1: Characters per page
    chars_per_page = char_count / page_count
    if chars_per_page < 50:
        density_score = 1.0
    elif chars_per_page < MIN_CHARS_PER_PAGE:
        density_score = 0.8
    elif chars_per_page < 200:
        density_score = 0.4
    else:
        density_score = 0.0

    # Metric 2: Lines per page
    lines_per_page = total_lines / page_count
    if lines_per_page < 5:
        lines_score = 1.0
    elif lines_per_page < 10:
        lines_score = 0.6
    elif lines_per_page < 20:
        lines_score = 0.3
    else:
        lines_score = 0.0

    # Metric 3: Handwritten content
    if total_spans > 0:
        handwritten_ratio = handwritten_spans / total_spans
        handwritten_score = handwritten_ratio
    else:
        handwritten_score = 0.0

    # Combine metrics (weighted)
    confidence = (
        0.5 * density_score +
        0.3 * lines_score +
        0.2 * handwritten_score
    )

    return min(1.0, confidence)

# =============================================================================
# CREATE TABLES
# =============================================================================

def create_all_tables():
    """Create all Delta tables needed for the pipeline."""

    print("\n[STEP 1] Creating Delta tables...")

    # 1. DENIAL LETTERS TABLE
    ddl_denial_letters = f"""
    CREATE TABLE IF NOT EXISTS {DENIAL_LETTERS_TABLE} (
        letter_id STRING NOT NULL,
        letter_text STRING,
        letter_pdf_path STRING,
        letter_format STRING,
        page_count INT,
        char_count INT,
        is_scanned BOOLEAN,
        scan_confidence FLOAT,
        ingested_at TIMESTAMP
    )
    USING DELTA
    TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
    COMMENT 'Denial letters ingested from payor communications'
    """
    spark.sql(ddl_denial_letters)
    print(f"  Created: {DENIAL_LETTERS_TABLE}")

    # 2. DENIAL EXTRACTS TABLE
    ddl_denial_extracts = f"""
    CREATE TABLE IF NOT EXISTS {DENIAL_EXTRACTS_TABLE} (
        extract_id STRING NOT NULL,
        letter_id STRING NOT NULL,
        condition_denied STRING,
        denial_category STRING,
        denial_subcategory STRING,
        denial_rationale STRING,
        denial_summary STRING,
        payer_quotes ARRAY<STRING>,
        denial_signals ARRAY<STRING>,
        denied_drg STRING,
        claimed_drg STRING,
        drg_weight_difference FLOAT,
        reviewer_name STRING,
        reviewer_credentials STRING,
        review_date STRING,
        admin_data MAP<STRING, STRING>,
        confidence_score FLOAT,
        human_validated BOOLEAN,
        extracted_at TIMESTAMP,
        model_version STRING
    )
    USING DELTA
    TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
    COMMENT 'Structured extracts from denial letters via Scan Agent'
    """
    spark.sql(ddl_denial_extracts)
    print(f"  Created: {DENIAL_EXTRACTS_TABLE}")

    # 3. CLINICAL CRITERIA TABLE
    ddl_clinical_criteria = f"""
    CREATE TABLE IF NOT EXISTS {CLINICAL_CRITERIA_TABLE} (
        criteria_id STRING NOT NULL,
        condition STRING NOT NULL,
        criteria_type STRING NOT NULL,
        criteria_subtype STRING,
        denial_relevance ARRAY<STRING>,
        source STRING NOT NULL,
        source_citation STRING,
        content STRING NOT NULL,
        content_embedding ARRAY<FLOAT>,
        effective_date DATE,
        supersedes_id STRING,
        metadata MAP<STRING, STRING>,
        created_at TIMESTAMP,
        updated_at TIMESTAMP
    )
    USING DELTA
    TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
    COMMENT 'Clinical criteria from Propel/ACDIS for retrieval'
    """
    spark.sql(ddl_clinical_criteria)
    print(f"  Created: {CLINICAL_CRITERIA_TABLE}")

    # 4. SOFA THRESHOLDS TABLE
    ddl_sofa = f"""
    CREATE TABLE IF NOT EXISTS {SOFA_THRESHOLDS_TABLE} (
        threshold_id STRING NOT NULL,
        organ_system STRING NOT NULL,
        sofa_score INT NOT NULL,
        measure STRING NOT NULL,
        operator STRING NOT NULL,
        threshold_value FLOAT,
        threshold_unit STRING,
        additional_conditions STRING,
        notes STRING,
        source STRING NOT NULL,
        source_version STRING,
        effective_date DATE,
        created_at TIMESTAMP
    )
    USING DELTA
    COMMENT 'Structured SOFA-2 thresholds for validation'
    """
    spark.sql(ddl_sofa)
    print(f"  Created: {SOFA_THRESHOLDS_TABLE}")

    # 5. GOLD TEMPLATES TABLE
    ddl_gold = f"""
    CREATE TABLE IF NOT EXISTS {GOLD_TEMPLATES_TABLE} (
        template_id STRING NOT NULL,
        condition STRING NOT NULL,
        denial_category STRING NOT NULL,
        denial_pattern STRING,
        template_text STRING NOT NULL,
        template_summary STRING,
        key_arguments ARRAY<STRING>,
        template_embedding ARRAY<FLOAT>,
        source_letter_ids ARRAY<STRING>,
        times_used INT,
        times_won INT,
        win_rate FLOAT,
        is_active BOOLEAN,
        created_at TIMESTAMP,
        updated_at TIMESTAMP
    )
    USING DELTA
    TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
    COMMENT 'Gold standard rebuttal letters for template matching'
    """
    spark.sql(ddl_gold)
    print(f"  Created: {GOLD_TEMPLATES_TABLE}")

    # 6. DRAFT LETTERS TABLE
    ddl_drafts = f"""
    CREATE TABLE IF NOT EXISTS {DRAFT_LETTERS_TABLE} (
        draft_id STRING NOT NULL,
        letter_id STRING NOT NULL,
        extract_id STRING,
        template_id STRING,
        draft_text STRING NOT NULL,
        draft_json STRING,
        evidence_sources STRING,
        citations_used STRING,
        confidence_score FLOAT,
        status STRING,
        reviewer_notes STRING,
        revision_count INT,
        created_at TIMESTAMP,
        updated_at TIMESTAMP
    )
    USING DELTA
    TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
    COMMENT 'Generated draft letters for human review'
    """
    spark.sql(ddl_drafts)
    print(f"  Created: {DRAFT_LETTERS_TABLE}")

    print("  All tables created successfully")

# =============================================================================
# INGEST DENIAL LETTERS
# =============================================================================

def ingest_denial_letters():
    """Read denial letters from DENIAL_LETTERS_PATH and write to table."""

    print(f"\n[STEP 2] Reading denial letters from: {DENIAL_LETTERS_PATH}")

    # List files in the folder
    try:
        files = dbutils.fs.ls(DENIAL_LETTERS_PATH)
        letter_files = [f.path for f in files if f.path.lower().endswith(('.pdf', '.docx', '.doc'))]
    except Exception as e:
        print(f"  ERROR: Could not list files in {DENIAL_LETTERS_PATH}: {e}")
        return

    if not letter_files:
        print(f"  No PDF/DOCX files found in {DENIAL_LETTERS_PATH}")
        return

    print(f"  Found {len(letter_files)} denial letter files")

    processed = []
    skipped = []

    for file_path in letter_files:
        # Convert dbfs path to local path for reading
        local_path = file_path.replace("dbfs:", "/dbfs")
        file_name = os.path.basename(file_path)

        print(f"  Processing: {file_name}...", end=" ")

        try:
            result = read_document_with_analysis(local_path)

            if result.is_usable:
                processed.append({
                    "letter_id": str(uuid.uuid4()),
                    "letter_text": result.text,
                    "letter_pdf_path": file_path,
                    "letter_format": result.file_type,
                    "page_count": result.page_count,
                    "char_count": result.char_count,
                    "is_scanned": False,
                    "scan_confidence": result.scan_confidence,
                    "ingested_at": datetime.now()
                })
                print(f"OK ({result.char_count} chars)")
            else:
                skipped.append({
                    "file": file_name,
                    "reason": result.skip_reason,
                    "scan_confidence": result.scan_confidence
                })
                print(f"SKIPPED - {result.skip_reason}")

        except Exception as e:
            skipped.append({
                "file": file_name,
                "reason": str(e),
                "scan_confidence": None
            })
            print(f"ERROR - {e}")

    # Write processed letters to table
    if processed:
        print(f"\n  Writing {len(processed)} letters to {DENIAL_LETTERS_TABLE}...")

        schema = StructType([
            StructField("letter_id", StringType(), False),
            StructField("letter_text", StringType(), True),
            StructField("letter_pdf_path", StringType(), True),
            StructField("letter_format", StringType(), True),
            StructField("page_count", IntegerType(), True),
            StructField("char_count", IntegerType(), True),
            StructField("is_scanned", BooleanType(), True),
            StructField("scan_confidence", FloatType(), True),
            StructField("ingested_at", TimestampType(), True),
        ])

        df = spark.createDataFrame(processed, schema)
        df.write.mode("append").saveAsTable(DENIAL_LETTERS_TABLE)
        print(f"  Successfully wrote {len(processed)} letters")

    # Summary
    print(f"\n  Summary: Processed={len(processed)}, Skipped={len(skipped)}")
    if skipped:
        print("  Skipped files:")
        for s in skipped:
            print(f"    - {s['file']}: {s['reason']}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

print("=" * 60)
print("FEATURIZATION PIPELINE (STANDALONE)")
print("=" * 60)

# Step 1: Create tables
create_all_tables()

# Step 2: Ingest denial letters
ingest_denial_letters()

# Step 3: Verify
print("\n[STEP 3] Verification...")
count = spark.sql(f"SELECT COUNT(*) as cnt FROM {DENIAL_LETTERS_TABLE}").collect()[0].cnt
print(f"  Total letters in {DENIAL_LETTERS_TABLE}: {count}")

print("\n" + "=" * 60)
print("FEATURIZATION COMPLETE")
print("=" * 60)
print(f"\nNext: Run inference_standalone.py to generate rebuttal letters")
