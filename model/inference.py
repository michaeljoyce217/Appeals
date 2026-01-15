# model/inference.py
# Sepsis Appeal Engine - Single Letter Processing
#
# END-TO-END PIPELINE FOR ONE DENIAL LETTER:
# 1. Parse denial PDF → Extract text, account ID, payor, DRGs, sepsis flag
# 2. Query Clarity → Get clinical notes for this one account
# 3. Vector search → Find best matching gold letter
# 4. Generate appeal → Use gold letter as template, clinical notes as evidence
# 5. Output → DOCX file for CDI review
#
# PRODUCTION-LIKE WORKFLOW:
# This mirrors how production will work - one denial at a time from Epic workqueue.
# No batch processing, no intermediate tables, no driver memory issues.
#
# Run on Databricks Runtime 15.4 LTS ML

# =============================================================================
# CELL 1: Install Dependencies (run this cell FIRST, then restart)
# =============================================================================
# IMPORTANT: Run this cell by itself, then run the rest of the notebook.
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
import math
from datetime import datetime, date
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# -----------------------------------------------------------------------------
# INPUT CONFIGURATION - Set the denial PDF to process
# -----------------------------------------------------------------------------
# Path to the denial letter PDF to process
# In production, this would come from Epic workqueue
DENIAL_PDF_PATH = "/Workspace/Repos/mijo8881@mercy.net/fudgesicle/utils/sample_denial_letters/example_denial.pdf"

# If account ID is known (production), set it here. Otherwise LLM will extract from PDF.
# In production: Epic workqueue provides this
# In POC: Set to None to extract from PDF
KNOWN_ACCOUNT_ID = None  # e.g., "12345678" or None to extract

# -----------------------------------------------------------------------------
# Processing Configuration
# -----------------------------------------------------------------------------
SCOPE_FILTER = "sepsis"
SEPSIS_DRG_CODES = ["870", "871", "872"]
MATCH_SCORE_THRESHOLD = 0.7
NOTE_EXTRACTION_THRESHOLD = 8000  # Chars - notes longer than this get extracted
EMBEDDING_MODEL = "text-embedding-ada-002"

# Default template path (fallback when no gold letter matches)
DEFAULT_TEMPLATE_PATH = "/Workspace/Repos/mijo8881@mercy.net/fudgesicle/utils/gold_standard_appeals/default_sepsis_appeal_template.docx"

# Output configuration
EXPORT_TO_DOCX = True
DOCX_OUTPUT_BASE = "/Workspace/Repos/mijo8881@mercy.net/fudgesicle/utils/outputs"

# =============================================================================
# CELL 3: Environment Setup
# =============================================================================
trgt_cat = os.environ.get('trgt_cat', 'dev')
if trgt_cat == 'dev':
    spark.sql('USE CATALOG prod;')
else:
    spark.sql(f'USE CATALOG {trgt_cat};')

# Table names (for knowledge base - gold letters and propel definitions)
GOLD_LETTERS_TABLE = f"{trgt_cat}.fin_ds.fudgesicle_gold_letters"
PROPEL_DATA_TABLE = f"{trgt_cat}.fin_ds.fudgesicle_propel_data"

print(f"Catalog: {trgt_cat}")
print(f"Denial PDF: {DENIAL_PDF_PATH}")

# =============================================================================
# CELL 4: Azure Credentials and Clients
# =============================================================================
from openai import AzureOpenAI
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential

# Load credentials
AZURE_OPENAI_KEY = dbutils.secrets.get(scope='idp_etl', key='az-openai-key1')
AZURE_OPENAI_ENDPOINT = dbutils.secrets.get(scope='idp_etl', key='az-openai-base')
AZURE_DOC_INTEL_KEY = dbutils.secrets.get(scope='idp_etl', key='az-aidcmntintel-key1')
AZURE_DOC_INTEL_ENDPOINT = dbutils.secrets.get(scope='idp_etl', key='az-aidcmntintel-endpoint')

# Initialize clients
openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2024-10-21"
)

doc_intel_client = DocumentIntelligenceClient(
    endpoint=AZURE_DOC_INTEL_ENDPOINT,
    credential=AzureKeyCredential(AZURE_DOC_INTEL_KEY)
)

print("Azure clients initialized")

# =============================================================================
# CELL 5: Core Functions - PDF Parsing and Denial Extraction
# =============================================================================

def extract_text_from_pdf(file_path):
    """Extract text from PDF using Document Intelligence."""
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


def generate_embedding(text):
    """Generate embedding vector for text."""
    if len(text) > 20000:
        text = text[:20000]

    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


DENIAL_PARSER_PROMPT = '''Extract key information from this denial letter.

# Denial Letter Text
{denial_text}

# Instructions
Find:
1. HOSPITAL ACCOUNT ID - starts with "H" followed by digits (e.g., H1234567890)
2. INSURANCE PAYOR - the company that sent this denial
3. ORIGINAL DRG - the DRG code the hospital billed (e.g., 871). ONLY if explicitly stated as a number.
4. PROPOSED DRG - the DRG the payor wants to change it to (e.g., 872). ONLY if explicitly stated as a number.
5. IS SEPSIS RELATED - does this denial involve sepsis, severe sepsis, or septic shock?

CRITICAL: For DRG codes, return NONE unless you see an actual 3-digit DRG number explicitly written in the letter.
Do NOT guess or infer DRG codes. If the letter just says "adjusted" or "changed" without specific numbers, return NONE.

Return ONLY these lines (no JSON):
ACCOUNT_ID: [H-prefixed number or NONE]
PAYOR: [insurance company name]
ORIGINAL_DRG: [3-digit code ONLY if explicitly stated, otherwise NONE]
PROPOSED_DRG: [3-digit code ONLY if explicitly stated, otherwise NONE]
IS_SEPSIS: [YES or NO]'''


def transform_hsp_account_id(raw_id):
    """Transform HSP_ACCOUNT_ID from denial letter format to Clarity format."""
    if not raw_id:
        return None

    cleaned = str(raw_id).strip()

    if not cleaned.upper().startswith('H'):
        print(f"  Skipping non-H account ID: {cleaned}")
        return None

    cleaned = cleaned[1:]  # Remove H prefix

    if len(cleaned) > 2:
        cleaned = cleaned[:-2]  # Remove last 2 digits
    else:
        return None

    if cleaned and cleaned.isdigit():
        return cleaned

    return None


def extract_denial_info_llm(text):
    """Use LLM to extract denial info from text."""
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
        print(f"  LLM extraction error: {e}")
        return {
            "hsp_account_id": None,
            "payor": "Unknown",
            "original_drg": None,
            "proposed_drg": None,
            "is_sepsis": False
        }


print("Core functions loaded")

# =============================================================================
# CELL 6: Clarity Query Function (Single Account) - OPTIMIZED
# =============================================================================

# Note type mapping for pivoting
NOTE_TYPE_MAP = {
    'H&P': ('hp_note_id', 'hp_note_csn_id', 'hp_note_text'),
    'Discharge Summary': ('discharge_summary_note_id', 'discharge_note_csn_id', 'discharge_summary_text'),
    'Progress Notes': ('progress_note_id', 'progress_note_csn_id', 'progress_note_text'),
    'Consults': ('consult_note_id', 'consult_note_csn_id', 'consult_note_text'),
    'ED Notes': ('ed_notes_id', 'ed_notes_csn_id', 'ed_notes_text'),
    'Initial Assessments': ('initial_assessment_id', 'initial_assessment_csn_id', 'initial_assessment_text'),
    'ED Triage Notes': ('ed_triage_id', 'ed_triage_csn_id', 'ed_triage_text'),
    'ED Provider Notes': ('ed_provider_note_id', 'ed_provider_note_csn_id', 'ed_provider_note_text'),
    'Addendum Note': ('addendum_note_id', 'addendum_note_csn_id', 'addendum_note_text'),
    'Hospital Course': ('hospital_course_id', 'hospital_course_csn_id', 'hospital_course_text'),
    'Subjective & Objective': ('subjective_objective_id', 'subjective_objective_csn_id', 'subjective_objective_text'),
    'Assessment & Plan Note': ('assessment_plan_id', 'assessment_plan_csn_id', 'assessment_plan_text'),
    'Nursing Note': ('nursing_note_id', 'nursing_note_csn_id', 'nursing_note_text'),
    'Code Documentation': ('code_documentation_id', 'code_documentation_csn_id', 'code_documentation_text'),
}

def query_clarity_for_account(account_id):
    """
    Query Clarity for clinical notes for a single account.
    Uses two simple queries instead of one complex query for better performance.
    Returns dict with patient info and 14 clinical note types.
    """
    print(f"  Querying Clarity for account {account_id}...")

    # Query 1: Get patient info (fast, simple query)
    patient_query = f"""
    SELECT ha.hsp_account_id, patient.pat_id, patient.pat_mrn_id,
           CONCAT(patient.pat_first_name, ' ', patient.pat_last_name) AS formatted_name,
           DATE_FORMAT(patient.birth_date, 'MM/dd/yyyy') AS formatted_birthdate,
           'Mercy Hospital' AS facility_name,
           DATEDIFF(ha.disch_date_time, DATE(ha.adm_date_time)) AS number_of_midnights,
           CONCAT(DATE_FORMAT(ha.adm_date_time, 'MM/dd/yyyy'), '-', DATE_FORMAT(ha.disch_date_time, 'MM/dd/yyyy')) AS formatted_date_of_service
    FROM clarity_cur.hsp_account_enh ha
    INNER JOIN clarity_cur.patient_enh patient ON ha.pat_id = patient.pat_id
    WHERE ha.hsp_account_id = '{account_id}'
    """

    patient_rows = spark.sql(patient_query).collect()
    if not patient_rows:
        print(f"  WARNING: No patient found in Clarity for account {account_id}")
        return None

    clinical_data = patient_rows[0].asDict()
    print(f"  Found patient: {clinical_data.get('formatted_name', 'Unknown')}")

    # Query 2: Get all notes for this account (simpler query, pivot in Python)
    notes_query = f"""
    SELECT
        nte.ip_note_type,
        nte.note_id,
        nte.note_csn_id,
        nte.contact_date,
        nte.ent_inst_local_dttm,
        CONCAT_WS('\\n', SORT_ARRAY(COLLECT_LIST(STRUCT(nte.line, nte.note_text))).note_text) AS note_text
    FROM clarity_cur.pat_enc_hsp_har_enh peh
    INNER JOIN clarity_cur.hno_note_text_enh nte USING(pat_enc_csn_id)
    WHERE peh.hsp_account_id = '{account_id}'
      AND nte.ip_note_type IN (
        'Progress Notes', 'Consults', 'H&P', 'Discharge Summary',
        'ED Notes', 'Initial Assessments', 'ED Triage Notes', 'ED Provider Notes',
        'Addendum Note', 'Hospital Course', 'Subjective & Objective',
        'Assessment & Plan Note', 'Nursing Note', 'Code Documentation'
      )
    GROUP BY nte.ip_note_type, nte.note_id, nte.note_csn_id, nte.contact_date, nte.ent_inst_local_dttm
    """

    print(f"  Fetching clinical notes...")
    notes_rows = spark.sql(notes_query).collect()
    print(f"  Retrieved {len(notes_rows)} notes")

    # Pivot in Python: get most recent note per type
    notes_by_type = {}
    for row in notes_rows:
        note_type = row['ip_note_type']
        # Use (contact_date, entry_datetime) as sort key for most recent
        sort_key = (row['contact_date'], row['ent_inst_local_dttm'])

        if note_type not in notes_by_type or sort_key > notes_by_type[note_type][0]:
            notes_by_type[note_type] = (sort_key, row)

    # Map notes to expected column names
    for note_type, (id_col, csn_col, text_col) in NOTE_TYPE_MAP.items():
        if note_type in notes_by_type:
            _, row = notes_by_type[note_type]
            clinical_data[id_col] = str(row['note_id']) if row['note_id'] else 'no id available'
            clinical_data[csn_col] = str(row['note_csn_id']) if row['note_csn_id'] else 'no id available'
            clinical_data[text_col] = row['note_text'] if row['note_text'] else 'No Note Available'
        else:
            clinical_data[id_col] = 'no id available'
            clinical_data[csn_col] = 'no id available'
            clinical_data[text_col] = 'No Note Available'

    return clinical_data


print("Clarity query function loaded")

# =============================================================================
# CELL 7: Clinical Note Extraction (for long notes)
# =============================================================================

NOTE_EXTRACTION_PROMPT = '''Extract clinically relevant information from this {note_type}.

CRITICAL: For EVERY piece of information you extract, include the associated date/time if available.
Format timestamps consistently as: MM/DD/YYYY HH:MM or MM/DD/YYYY if time not available.

# Clinical Note
{note_text}

# What to Extract (with timestamps)
## Vital Signs, Laboratory Values, Infection Evidence, Organ Dysfunction, Clinical Events, Physician Assessments

# Output Format
Return a structured summary with timestamps. Example:
VITAL SIGNS:
- 03/15/2024 08:00: Temp 38.9°C, HR 112, BP 85/52 (MAP 63)

LABS:
- 03/15/2024 06:30: Lactate 4.2, WBC 18.5, Creatinine 2.1

Only include sections that have relevant data. Be thorough but concise.'''


def extract_clinical_data(note_text, note_type):
    """Extract clinically relevant data with timestamps from a long clinical note."""
    if not note_text or note_text == "No Note Available":
        return note_text

    if len(note_text) < NOTE_EXTRACTION_THRESHOLD:
        return note_text

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a clinical data extraction specialist. Extract relevant medical information with precise timestamps."},
                {"role": "user", "content": NOTE_EXTRACTION_PROMPT.format(note_type=note_type, note_text=note_text)}
            ],
            temperature=0.1,
            max_tokens=3000
        )

        extracted = response.choices[0].message.content.strip()
        print(f"    Extracted {note_type}: {len(note_text)} chars → {len(extracted)} chars")
        return extracted

    except Exception as e:
        print(f"    Warning: Extraction failed for {note_type}: {e}")
        return note_text[:NOTE_EXTRACTION_THRESHOLD] + "\n\n[Note truncated]"


def extract_notes_for_case(clinical_data):
    """Extract clinical data from all long notes for a case."""
    note_types = {
        "discharge_summary": ("discharge_summary_text", "Discharge Summary"),
        "hp_note": ("hp_note_text", "History & Physical"),
        "progress_note": ("progress_note_text", "Progress Notes"),
        "consult_note": ("consult_note_text", "Consult Notes"),
        "ed_notes": ("ed_notes_text", "ED Notes"),
        "initial_assessment": ("initial_assessment_text", "Initial Assessments"),
        "ed_triage": ("ed_triage_text", "ED Triage Notes"),
        "ed_provider_note": ("ed_provider_note_text", "ED Provider Notes"),
        "addendum_note": ("addendum_note_text", "Addendum Note"),
        "hospital_course": ("hospital_course_text", "Hospital Course"),
        "subjective_objective": ("subjective_objective_text", "Subjective & Objective"),
        "assessment_plan": ("assessment_plan_text", "Assessment & Plan Note"),
        "nursing_note": ("nursing_note_text", "Nursing Note"),
        "code_documentation": ("code_documentation_text", "Code Documentation"),
    }

    extracted_notes = {}
    notes_to_extract = []

    for key, (col_name, display_name) in note_types.items():
        note_text = clinical_data.get(col_name, "No Note Available")
        if note_text and note_text != "No Note Available":
            if len(note_text) >= NOTE_EXTRACTION_THRESHOLD:
                notes_to_extract.append((key, col_name, display_name, note_text))
            else:
                extracted_notes[key] = note_text
        else:
            extracted_notes[key] = "Not available"

    if notes_to_extract:
        print(f"  Extracting from {len(notes_to_extract)} long notes...")
        for key, col_name, display_name, note_text in notes_to_extract:
            extracted_notes[key] = extract_clinical_data(note_text, display_name)

    return extracted_notes


print("Note extraction functions loaded")

# =============================================================================
# CELL 8: Load Knowledge Base (Gold Letters + Propel)
# =============================================================================
print("\n" + "="*60)
print("LOADING KNOWLEDGE BASE")
print("="*60)

# Load gold letters
print("\nLoading gold standard letters...")
gold_letters_cache = []
try:
    gold_letters_df = spark.sql(f"""
        SELECT letter_id, source_file, payor, denial_text, rebuttal_text, denial_embedding, metadata
        FROM {GOLD_LETTERS_TABLE}
    """)
    gold_letters = gold_letters_df.collect()

    gold_letters_cache = [
        {
            "letter_id": row["letter_id"],
            "source_file": row["source_file"],
            "payor": row["payor"],
            "denial_text": row["denial_text"],
            "appeal_text": row["rebuttal_text"],
            "denial_embedding": list(row["denial_embedding"]) if row["denial_embedding"] else None,
            "metadata": dict(row["metadata"]) if row["metadata"] else {},
        }
        for row in gold_letters
    ]
    print(f"  Loaded {len(gold_letters_cache)} gold standard letters")
except Exception as e:
    print(f"  Warning: Could not load gold letters: {e}")

# Load default template
print("\nLoading default template...")
default_template = None
try:
    if os.path.exists(DEFAULT_TEMPLATE_PATH):
        from docx import Document as DocxDocument
        doc = DocxDocument(DEFAULT_TEMPLATE_PATH)
        template_text = "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])

        default_template = {
            "letter_id": "default_template",
            "source_file": os.path.basename(DEFAULT_TEMPLATE_PATH),
            "payor": "Generic",
            "denial_text": None,
            "appeal_text": template_text,
            "denial_embedding": None,
            "metadata": {"is_default_template": "true"},
        }
        print(f"  Loaded default template: {len(template_text)} chars")
    else:
        print(f"  Warning: Default template not found at {DEFAULT_TEMPLATE_PATH}")
except Exception as e:
    print(f"  Warning: Could not load default template: {e}")

# Load Propel definitions
print("\nLoading Propel clinical definitions...")
propel_definitions = {}
try:
    propel_df = spark.sql(f"""
        SELECT condition_name, definition_summary, definition_text
        FROM {PROPEL_DATA_TABLE}
    """)
    for row in propel_df.collect():
        definition = row["definition_summary"] or row["definition_text"]
        propel_definitions[row["condition_name"]] = definition
        print(f"  {row['condition_name']}: {len(definition)} chars")
except Exception as e:
    print(f"  Warning: Could not load propel definitions: {e}")

# =============================================================================
# CELL 9: Writer Prompt
# =============================================================================
WRITER_PROMPT = '''You are a clinical coding expert writing a DRG validation appeal letter for Mercy Hospital.

# Original Denial Letter
{denial_letter_text}

# Clinical Notes (PRIMARY EVIDENCE)
## Discharge Summary
{discharge_summary}

## H&P Note
{hp_note}

## Progress Notes
{progress_note}

## Consult Notes
{consult_note}

## ED Notes
{ed_notes}

## Initial Assessments
{initial_assessment}

## ED Triage Notes
{ed_triage}

## ED Provider Notes
{ed_provider_note}

## Addendum Note
{addendum_note}

## Hospital Course
{hospital_course}

## Subjective & Objective
{subjective_objective}

## Assessment & Plan Note
{assessment_plan}

## Nursing Note
{nursing_note}

## Code Documentation
{code_documentation}

# Official Clinical Definition
{clinical_definition_section}

# Gold Standard Letter
{gold_letter_section}

# Patient Information
Name: {patient_name}
DOB: {patient_dob}
Hospital Account #: {hsp_account_id}
Date of Service: {date_of_service}
Facility: {facility_name}
Original DRG: {original_drg}
Proposed DRG: {proposed_drg}
Payor: {payor}

# Instructions
{gold_letter_instructions}
1. READ THE DENIAL LETTER - extract the payor address, reviewer name, claim numbers
2. ADDRESS EACH DENIAL ARGUMENT - quote the payer, then refute
3. CITE CLINICAL EVIDENCE from provider notes FIRST
4. INCLUDE TIMESTAMPS with clinical values
5. Follow the Mercy Hospital template structure exactly

Return ONLY the letter text.'''

# =============================================================================
# CELL 10: Main Processing - Single Letter
# =============================================================================
print("\n" + "="*60)
print("PROCESSING DENIAL LETTER")
print("="*60)

# Check input file exists
if not os.path.exists(DENIAL_PDF_PATH):
    print(f"\nERROR: Denial PDF not found: {DENIAL_PDF_PATH}")
    print("Set DENIAL_PDF_PATH to a valid PDF file path.")
else:
    print(f"\nInput: {os.path.basename(DENIAL_PDF_PATH)}")

    # ---------------------------------------------------------------------
    # STEP 1: Parse denial PDF
    # ---------------------------------------------------------------------
    print("\nStep 1: Parsing denial PDF...")
    pages = extract_text_from_pdf(DENIAL_PDF_PATH)
    denial_text = "\n\n".join(pages)
    print(f"  Extracted {len(pages)} pages, {len(denial_text)} chars")

    # ---------------------------------------------------------------------
    # STEP 2: Find similar gold standard letter (only needs denial text)
    # ---------------------------------------------------------------------
    print("\nStep 2: Finding similar gold standard letter...")
    denial_embedding = generate_embedding(denial_text)
    print(f"  Generated embedding ({len(denial_embedding)} dims)")

    gold_letter = None
    gold_letter_score = 0.0

    if gold_letters_cache:
        best_score = 0.0
        best_match = None

        for letter in gold_letters_cache:
            if letter["denial_embedding"]:
                vec1 = denial_embedding
                vec2 = letter["denial_embedding"]
                dot_product = sum(a * b for a, b in zip(vec1, vec2))
                norm1 = math.sqrt(sum(a * a for a in vec1))
                norm2 = math.sqrt(sum(b * b for b in vec2))
                if norm1 > 0 and norm2 > 0:
                    similarity = dot_product / (norm1 * norm2)
                    if similarity > best_score:
                        best_score = similarity
                        best_match = letter

        gold_letter_score = best_score

        if best_match and best_score >= MATCH_SCORE_THRESHOLD:
            gold_letter = best_match
            print(f"  Found match: {gold_letter['source_file']} (score: {best_score:.3f})")
        elif default_template:
            gold_letter = default_template
            gold_letter_score = 0.0
            print(f"  No good match (best: {best_score:.3f}) - using default template")
        else:
            print(f"  No good match (best: {best_score:.3f})")

    # ---------------------------------------------------------------------
    # STEP 3: Extract denial info via LLM
    # ---------------------------------------------------------------------
    print("\nStep 3: Extracting denial info via LLM...")
    denial_info = extract_denial_info_llm(denial_text[:15000])
    print(f"  Account ID: {denial_info['hsp_account_id'] or 'NOT FOUND'}")
    print(f"  Payor: {denial_info['payor']}")
    print(f"  DRG: {denial_info['original_drg']} → {denial_info['proposed_drg']}")
    print(f"  Sepsis: {denial_info['is_sepsis']}")

    # Use known account ID if provided (production mode)
    account_id = KNOWN_ACCOUNT_ID or denial_info['hsp_account_id']

    if not account_id:
        print("\nERROR: No account ID found. Set KNOWN_ACCOUNT_ID or ensure denial letter contains H-prefixed account number.")
    else:
        # Check scope
        if SCOPE_FILTER == "sepsis" and not denial_info['is_sepsis']:
            print("\nWARNING: Denial not identified as sepsis-related. Proceeding anyway (set SCOPE_FILTER='all' to skip this check).")

        # ---------------------------------------------------------------------
        # STEP 4: Query Clarity for clinical notes
        # ---------------------------------------------------------------------
        print(f"\nStep 4: Querying Clarity for account {account_id}...")
        clinical_data = query_clarity_for_account(account_id)

        if clinical_data:
            # ---------------------------------------------------------------------
            # STEP 5: Extract clinical notes
            # ---------------------------------------------------------------------
            print("\nStep 5: Processing clinical notes...")
            extracted_notes = extract_notes_for_case(clinical_data)

            # ---------------------------------------------------------------------
            # STEP 6: Generate appeal letter
            # ---------------------------------------------------------------------
            print("\nStep 6: Generating appeal letter...")

            # Build gold letter section
            if gold_letter:
                is_default = gold_letter.get("metadata", {}).get("is_default_template") == "true"
                if is_default:
                    gold_letter_section = f"## APPEAL TEMPLATE - USE AS STRUCTURAL GUIDE\n{gold_letter['appeal_text']}"
                    gold_letter_instructions = "**NOTE: Using default template as structural guide.**\n"
                else:
                    gold_letter_section = f"## WINNING APPEAL (Score: {gold_letter_score:.3f})\nPayor: {gold_letter.get('payor')}\n\n{gold_letter['appeal_text']}"
                    gold_letter_instructions = "**CRITICAL: Learn from this winning appeal - adapt to this patient.**\n"
            else:
                gold_letter_section = "No template available."
                gold_letter_instructions = ""

            # Clinical definition section
            if denial_info['is_sepsis'] and "sepsis" in propel_definitions:
                clinical_definition_section = f"## OFFICIAL SEPSIS DEFINITION\n{propel_definitions['sepsis']}"
            else:
                clinical_definition_section = "No specific definition loaded."

            # Build prompt
            writer_prompt = WRITER_PROMPT.format(
                denial_letter_text=denial_text,
                discharge_summary=extracted_notes.get("discharge_summary", "Not available"),
                hp_note=extracted_notes.get("hp_note", "Not available"),
                progress_note=extracted_notes.get("progress_note", "Not available"),
                consult_note=extracted_notes.get("consult_note", "Not available"),
                ed_notes=extracted_notes.get("ed_notes", "Not available"),
                initial_assessment=extracted_notes.get("initial_assessment", "Not available"),
                ed_triage=extracted_notes.get("ed_triage", "Not available"),
                ed_provider_note=extracted_notes.get("ed_provider_note", "Not available"),
                addendum_note=extracted_notes.get("addendum_note", "Not available"),
                hospital_course=extracted_notes.get("hospital_course", "Not available"),
                subjective_objective=extracted_notes.get("subjective_objective", "Not available"),
                assessment_plan=extracted_notes.get("assessment_plan", "Not available"),
                nursing_note=extracted_notes.get("nursing_note", "Not available"),
                code_documentation=extracted_notes.get("code_documentation", "Not available"),
                clinical_definition_section=clinical_definition_section,
                gold_letter_section=gold_letter_section,
                gold_letter_instructions=gold_letter_instructions,
                patient_name=clinical_data.get("formatted_name", ""),
                patient_dob=clinical_data.get("formatted_birthdate", ""),
                hsp_account_id=account_id,
                date_of_service=clinical_data.get("formatted_date_of_service", ""),
                facility_name=clinical_data.get("facility_name", "Mercy Hospital"),
                original_drg=denial_info['original_drg'] or "Unknown",
                proposed_drg=denial_info['proposed_drg'] or "Unknown",
                payor=denial_info['payor'],
            )

            # Generate letter
            writer_response = openai_client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "You are a clinical coding expert writing DRG appeal letters. Prioritize evidence from provider notes."},
                    {"role": "user", "content": writer_prompt}
                ],
                temperature=0.2,
                max_tokens=4000
            )

            letter_text = writer_response.choices[0].message.content.strip()
            print(f"  Generated {len(letter_text)} character letter")

            # ---------------------------------------------------------------------
            # STEP 7: Export to DOCX
            # ---------------------------------------------------------------------
            if EXPORT_TO_DOCX:
                print("\nStep 7: Exporting to DOCX...")
                from docx import Document
                from docx.shared import Pt

                # Ensure output folder exists
                os.makedirs(DOCX_OUTPUT_BASE, exist_ok=True)

                # Create document
                doc = Document()
                doc.add_heading('Appeal Letter', level=1)

                # Metadata section
                meta = doc.add_paragraph()
                meta.add_run("Generated: ").bold = True
                meta.add_run(f"{datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
                meta.add_run("Patient: ").bold = True
                meta.add_run(f"{clinical_data.get('formatted_name', 'Unknown')}\n")
                meta.add_run("Payor: ").bold = True
                meta.add_run(f"{denial_info['payor']}\n")
                meta.add_run("DRG: ").bold = True
                meta.add_run(f"{denial_info['original_drg']} → {denial_info['proposed_drg']}\n")
                meta.add_run("Gold Letter: ").bold = True
                meta.add_run(f"{gold_letter['source_file'] if gold_letter else 'None'} (score: {gold_letter_score:.3f})\n")

                doc.add_paragraph("─" * 60)

                # Letter content
                for paragraph in letter_text.split('\n\n'):
                    if paragraph.strip():
                        p = doc.add_paragraph(paragraph.strip())
                        p.paragraph_format.space_after = Pt(12)

                # Save
                patient_name = clinical_data.get('formatted_name', 'Unknown')
                safe_name = "".join(c for c in patient_name if c.isalnum() or c in (' ', '-', '_')).strip()
                filename = f"{account_id}_{safe_name}_appeal.docx"
                filepath = os.path.join(DOCX_OUTPUT_BASE, filename)
                doc.save(filepath)

                print(f"  Saved: {filepath}")

            # ---------------------------------------------------------------------
            # SUMMARY
            # ---------------------------------------------------------------------
            print("\n" + "="*60)
            print("PROCESSING COMPLETE")
            print("="*60)
            print(f"Patient: {clinical_data.get('formatted_name', 'Unknown')}")
            print(f"Account: {account_id}")
            print(f"Letter length: {len(letter_text)} chars")
            if EXPORT_TO_DOCX:
                print(f"Output: {filepath}")

        else:
            print("\nERROR: Could not retrieve clinical data from Clarity.")
            print("Check that the account ID exists and has clinical notes.")

print("\nInference complete.")
