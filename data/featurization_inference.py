# data/featurization_inference.py
# Sepsis Appeal Engine - Per-Case Data Preparation
#
# PER-CASE FEATURIZATION: Prepares all data needed for a single appeal:
# 1. Parse denial PDF → Extract text, account ID, payor, DRGs
# 2. Query clinical notes → ALL notes from 14 types from Epic Clarity
# 3. Extract clinical notes → LLM summarization of long notes
# 4. Query structured data → Labs, vitals, meds, diagnoses
# 5. Extract structured data → LLM summarization for sepsis evidence
# 6. Conflict detection → Flag discrepancies between notes and structured data
# 7. Write to case tables → Ready for inference.py to read
#
# OUTPUT: Case tables in Unity Catalog for inference.py
#
# Run on Databricks Runtime 15.4 LTS ML

# Uncomment and run ONCE per cluster session:
# %pip install azure-ai-documentintelligence==1.0.2 openai python-docx
# dbutils.library.restartPython()

# =============================================================================
# CELL 1: Configuration
# =============================================================================
import os
import re
import json
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType, BooleanType, TimestampType

spark = SparkSession.builder.getOrCreate()

# -----------------------------------------------------------------------------
# INPUT: Set the denial PDF to process
# -----------------------------------------------------------------------------
DENIAL_PDF_PATH = "/Workspace/Repos/mijo8881@mercy.net/fudgesicle/utils/sample_denial_letters/example_denial.pdf"

# If account ID is known (production), set it here. Otherwise LLM will extract from PDF.
KNOWN_ACCOUNT_ID = None  # e.g., "12345678" or None to extract

# -----------------------------------------------------------------------------
# Processing Configuration
# -----------------------------------------------------------------------------
NOTE_EXTRACTION_THRESHOLD = 8000  # Chars - notes longer than this get LLM extraction
EMBEDDING_MODEL = "text-embedding-ada-002"

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
# NOTE: Data lives in prod catalog, but we write to our environment's catalog.
# This is intentional - we query from prod but can only write to our own env.
trgt_cat = os.environ.get('trgt_cat', 'dev')
spark.sql('USE CATALOG prod;')

# -----------------------------------------------------------------------------
# Output Tables (case data for inference.py to read)
# -----------------------------------------------------------------------------
CASE_DENIAL_TABLE = f"{trgt_cat}.fin_ds.fudgesicle_case_denial"
CASE_CLINICAL_TABLE = f"{trgt_cat}.fin_ds.fudgesicle_case_clinical"
CASE_STRUCTURED_SUMMARY_TABLE = f"{trgt_cat}.fin_ds.fudgesicle_case_structured_summary"
CASE_CONFLICTS_TABLE = f"{trgt_cat}.fin_ds.fudgesicle_case_conflicts"

# Structured data tables (intermediate)
LABS_TABLE = f"{trgt_cat}.fin_ds.fudgesicle_labs"
VITALS_TABLE = f"{trgt_cat}.fin_ds.fudgesicle_vitals"
MEDS_TABLE = f"{trgt_cat}.fin_ds.fudgesicle_meds"
DIAGNOSIS_TABLE = f"{trgt_cat}.fin_ds.fudgesicle_diagnoses"
MERGED_TABLE = f"{trgt_cat}.fin_ds.fudgesicle_structured_timeline"

print(f"Denial PDF: {DENIAL_PDF_PATH}")
print(f"Catalog: {trgt_cat}")

# =============================================================================
# CELL 2: Azure Credentials and Clients
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
# CELL 3: PDF Parsing and Denial Extraction Functions
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
    # text-embedding-ada-002 has 8191 token limit (~32k chars). Use 30k for safety buffer.
    if len(text) > 30000:
        print(f"  Warning: Text truncated from {len(text)} to 30000 chars for embedding")
        text = text[:30000]

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


print("PDF parsing functions loaded")

# =============================================================================
# CELL 4: Clinical Notes Query Functions
# =============================================================================

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
    Returns dict with patient info and ALL notes for each of 14 clinical note types.
    Notes are concatenated chronologically with timestamps.
    """
    print(f"  Querying Clarity for account {account_id}...")

    # Query 1: Get patient info
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

    # Query 2: Get ALL notes for this account (all encounters associated with this HSP_ACCOUNT_ID)
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
    ORDER BY nte.contact_date ASC, nte.ent_inst_local_dttm ASC
    """

    print(f"  Fetching clinical notes...")
    notes_rows = spark.sql(notes_query).collect()
    print(f"  Retrieved {len(notes_rows)} total notes")

    # Group ALL notes by type (not just most recent)
    notes_by_type = {}
    for row in notes_rows:
        note_type = row['ip_note_type']
        if note_type not in notes_by_type:
            notes_by_type[note_type] = []
        notes_by_type[note_type].append(row)

    # Report counts per type
    for note_type, notes in notes_by_type.items():
        print(f"    {note_type}: {len(notes)} notes")

    # Concatenate all notes of each type with timestamps
    for note_type, (id_col, csn_col, text_col) in NOTE_TYPE_MAP.items():
        if note_type in notes_by_type:
            notes_list = notes_by_type[note_type]
            # Concatenate all notes with timestamps
            combined_text_parts = []
            note_ids = []
            csn_ids = []
            for row in notes_list:
                timestamp = row['ent_inst_local_dttm'] or row['contact_date'] or 'Unknown time'
                note_text = row['note_text'] if row['note_text'] else ''
                if note_text:
                    combined_text_parts.append(f"[{timestamp}]\n{note_text}")
                if row['note_id']:
                    note_ids.append(str(row['note_id']))
                if row['note_csn_id']:
                    csn_ids.append(str(row['note_csn_id']))

            clinical_data[id_col] = ', '.join(note_ids) if note_ids else 'no id available'
            clinical_data[csn_col] = ', '.join(csn_ids) if csn_ids else 'no id available'
            clinical_data[text_col] = '\n\n---\n\n'.join(combined_text_parts) if combined_text_parts else 'No Note Available'
        else:
            clinical_data[id_col] = 'no id available'
            clinical_data[csn_col] = 'no id available'
            clinical_data[text_col] = 'No Note Available'

    return clinical_data


print("Clinical notes query functions loaded")

# =============================================================================
# CELL 5: Clinical Note Extraction (LLM)
# =============================================================================

NOTE_EXTRACTION_PROMPT = '''Extract clinically relevant information from this {note_type}.

CRITICAL: For EVERY piece of information you extract, include the associated date/time if available.
Format timestamps consistently as: MM/DD/YYYY HH:MM or MM/DD/YYYY if time not available.

# Clinical Note
{note_text}

# What to Extract (with timestamps)

## SOFA Score Components (PRIORITY - extract ALL available)
- Respiration: PaO2/FiO2 ratio, SpO2/FiO2, oxygen requirements, ventilator settings
- Coagulation: Platelet count
- Liver: Bilirubin (total)
- Cardiovascular: MAP, hypotension, vasopressor use (drug, dose)
- CNS: GCS (Glasgow Coma Scale), mental status changes
- Renal: Creatinine, urine output

## Other Sepsis Markers
- Lactate levels (CRITICAL - include all values with times)
- WBC count, bands
- Temperature (fever, hypothermia)
- Heart rate, respiratory rate
- Blood culture results, infection source
- Antibiotic administration times

## Clinical Events
- Fluid resuscitation (volume, timing)
- ICU admission/transfer
- Physician assessments mentioning sepsis, SIRS, infection

# Output Format
Return a structured summary with timestamps. Be thorough but concise.'''


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
            temperature=0,
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


print("Clinical note extraction functions loaded")

# =============================================================================
# CELL 6: Structured Data Query Functions
# =============================================================================

def create_target_encounter_view(account_id):
    """Create temp view for target encounter."""
    spark.sql(f"""
    CREATE OR REPLACE TEMP VIEW target_encounter AS
    SELECT
        peh.HSP_ACCOUNT_ID,
        peh.PAT_ENC_CSN_ID,
        peh.PAT_ID,
        peh.ADM_DATE_TIME AS ENCOUNTER_START,
        peh.DISCH_DATE_TIME AS ENCOUNTER_END
    FROM prod.clarity_cur.pat_enc_hsp_har_enh peh
    WHERE peh.HSP_ACCOUNT_ID = {account_id}
    """)
    spark.sql("CACHE TABLE target_encounter")


def query_labs(account_id):
    """Query all labs for account."""
    spark.sql(f"""
    CREATE OR REPLACE TABLE {LABS_TABLE} AS
    SELECT
        t.HSP_ACCOUNT_ID,
        t.PAT_ENC_CSN_ID,
        CAST(res_comp.COMP_VERIF_DTTM AS TIMESTAMP) AS EVENT_TIMESTAMP,
        cc.NAME AS LAB_NAME,
        CAST(REGEXP_REPLACE(res_comp.COMPONENT_VALUE, '>', '') AS STRING) AS lab_value,
        res_comp.component_units AS lab_units,
        zsab.NAME AS abnormal_flag
    FROM target_encounter t
    INNER JOIN prod.clarity.order_proc op ON t.PAT_ENC_CSN_ID = op.PAT_ENC_CSN_ID
    INNER JOIN prod.clarity.RES_DB_MAIN rdm ON rdm.RES_ORDER_ID = op.ORDER_PROC_ID
    INNER JOIN prod.clarity.res_components res_comp ON res_comp.result_id = rdm.result_id
    INNER JOIN prod.clarity.clarity_component cc ON cc.component_id = res_comp.component_id
    LEFT JOIN prod.clarity.zc_stat_abnorms zsab ON zsab.stat_abnorms_c = res_comp.component_abn_c
    WHERE op.order_status_c = 5
      AND op.lab_status_c IN (3, 5)
      AND rdm.res_val_status_c = 9
      AND res_comp.COMPONENT_VALUE IS NOT NULL
      AND res_comp.COMPONENT_VALUE <> '-1'
    ORDER BY res_comp.COMP_VERIF_DTTM ASC
    """)
    count = spark.sql(f"SELECT COUNT(*) as cnt FROM {LABS_TABLE}").collect()[0]["cnt"]
    print(f"  Labs: {count} rows")


def query_vitals(account_id):
    """Query vitals for account."""
    spark.sql(f"""
    CREATE OR REPLACE TABLE {VITALS_TABLE} AS
    SELECT
        t.HSP_ACCOUNT_ID,
        t.PAT_ENC_CSN_ID,
        CAST(to_timestamp(substring(v.RECORDED_TIME, 1, 19), 'yyyy-MM-dd HH:mm:ss') AS TIMESTAMP) AS EVENT_TIMESTAMP,
        v.FLO_MEAS_NAME AS VITAL_NAME,
        v.MEAS_VALUE AS vital_value
    FROM target_encounter t
    INNER JOIN prod.clarity_cur.ip_flwsht_rec_enh v ON t.PAT_ENC_CSN_ID = v.IP_DATA_STORE_EPT_CSN
    WHERE v.FLO_MEAS_ID IN ('5', '6', '8', '9', '10', '11', '14')
      AND v.MEAS_VALUE IS NOT NULL
    ORDER BY EVENT_TIMESTAMP ASC
    """)
    count = spark.sql(f"SELECT COUNT(*) as cnt FROM {VITALS_TABLE}").collect()[0]["cnt"]
    print(f"  Vitals: {count} rows")


def query_meds(account_id):
    """Query medications for account."""
    spark.sql(f"""
    CREATE OR REPLACE TABLE {MEDS_TABLE} AS
    SELECT
        t.HSP_ACCOUNT_ID,
        t.PAT_ENC_CSN_ID,
        CAST(mar.TAKEN_TIME AS TIMESTAMP) AS EVENT_TIMESTAMP,
        om.SIMPLE_GENERIC_NAME AS MED_NAME,
        CAST(om.HV_DISCRETE_DOSE AS STRING) AS MED_DOSE,
        om.DOSE_UNIT AS MED_UNITS,
        mar.ROUTE AS MED_ROUTE,
        mar.ACTION AS ADMIN_ACTION
    FROM target_encounter t
    INNER JOIN prod.clarity_cur.order_med_enh om ON t.PAT_ENC_CSN_ID = om.PAT_ENC_CSN_ID
    INNER JOIN prod.clarity_cur.mar_admin_info_enh mar ON om.ORDER_MED_ID = mar.ORDER_MED_ID
    WHERE mar.ACTION IN (
        'Given', 'Patient/Family Admin', 'Given-See Override',
        'Admin by Another Clinician (Comment)', 'New Bag', 'Bolus', 'Push',
        'Started by Another Clinician', 'Bag Switched', 'Clinic Sample Administered',
        'Applied', 'Feeding Started', 'Acknowledged', 'Contrast Given',
        'New Bag-See Override', 'Bolus from Bag'
    )
    ORDER BY EVENT_TIMESTAMP ASC
    """)
    count = spark.sql(f"SELECT COUNT(*) as cnt FROM {MEDS_TABLE}").collect()[0]["cnt"]
    print(f"  Medications: {count} rows")


def query_diagnoses(account_id):
    """
    Query diagnosis records for account (DX_NAME is the granular clinical description).
    All diagnoses include their timestamp - LLM decides relevance based on date.
    """
    spark.sql(f"""
    CREATE OR REPLACE TEMP VIEW encounter_diagnoses AS
    -- Outpatient encounter diagnoses
    SELECT te.HSP_ACCOUNT_ID, te.PAT_ENC_CSN_ID,
           dd.DX_ID,
           edg.DX_NAME,
           CAST(pe.CONTACT_DATE AS TIMESTAMP) AS EVENT_TIMESTAMP,
           'OUTPATIENT_ENC_DX' AS source
    FROM target_encounter te
    JOIN prod.clarity_cur.pat_enc_dx_enh dd ON dd.PAT_ENC_CSN_ID = te.PAT_ENC_CSN_ID
    JOIN prod.clarity_cur.pat_enc_enh pe ON pe.PAT_ENC_CSN_ID = dd.PAT_ENC_CSN_ID
    LEFT JOIN prod.clarity.clarity_edg edg ON dd.DX_ID = edg.DX_ID
    WHERE dd.DX_ID IS NOT NULL
    UNION ALL
    -- Inpatient hospital account diagnoses
    SELECT te.HSP_ACCOUNT_ID, te.PAT_ENC_CSN_ID,
           dx.DX_ID,
           edg.DX_NAME,
           CAST(ha.DISCH_DATE_TIME AS TIMESTAMP) AS EVENT_TIMESTAMP,
           'INPATIENT_ACCT_DX' AS source
    FROM target_encounter te
    JOIN prod.clarity_cur.hsp_acct_dx_list_enh dx ON dx.PAT_ID = te.PAT_ID
    JOIN prod.clarity_cur.pat_enc_hsp_har_enh ha ON ha.HSP_ACCOUNT_ID = dx.HSP_ACCOUNT_ID
    LEFT JOIN prod.clarity.clarity_edg edg ON dx.DX_ID = edg.DX_ID
    WHERE dx.DX_ID IS NOT NULL
    UNION ALL
    -- Problem list history (uses HX fields directly)
    SELECT te.HSP_ACCOUNT_ID, te.PAT_ENC_CSN_ID,
           phx.HX_PROBLEM_ID AS DX_ID,
           phx.HX_PROBLEM_DX_NAME AS DX_NAME,
           CAST(phx.HX_DATE_OF_ENTRY AS TIMESTAMP) AS EVENT_TIMESTAMP,
           'PROBLEM_LIST' AS source
    FROM target_encounter te
    JOIN prod.clarity_cur.problem_list_hx_enh phx ON phx.PAT_ID = te.PAT_ID
    WHERE phx.HX_PROBLEM_ID IS NOT NULL AND phx.HX_STATUS = 'Active'
    """)

    spark.sql(f"""
    CREATE OR REPLACE TABLE {DIAGNOSIS_TABLE} AS
    SELECT DISTINCT HSP_ACCOUNT_ID, PAT_ENC_CSN_ID, DX_ID, DX_NAME,
           EVENT_TIMESTAMP, source
    FROM encounter_diagnoses
    WHERE DX_NAME IS NOT NULL
    ORDER BY EVENT_TIMESTAMP ASC
    """)
    count = spark.sql(f"SELECT COUNT(*) as cnt FROM {DIAGNOSIS_TABLE}").collect()[0]["cnt"]
    print(f"  Diagnoses: {count} rows")


def create_merged_timeline(account_id):
    """Merge all structured data into chronological timeline."""
    spark.sql(f"""
    CREATE OR REPLACE TABLE {MERGED_TABLE} AS
    WITH RawEvents AS (
        SELECT HSP_ACCOUNT_ID, PAT_ENC_CSN_ID, EVENT_TIMESTAMP, 'LAB' AS event_type,
               CONCAT(LAB_NAME, ': ', lab_value, ' ', COALESCE(lab_units, ''),
                      CASE WHEN abnormal_flag IS NOT NULL THEN CONCAT(' (', abnormal_flag, ')') ELSE '' END
               ) AS event_detail,
               CASE WHEN abnormal_flag IS NOT NULL THEN 1 ELSE 0 END AS is_abnormal
        FROM {LABS_TABLE}
        UNION ALL
        SELECT HSP_ACCOUNT_ID, PAT_ENC_CSN_ID, EVENT_TIMESTAMP, 'VITAL' AS event_type,
               CONCAT(VITAL_NAME, ': ', vital_value) AS event_detail, 0 AS is_abnormal
        FROM {VITALS_TABLE}
        UNION ALL
        SELECT HSP_ACCOUNT_ID, PAT_ENC_CSN_ID, EVENT_TIMESTAMP, 'MEDICATION' AS event_type,
               CONCAT(MED_NAME, ' ', COALESCE(MED_DOSE, ''), ' ', COALESCE(MED_UNITS, ''),
                      ' via ', COALESCE(MED_ROUTE, 'unknown'), ' - ', ADMIN_ACTION
               ) AS event_detail, 0 AS is_abnormal
        FROM {MEDS_TABLE}
        UNION ALL
        SELECT HSP_ACCOUNT_ID, PAT_ENC_CSN_ID, EVENT_TIMESTAMP, 'DIAGNOSIS' AS event_type,
               COALESCE(DX_NAME, 'Unknown diagnosis') AS event_detail,
               0 AS is_abnormal
        FROM {DIAGNOSIS_TABLE}
        WHERE EVENT_TIMESTAMP IS NOT NULL
    ),
    Deduplicated AS (
        SELECT *, ROW_NUMBER() OVER (
            PARTITION BY HSP_ACCOUNT_ID, PAT_ENC_CSN_ID, EVENT_TIMESTAMP, event_type, event_detail
            ORDER BY EVENT_TIMESTAMP
        ) as rn
        FROM RawEvents
    )
    SELECT HSP_ACCOUNT_ID, PAT_ENC_CSN_ID, EVENT_TIMESTAMP, event_type, event_detail, is_abnormal
    FROM Deduplicated WHERE rn = 1
    ORDER BY EVENT_TIMESTAMP ASC
    """)
    count = spark.sql(f"SELECT COUNT(*) as cnt FROM {MERGED_TABLE}").collect()[0]["cnt"]
    print(f"  Merged timeline: {count} rows")


print("Structured data query functions loaded")

# =============================================================================
# CELL 7: Structured Data Extraction (LLM)
# =============================================================================

STRUCTURED_DATA_EXTRACTION_PROMPT = '''You are a clinical data analyst extracting sepsis-relevant information from structured EHR data.

**Context on Diagnosis Records:**
The diagnosis names are the granular clinical descriptions from Epic's diagnosis dictionary. Quote these directly in appeals - they are the specific documented diagnoses. For example:
- "Severe sepsis with septic shock due to Methicillin-susceptible Staphylococcus aureus"
- "Sepsis due to Escherichia coli"

Multiple diagnosis records may describe the same condition at different levels of specificity. Use the most specific documented diagnosis that is supported by clinical evidence.

Diagnoses include timestamps - use these to understand if a condition is pre-existing (before admission) or documented during the encounter.

**Your Task:**
Extract a focused summary of sepsis-relevant clinical data from this timeline. Prioritize:

1. **SOFA Score Components** (with timestamps and trends):
   - Respiratory: PaO2/FiO2, SpO2, oxygen requirements
   - Coagulation: Platelet count
   - Liver: Bilirubin
   - Cardiovascular: MAP, vasopressor use with doses
   - CNS: GCS
   - Renal: Creatinine, urine output

2. **Sepsis Bundle Compliance**:
   - Time of suspected infection
   - Antibiotic administration (within 3 hours?)
   - Lactate measurement and remeasurement (within 6 hours if elevated?)
   - Fluid resuscitation (30 mL/kg within 3 hours if hypotensive/lactate ≥4?)
   - Vasopressor initiation (if MAP <65 after fluids?)

3. **Clinical Trajectory**:
   - When did patient meet sepsis criteria?
   - Worst values and when they occurred
   - Evidence of organ dysfunction

4. **Relevant Diagnoses** (with dates - note which are pre-existing vs new)

**Structured Timeline:**
{structured_timeline}

**Output Format:**
Provide a concise clinical summary (500-800 words) organized by the categories above, with specific timestamps and values. Flag any data gaps.'''


def extract_structured_data_summary(account_id):
    """Extract sepsis-relevant summary from structured data timeline."""
    print("  Extracting structured data summary...")

    # Get timeline data
    timeline_df = spark.sql(f"""
        SELECT EVENT_TIMESTAMP, event_type, event_detail, is_abnormal
        FROM {MERGED_TABLE}
        ORDER BY EVENT_TIMESTAMP
        LIMIT 500
    """)
    timeline_rows = timeline_df.collect()

    if not timeline_rows:
        return "No structured data available for this encounter."

    # Format timeline for LLM
    timeline_text = "\n".join([
        f"[{row['EVENT_TIMESTAMP']}] {row['event_type']}: {row['event_detail']}"
        + (" (ABNORMAL)" if row['is_abnormal'] else "")
        for row in timeline_rows
    ])

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a clinical data analyst specializing in sepsis cases."},
                {"role": "user", "content": STRUCTURED_DATA_EXTRACTION_PROMPT.format(structured_timeline=timeline_text)}
            ],
            temperature=0,
            max_tokens=2000
        )
        summary = response.choices[0].message.content.strip()
        print(f"  Structured data summary: {len(summary)} chars")
        return summary

    except Exception as e:
        print(f"  Warning: Structured data extraction failed: {e}")
        return f"Extraction failed. Raw timeline has {len(timeline_rows)} events."


print("Structured data extraction functions loaded")

# =============================================================================
# CELL 8: Conflict Detection
# =============================================================================

CONFLICT_DETECTION_PROMPT = '''You are comparing clinical documentation from two sources for the same patient encounter:

1. **PHYSICIAN NOTES** (primary source - clinical interpretation):
{notes_summary}

2. **STRUCTURED DATA** (objective measurements):
{structured_summary}

**Your Task:**
Identify any CONFLICTS where the physician notes say one thing but the structured data shows something different.

Examples of conflicts:
- Note says "MAP maintained >65" but vitals show MAP values below 65
- Note says "lactate normalized" but labs show lactate still elevated (>2.0)
- Note says "no vasopressors needed" but meds show vasopressor administration
- Note says "patient alert and oriented" but GCS recorded as <15
- Note says "afebrile" but temps >38°C documented

**Important:**
- Only flag CLEAR contradictions, not missing information
- Note the specific values from each source
- Consider timing - data from different times is not a conflict

**Output Format:**
If conflicts found, list each one:
CONFLICT 1: [Notes say X, but structured data shows Y]
CONFLICT 2: [Notes say X, but structured data shows Y]

If no conflicts: "NO CONFLICTS DETECTED"

Then provide:
RECOMMENDATION: [Brief guidance for CDI reviewer]'''


def detect_conflicts(notes_summary, structured_summary):
    """Detect conflicts between clinical notes and structured data."""
    print("  Running conflict detection...")

    if not notes_summary or not structured_summary:
        return {"conflicts": [], "recommendation": "Insufficient data for conflict detection"}

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a clinical documentation integrity specialist."},
                {"role": "user", "content": CONFLICT_DETECTION_PROMPT.format(
                    notes_summary=notes_summary[:8000],
                    structured_summary=structured_summary[:8000]
                )}
            ],
            temperature=0,
            max_tokens=1000
        )

        result_text = response.choices[0].message.content.strip()

        # Parse conflicts - only capture lines with actual content after "CONFLICT X:"
        conflicts = []
        recommendation = ""

        lines = result_text.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("CONFLICT"):
                # Check if there's actual content after "CONFLICT X:"
                if ":" in line:
                    after_colon = line.split(":", 1)[1].strip() if len(line.split(":", 1)) > 1 else ""
                    if after_colon:  # Only add if there's content after the colon
                        conflicts.append(line)
            elif line.startswith("RECOMMENDATION:"):
                recommendation = line.replace("RECOMMENDATION:", "").strip()

        if "NO CONFLICTS DETECTED" in result_text:
            print("  No conflicts detected")
        else:
            print(f"  Found {len(conflicts)} conflicts")

        return {
            "conflicts": conflicts,
            "recommendation": recommendation,
            "raw_response": result_text
        }

    except Exception as e:
        print(f"  Warning: Conflict detection failed: {e}")
        return {"conflicts": [], "recommendation": f"Detection failed: {e}"}


print("Conflict detection functions loaded")

# =============================================================================
# CELL 9: Write to Case Tables
# =============================================================================

def write_case_denial_table(account_id, denial_text, denial_embedding, denial_info):
    """Write denial data to case table."""
    spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {CASE_DENIAL_TABLE} (
        account_id STRING,
        denial_text STRING,
        denial_embedding ARRAY<FLOAT>,
        payor STRING,
        original_drg STRING,
        proposed_drg STRING,
        is_sepsis BOOLEAN,
        created_at TIMESTAMP
    ) USING DELTA
    """)

    record = [{
        "account_id": account_id,
        "denial_text": denial_text,
        "denial_embedding": denial_embedding,
        "payor": denial_info.get("payor", "Unknown"),
        "original_drg": denial_info.get("original_drg"),
        "proposed_drg": denial_info.get("proposed_drg"),
        "is_sepsis": denial_info.get("is_sepsis", False),
        "created_at": datetime.now()
    }]

    schema = StructType([
        StructField("account_id", StringType(), False),
        StructField("denial_text", StringType(), True),
        StructField("denial_embedding", ArrayType(FloatType()), True),
        StructField("payor", StringType(), True),
        StructField("original_drg", StringType(), True),
        StructField("proposed_drg", StringType(), True),
        StructField("is_sepsis", BooleanType(), True),
        StructField("created_at", TimestampType(), True)
    ])

    df = spark.createDataFrame(record, schema)
    df.write.format("delta").mode("overwrite").saveAsTable(CASE_DENIAL_TABLE)
    print(f"  Written to {CASE_DENIAL_TABLE}")


def write_case_clinical_table(account_id, clinical_data, extracted_notes):
    """Write clinical notes data to case table."""
    spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {CASE_CLINICAL_TABLE} (
        account_id STRING,
        patient_name STRING,
        patient_dob STRING,
        facility_name STRING,
        date_of_service STRING,
        extracted_notes STRING,
        created_at TIMESTAMP
    ) USING DELTA
    """)

    record = [{
        "account_id": account_id,
        "patient_name": clinical_data.get("formatted_name", "Unknown"),
        "patient_dob": clinical_data.get("formatted_birthdate", ""),
        "facility_name": clinical_data.get("facility_name", "Mercy Hospital"),
        "date_of_service": clinical_data.get("formatted_date_of_service", ""),
        "extracted_notes": json.dumps(extracted_notes),
        "created_at": datetime.now()
    }]

    df = spark.createDataFrame(record)
    df.write.format("delta").mode("overwrite").saveAsTable(CASE_CLINICAL_TABLE)
    print(f"  Written to {CASE_CLINICAL_TABLE}")


def write_case_structured_summary_table(account_id, structured_summary):
    """Write structured data summary to case table."""
    spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {CASE_STRUCTURED_SUMMARY_TABLE} (
        account_id STRING,
        structured_summary STRING,
        created_at TIMESTAMP
    ) USING DELTA
    """)

    record = [{
        "account_id": account_id,
        "structured_summary": structured_summary,
        "created_at": datetime.now()
    }]

    df = spark.createDataFrame(record)
    df.write.format("delta").mode("overwrite").saveAsTable(CASE_STRUCTURED_SUMMARY_TABLE)
    print(f"  Written to {CASE_STRUCTURED_SUMMARY_TABLE}")


def write_case_conflicts_table(account_id, conflicts_result):
    """Write conflicts data to case table."""
    spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {CASE_CONFLICTS_TABLE} (
        account_id STRING,
        conflicts STRING,
        recommendation STRING,
        created_at TIMESTAMP
    ) USING DELTA
    """)

    record = [{
        "account_id": account_id,
        "conflicts": json.dumps(conflicts_result.get("conflicts", [])),
        "recommendation": conflicts_result.get("recommendation", ""),
        "created_at": datetime.now()
    }]

    df = spark.createDataFrame(record)
    df.write.format("delta").mode("overwrite").saveAsTable(CASE_CONFLICTS_TABLE)
    print(f"  Written to {CASE_CONFLICTS_TABLE}")


print("Case table write functions loaded")

# =============================================================================
# CELL 10: Main Processing Pipeline
# =============================================================================
print("\n" + "="*60)
print("FEATURIZATION - PER-CASE DATA PREPARATION")
print("="*60)

# Check input file exists
if not os.path.exists(DENIAL_PDF_PATH):
    print(f"\nERROR: Denial PDF not found: {DENIAL_PDF_PATH}")
    print("Set DENIAL_PDF_PATH to a valid PDF file path.")
else:
    print(f"\nInput: {os.path.basename(DENIAL_PDF_PATH)}")

    # -------------------------------------------------------------------------
    # STEP 1: Parse denial PDF
    # -------------------------------------------------------------------------
    print("\nStep 1: Parsing denial PDF...")
    pages = extract_text_from_pdf(DENIAL_PDF_PATH)
    denial_text = "\n\n".join(pages)
    denial_embedding = generate_embedding(denial_text)
    print(f"  Extracted {len(pages)} pages, {len(denial_text)} chars")
    print(f"  Generated embedding ({len(denial_embedding)} dims)")

    # -------------------------------------------------------------------------
    # STEP 2: Extract denial info via LLM
    # -------------------------------------------------------------------------
    print("\nStep 2: Extracting denial info...")
    denial_info = extract_denial_info_llm(denial_text[:15000])
    print(f"  Account ID: {denial_info['hsp_account_id'] or 'NOT FOUND'}")
    print(f"  Payor: {denial_info['payor']}")
    print(f"  DRG: {denial_info['original_drg']} → {denial_info['proposed_drg']}")
    print(f"  Sepsis: {denial_info['is_sepsis']}")

    # Use known account ID if provided
    account_id = KNOWN_ACCOUNT_ID or denial_info['hsp_account_id']

    if not account_id:
        print("\nERROR: No account ID found. Set KNOWN_ACCOUNT_ID or ensure denial letter contains H-prefixed account number.")
    else:
        # -------------------------------------------------------------------------
        # STEP 3: Query clinical notes
        # -------------------------------------------------------------------------
        print(f"\nStep 3: Querying clinical notes for account {account_id}...")
        clinical_data = query_clarity_for_account(account_id)

        if clinical_data:
            # -------------------------------------------------------------------------
            # STEP 4: Extract clinical notes
            # -------------------------------------------------------------------------
            print("\nStep 4: Extracting clinical notes...")
            extracted_notes = extract_notes_for_case(clinical_data)

            # Create notes summary for conflict detection
            notes_summary = "\n\n".join([
                f"## {key}\n{value[:2000]}"
                for key, value in extracted_notes.items()
                if value and value != "Not available"
            ])

            # -------------------------------------------------------------------------
            # STEP 5: Query structured data
            # -------------------------------------------------------------------------
            print(f"\nStep 5: Querying structured data for account {account_id}...")
            create_target_encounter_view(account_id)
            query_labs(account_id)
            query_vitals(account_id)
            query_meds(account_id)
            query_diagnoses(account_id)
            create_merged_timeline(account_id)

            # -------------------------------------------------------------------------
            # STEP 6: Extract structured data summary
            # -------------------------------------------------------------------------
            print("\nStep 6: Extracting structured data summary...")
            structured_summary = extract_structured_data_summary(account_id)

            # -------------------------------------------------------------------------
            # STEP 7: Detect conflicts
            # -------------------------------------------------------------------------
            print("\nStep 7: Detecting conflicts...")
            conflicts_result = detect_conflicts(notes_summary, structured_summary)

            # -------------------------------------------------------------------------
            # STEP 8: Write to case tables
            # -------------------------------------------------------------------------
            print("\nStep 8: Writing to case tables...")
            write_case_denial_table(account_id, denial_text, denial_embedding, denial_info)
            write_case_clinical_table(account_id, clinical_data, extracted_notes)
            write_case_structured_summary_table(account_id, structured_summary)
            write_case_conflicts_table(account_id, conflicts_result)

            # -------------------------------------------------------------------------
            # SUMMARY
            # -------------------------------------------------------------------------
            print("\n" + "="*60)
            print("FEATURIZATION COMPLETE")
            print("="*60)
            print(f"Account: {account_id}")
            print(f"Patient: {clinical_data.get('formatted_name', 'Unknown')}")
            print(f"Clinical notes extracted: {len([v for v in extracted_notes.values() if v != 'Not available'])}")
            print(f"Structured summary: {len(structured_summary)} chars")
            print(f"Conflicts: {len(conflicts_result.get('conflicts', []))}")
            if conflicts_result.get('conflicts'):
                print("\nConflicts found:")
                for conflict in conflicts_result['conflicts']:
                    print(f"  - {conflict}")
            print(f"\nCase tables written - ready for inference.py")

        else:
            print("\nERROR: Could not retrieve clinical data from Clarity.")

print("\nFeaturization inference complete.")
