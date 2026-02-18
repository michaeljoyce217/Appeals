# poc/featurization.py
# Rebuttal Engine v2 - Featurization (Data Gathering)
#
# MLFlow Structure: All data gathering happens here
# Output: fudgesicle_inference table with denial letters + clinical data
#
# This file reads denial letters and clinical data, preparing everything
# for inference.py to process.

import os
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

# =============================================================================
# CELL 1: Install Dependencies (run once)
# =============================================================================
# %pip install azure-ai-documentintelligence azure-core
# %restart_python

# =============================================================================
# CELL 2: Imports and Configuration
# =============================================================================
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, lit

# Get spark session (available in Databricks)
spark = SparkSession.builder.getOrCreate()

# -----------------------------------------------------------------------------
# SCOPE FILTER CONSTANTS - Change these to expand beyond sepsis
# -----------------------------------------------------------------------------
SCOPE_FILTER = "sepsis"  # "sepsis" | "all"

SEPSIS_DRG_CODES = ["870", "871", "872"]
SEPSIS_ICD10_CODES = [
    "A41.9", "A41.0", "A41.1", "A41.2", "A41.50", "A41.51", "A41.52", "A41.53",
    "R65.20", "R65.21",
]

# =============================================================================
# CELL 3: Environment Configuration
# =============================================================================
# Set the catalog
trgt_cat = os.environ.get('trgt_cat', 'dev')
spark.sql('use catalog prod;') if trgt_cat == 'dev' else spark.sql(f'use catalog {trgt_cat};')

# Paths - UPDATE THESE for your environment
DENIAL_LETTERS_BASE = "/Workspace/Repos/mijo8881@mercy.net/fudgsicle/eda/POC/Sample_Denials"

# Target accounts for POC
# Format: (HSP_ACCOUNT_ID, denial_letter_filename)
TARGET_ACCOUNTS: List[Tuple[str, str]] = [
    # UPDATE THESE with your 10 test cases
    # ("123456789", "denial_letter_1.pdf"),
    # ("987654321", "denial_letter_2.docx"),
]

print(f"Catalog: {trgt_cat}")
print(f"Scope Filter: {SCOPE_FILTER}")
print(f"Target Accounts: {len(TARGET_ACCOUNTS)}")

# =============================================================================
# CELL 4: Azure AI Document Intelligence Setup
# =============================================================================
class DocumentReader:
    """Reads documents using Azure AI Document Intelligence."""

    def __init__(self, endpoint: str, api_key: str):
        self.endpoint = endpoint
        self.api_key = api_key
        self._client = None

    @classmethod
    def from_databricks_secrets(cls, scope: str = "idp_etl",
                                  endpoint_key: str = "az-doc-intelligence-endpoint",
                                  api_key_key: str = "az-doc-intelligence-key"):
        """Create reader from Databricks secrets."""
        endpoint = dbutils.secrets.get(scope=scope, key=endpoint_key)
        api_key = dbutils.secrets.get(scope=scope, key=api_key_key)
        return cls(endpoint=endpoint, api_key=api_key)

    @property
    def client(self):
        """Lazy initialization of Document Intelligence client."""
        if self._client is None:
            from azure.ai.documentintelligence import DocumentIntelligenceClient
            from azure.core.credentials import AzureKeyCredential
            self._client = DocumentIntelligenceClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.api_key)
            )
        return self._client

    def read_document(self, file_path: str) -> str:
        """Read a document and extract text content."""
        _, ext = os.path.splitext(file_path.lower())

        # .txt files: simple read
        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

        # All other formats: use Azure AI Document Intelligence
        from azure.ai.documentintelligence.models import AnalyzeDocumentRequest

        with open(file_path, "rb") as f:
            document_bytes = f.read()

        poller = self.client.begin_analyze_document(
            model_id="prebuilt-read",
            body=AnalyzeDocumentRequest(bytes_source=document_bytes),
        )
        result = poller.result()

        text_parts = []
        for page in result.pages:
            for line in page.lines:
                text_parts.append(line.content)

        return "\n".join(text_parts)


# Initialize document reader
try:
    doc_reader = DocumentReader.from_databricks_secrets()
    print("Document reader initialized")
except Exception as e:
    print(f"Warning: Document reader init failed: {e}")
    doc_reader = None

# =============================================================================
# CELL 5: Clinical Data Query
# =============================================================================
def get_clinical_data_for_accounts(account_ids: List[str]) -> pd.DataFrame:
    """
    Query clinical data for multiple accounts from Clarity.

    Returns DataFrame with:
    - discharge_summary_text, hp_note_text (provider notes - PRIMARY evidence)
    - Patient demographics
    - Claim info
    """
    if not account_ids:
        return pd.DataFrame()

    account_list = ",".join(f"'{a}'" for a in account_ids)

    query = f"""
    WITH target_accounts AS (
        SELECT explode(array({account_list})) AS hsp_account_id
    ),
    encounters AS (
        SELECT
            pe.pat_enc_csn_id,
            pe.pat_id,
            pe.hsp_account_id,
            pe.hosp_admsn_time,
            pe.hosp_disch_time,
            ROW_NUMBER() OVER (PARTITION BY pe.hsp_account_id ORDER BY pe.hosp_admsn_time DESC) AS rn
        FROM {trgt_cat}.clarity_cur.pat_enc_hsp_har_enh pe
        INNER JOIN target_accounts ta ON pe.hsp_account_id = ta.hsp_account_id
        WHERE pe.hosp_admsn_time IS NOT NULL
    ),
    latest_encounters AS (
        SELECT * FROM encounters WHERE rn = 1
    ),
    discharge_notes AS (
        SELECT
            e.hsp_account_id,
            hno.note_id AS discharge_note_id,
            hno.pat_enc_csn_id AS discharge_note_csn_id,
            CONCAT_WS(' ', COLLECT_LIST(hnt.note_text)) AS discharge_summary_text
        FROM {trgt_cat}.clarity_cur.hno_info_enh hno
        INNER JOIN latest_encounters e ON hno.pat_enc_csn_id = e.pat_enc_csn_id
        INNER JOIN {trgt_cat}.clarity_cur.hno_note_text_enh hnt ON hno.note_id = hnt.note_id
        WHERE hno.note_type_c = 2
        GROUP BY e.hsp_account_id, hno.note_id, hno.pat_enc_csn_id
    ),
    hp_notes AS (
        SELECT
            e.hsp_account_id,
            hno.note_id AS hp_note_id,
            hno.pat_enc_csn_id AS hp_note_csn_id,
            CONCAT_WS(' ', COLLECT_LIST(hnt.note_text)) AS hp_note_text
        FROM {trgt_cat}.clarity_cur.hno_info_enh hno
        INNER JOIN latest_encounters e ON hno.pat_enc_csn_id = e.pat_enc_csn_id
        INNER JOIN {trgt_cat}.clarity_cur.hno_note_text_enh hnt ON hno.note_id = hnt.note_id
        WHERE hno.note_type_c = 1
        GROUP BY e.hsp_account_id, hno.note_id, hno.pat_enc_csn_id
    ),
    patient_info AS (
        SELECT
            e.hsp_account_id,
            p.pat_id,
            p.pat_mrn_id,
            CONCAT(p.pat_last_name, ', ', p.pat_first_name) AS formatted_name,
            DATE_FORMAT(p.birth_date, 'MM/dd/yyyy') AS formatted_birthdate
        FROM {trgt_cat}.clarity_cur.patient_enh p
        INNER JOIN latest_encounters e ON p.pat_id = e.pat_id
    ),
    account_info AS (
        SELECT
            ha.hsp_account_id,
            ha.prim_enc_csn_id,
            f.facility_name,
            DATEDIFF(ha.disch_date_time, ha.adm_date_time) AS number_of_midnights,
            DATE_FORMAT(ha.adm_date_time, 'MM/dd/yyyy') AS adm_date_formatted,
            DATE_FORMAT(ha.disch_date_time, 'MM/dd/yyyy') AS disch_date_formatted,
            CONCAT(DATE_FORMAT(ha.adm_date_time, 'MM/dd/yyyy'), ' - ',
                   DATE_FORMAT(ha.disch_date_time, 'MM/dd/yyyy')) AS formatted_date_of_service
        FROM {trgt_cat}.clarity_cur.hsp_account_enh ha
        INNER JOIN target_accounts ta ON ha.hsp_account_id = ta.hsp_account_id
        LEFT JOIN {trgt_cat}.clarity.zc_loc_facility f ON ha.loc_id = f.facility_id
    ),
    claim_info AS (
        SELECT
            hcd.hsp_account_id,
            FIRST(hcd.claim_number) AS claim_number,
            FIRST(hcd.tax_id) AS tax_id,
            FIRST(hcd.npi) AS npi
        FROM {trgt_cat}.clarity.hsp_claim_detail hcd
        INNER JOIN target_accounts ta ON hcd.hsp_account_id = ta.hsp_account_id
        GROUP BY hcd.hsp_account_id
    ),
    diagnosis AS (
        SELECT
            hadl.hsp_account_id,
            FIRST(edg.code) AS code,
            FIRST(edg.dx_name) AS dx_name
        FROM {trgt_cat}.clarity_cur.hsp_acct_dx_list_enh hadl
        INNER JOIN target_accounts ta ON hadl.hsp_account_id = ta.hsp_account_id
        INNER JOIN {trgt_cat}.clarity_cur.edg_current_icd10_enh edg ON hadl.dx_id = edg.dx_id
        WHERE hadl.line = 1
        GROUP BY hadl.hsp_account_id
    )
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
        dn.discharge_note_id AS discharge_summary_note_id,
        dn.discharge_note_csn_id AS discharge_note_csn_id,
        dn.discharge_summary_text,
        hp.hp_note_id,
        hp.hp_note_csn_id,
        hp.hp_note_text
    FROM target_accounts ta
    LEFT JOIN patient_info pi ON ta.hsp_account_id = pi.hsp_account_id
    LEFT JOIN account_info ai ON ta.hsp_account_id = ai.hsp_account_id
    LEFT JOIN claim_info ci ON ta.hsp_account_id = ci.hsp_account_id
    LEFT JOIN diagnosis d ON ta.hsp_account_id = d.hsp_account_id
    LEFT JOIN discharge_notes dn ON ta.hsp_account_id = dn.hsp_account_id
    LEFT JOIN hp_notes hp ON ta.hsp_account_id = hp.hsp_account_id
    """

    return spark.sql(query).toPandas()


# =============================================================================
# CELL 6: Read Denial Letters
# =============================================================================
def read_denial_letters(accounts: List[Tuple[str, str]]) -> dict:
    """
    Read denial letters for all target accounts.

    Returns dict mapping hsp_account_id -> denial_letter_text
    """
    if doc_reader is None:
        print("Warning: Document reader not available")
        return {}

    denial_texts = {}

    for hsp_account_id, filename in accounts:
        file_path = os.path.join(DENIAL_LETTERS_BASE, filename)
        try:
            text = doc_reader.read_document(file_path)
            denial_texts[hsp_account_id] = text
            print(f"  Read {filename}: {len(text)} chars")
        except Exception as e:
            print(f"  ERROR reading {filename}: {e}")
            denial_texts[hsp_account_id] = None

    return denial_texts


# =============================================================================
# CELL 6B: Query Structured Clinical Data (Labs, Vitals, Meds, Procedures)
# =============================================================================
def get_structured_data_for_accounts(account_ids: List[str]) -> pd.DataFrame:
    """
    Query structured clinical data for accounts.

    This pulls BROADLY - the Research Agent will filter to what's relevant.

    Returns DataFrame with:
    - Labs (lactate, WBC, procalcitonin, creatinine, bilirubin, platelets, INR, etc.)
    - Vitals (temp, HR, RR, BP, MAP, O2 sat)
    - Meds (antibiotics, vasopressors, fluids)
    - Procedures (blood cultures, imaging, lines)
    """
    if not account_ids:
        return pd.DataFrame()

    account_list = ",".join(f"'{a}'" for a in account_ids)

    # -------------------------------------------------------------------------
    # LABS QUERY (DUMMY - replace with actual Clarity tables)
    # -------------------------------------------------------------------------
    labs_query = f"""
    -- DUMMY: Replace with actual lab query
    -- Expected tables: clarity.order_results, clarity.order_proc, etc.
    SELECT
        ta.hsp_account_id,
        'labs' AS data_type,
        NULL AS result_time,
        NULL AS component_name,
        NULL AS ord_value,
        NULL AS reference_unit
    FROM (SELECT explode(array({account_list})) AS hsp_account_id) ta
    WHERE 1=0  -- Returns empty until real query is added

    /*
    -- REAL QUERY TEMPLATE:
    SELECT
        pe.hsp_account_id,
        'labs' AS data_type,
        ore.result_time,
        orc.component_name,
        ore.ord_value,
        orc.reference_unit
    FROM {{catalog}}.clarity.order_results ore
    INNER JOIN {{catalog}}.clarity.order_proc op ON ore.order_proc_id = op.order_proc_id
    INNER JOIN {{catalog}}.clarity.order_res_comp_cmt orc ON ore.result_id = orc.result_id
    INNER JOIN encounters e ON op.pat_enc_csn_id = e.pat_enc_csn_id
    WHERE orc.component_name IN (
        'LACTATE', 'WBC', 'PROCALCITONIN', 'CRP',
        'CREATININE', 'BILIRUBIN', 'PLATELETS', 'INR', 'APTT',
        'HEMOGLOBIN', 'GLUCOSE', 'A1C', 'BNP', 'TROPONIN'
    )
    */
    """

    # -------------------------------------------------------------------------
    # VITALS QUERY (DUMMY - replace with actual Clarity tables)
    # -------------------------------------------------------------------------
    vitals_query = f"""
    -- DUMMY: Replace with actual vitals query
    -- Expected tables: clarity.ip_flwsht_meas, clarity.ip_flo_gp_data, etc.
    SELECT
        ta.hsp_account_id,
        'vitals' AS data_type,
        NULL AS recorded_time,
        NULL AS vital_name,
        NULL AS meas_value
    FROM (SELECT explode(array({account_list})) AS hsp_account_id) ta
    WHERE 1=0  -- Returns empty until real query is added

    /*
    -- REAL QUERY TEMPLATE:
    SELECT
        e.hsp_account_id,
        'vitals' AS data_type,
        fm.recorded_time,
        fgd.disp_name AS vital_name,
        fm.meas_value
    FROM {{catalog}}.clarity.ip_flwsht_meas fm
    INNER JOIN {{catalog}}.clarity.ip_flo_gp_data fgd ON fm.flo_meas_id = fgd.flo_meas_id
    INNER JOIN encounters e ON fm.pat_enc_csn_id = e.pat_enc_csn_id
    WHERE fgd.disp_name IN (
        'TEMPERATURE', 'HEART RATE', 'RESPIRATORY RATE',
        'SYSTOLIC BP', 'DIASTOLIC BP', 'MAP', 'O2 SAT', 'GCS'
    )
    */
    """

    # -------------------------------------------------------------------------
    # MEDS QUERY (DUMMY - replace with actual Clarity tables)
    # -------------------------------------------------------------------------
    meds_query = f"""
    -- DUMMY: Replace with actual meds query
    -- Expected tables: clarity.mar_admin_info, clarity.order_med, etc.
    SELECT
        ta.hsp_account_id,
        'meds' AS data_type,
        NULL AS admin_time,
        NULL AS medication_name,
        NULL AS dose,
        NULL AS route
    FROM (SELECT explode(array({account_list})) AS hsp_account_id) ta
    WHERE 1=0  -- Returns empty until real query is added

    /*
    -- REAL QUERY TEMPLATE:
    SELECT
        e.hsp_account_id,
        'meds' AS data_type,
        mai.taken_time AS admin_time,
        om.description AS medication_name,
        mai.dose,
        zar.name AS route
    FROM {{catalog}}.clarity.mar_admin_info mai
    INNER JOIN {{catalog}}.clarity.order_med om ON mai.order_med_id = om.order_med_id
    INNER JOIN encounters e ON mai.pat_enc_csn_id = e.pat_enc_csn_id
    LEFT JOIN {{catalog}}.clarity.zc_admin_route zar ON mai.route_c = zar.route_c
    WHERE om.thera_class_c IN (
        -- Antibiotics, vasopressors, IV fluids, etc.
        'ANTIBIOTICS', 'VASOPRESSORS', 'IV FLUIDS'
    )
    */
    """

    # -------------------------------------------------------------------------
    # PROCEDURES QUERY (DUMMY - replace with actual Clarity tables)
    # -------------------------------------------------------------------------
    procedures_query = f"""
    -- DUMMY: Replace with actual procedures query
    -- Expected tables: clarity.order_proc, clarity.or_log, etc.
    SELECT
        ta.hsp_account_id,
        'procedures' AS data_type,
        NULL AS proc_time,
        NULL AS proc_name,
        NULL AS proc_code
    FROM (SELECT explode(array({account_list})) AS hsp_account_id) ta
    WHERE 1=0  -- Returns empty until real query is added

    /*
    -- REAL QUERY TEMPLATE:
    SELECT
        e.hsp_account_id,
        'procedures' AS data_type,
        op.proc_start_time AS proc_time,
        op.description AS proc_name,
        op.proc_code
    FROM {{catalog}}.clarity.order_proc op
    INNER JOIN encounters e ON op.pat_enc_csn_id = e.pat_enc_csn_id
    WHERE op.proc_code IN (
        -- Blood cultures, imaging, central lines, etc.
    )
    */
    """

    # Execute queries and combine (all return empty for now)
    try:
        labs_df = spark.sql(labs_query).toPandas()
        vitals_df = spark.sql(vitals_query).toPandas()
        meds_df = spark.sql(meds_query).toPandas()
        procs_df = spark.sql(procedures_query).toPandas()

        # Combine all structured data
        combined = pd.concat([labs_df, vitals_df, meds_df, procs_df], ignore_index=True)
        return combined
    except Exception as e:
        print(f"Warning: Structured data query failed: {e}")
        return pd.DataFrame()


# =============================================================================
# CELL 7: Build Featurized Dataset
# =============================================================================
def build_featurized_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the complete featurized dataset for inference.

    Combines:
    - Denial letter text (from Azure Doc Intelligence)
    - Clinical notes (from Clarity - PRIMARY evidence)
    - Patient demographics
    - Claim info
    - Structured data (labs, vitals, meds, procedures - SECONDARY evidence)

    Returns:
    - clinical_df: Main patient/encounter data
    - structured_df: Structured clinical data (separate table, joined by hsp_account_id)
    """
    if not TARGET_ACCOUNTS:
        print("No target accounts configured")
        return pd.DataFrame(), pd.DataFrame()

    print(f"Building featurized dataset for {len(TARGET_ACCOUNTS)} accounts...")

    # 1. Get account IDs
    account_ids = [a[0] for a in TARGET_ACCOUNTS]

    # 2. Read denial letters
    print("\nReading denial letters...")
    denial_texts = read_denial_letters(TARGET_ACCOUNTS)

    # 3. Query clinical data (notes, demographics)
    print("\nQuerying clinical data from Clarity...")
    clinical_df = get_clinical_data_for_accounts(account_ids)
    print(f"  Retrieved data for {len(clinical_df)} accounts")

    # 4. Query structured data (labs, vitals, meds, procedures)
    print("\nQuerying structured clinical data...")
    structured_df = get_structured_data_for_accounts(account_ids)
    print(f"  Retrieved {len(structured_df)} structured data records")

    # 5. Add denial letter text
    clinical_df['denial_letter_text'] = clinical_df['hsp_account_id'].map(
        lambda x: denial_texts.get(x, None)
    )

    # 6. Add metadata
    clinical_df['scope_filter'] = SCOPE_FILTER
    clinical_df['featurization_timestamp'] = datetime.now().isoformat()

    # 7. Add denial letter filename for reference
    filename_map = {a[0]: a[1] for a in TARGET_ACCOUNTS}
    clinical_df['denial_letter_filename'] = clinical_df['hsp_account_id'].map(filename_map)

    # 8. Serialize structured data as JSON per account (for storage in main table)
    # Research Agent will deserialize and filter during inference
    def serialize_structured_data(acct_id):
        acct_data = structured_df[structured_df['hsp_account_id'] == acct_id]
        if acct_data.empty:
            return None
        return acct_data.to_json(orient='records')

    clinical_df['structured_data_json'] = clinical_df['hsp_account_id'].map(serialize_structured_data)

    print(f"\nFeaturized dataset: {len(clinical_df)} rows")
    print(f"Columns: {list(clinical_df.columns)}")

    return clinical_df, structured_df


# =============================================================================
# CELL 8: Write to Delta Table
# =============================================================================
def write_to_inference_table(df: pd.DataFrame):
    """Write featurized data to fudgesicle_inference table."""
    if df.empty:
        print("No data to write")
        return

    table_name = f"{trgt_cat}.fin_ds.fudgesicle_inference"

    # Convert to Spark DataFrame
    spark_df = spark.createDataFrame(df)
    spark_df = spark_df.withColumn("insert_tsp", current_timestamp())

    # Write to table
    spark_df.write.mode("append").saveAsTable(table_name)
    print(f"Wrote {len(df)} rows to {table_name}")


# =============================================================================
# CELL 9: Main Execution
# =============================================================================
if __name__ == "__main__" or True:  # Always run in notebook
    # Check configuration
    if len(TARGET_ACCOUNTS) == 0:
        print("="*60)
        print("SETUP REQUIRED")
        print("="*60)
        print("Update TARGET_ACCOUNTS in Cell 3 with your test cases:")
        print('  TARGET_ACCOUNTS = [')
        print('      ("123456789", "denial_letter_1.pdf"),')
        print('      ("987654321", "denial_letter_2.docx"),')
        print('  ]')
    else:
        # Build and write featurized data
        clinical_df, structured_df = build_featurized_dataset()

        if not clinical_df.empty:
            print("\n" + "="*60)
            print("PREVIEW")
            print("="*60)
            print(clinical_df[['hsp_account_id', 'formatted_name', 'denial_letter_filename']].to_string())

            print(f"\nStructured data records: {len(structured_df)}")

            # Uncomment to write to table:
            # write_to_inference_table(clinical_df)
            print("\nTo write to table, uncomment: write_to_inference_table(clinical_df)")

# =============================================================================
# OUTPUT SCHEMA (for inference.py)
# =============================================================================
"""
fudgesicle_inference table columns:
- hsp_account_id
- pat_id, pat_mrn_id
- formatted_name, formatted_birthdate
- facility_name, number_of_midnights, formatted_date_of_service
- claim_number, tax_id, npi
- code, dx_name (principal diagnosis)
- discharge_summary_note_id, discharge_note_csn_id, discharge_summary_text
- hp_note_id, hp_note_csn_id, hp_note_text
- denial_letter_text (from Azure Doc Intelligence)
- denial_letter_filename
- structured_data_json (labs, vitals, meds, procedures - Research Agent filters this)
- scope_filter
- featurization_timestamp
- insert_tsp

Evidence Priority (used by Writer Agent):
1. Provider notes (discharge_summary_text, hp_note_text) - BEST
2. Structured data (structured_data_json, filtered by Research Agent) - BACKUP
3. Inference (conclusions drawn from above) - LEAST IMPORTANT
"""
