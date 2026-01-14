# model/inference.py
# Rebuttal Engine v2 - Inference (Letter Generation)
#
# PIPELINE OVERVIEW:
# 1. Parse denial letter → Extract structured info via LLM (DRG, payor, denial reasons)
# 2. Vector search → Compare pre-computed embeddings (apples-to-apples)
# 3. Generate rebuttal → Use winning rebuttal as template, clinical notes as evidence
# 4. Output → DOCX files for POC, Delta table for production
#
# KEY INSIGHT:
# We match new denials to PAST denials (not rebuttals) because denial letters
# from the same payor with similar arguments tend to need similar rebuttals.
# The gold letter's winning rebuttal becomes our "template to learn from."
#
# RIGOROUS ARCHITECTURE:
# Both new denials AND gold letter denials are embedded using the SAME
# generate_embedding() function in featurization.py. The denial_embedding
# column in the inference table is pre-computed, ensuring true apples-to-apples
# comparison with gold letter embeddings.
#
# Run on Databricks Runtime 15.4 LTS ML

# =============================================================================
# CELL 1: Install Dependencies (run once per cluster)
# =============================================================================
# Only openai is needed here - Document Intelligence was used in featurization.
# The denial text is already extracted and stored in the inference table.
#
# %pip install openai python-docx
# dbutils.library.restartPython()

# =============================================================================
# CELL 2: Imports and Configuration
# =============================================================================
import os
import json
import math
from datetime import datetime, date
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp

# Get or create Spark session (already exists in Databricks notebooks)
spark = SparkSession.builder.getOrCreate()

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# SCOPE_FILTER: Which denials to process
# - "sepsis": Only process sepsis-related denials (our POC scope)
# - "all": Process all denial types (future expansion)
SCOPE_FILTER = "sepsis"

# DRG codes for sepsis cases:
# - 870: Sepsis with MV >96 hours (highest severity/reimbursement)
# - 871: Sepsis without MV >96 hours
# - 872: Sepsis without MCC (lowest severity)
# Payors often try to downgrade from 871 to 872 - that's what we're fighting.
SEPSIS_DRG_CODES = ["870", "871", "872"]

# Minimum cosine similarity to use a gold letter as reference.
# 0.7 means the denial must be reasonably similar.
# Lower = more matches but potentially less relevant templates.
# Higher = fewer matches but higher quality templates.
MATCH_SCORE_THRESHOLD = 0.7

# =============================================================================
# CELL 3: Environment Setup
# =============================================================================
# Determine which Unity Catalog to use.
# Same logic as featurization.py - dev defaults to prod catalog.
trgt_cat = os.environ.get('trgt_cat', 'dev')
if trgt_cat == 'dev':
    spark.sql('USE CATALOG prod;')
else:
    spark.sql(f'USE CATALOG {trgt_cat};')

# Table names - must match featurization.py
GOLD_LETTERS_TABLE = f"{trgt_cat}.fin_ds.fudgesicle_gold_letters"
INFERENCE_TABLE = f"{trgt_cat}.fin_ds.fudgesicle_inference"
INFERENCE_SCORE_TABLE = f"{trgt_cat}.fin_ds.fudgesicle_inference_score"

print(f"Catalog: {trgt_cat}")

# =============================================================================
# CELL 4: Azure OpenAI Setup
# =============================================================================
from openai import AzureOpenAI

# Load credentials from Databricks secrets
api_key = dbutils.secrets.get(scope='idp_etl', key='az-openai-key1')
azure_endpoint = dbutils.secrets.get(scope='idp_etl', key='az-openai-base')
api_version = '2024-10-21'

# Model names deployed in our Azure OpenAI resource
model = 'gpt-4.1'                    # For parsing denial info and letter generation
# NOTE: Embedding generation moved to featurization.py for rigorous architecture.
# The denial_embedding is pre-computed using text-embedding-ada-002 (1536 dims).

# Initialize the client
client = AzureOpenAI(api_key=api_key, azure_endpoint=azure_endpoint, api_version=api_version)
print(f"Azure OpenAI client initialized (model: {model})")

# =============================================================================
# CELL 5: Create Score Table (run once)
# =============================================================================
# This table stores the generated rebuttal letters along with metadata
# about how they were generated (which gold letter was used, etc.)
#
# Schema:
# - hsp_account_id: Links back to the original denial case
# - letter_text: The generated rebuttal letter
# - gold_letter_used: Which gold letter we learned from (for auditability)
# - gold_letter_score: Cosine similarity score (how good was the match?)
# - denial_info_json: Parsed denial info (for debugging/analysis)
create_score_table_sql = f"""
CREATE TABLE IF NOT EXISTS {INFERENCE_SCORE_TABLE} (
    hsp_account_id STRING,
    pat_mrn_id STRING,
    formatted_name STRING,
    discharge_summary_note_id STRING,
    discharge_note_csn_id STRING,
    hp_note_id STRING,
    hp_note_csn_id STRING,
    letter_type STRING,
    letter_text STRING,
    letter_curated_date DATE,
    denial_info_json STRING,
    gold_letter_used STRING,
    gold_letter_score FLOAT,
    pipeline_version STRING,
    insert_tsp TIMESTAMP
)
USING DELTA
COMMENT 'Generated rebuttal letters'
"""

spark.sql(create_score_table_sql)
print(f"Table {INFERENCE_SCORE_TABLE} ready")

# =============================================================================
# CELL 6: Check for New Records to Process
# =============================================================================
# Find records that haven't been processed yet.
# A record needs processing if:
# 1. It was inserted after the last processing run, OR
# 2. It has never been processed (not in score table)

# Get timestamp of last processing run
try:
    last_processed_ts = spark.sql(f"""
        SELECT COALESCE(MAX(insert_tsp), TIMESTAMP'2020-01-01 00:00:00') AS last_ts
        FROM {INFERENCE_SCORE_TABLE}
    """).collect()[0]["last_ts"]
except Exception:
    # Table might not exist yet on first run
    last_processed_ts = "2020-01-01 00:00:00"

# Count new rows needing processing
n_new_rows = spark.sql(f"""
    WITH scored_accounts AS (
        -- Get all accounts that have already been processed
        SELECT DISTINCT hsp_account_id
        FROM {INFERENCE_SCORE_TABLE}
    )
    SELECT COUNT(*) AS cnt
    FROM {INFERENCE_TABLE} src
    LEFT JOIN scored_accounts sa ON src.hsp_account_id = sa.hsp_account_id
    WHERE src.insert_tsp > TIMESTAMP'{last_processed_ts}'  -- New since last run
       OR sa.hsp_account_id IS NULL                        -- Never processed
""").collect()[0]["cnt"]

print(f"New rows to process: {n_new_rows}")

# =============================================================================
# CELL 7: Load Gold Letters into Memory (for vector search)
# =============================================================================
# We load all gold letters into memory for fast similarity search.
# With ~30 letters, this is efficient. For 1000s, we'd need a vector DB.
print("\nLoading gold standard letters...")

try:
    gold_letters_df = spark.sql(f"""
        SELECT letter_id, source_file, payor, denial_text, rebuttal_text, denial_embedding, metadata
        FROM {GOLD_LETTERS_TABLE}
    """)
    gold_letters = gold_letters_df.collect()

    # Convert Spark Rows to Python dicts for easier access
    gold_letters_cache = [
        {
            "letter_id": row["letter_id"],
            "source_file": row["source_file"],
            "payor": row["payor"],
            "denial_text": row["denial_text"],
            "rebuttal_text": row["rebuttal_text"],
            # Convert Spark array to Python list
            "denial_embedding": list(row["denial_embedding"]) if row["denial_embedding"] else None,
            "metadata": dict(row["metadata"]) if row["metadata"] else {},
        }
        for row in gold_letters
    ]
    print(f"Loaded {len(gold_letters_cache)} gold standard letters")
except Exception as e:
    print(f"Warning: Could not load gold letters: {e}")
    print("Will generate letters without gold letter reference.")
    gold_letters_cache = []

# =============================================================================
# CELL 8: Parser Prompt - Extract Structured Data from Denial Letter
# =============================================================================
# This prompt instructs the LLM to extract key information from denial letters.
# The output is JSON that we can use to:
# 1. Filter cases (is it sepsis-related?)
# 2. Populate the rebuttal letter template
# 3. Target specific denial arguments in our response
PARSER_PROMPT = '''You are a medical billing expert extracting information from denial letters.

# Task
Extract ALL relevant information from this denial letter into structured JSON.

# Denial Letter
{denial_letter_text}

# Output Format
Return ONLY valid JSON (no markdown):
{{
  "denial_date": "YYYY-MM-DD or null",
  "payer_name": "Insurance company name",
  "payer_address": "Full mailing address or null",
  "reviewer_name": "Name and credentials or null",
  "original_drg": "Billed DRG (e.g., '871')",
  "proposed_drg": "Payer's proposed DRG (e.g., '872')",
  "administrative_data": {{
    "claim_reference_number": "Primary claim/reference number",
    "member_id": "Patient member ID",
    "authorization_number": "Prior auth number or null",
    "date_of_service": "Admission date or date range",
    "other_identifiers": {{}}
  }},
  "denial_reasons": [{{
    "type": "clinical_validation | medical_necessity | level_of_care | coding | other",
    "summary": "Brief summary",
    "specific_arguments": ["Each specific argument made"],
    "payer_quote": "Direct quote if available"
  }}],
  "is_sepsis_related": true/false,
  "is_single_issue": true/false
}}'''

# =============================================================================
# CELL 9: Writer Prompt - Generate the Rebuttal Letter
# =============================================================================
# This is the main generation prompt. Key design decisions:
#
# 1. CLINICAL NOTES FIRST: We prioritize clinical evidence from the actual
#    patient encounter. This is the most defensible evidence.
#
# 2. GOLD LETTER AS GUIDE: If we found a similar past denial, we include
#    the winning rebuttal as a template. The LLM learns the style/approach
#    but adapts it to THIS patient's specific clinical data.
#
# 3. TEMPLATE STRUCTURE: We enforce the Mercy Hospital letter format
#    for consistency and professional appearance.
WRITER_PROMPT = '''You are a clinical coding expert writing a DRG validation appeal letter for Mercy Hospital.

# Denial Information
{denial_info_json}

# Clinical Notes (PRIMARY EVIDENCE - use these first)
## Discharge Summary
{discharge_summary}

## H&P Note
{hp_note}

# Gold Standard Letter (WINNING REBUTTAL - learn from this)
{gold_letter_section}

# Patient Information
{patient_info_json}

# Instructions
{gold_letter_instructions}
1. ADDRESS EACH DENIAL ARGUMENT - quote the payer, then refute
2. CITE CLINICAL EVIDENCE from provider notes FIRST (best source)
3. Follow the Mercy Hospital template structure exactly
4. Include specific clinical values (lactate 2.4, MAP 62, etc.)
5. DELETE sections that don't apply to this patient

# Template Structure
Return the complete letter text following this structure:

Mercy Hospital
Payor Audits & Denials Dept
ATTN: Compliance Manager
2115 S Fremont Ave - Ste LL1
Springfield, MO 65804

{current_date}

[PAYOR ADDRESS]

First Level Appeal

Beneficiary Name: [NAME]
DOB: [DOB]
Claim reference #: [CLAIM_REF]
Hospital Account #: [HSP_ACCOUNT_ID]
Date of Service: [DOS]

Dear [REVIEWER]:

[Opening paragraph about receiving DRG review...]

Justification for Appeal:
[Why we disagree...]

Rationale:
[Quote payer's argument, then provide clinical evidence...]

Infection Source: [...]
Organ Dysfunction: [List each with values...]
SIRS Criteria Met: [List each with values...]
[Other relevant sections...]

Hospital Course:
[Narrative from clinical notes...]

[Summary paragraph...]

Conclusion:
We anticipate our original DRG of [X] will be approved.

[Contact info and signature...]

Return ONLY the letter text, no JSON wrapper.'''

# =============================================================================
# CELL 10: Main Processing Loop
# =============================================================================
# Process each denial case one at a time.
# For each case:
# 1. Parse the denial letter
# 2. Check if it's in scope (sepsis-related)
# 3. Find the best matching gold letter
# 4. Generate the rebuttal

if n_new_rows == 0:
    print("No new rows to process")
else:
    print(f"\nProcessing {n_new_rows} rows...")

    # Pull unprocessed rows from the inference table
    df = spark.sql(f"""
        WITH scored_accounts AS (
            SELECT DISTINCT hsp_account_id
            FROM {INFERENCE_SCORE_TABLE}
        )
        SELECT src.*
        FROM {INFERENCE_TABLE} src
        LEFT JOIN scored_accounts sa ON src.hsp_account_id = sa.hsp_account_id
        WHERE src.insert_tsp > TIMESTAMP'{last_processed_ts}'
           OR sa.hsp_account_id IS NULL
    """).toPandas()

    print(f"Pulled {len(df)} rows for processing")

    # Collect results for all processed rows
    results = []

    # Process each row individually
    for idx, row in df.iterrows():
        hsp_account_id = row.get("hsp_account_id", "unknown")
        print(f"\n{'='*60}")
        print(f"Processing {idx+1}/{len(df)}: {hsp_account_id}")
        print(f"{'='*60}")

        # Initialize result dict for this case
        result = {
            "hsp_account_id": hsp_account_id,
            "status": None,
            "letter_text": None,
            "denial_info": None,
            "gold_letter_used": None,
            "gold_letter_score": None,
        }

        # ---------------------------------------------------------------------
        # STEP 1: Parse the denial letter
        # ---------------------------------------------------------------------
        # Extract structured information from the denial letter text.
        # This gives us: payor name, DRG codes, denial reasons, etc.
        denial_text = row.get("denial_letter_text", "")

        # Skip if no denial text (can't process without it)
        if not denial_text or str(denial_text).strip() == "" or denial_text is None:
            result["status"] = "no_denial_text"
            result["error"] = "No denial letter text"
            results.append(result)
            print("  SKIP: No denial letter text")
            continue

        print("  Step 1: Parsing denial letter...")

        # Call LLM to parse the denial letter
        parser_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Extract information accurately. Return only valid JSON."},
                {"role": "user", "content": PARSER_PROMPT.format(denial_letter_text=denial_text)}
            ],
            temperature=0,    # Zero temp for consistent extraction
            max_tokens=2000   # Denial info is typically <1000 tokens
        )

        # Parse the JSON response (handle markdown code blocks)
        raw_parser = parser_response.choices[0].message.content.strip()
        json_str = raw_parser
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()

        try:
            denial_info = json.loads(json_str)
            # Handle case where LLM returns array instead of object
            if isinstance(denial_info, list):
                if len(denial_info) > 0:
                    denial_info = denial_info[0]
                else:
                    raise ValueError("Empty array returned")
        except (json.JSONDecodeError, ValueError) as e:
            result["status"] = "parse_error"
            result["error"] = f"JSON parse failed: {e}"
            results.append(result)
            print(f"  ERROR: JSON parse failed")
            continue

        result["denial_info"] = denial_info
        print(f"  Parsed: DRG {denial_info.get('original_drg')} -> {denial_info.get('proposed_drg')}")

        # ---------------------------------------------------------------------
        # STEP 2: Check if case is in scope
        # ---------------------------------------------------------------------
        # For POC, we only process sepsis-related denials.
        # The parser extracts is_sepsis_related from the denial content.
        if SCOPE_FILTER == "sepsis":
            if not denial_info.get("is_sepsis_related"):
                result["status"] = "out_of_scope"
                result["reason"] = "Not sepsis-related"
                results.append(result)
                print("  SKIP: Not sepsis-related")
                continue

        # ---------------------------------------------------------------------
        # STEP 3: Vector search for best matching gold letter
        # ---------------------------------------------------------------------
        # RIGOROUS ARCHITECTURE:
        # The denial_embedding was pre-computed in featurization.py using the
        # SAME generate_embedding() function as gold letters. This ensures
        # apples-to-apples comparison (no embedding generation here).
        print("  Step 2: Finding similar gold standard letter...")

        gold_letter = None        # Will hold the best match (if found)
        gold_letter_score = 0.0   # Cosine similarity score

        if gold_letters_cache:
            # Get pre-computed embedding from inference table (computed in featurization.py)
            query_embedding = row.get("denial_embedding")

            if query_embedding is None:
                print("  WARNING: No denial_embedding in inference table - skipping vector search")
                print("  (Re-run featurization.py to compute embeddings)")
            else:
                # Convert to list if it's a Spark array
                if hasattr(query_embedding, 'tolist'):
                    query_embedding = query_embedding.tolist()
                elif not isinstance(query_embedding, list):
                    query_embedding = list(query_embedding)

                print(f"  Using pre-computed embedding ({len(query_embedding)} dims)")

                # Compare against all gold letters using cosine similarity
                best_score = 0.0
                best_match = None

                for letter in gold_letters_cache:
                    if letter["denial_embedding"]:
                        # Cosine similarity = dot(a,b) / (||a|| * ||b||)
                        vec1 = query_embedding
                        vec2 = letter["denial_embedding"]

                        # Dot product
                        dot_product = sum(a * b for a, b in zip(vec1, vec2))

                        # Vector norms (magnitudes)
                        norm1 = math.sqrt(sum(a * a for a in vec1))
                        norm2 = math.sqrt(sum(b * b for b in vec2))

                        # Cosine similarity (avoid div by zero)
                        if norm1 > 0 and norm2 > 0:
                            similarity = dot_product / (norm1 * norm2)
                            if similarity > best_score:
                                best_score = similarity
                                best_match = letter

                gold_letter_score = best_score

                # Only use the match if it meets our quality threshold
                if best_match and best_score >= MATCH_SCORE_THRESHOLD:
                    gold_letter = best_match
                    print(f"  Found match: {gold_letter['source_file']} | Payor: {gold_letter.get('payor', 'Unknown')} | Score: {best_score:.3f}")
                else:
                    print(f"  No good match (best score: {best_score:.3f}, threshold: {MATCH_SCORE_THRESHOLD})")

        else:
            print("  No gold letters loaded - using template only")

        # Record which gold letter we used (for auditability)
        result["gold_letter_used"] = gold_letter["letter_id"] if gold_letter else None
        result["gold_letter_score"] = gold_letter_score

        # ---------------------------------------------------------------------
        # STEP 4: Generate the rebuttal letter
        # ---------------------------------------------------------------------
        # Combine everything into the Writer prompt and generate.
        print("  Step 3: Generating rebuttal letter...")

        current_date_str = date.today().strftime("%m/%d/%Y")

        # Patient info for the letter header
        patient_info = {
            "formatted_name": row.get("formatted_name", ""),
            "formatted_birthdate": row.get("formatted_birthdate", ""),
            "hsp_account_id": row.get("hsp_account_id", ""),
            "claim_number": row.get("claim_number", ""),
            "formatted_date_of_service": row.get("formatted_date_of_service", ""),
            "facility_name": row.get("facility_name", ""),
            "number_of_midnights": row.get("number_of_midnights", ""),
            "code": row.get("code", ""),
            "dx_name": row.get("dx_name", ""),
        }

        # Build the gold letter section for the prompt
        # If we found a matching gold letter, include it with strong instructions
        if gold_letter:
            gold_letter_section = f"""## THIS LETTER WON A SIMILAR APPEAL - LEARN FROM IT
Source: {gold_letter.get('source_file', 'Unknown')}
Payor: {gold_letter.get('payor', 'Unknown')}
Match Score: {gold_letter_score:.3f}

### Winning Rebuttal Text:
{gold_letter['rebuttal_text']}
"""
            # Instructions emphasizing learning from the gold letter
            gold_letter_instructions = """**CRITICAL: A gold standard letter that won a similar appeal is provided above.**
- Study how it structures arguments and presents clinical evidence
- Emulate its persuasive techniques and medical reasoning
- Adapt its successful patterns to this patient's specific situation
- Do NOT copy verbatim - adapt the approach with this patient's actual clinical data

"""
        else:
            gold_letter_section = "No similar winning rebuttal available. Use the Mercy template structure."
            gold_letter_instructions = ""

        # Assemble the full Writer prompt
        writer_prompt = WRITER_PROMPT.format(
            denial_info_json=json.dumps(denial_info, indent=2, default=str),
            discharge_summary=row.get("discharge_summary_text", "Not available"),
            hp_note=row.get("hp_note_text", "Not available"),
            gold_letter_section=gold_letter_section,
            gold_letter_instructions=gold_letter_instructions,
            patient_info_json=json.dumps(patient_info, indent=2),
            current_date=current_date_str,
        )

        # Generate the rebuttal letter
        writer_response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a clinical coding expert writing DRG appeal letters. "
                        "Prioritize evidence from provider notes. Be thorough and specific."
                    )
                },
                {"role": "user", "content": writer_prompt}
            ],
            temperature=0.2,  # Slight variation for natural language
            max_tokens=4000   # Letters can be long
        )

        letter_text = writer_response.choices[0].message.content.strip()
        result["status"] = "success"
        result["letter_text"] = letter_text
        results.append(result)

        print(f"  SUCCESS: Generated {len(letter_text)} character letter")

    # -------------------------------------------------------------------------
    # SUMMARY - Print processing stats
    # -------------------------------------------------------------------------
    success = sum(1 for r in results if r["status"] == "success")
    out_of_scope = sum(1 for r in results if r["status"] == "out_of_scope")
    errors = sum(1 for r in results if r["status"] not in ("success", "out_of_scope"))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Success: {success}")
    print(f"  Out of Scope: {out_of_scope}")
    print(f"  Errors/No Data: {errors}")

# =============================================================================
# CELL 11: Write Results to Delta Table
# =============================================================================
# For production: Set WRITE_TO_TABLE = True to persist results.
# The score table becomes the source of truth for generated letters.
WRITE_TO_TABLE = False

if n_new_rows > 0 and 'results' in dir() and len(results) > 0:
    # Build output rows from successful results
    output_rows = []

    for r, orig_row in zip(results, df.itertuples()):
        if r["status"] == "success":
            output_rows.append({
                "hsp_account_id": r["hsp_account_id"],
                "pat_mrn_id": getattr(orig_row, 'pat_mrn_id', None),
                "formatted_name": getattr(orig_row, 'formatted_name', None),
                "discharge_summary_note_id": getattr(orig_row, 'discharge_summary_note_id', None),
                "discharge_note_csn_id": getattr(orig_row, 'discharge_note_csn_id', None),
                "hp_note_id": getattr(orig_row, 'hp_note_id', None),
                "hp_note_csn_id": getattr(orig_row, 'hp_note_csn_id', None),
                "letter_type": "Sepsis_v2",  # Identifies this pipeline version
                "letter_text": r["letter_text"],
                "letter_curated_date": datetime.now().date(),
                "denial_info_json": json.dumps(r.get("denial_info", {}), default=str),
                "gold_letter_used": r.get("gold_letter_used"),
                "gold_letter_score": r.get("gold_letter_score"),
                "pipeline_version": "rebuttal_engine_v2",
            })

    if output_rows:
        print(f"\n{len(output_rows)} letters ready to write")

        if WRITE_TO_TABLE:
            import pandas as pd
            output_df = pd.DataFrame(output_rows)
            spark_df = spark.createDataFrame(output_df)
            spark_df = spark_df.withColumn("insert_tsp", current_timestamp())
            spark_df.write.mode("append").saveAsTable(INFERENCE_SCORE_TABLE)
            print(f"Wrote {len(output_df)} letters to {INFERENCE_SCORE_TABLE}")
        else:
            print("To write to table, set WRITE_TO_TABLE = True")

        # Preview the first letter (truncated for display)
        print(f"\n{'='*60}")
        print("PREVIEW: First Generated Letter")
        print(f"{'='*60}")
        print(output_rows[0]["letter_text"][:2000])
        print("...")
    else:
        print("No successful letters to write")

# =============================================================================
# CELL 12: POC - Export to DOCX for User Feedback
# =============================================================================
# For the POC phase, we export letters as Word documents.
# This lets users review and provide feedback on quality.
# In production, we'd skip this and just write to the Delta table.
EXPORT_TO_DOCX = True
DOCX_OUTPUT_PATH = "/Workspace/Repos/mijo8881@mercy.net/fudgesicle/output"

if EXPORT_TO_DOCX and 'output_rows' in dir() and len(output_rows) > 0:
    print(f"\n{'='*60}")
    print("EXPORTING TO DOCX")
    print(f"{'='*60}")

    # Import python-docx for Word document creation
    from docx import Document
    from docx.shared import Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    # Ensure output directory exists
    os.makedirs(DOCX_OUTPUT_PATH, exist_ok=True)

    # Create one DOCX per letter
    for row in output_rows:
        hsp_account_id = row["hsp_account_id"]
        patient_name = row.get("formatted_name", "Unknown")
        letter_text = row["letter_text"]
        denial_info = json.loads(row["denial_info_json"]) if row["denial_info_json"] else {}

        # Create new Word document
        doc = Document()

        # Add title
        title = doc.add_heading(f'Rebuttal Letter', level=1)

        # Add metadata section (for reviewer context)
        meta = doc.add_paragraph()
        meta.add_run("Generated: ").bold = True
        meta.add_run(f"{datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        meta.add_run("Account: ").bold = True
        meta.add_run(f"{hsp_account_id}\n")
        meta.add_run("Patient: ").bold = True
        meta.add_run(f"{patient_name}\n")
        meta.add_run("Payor: ").bold = True
        meta.add_run(f"{denial_info.get('payer_name', 'Unknown')}\n")
        meta.add_run("DRG: ").bold = True
        meta.add_run(f"{denial_info.get('original_drg', '?')} → {denial_info.get('proposed_drg', '?')}\n")
        meta.add_run("Gold Letter: ").bold = True
        if row.get("gold_letter_used"):
            meta.add_run(f"{row['gold_letter_used']} (score: {row.get('gold_letter_score', 0):.3f})\n")
        else:
            meta.add_run("None\n")

        # Visual separator
        doc.add_paragraph("─" * 60)

        # Add the actual letter content (split by double newlines for paragraphs)
        for paragraph in letter_text.split('\n\n'):
            if paragraph.strip():
                p = doc.add_paragraph(paragraph.strip())
                p.paragraph_format.space_after = Pt(12)

        # Generate safe filename (remove special characters)
        safe_name = "".join(c for c in patient_name if c.isalnum() or c in (' ', '-', '_')).strip()
        filename = f"{hsp_account_id}_{safe_name}_rebuttal.docx"
        filepath = os.path.join(DOCX_OUTPUT_PATH, filename)

        # Save the document
        doc.save(filepath)
        print(f"  Saved: {filename}")

    print(f"\nDOCX files saved to: {DOCX_OUTPUT_PATH}")

print("\nInference complete.")
