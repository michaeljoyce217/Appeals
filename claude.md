# DRG Appeal Engine - Master Prompt

**Last Updated:** 2026-02-17
**Repo:** https://github.com/michaeljoyce217/SEPSIS

---

## Project Overview

**Goal:** Automated generation of DRG appeal letters for condition-specific insurance denials. Currently supports sepsis (DRG 870/871/872) and acute respiratory failure (DRG 189/190/191/207/208), extensible to other conditions via condition profiles.

**Architecture:** Three-file pipeline using Azure OpenAI GPT-4.1, with pluggable condition profiles

**Platform:** Databricks on Azure with Unity Catalog

**Status:** POC Complete - Ready for production Epic workqueue integration

---

## Repository Structure

```
SEPSIS/
├── condition_profiles/
│   ├── sepsis.py                     # Sepsis profile (config + validation + SOFA scorer + DOCX rendering)
│   └── respiratory_failure.py        # ARF profile (config + validation + conditional rebuttals)
├── data/
│   ├── featurization_train.py        # ONE-TIME: Knowledge base ingestion (gold letters + propel)
│   └── featurization_inference.py    # PER-CASE: Data prep (denial + notes + structured data)
├── model/
│   └── inference.py                  # GENERATION: Vector search, write, assess, export
├── utils/
│   ├── gold_standard_appeals.gold_standard_appeals__sepsis_only/  # Current gold letters + default template
│   ├── gold_standard_appeals_sepsis_multiple/ # Future use
│   ├── sample_denial_letters/        # New denial letters to process (PDFs)
│   ├── propel_data/                  # Clinical criteria definitions (PDFs)
│   └── outputs/                      # Generated appeal letters (DOCX files)
├── docs/
│   ├── plans/                        # Design documents
│   ├── rebuttal-engine-overview.html # Technical overview (tabbed, detailed)
│   └── appeals-team-overview.html    # Simplified overview for appeals team
├── test_queries.sql                  # Validation queries for Unity Catalog
├── README.md                         # Project documentation
└── claude.md                         # This file (master prompt)
```

---

## Environment Note

**Catalog Access Pattern:** Data lives in the `prod` catalog, but we can only write to our current environment's catalog (`dev` or `prod`). This is intentional - all code uses `USE CATALOG prod;` for queries but writes tables with the `trgt_cat` prefix (e.g., `dev.fin_ds.fudgesicle_*`).

---

## Pipeline Architecture

### One-Time Setup: featurization_train.py
Run once to populate knowledge base tables:

| Step | Technology | Function |
|------|------------|----------|
| Gold Letter Parsing | Azure AI Document Intelligence + GPT-4.1 | Extract appeal/denial from gold PDFs |
| Denial Embedding | text-embedding-ada-002 | Generate 1536-dim vectors for similarity search |
| Propel Extraction | GPT-4.1 | Extract key clinical criteria from Propel PDFs |

### Per-Case Data Prep: featurization_inference.py
All data gathering for a single case. **Writes to case tables for inference.py to read.**

| Step | Technology | Function |
|------|------------|----------|
| 1. Parse Denial PDF | Azure AI Document Intelligence | OCR extraction from denial PDF |
| 2. Extract Denial Info | GPT-4.1 | Extract: account_id, payor, DRGs, condition relevance |
| 3. Query Clinical Notes | Spark SQL | Get ALL notes from 47 types from Epic Clarity |
| 4. Extract Clinical Notes | GPT-4.1 (parallel) | Extract SOFA components + clinical data with timestamps |
| 5. Query Structured Data | Spark SQL | Get labs, vitals, meds, diagnoses from Clarity |
| 5.5. Calculate SOFA Scores | Python (deterministic) | Programmatic SOFA scoring from raw labs/vitals/meds — zero LLM calls |
| 5.7. Numeric Cross-Check | GPT-4.1 + Python | Extract numeric claims from notes, compare against raw structured data |
| 6. Extract Structured Summary | GPT-4.1 | Summarize sepsis-relevant data with diagnosis descriptions |
| 7. Detect Conflicts | GPT-4.1 | Compare notes vs structured data for discrepancies (includes numeric mismatches) |
| 8. Write Case Tables | Spark SQL | Write all outputs to case tables |

### Generation: inference.py
Reads prepared data from case tables (run featurization_inference.py first):

| Step | Technology | Function |
|------|------------|----------|
| 1. Load Case Data | Spark SQL | Read from case tables written by featurization_inference.py |
| 2. Vector Search | Cosine Similarity | Find best-matching gold letter (uses denial embedding) |
| 3. Letter Generation | GPT-4.1 | Generate appeal using gold letter + notes + structured data |
| 4. Strength Assessment | GPT-4.1 | Evaluate letter against Propel criteria, argument structure, evidence quality |
| 5. Export | python-docx | Output DOCX with assessment + conflicts appendix |

---

## Condition Profile Architecture

The base engine is condition-agnostic. All condition-specific logic lives in `condition_profiles/<condition>.py`. Each notebook has a `CONDITION_PROFILE` setting at the top:

```python
CONDITION_PROFILE = "sepsis"  # "sepsis", "respiratory_failure", etc.
profile = importlib.import_module(f"condition_profiles.{CONDITION_PROFILE}")
```

### What a Condition Profile Provides

| Category | Examples (sepsis) |
|----------|-------------------|
| **Identity** | `CONDITION_NAME`, `DRG_CODES`, paths to gold letters/Propel/templates |
| **Denial Parser** | `DENIAL_CONDITION_QUESTION`, `DENIAL_CONDITION_FIELD` |
| **Note Extraction** | `NOTE_EXTRACTION_TARGETS` (fallback when Propel unavailable) |
| **Structured Data** | `STRUCTURED_DATA_CONTEXT`, `STRUCTURED_DATA_SYSTEM_MESSAGE` |
| **Conflict Detection** | `CONFLICT_EXAMPLES` |
| **Numeric Cross-Check** | `PARAM_TO_CATEGORY`, `LAB_VITAL_MATCHERS` |
| **Writer Prompt** | `WRITER_SCORING_INSTRUCTIONS` |
| **Assessment** | `ASSESSMENT_CONDITION_LABEL`, `ASSESSMENT_CRITERIA_LABEL` |
| **Clinical Scorer** (optional) | `calculate_clinical_scores()`, `write_clinical_scores_table()` |
| **DOCX Rendering** (optional) | `format_scores_for_prompt()`, `render_scores_in_docx()`, `render_scores_status_note()` |
| **Conditional Rebuttals** (optional) | `CONDITIONAL_REBUTTALS` (list of rebuttal dicts) |

### Dynamic Note Extraction

The base engine loads the Propel `definition_summary` for the current condition and uses it to drive note extraction targets dynamically. The profile's `NOTE_EXTRACTION_TARGETS` is only used as a fallback when Propel data is unavailable. This means adding a new condition mostly requires data files + config, not code.

### Adding a New Condition

1. Copy an existing profile (e.g., `condition_profiles/respiratory_failure.py`) to `condition_profiles/<your_condition>.py`
2. Fill in all required constants (see `REQUIRED_ATTRIBUTES` list in the profile)
3. Add gold standard letters to a `utils/` subdirectory
4. Add Propel definition PDF to `utils/propel_data/`
5. Optionally implement a clinical scorer (like SOFA for sepsis)
6. Set `CONDITION_PROFILE = "<your_condition>"` in each notebook
7. Run `featurization_train.py` to ingest gold letters and Propel data
8. Process cases normally with `featurization_inference.py` → `inference.py`

### Schema Changes

The denial table now uses `is_condition_match` (boolean) and `condition_name` (string) instead of the previous `is_sepsis` column.

---

## Unity Catalog Tables

### Knowledge Base (populated by featurization_train.py)

| Table | Purpose |
|-------|---------|
| `fudgesicle_gold_letters` | Past winning appeals with denial embeddings |
| `fudgesicle_propel_data` | Official clinical criteria (definition_summary for prompts) |

### Case Data (populated by featurization_inference.py, read by inference.py)

| Table | Purpose |
|-------|---------|
| `fudgesicle_case_denial` | Denial text, embedding, payor, DRGs, is_condition_match flag, condition_name |
| `fudgesicle_case_clinical` | Patient info + extracted clinical notes (JSON) |
| `fudgesicle_case_structured_summary` | LLM summary of structured data |
| `fudgesicle_case_sofa_scores` | Programmatic SOFA scores per organ system (JSON), total score, vasopressor detail |
| `fudgesicle_case_conflicts` | Detected conflicts + recommendation (includes numeric mismatches) |

### Intermediate Data (populated by featurization_inference.py)

| Table | Purpose |
|-------|---------|
| `fudgesicle_labs` | Lab results with timestamps |
| `fudgesicle_vitals` | Vital signs with timestamps |
| `fudgesicle_meds` | Medication administrations |
| `fudgesicle_diagnoses` | DX records with timestamps |
| `fudgesicle_structured_timeline` | Merged chronological timeline |

Note: All tables use the `{trgt_cat}.fin_ds.` prefix (e.g., `dev.fin_ds.fudgesicle_*`).

---

## Key Features

### Evidence Hierarchy
- **Primary Evidence:** Physician notes (clinical interpretation with medical judgment)
- **Supporting Evidence:** Structured data (objective lab values, vitals, medications)
- **Conflict Detection:** When structured data contradicts physician notes, flagged for CDI review

### 47 Clinical Note Types (from Epic Clarity)
Progress Notes, Consults, H&P, Discharge Summary, ED Notes, Initial Assessments, ED Triage Notes, ED Provider Notes, Addendum Note, Hospital Course, Subjective & Objective, Assessment & Plan Note, Nursing Note, Code Documentation, Anesthesia Preprocedure Evaluation, Anesthesia Postprocedure Evaluation, H&P (View-Only), Internal H&P Note, Anesthesia Procedure Notes, L&D Delivery Note, Pre-Procedure Assessment, Inpatient Medication Chart, Hospice, Hospice Plan of Care, Hospice Non-Covered, OR Post-Procedure Note, Peri-OP, Treatment Plan, Delivery, Brief Op Note, Operative Report, Scanned Form, Therapy Evaluation, Therapy Treatment, Therapy Discharge, Therapy Progress Note, Wound Care, Anesthesia Post Evaluation, Query, Anesthesia Post-Op Follow-up Note, Anesthesia Handoff, Anesthesia PAT Evaluation, Anesthesiology, ED Attestation Note, ED Procedure Note, ED Re-evaluation Note, CDU Provider Note

**Note:** ALL notes from the encounter are retrieved (not just most recent), concatenated chronologically with timestamps.

### Programmatic SOFA Scoring
SOFA scores are calculated deterministically from raw structured data (zero LLM calls). The calculator reads directly from `fudgesicle_labs`, `fudgesicle_vitals`, and `fudgesicle_meds` tables, applies standard SOFA thresholds, and outputs per-organ scores with source values and timestamps:
- **Respiration:** PaO2/FiO2 ratio (closest-in-time pairing)
- **Coagulation:** Worst platelet count
- **Liver:** Worst total bilirubin
- **Cardiovascular:** MAP + vasopressor-based scoring (dopamine, dobutamine, epinephrine, norepinephrine dose thresholds)
- **CNS:** GCS (Glasgow Coma Scale) — uses FLO_MEAS_ID '1525'
- **Renal:** Worst creatinine

When SOFA total >= 2, a formatted table is appended after the letter body (rendered programmatically in DOCX, not by the LLM). The LLM references the scores narratively but does not include a table in the letter text. When the SOFA table is omitted (score < 2 or insufficient data), the internal review section explains why.

### SOFA Score Extraction (Notes)
Note extraction still prioritizes organ dysfunction data for qualitative clinical context:
- Lactate trends, infection evidence, antibiotic timing
- Physician assessments of sepsis severity

### Structured Data Summary
Labs, vitals, and medications are queried from Clarity and summarized by LLM for sepsis-relevant data:
- **Labs:** Lactate trends, WBC, procalcitonin, cultures, organ function (creatinine, bilirubin, platelets)
- **Vitals:** Temperature, MAP, heart rate, respiratory rate, SpO2, GCS
- **Meds:** Antibiotic timing (SEP-1 compliance), vasopressor initiation, fluid resuscitation

### Diagnosis Records (DX_NAME + ICD10_CODE)
We query DX records from three Epic sources (outpatient encounter DX, inpatient account DX, problem list history) and include both the ICD-10 code and the granular clinical description:
- **ICD10_CODE** is the standard billing code (e.g., "J96.01")
- **DX_NAME** is the specific clinical description (e.g., "Acute respiratory failure with hypoxia")
- **DX_ID** provides traceability to the source record
- All diagnoses include timestamps - LLM decides relevance based on date
- In the merged timeline, diagnoses appear as "ICD10_CODE - DX_NAME" (e.g., "J96.01 - Acute respiratory failure with hypoxia")
- Both codes are forwarded to the writing agent for citation in appeals

### LLM Note Extraction (Parallel)
All clinical notes are extracted via LLM in parallel (ThreadPoolExecutor) to pull relevant clinical data WITH timestamps in a consistent structured format (e.g., "03/15/2024 08:00: Lactate 4.2, MAP 63"). This ensures homogeneous output regardless of note length and keeps latency flat despite the large number of note types.

### Conflict Detection
Compares physician notes vs structured data to identify discrepancies:
- Note says "MAP maintained >65" but vitals show MAP <65
- Note says "lactate normalized" but labs show lactate still elevated
- **Numeric cross-check:** LLM extracts all numeric claims from notes, then Python compares each against the closest-in-time raw value (>10% relative difference for continuous values, exact match for integer scales like GCS)
- Conflicts (including numeric mismatches) appear in DOCX appendix for CDI review

### Assertion-Only Language
The writer prompt enforces strict language rules to prevent giving payors anything to seize on:
- Every sentence must advance the argument — state what IS documented, never what is missing
- Forbidden phrases: "although", "despite", "however", "only", "merely", "may/might/could" and any hedging, minimizing, or conceding language
- Never concede any point from the denial letter — refute or ignore, never agree
- If data for a parameter is absent, it is omitted entirely — never mentioned or hedged
- Every paragraph must contain at least one specific clinical value with timestamp
- Only cite Propel-approved references — do not list or comment on the payor's cited references unless directly refuting a specific clinical claim
- Do not include a "Summary Table of Key Clinical Data" or similar table at the end of the letter

### Conditional Rebuttals
Some conditions have specific rebuttal templates for common payor denial arguments (e.g., respiratory failure has SpO2 reading rebuttals from Dr. Gharfeh and Dr. Bourland). These are defined in the condition profile as `CONDITIONAL_REBUTTALS` and are conditionally injected into both the writer prompt and assessment prompt. The LLM applies ONLY the rebuttals that match the payor's actual argument — it does not apply rebuttals for arguments the payor did not make.

### Appeal Strength Assessment
After letter generation, an LLM evaluates the appeal and produces:
- **Overall score** (1-10) with LOW/MODERATE/HIGH rating
- **Summary** (2-3 sentences explaining the score)
- **Detailed breakdown** scoring five dimensions:
  - Source Fidelity (did the letter use ONLY Propel-approved criteria and references?)
  - Propel Criteria Coverage (from: Propel definitions)
  - Argument Structure (from: denial letter, gold template; rebuttals applied correctly?)
  - Evidence Quality (from: clinical notes AND structured data)
  - Formatting Compliance (no summary table at end of letter)

Each finding is marked ✓ present, △ could strengthen, or ✗ missing. The "missing" items in Evidence Quality flag specific data points that weren't cited in the letter.

**Scoping Rules:** Each assessment dimension is explicitly constrained to evaluate against its designated evidence source. The `propel_criteria` dimension evaluates ONLY against criteria stated in the Propel definition — the LLM is instructed not to infer additional criteria from clinical evidence, denial letters, gold letters, or general medical knowledge. This prevents "bleed" where the assessor flags items that weren't actually in the Propel input.

Assessment appears in DOCX before the letter body for CDI reviewer reference. Includes clinical scores status note when the scores table is omitted.

### Conservative DRG Extraction
Parser only extracts DRG codes if explicitly stated as numbers in the denial letter. Returns "Unknown" rather than hallucinating plausible codes.

### Markdown Bold Parsing
DOCX export converts markdown bold (`**text**`) to actual Word bold formatting for professional output.

### Propel Definition Summary
Full Propel PDFs are processed at ingestion time - LLM extracts key clinical criteria into `definition_summary` field for efficient prompt inclusion.

### Default Template Fallback
When vector search doesn't find a good match (score < 0.7), the system falls back to `default_sepsis_appeal_template.docx` as a structural guide. Score displays as "N/A" in output.

### Single-Letter Processing
Each denial is processed end-to-end in one run - no batch processing. This:
- Eliminates driver memory issues
- Matches production workflow (Epic workqueue feeds one case at a time)
- Simplifies debugging and testing

### Output Location
Appeal letters are saved to `utils/outputs/` with filename format: `{account_id}_{patient_name}_appeal.docx`

---

## Configuration

### featurization_train.py Flags
| Flag | Default | Description |
|------|---------|-------------|
| `RUN_GOLD_INGESTION` | False | Process gold standard letter PDFs |
| `RUN_PROPEL_INGESTION` | False | Process Propel definition PDFs |

### featurization_inference.py Settings
| Setting | Default | Description |
|---------|---------|-------------|
| `DENIAL_PDF_PATH` | (required) | Path to denial letter PDF to process |
| `KNOWN_ACCOUNT_ID` | None | Account ID (if known from Epic workqueue) |

### inference.py Settings
| Setting | Default | Description |
|---------|---------|-------------|
| `MATCH_SCORE_THRESHOLD` | 0.7 | Minimum similarity for gold letter match |
| `EXPORT_TO_DOCX` | True | Export as Word documents |

---

## Cost Estimates

Based on Azure OpenAI GPT-4.1 standard pricing ($2.20/1M input, $8.80/1M output):

### Per Appeal Letter (~$0.50)
| Step | Input Tokens | Output Tokens | Cost |
|------|-------------|---------------|------|
| Denial info extraction | ~4,000 | ~100 | $0.01 |
| Note extraction (~20 calls avg, parallel) | ~50,000 | ~12,000 | $0.22 |
| SOFA calculation | 0 | 0 | $0.00 |
| Numeric cross-check | ~5,000 | ~1,500 | $0.02 |
| Structured data extraction | ~8,000 | ~1,500 | $0.03 |
| Conflict detection | ~6,000 | ~500 | $0.02 |
| Appeal letter generation | ~60,000 | ~3,000 | $0.16 |
| Strength assessment | ~15,000 | ~1,500 | $0.05 |
| **Total** | ~148,000 | ~20,100 | **~$0.50** |

Note: 47 note types are queried but ~20 typically have content for a given sepsis case. All extractions run in parallel so wall-clock time is similar to a single call.

### Monthly Projections
| Volume | LLM Cost |
|--------|----------|
| 100 appeals/month | ~$50 |
| 500 appeals/month | ~$250 |
| 1,000 appeals/month | ~$500 |

**One-time setup:** <$1 for gold letter + Propel ingestion

---

### POC vs Production

**POC Mode:** Set `KNOWN_ACCOUNT_ID = None` - LLM extracts account ID from denial letter text. Some generic denials may lack identifiable information.

**Production Mode:** Set `KNOWN_ACCOUNT_ID = "12345678"` - Epic workqueue provides account ID directly, enabling 100% coverage.

---

## Quick Start (Databricks)

1. **Install dependencies** (run Cell 1 alone, then restart):
   ```python
   %pip install azure-ai-documentintelligence==1.0.2 openai python-docx
   dbutils.library.restartPython()
   ```

2. **One-time setup** - Run `featurization_train.py` with flags enabled:
   ```python
   RUN_GOLD_INGESTION = True   # First run
   RUN_PROPEL_INGESTION = True # First run
   ```

3. **Prepare case data** - In `featurization_inference.py`, set the PDF path:
   ```python
   DENIAL_PDF_PATH = "/path/to/denial.pdf"
   ```
   Run the notebook to prepare all case data (writes to case tables).

4. **Generate appeal** - Run `inference.py`:
   - Reads prepared case data from tables
   - Generates appeal letter
   - Outputs DOCX to `utils/outputs/`

5. **Review output** - DOCX structure:
   - Internal review: assessment + SOFA status note + conflicts (if any)
   - Letter body (LLM-generated)
   - Appendix: SOFA Score Table (if total >= 2)

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Platform | Databricks on Azure |
| LLM | Azure OpenAI GPT-4.1 |
| Embeddings | text-embedding-ada-002 (1536 dims, 30k char limit) |
| Document OCR | Azure AI Document Intelligence |
| Storage | Delta Lake tables in Unity Catalog |
| Clinical Data | Epic Clarity |

---

## Verification Rules

After writing or modifying code in this project, verify:
1. All referenced Unity Catalog table names use the correct `{trgt_cat}.fin_ds.fudgesicle_*` pattern.
2. SQL queries use `USE CATALOG prod` for reading source data.
3. Condition profile references match the `CONDITION_PROFILE` setting.
4. LLM prompts follow assertion-only language rules — no hedging, conceding, or mentioning absent data.
5. SOFA scoring logic uses deterministic Python, not LLM calls.
6. Output would match the previous version unless a change was explicitly requested.

## Team

**Financial Data Science** | Mercy Hospital
