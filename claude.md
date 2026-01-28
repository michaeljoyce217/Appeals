# Sepsis Appeal Engine - Master Prompt

**Last Updated:** 2026-01-27
**Repo:** https://github.com/michaeljoyce217/SEPSIS

---

## Project Overview

**Goal:** Automated generation of DRG appeal letters for sepsis-related insurance denials (DRG 870/871/872).

**Architecture:** Three-file pipeline using Azure OpenAI GPT-4.1

**Platform:** Databricks on Azure with Unity Catalog

**Status:** POC Complete - Ready for production Epic workqueue integration

---

## Repository Structure

```
SEPSIS/
├── data/
│   ├── featurization_train.py        # ONE-TIME: Knowledge base ingestion (gold letters + propel)
│   └── featurization_inference.py    # PER-CASE: Data prep (denial + notes + structured data)
├── model/
│   └── inference.py                  # GENERATION: Vector search, write, assess, export
├── utils/
│   ├── gold_standard_appeals_sepsis_only/    # Current gold letters + default template
│   ├── gold_standard_appeals_sepsis_multiple/ # Future use
│   ├── sample_denial_letters/        # New denial letters to process (PDFs)
│   ├── propel_data/                  # Clinical criteria definitions (PDFs)
│   └── outputs/                      # Generated appeal letters (DOCX files)
├── docs/
│   ├── plans/                        # Design documents
│   └── rebuttal-engine-overview.html # Executive overview with Technical Architecture
├── compare_denials.py                # Utility: check for duplicate denials
├── test_queries.sql                  # Validation queries for Unity Catalog
├── README.md                         # Project documentation
└── claude.md                         # This file (master prompt)
```

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
All data gathering for a single case:

| Step | Technology | Function |
|------|------------|----------|
| 1. Parse Denial PDF | Azure AI Document Intelligence | OCR extraction from denial PDF |
| 2. Extract Denial Info | GPT-4.1 | Extract: account_id, payor, DRGs, is_sepsis |
| 3. Query Clinical Notes | Spark SQL | Get 14 note types from Epic Clarity |
| 4. Extract Clinical Notes | GPT-4.1 | Extract SOFA components + clinical data with timestamps |
| 5. Query Structured Data | Spark SQL | Get labs, vitals, meds, diagnoses from Clarity |
| 6. Extract Structured Summary | GPT-4.1 | Summarize sepsis-relevant labs/vitals/meds |
| 7. Detect Conflicts | GPT-4.1 | Compare notes vs structured data for discrepancies |

### Generation: inference.py
Run for each denial letter (imports functions from featurization_inference.py):

| Step | Technology | Function |
|------|------------|----------|
| 1. Data Prep | featurization_inference.py | Run steps 1-7 above |
| 2. Vector Search | Cosine Similarity | Find best-matching gold letter (uses denial text only) |
| 3. Letter Generation | GPT-4.1 | Generate appeal using gold letter + notes + structured data |
| 4. Strength Assessment | GPT-4.1 | Evaluate letter against Propel criteria, argument structure, evidence quality |
| 5. Export | python-docx | Output DOCX with assessment + conflicts appendix |

---

## Unity Catalog Tables

### Knowledge Base (populated by featurization_train.py)

| Table | Purpose |
|-------|---------|
| `dev.fin_ds.fudgesicle_gold_letters` | Past winning appeals with denial embeddings |
| `dev.fin_ds.fudgesicle_propel_data` | Official clinical criteria (definition_summary for prompts) |

Note: Structured data is queried directly from Clarity at inference time - no intermediate tables needed.

---

## Key Features

### Evidence Hierarchy
- **Primary Evidence:** Physician notes (clinical interpretation with medical judgment)
- **Supporting Evidence:** Structured data (objective lab values, vitals, medications)
- **Conflict Detection:** When structured data contradicts physician notes, flagged for CDI review

### 14 Clinical Note Types (from Epic Clarity)
Progress Notes, Consults, H&P, Discharge Summary, ED Notes, Initial Assessments, ED Triage Notes, ED Provider Notes, Addendum Note, Hospital Course, Subjective & Objective, Assessment & Plan Note, Nursing Note, Code Documentation

### SOFA Score Extraction
Note extraction prioritizes organ dysfunction data for quantifying sepsis severity:
- **Respiration:** PaO2/FiO2 ratio, oxygen requirements
- **Coagulation:** Platelet count
- **Liver:** Bilirubin
- **Cardiovascular:** MAP, vasopressor use with doses
- **CNS:** GCS (Glasgow Coma Scale)
- **Renal:** Creatinine, urine output
- **Plus:** Lactate trends, infection evidence, antibiotic timing

### Structured Data Summary
Labs, vitals, and medications are queried from Clarity and summarized by LLM for sepsis-relevant data:
- **Labs:** Lactate trends, WBC, procalcitonin, cultures, organ function (creatinine, bilirubin, platelets)
- **Vitals:** Temperature, MAP, heart rate, respiratory rate, SpO2, GCS
- **Meds:** Antibiotic timing (SEP-1 compliance), vasopressor initiation, fluid resuscitation

### Smart Note Extraction
Notes >8,000 chars are automatically extracted via LLM to pull relevant clinical data WITH timestamps (e.g., "03/15/2024 08:00: Lactate 4.2, MAP 63").

### Conflict Detection
Compares physician notes vs structured data to identify discrepancies:
- Note says "MAP maintained >65" but vitals show MAP <65
- Note says "lactate normalized" but labs show lactate still elevated
- Conflicts appear in DOCX appendix for CDI review

### Appeal Strength Assessment
After letter generation, an LLM evaluates the appeal and produces:
- **Overall score** (1-10) with LOW/MODERATE/HIGH rating
- **Summary** (2-3 sentences explaining the score)
- **Detailed breakdown** scoring three dimensions:
  - Propel Criteria Coverage (from: Propel definitions)
  - Argument Structure (from: denial letter, gold template)
  - Evidence Quality (from: clinical notes AND structured data)

Each finding is marked ✓ present, △ could strengthen, or ✗ missing. The "missing" items in Evidence Quality flag specific data points that weren't cited in the letter.

Assessment appears in DOCX before the letter body for CDI reviewer reference.

### Conservative DRG Extraction
Parser only extracts DRG codes if explicitly stated as numbers in the denial letter. Returns "Unknown" rather than hallucinating plausible codes.

### Markdown Bold Parsing
DOCX export converts markdown bold (`**text**`) to actual Word bold formatting for professional output.

### Propel Definition Summary
Full Propel PDFs are processed at ingestion time - LLM extracts key clinical criteria into `definition_summary` field for efficient prompt inclusion.

### Default Template Fallback
When vector search doesn't find a good match (score < 0.7), the system falls back to `default_sepsis_appeal_template.docx` as a structural guide.

### Single-Letter Processing
Each denial is processed end-to-end in one run - no batch processing, no intermediate tables. This:
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

### inference.py Settings
| Setting | Default | Description |
|---------|---------|-------------|
| `DENIAL_PDF_PATH` | (required) | Path to denial letter PDF to process |
| `KNOWN_ACCOUNT_ID` | None | Account ID (if known from Epic workqueue) |
| `SCOPE_FILTER` | "sepsis" | Which denial types to process |
| `MATCH_SCORE_THRESHOLD` | 0.7 | Minimum similarity for gold letter match |
| `NOTE_EXTRACTION_THRESHOLD` | 8000 | Char limit before LLM extraction |
| `EXPORT_TO_DOCX` | True | Export as Word documents |

### featurization_inference.py (called by inference.py)
| Setting | Default | Description |
|---------|---------|-------------|
| `NOTE_EXTRACTION_THRESHOLD` | 8000 | Char limit before LLM extraction |
| `RUN_STRUCTURED_DATA` | True | Query and extract structured data |
| `RUN_CONFLICT_DETECTION` | True | Compare notes vs structured data |

---

## Cost Estimates

Based on Azure OpenAI GPT-4.1 standard pricing ($2.20/1M input, $8.80/1M output):

### Per Appeal Letter (~$0.30)
| Step | Input Tokens | Output Tokens | Cost |
|------|-------------|---------------|------|
| Denial info extraction | ~4,000 | ~100 | $0.01 |
| Note extraction (4 calls avg) | ~12,000 | ~3,200 | $0.05 |
| Structured data extraction | ~8,000 | ~1,500 | $0.03 |
| Conflict detection | ~6,000 | ~500 | $0.02 |
| Appeal letter generation | ~55,000 | ~3,000 | $0.15 |
| Strength assessment | ~15,000 | ~800 | $0.04 |
| **Total** | ~100,000 | ~9,100 | **~$0.30** |

### Monthly Projections
| Volume | LLM Cost |
|--------|----------|
| 100 appeals/month | ~$30 |
| 500 appeals/month | ~$150 |
| 1,000 appeals/month | ~$300 |

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

3. **Process a denial** - In `inference.py`, set the PDF path:
   ```python
   DENIAL_PDF_PATH = "/path/to/denial.pdf"
   ```
   This will automatically run `featurization_inference.py` to gather all case data.

4. **Review output** in `utils/outputs/` - includes assessment section and conflicts appendix (if any)

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Platform | Databricks on Azure |
| LLM | Azure OpenAI GPT-4.1 |
| Embeddings | text-embedding-ada-002 (1536 dims) |
| Document OCR | Azure AI Document Intelligence |
| Storage | Delta Lake tables in Unity Catalog |
| Clinical Data | Epic Clarity |

---

## Team

**Financial Data Science** | Mercy Hospital
