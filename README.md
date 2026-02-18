# DRG Appeal Engine

**Multi-Condition AI for DRG Appeal Letter Automation**

Automated generation of DRG appeal letters for condition-specific insurance denials. Currently supports **sepsis** (DRG 870/871/872) and **acute respiratory failure** (DRG 189/190/191/207/208), extensible to other conditions via pluggable condition profiles. Built by the Financial Data Science team at Mercy Hospital.

---

## Overview

When insurance payors deny or downgrade DRG claims, this system generates professional appeal letters by:

1. **Parsing denial letters** - OCR extracts text from denial PDF
2. **Extracting denial info** - LLM extracts account ID, payor, DRG codes, and condition relevance
3. **Querying clinical data** - Pulls 47 note types + structured data (labs, vitals, meds, diagnoses) from Epic Clarity
4. **Calculating clinical scores** - Programmatic scoring from raw data (e.g., SOFA for sepsis) — zero LLM calls
5. **Numeric cross-check** - LLM extracts numeric claims from notes, Python validates against raw structured data
6. **Detecting conflicts** - Compares physician notes vs structured data for discrepancies
7. **Vector search** - Finds the most similar past winning appeal as a template
8. **Generating appeals** - Creates patient-specific appeal letters with source fidelity rules and condition-specific rebuttals
9. **Assessing strength** - Evaluates the letter across five dimensions: source fidelity, Propel criteria, argument structure, evidence quality, and formatting compliance

**Single-Letter Processing:** Each denial is processed end-to-end in one run. This matches production workflow (Epic workqueue feeds one case at a time) and eliminates batch processing memory issues.

> **POC vs Production:** In POC mode, the LLM extracts the account ID from denial letter text (some generic denials may lack this info). In production, Epic workqueue provides the account ID directly, enabling 100% coverage.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   KNOWLEDGE LAYER (featurization_train.py - run once)       │
│  ┌──────────────┐    ┌──────────────┐                                      │
│  │    Propel    │    │    Gold      │                                      │
│  │   Criteria   │    │   Letters    │                                      │
│  │  (propel_    │    │  (gold_      │                                      │
│  │   data)      │    │   letters)   │                                      │
│  └──────────────┘    └──────────────┘                                      │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│             DATA PREP LAYER (featurization_inference.py - per case)         │
│                                                                             │
│  [Denial PDF] ──► Parse & Extract Info ──► Condition Routing               │
│                                       ──► Query Clinical Notes             │
│                                       ──► Query Structured Data            │
│                                       ──► Calculate Clinical Scores        │
│                                       ──► Numeric Cross-Check              │
│                                       ──► Detect Conflicts                 │
│                                                                             │
│  Outputs: denial_text, denial_info, clinical_notes, structured_summary,    │
│           clinical_scores, conflicts (notes vs structured data)             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                   GENERATION LAYER (inference.py - per case)                │
│                                                                             │
│  Step 1: Vector search for best gold letter                                 │
│  Step 2: Generate appeal (source fidelity + conditional rebuttals)          │
│  Step 3: 5-dimension assessment (fidelity, criteria, structure, evidence,  │
│           formatting)                                                       │
│  Step 4: Export to DOCX (includes conflicts appendix if any)                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

[DOCX Letter] ──► CDI Review ──► Approved Letter ──► Send to Payor
```

## Supported Conditions

| Condition | DRG Codes | Clinical Scoring | Conditional Rebuttals |
|-----------|-----------|-----------------|----------------------|
| **Sepsis** | 870, 871, 872 | SOFA (6 organs, deterministic) | None |
| **Acute Respiratory Failure** | 189, 190, 191, 207, 208 | None | 3 physician-authored rebuttals |

### Respiratory Failure — Conditional Rebuttals

The ARF profile includes three physician-authored rebuttal templates (from Dr. Gharfeh and Dr. Bourland) that are conditionally injected when they match the payor's actual denial argument:

| Rebuttal | Triggers When |
|----------|--------------|
| **Consecutive SpO2 Readings** | Denial requires "consecutive" or "sequential" SpO2 readings |
| **Persistent Symptoms** | Denial argues symptoms were not "persistent" or "continuous" |
| **Proprietary Clinical Criteria** | Denial imposes thresholds beyond nationally recognized standards |

The engine reads the denial, decides which rebuttals apply, and weaves them into the letter. Rebuttals that don't match the denial are not applied.

## Key Features

| Feature | Description |
|---------|-------------|
| **Condition Profiles** | Pluggable condition modules — add a new condition by providing data files + a config module |
| **Conditional Rebuttals** | Physician-authored rebuttal templates injected only when they match the payor's actual argument |
| **Source Fidelity** | Letters only cite Propel-approved references — never list or validate the payor's cited sources |
| **Single-Letter Processing** | One denial at a time — no batch processing, no memory issues |
| **Evidence Hierarchy** | Physician notes (primary) + structured data (supporting) for comprehensive evidence |
| **Programmatic Clinical Scoring** | Deterministic scoring from raw data (e.g., SOFA for sepsis) — zero LLM calls |
| **Numeric Cross-Check** | LLM-extracted numeric claims validated against closest-in-time raw values |
| **Conflict Detection** | Flags discrepancies between notes and structured data for CDI review |
| **5-Dimension Assessment** | Source fidelity, Propel criteria, argument structure, evidence quality, formatting compliance |
| **Assertion-Only Language** | No hedging, conceding, or defensive phrasing — every sentence advances the argument |
| **Vector Search** | Embeddings-based similarity matching finds the most relevant past denial |
| **Gold Letter Learning** | Uses winning appeals as templates — proven arguments get reused |
| **Default Template Fallback** | When no good match found, uses default template as structural guide |
| **Propel Integration** | LLM extracts key criteria from Propel PDFs into concise summaries |
| **Dynamic Note Extraction** | Propel definition drives what the LLM extracts from notes — no hardcoded targets |
| **Comprehensive Clinical Notes** | Pulls 47 note types from Clarity |
| **Conservative DRG Extraction** | Only extracts DRGs if explicitly stated — no hallucination of plausible codes |
| **Markdown Bold Parsing** | `**text**` in LLM output renders as bold in DOCX |
| **Human-in-the-Loop** | All letters output as DOCX for CDI review before sending |
| **Production-Ready** | Supports Epic workqueue integration via KNOWN_ACCOUNT_ID |

## Condition Profile Architecture

The base engine is condition-agnostic. All condition-specific logic lives in `condition_profiles/<condition>.py`. Each notebook has a `CONDITION_PROFILE` setting:

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
| **Structured Data** | `STRUCTURED_DATA_CONTEXT`, `DIAGNOSIS_EXAMPLES` |
| **Conflict Detection** | `CONFLICT_EXAMPLES` |
| **Numeric Cross-Check** | `PARAM_TO_CATEGORY`, `LAB_VITAL_MATCHERS` |
| **Writer Prompt** | `WRITER_SCORING_INSTRUCTIONS` |
| **Assessment** | `ASSESSMENT_CONDITION_LABEL`, `ASSESSMENT_CRITERIA_LABEL` |
| **Clinical Scorer** (optional) | `calculate_clinical_scores()`, `write_clinical_scores_table()` |
| **DOCX Rendering** (optional) | `format_scores_for_prompt()`, `render_scores_in_docx()`, `render_scores_status_note()` |
| **Conditional Rebuttals** (optional) | `CONDITIONAL_REBUTTALS` (list of rebuttal dicts) |

### Adding a New Condition

1. Copy `condition_profiles/TEMPLATE.py` to `condition_profiles/<your_condition>.py`
2. Fill in all required constants (see TEMPLATE.py for documentation)
3. Add gold standard letters to a workspace subdirectory
4. Add Propel definition PDF to the propel_data directory
5. Optionally implement a clinical scorer (like SOFA for sepsis)
6. Optionally add conditional rebuttals for common payor denial arguments
7. Set `CONDITION_PROFILE = "<your_condition>"` in each notebook
8. Run `featurization_train.py` to ingest gold letters and Propel data
9. Process cases normally with `featurization_inference.py` then `inference.py`

## Appeal Strength Assessment

After letter generation, an LLM evaluates the appeal across five scoped dimensions:

| Dimension | Evaluates Against | Description |
|-----------|-------------------|-------------|
| **Source Fidelity** | Propel document | Did the letter use only Propel-approved criteria and references? Did it avoid citing the payor's references unnecessarily? |
| **Propel Criteria** | Propel definition only | Coverage of official clinical criteria. Scoped strictly to the Propel definition text — does not infer criteria from other sources. |
| **Argument Structure** | Denial letter + gold template | How well the letter rebuts the denial, follows proven structure, and applies conditional rebuttals correctly |
| **Evidence Quality** | Clinical notes + structured data | Whether specific values and timestamps are cited |
| **Formatting Compliance** | Letter format rules | No summary table at the end; all clinical data embedded in the argument |

Each finding is marked present, could strengthen, or missing. The overall score (1-10) with LOW/MODERATE/HIGH rating appears in the DOCX before the letter body for CDI reviewer reference.

## Repository Structure

```
SEPSIS/
├── condition_profiles/
│   ├── __init__.py                   # Profile validation (REQUIRED_ATTRIBUTES)
│   ├── sepsis.py                     # Sepsis profile (config + SOFA scorer + DOCX rendering)
│   ├── respiratory_failure.py        # ARF profile (config + conditional rebuttals)
│   └── TEMPLATE.py                   # Template for creating new condition profiles
├── data/
│   ├── featurization_train.py        # ONE-TIME: Knowledge base ingestion (gold letters + Propel)
│   ├── featurization_inference.py    # PER-CASE: Data prep (denial + notes + structured data)
│   └── structured_data_ingestion.py  # PER-CASE: Labs, vitals, meds, diagnoses from Clarity
├── model/
│   └── inference.py                  # GENERATION: Vector search, write, assess, export
├── tests/
│   └── test_condition_profiles.py    # Unit tests (35 tests — profiles, validation, prompt assembly)
├── docs/
│   ├── plans/                        # Design & implementation plans
│   ├── drg_appeal_engine_v2_synopsis.html  # Executive overview (multi-condition)
│   ├── rebuttal-engine-overview.html       # Technical overview (tabbed, detailed)
│   ├── appeals-team-overview.html          # Simplified overview for appeals team
│   └── technical-architecture.html         # Architecture deep-dive
├── archive/                          # Older versions, POC code, source materials
├── test_queries.sql                  # Validation queries for Unity Catalog
├── README.md                         # This file
└── claude.md                         # Master prompt (project documentation)
```

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
| `fudgesicle_case_sofa_scores` | Programmatic clinical scores per organ system (JSON), total score |
| `fudgesicle_case_conflicts` | Detected conflicts + recommendation (includes numeric mismatches) |

### Intermediate Data (populated by featurization_inference.py)

| Table | Purpose |
|-------|---------|
| `fudgesicle_labs` | Lab results with timestamps |
| `fudgesicle_vitals` | Vital signs with timestamps |
| `fudgesicle_meds` | Medication administrations |
| `fudgesicle_diagnoses` | DX records with timestamps |
| `fudgesicle_structured_timeline` | Merged chronological timeline |

All tables use the `{trgt_cat}.fin_ds.` prefix (e.g., `dev.fin_ds.fudgesicle_*`).

## Quick Start (Databricks)

### 1. Install Dependencies
Run Cell 1 alone, then restart:
```python
%pip install azure-ai-documentintelligence==1.0.2 openai python-docx
dbutils.library.restartPython()
```

### 2. One-Time Setup
In `featurization_train.py`:
```python
CONDITION_PROFILE = "respiratory_failure"  # or "sepsis"
RUN_GOLD_INGESTION = True
RUN_PROPEL_INGESTION = True
```
Run the notebook. This ingests gold standard letters and Propel clinical definitions.

### 3. Prepare Case Data
In `featurization_inference.py`, set the denial PDF path:
```python
CONDITION_PROFILE = "respiratory_failure"  # or "sepsis"
DENIAL_PDF_PATH = "/path/to/denial_letter.pdf"
```
Run the notebook to parse the denial, extract clinical notes, calculate clinical scores, and write all case data to tables.

### 4. Generate Appeal
Run `inference.py`:
```python
CONDITION_PROFILE = "respiratory_failure"  # or "sepsis"
```
This reads prepared case data, generates the appeal letter, assesses its strength, and exports a DOCX.

### 5. Review Output
DOCX structure:
- Internal review: assessment + clinical scores status (if applicable) + conflicts (if any)
- Letter body (LLM-generated)
- Appendix: Clinical Scores Table (if applicable, e.g., SOFA for sepsis)

## Configuration

### featurization_train.py
| Setting | Default | Description |
|---------|---------|-------------|
| `CONDITION_PROFILE` | `"sepsis"` | Which condition profile to use |
| `RUN_GOLD_INGESTION` | `False` | Process gold standard letter PDFs |
| `RUN_PROPEL_INGESTION` | `False` | Process Propel definition PDFs |

### featurization_inference.py
| Setting | Default | Description |
|---------|---------|-------------|
| `CONDITION_PROFILE` | `"sepsis"` | Which condition profile to use |
| `DENIAL_PDF_PATH` | (required) | Path to denial letter PDF to process |
| `KNOWN_ACCOUNT_ID` | `None` | Account ID (if known from Epic workqueue) |

### inference.py
| Setting | Default | Description |
|---------|---------|-------------|
| `CONDITION_PROFILE` | `"sepsis"` | Which condition profile to use |
| `MATCH_SCORE_THRESHOLD` | `0.7` | Minimum similarity for gold letter match |
| `EXPORT_TO_DOCX` | `True` | Export as Word documents |

## Technology Stack

| Component | Technology |
|-----------|------------|
| Platform | Databricks on Azure |
| LLM | Azure OpenAI GPT-4.1 |
| Embeddings | text-embedding-ada-002 (1536 dims, 30k char limit) |
| Document OCR | Azure AI Document Intelligence |
| Storage | Delta Lake tables in Unity Catalog |
| Clinical Data | Epic Clarity |

## Cost Estimates

Based on Azure OpenAI GPT-4.1 standard pricing ($2.20/1M input, $8.80/1M output):

### Per Appeal Letter (~$0.50)

| Step | Input Tokens | Output Tokens | Cost |
|------|-------------|---------------|------|
| Denial info extraction | ~4,000 | ~100 | $0.01 |
| Note extraction (~20 calls avg, parallel) | ~50,000 | ~12,000 | $0.22 |
| Clinical score calculation | 0 | 0 | $0.00 |
| Numeric cross-check | ~5,000 | ~1,500 | $0.02 |
| Structured data extraction | ~8,000 | ~1,500 | $0.03 |
| Conflict detection | ~6,000 | ~500 | $0.02 |
| Appeal letter generation | ~60,000 | ~3,000 | $0.16 |
| Strength assessment | ~15,000 | ~1,500 | $0.05 |
| **Total** | ~148,000 | ~20,100 | **~$0.50** |

47 note types are queried but ~20 typically have content for a given case. All extractions run in parallel so wall-clock time is similar to a single call.

### Monthly Projections

| Volume | LLM Cost |
|--------|----------|
| 100 appeals/month | ~$50 |
| 500 appeals/month | ~$250 |
| 1,000 appeals/month | ~$500 |

**One-time setup:** <$1 for gold letter + Propel ingestion

**Infrastructure:** $0 incremental (uses existing Databricks/Azure)

## Team

**Financial Data Science** | Mercy Hospital

---

*Built with Azure OpenAI, Databricks, and Epic Clarity integration.*
