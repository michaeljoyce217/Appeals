# Sepsis Appeal Engine

**Multi-Agent AI for DRG Appeal Letter Automation**

Automated generation of DRG appeal letters for sepsis-related insurance denials. Built by the Financial Data Science team at Mercy Hospital.

---

## Overview

When insurance payors deny or downgrade sepsis DRG claims (870/871/872), this system generates professional appeal letters by:

1. **Parsing denial letters** - OCR extracts text from denial PDF
2. **Vector search** - Finds the most similar past denial from our gold standard library (uses denial text only)
3. **Extracting denial info** - LLM extracts account ID, payor, DRG codes, and determines if sepsis-related
4. **Querying clinical data** - Pulls 14 note types from Epic Clarity for this specific account
5. **Learning from winners** - Uses the matched winning appeal as a template/guide
6. **Applying clinical criteria** - Includes official Propel sepsis definitions
7. **Generating appeals** - Creates patient-specific appeal letters using clinical notes

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
│  [Denial PDF] ──► Parse & Extract Info ──► Query Clinical Notes            │
│                                       ──► Query Structured Data             │
│                                       ──► Detect Conflicts                  │
│                                                                             │
│  Outputs: denial_text, denial_info, clinical_notes, structured_summary,    │
│           conflicts (notes vs structured data)                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                   GENERATION LAYER (inference.py - per case)                │
│                                                                             │
│  Step 1: Vector search for best gold letter                                 │
│  Step 2: Generate appeal (LLM: gold letter + notes + structured data)       │
│  Step 3: Assess strength (LLM: Propel criteria, evidence quality)           │
│  Step 4: Export to DOCX (includes conflicts appendix if any)                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

[DOCX Letter] ──► CDI Review ──► Approved Letter ──► Send to Payor
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Single-Letter Processing** | One denial at a time - no batch processing, no memory issues |
| **Evidence Hierarchy** | Physician notes (primary) + structured data (supporting) for comprehensive evidence |
| **Conflict Detection** | Flags discrepancies between notes and structured data for CDI review |
| **Vector Search** | Embeddings-based similarity matching finds the most relevant past denial |
| **Gold Letter Learning** | Uses winning appeals as templates - proven arguments get reused |
| **Default Template Fallback** | When no good match found, uses default template as structural guide |
| **Propel Integration** | LLM extracts key criteria from Propel PDFs into concise summaries |
| **Comprehensive Clinical Notes** | Pulls 14 note types from Clarity (see below) |
| **Structured Data Summary** | Labs, vitals, meds queried from Clarity and summarized for sepsis relevance |
| **Smart Note Extraction** | Long notes (>8k chars) auto-extracted with timestamps via LLM |
| **SOFA Score Extraction** | Prioritizes organ dysfunction data: lactate, MAP, creatinine, platelets, bilirubin, GCS, PaO2/FiO2 |
| **Conservative DRG Extraction** | Only extracts DRGs if explicitly stated - no hallucination of plausible codes |
| **Markdown Bold Parsing** | `**text**` in LLM output renders as bold in DOCX |
| **Conflicts Appendix** | DOCX includes appendix listing any notes vs structured data conflicts |
| **Human-in-the-Loop** | All letters output as DOCX for CDI review before sending |
| **Production-Ready** | Supports Epic workqueue integration via KNOWN_ACCOUNT_ID |

## Clinical Notes (from Epic Clarity)

The system pulls **14 sepsis-relevant note types** for comprehensive clinical evidence:

| Code | Note Type | Purpose |
|------|-----------|---------|
| 1 | **Progress Notes** | Daily physician documentation |
| 2 | **Consults** | Specialist consultations (ID, Pulm, etc.) |
| 4 | **H&P** | History & Physical - admission assessment |
| 5 | **Discharge Summary** | Complete hospitalization summary |
| 6 | **ED Notes** | Emergency department notes |
| 7 | **Initial Assessments** | Early clinical picture |
| 8 | **ED Triage Notes** | Arrival vitals, chief complaint |
| 19 | **ED Provider Notes** | ED physician assessment |
| 29 | **Addendum Note** | Updates/corrections to notes |
| 32 | **Hospital Course** | Timeline narrative |
| 33 | **Subjective & Objective** | Clinical findings (S&O) |
| 38 | **Assessment & Plan Note** | Physician reasoning |
| 70 | **Nursing Note** | Vital signs, observations |
| 10000 | **Code Documentation** | Code events (if applicable) |

**Note Extraction**: Long notes (>8k chars) are automatically extracted via LLM to pull relevant clinical data with timestamps, reducing token usage while preserving key evidence.

## Repository Structure

```
SEPSIS/
├── data/
│   ├── featurization_train.py        # ONE-TIME: Knowledge base ingestion (gold letters + Propel)
│   └── featurization_inference.py    # PER-CASE: Data prep (denial + notes + structured data)
├── model/
│   └── inference.py                  # GENERATION: Vector search, write, assess, export
├── utils/
│   ├── gold_standard_appeals.gold_standard_appeals__sepsis_only/    # Current gold letters + default template
│   ├── gold_standard_appeals_sepsis_multiple/ # Future use
│   ├── sample_denial_letters/  # Denial letters to test with (PDFs)
│   ├── propel_data/            # Clinical criteria definitions (PDFs)
│   └── outputs/                # Generated appeal letters (DOCX)
├── docs/
│   └── rebuttal-engine-overview.html  # Executive overview
├── compare_denials.py        # Utility: check for duplicate denials
├── test_queries.sql          # Validation queries for Unity Catalog
└── README.md
```

## Unity Catalog Tables

| Table | Purpose |
|-------|---------|
| `dev.fin_ds.fudgesicle_gold_letters` | Past winning appeals with denial embeddings |
| `dev.fin_ds.fudgesicle_propel_data` | Official clinical criteria (sepsis definition) |

Note: Structured data (labs, vitals, meds) is queried directly from Clarity at inference time - no intermediate tables needed.

## Quick Start (Databricks)

### 1. Initial Setup

Copy files to Databricks notebooks and set the paths:
```python
GOLD_LETTERS_PATH = "/Workspace/Repos/your-user/fudgesicle/utils/gold_standard_appeals.gold_standard_appeals__sepsis_only"
PROPEL_DATA_PATH = "/Workspace/Repos/your-user/fudgesicle/utils/propel_data"
```

### 2. Ingest Gold Standard Letters (one-time)

In `featurization_train.py`:
```python
RUN_GOLD_INGESTION = True
```
Run the notebook. This extracts appeals and denials from gold letter PDFs, generates embeddings, and stores in `fudgesicle_gold_letters`.

### 3. Ingest Propel Definitions (one-time)

In `featurization_train.py`:
```python
RUN_PROPEL_INGESTION = True
```
Run the notebook. This reads PDF files from `propel_data/`, extracts key clinical criteria via LLM, and stores in `fudgesicle_propel_data`.

### 4. Process a Denial Letter

In `inference.py`, set the input:
```python
# Path to the denial PDF
DENIAL_PDF_PATH = "/path/to/denial_letter.pdf"

# Optional: If account ID is known (production mode)
KNOWN_ACCOUNT_ID = None  # or "12345678"
```

Run the notebook. For this denial:
- Parses PDF and extracts denial info (account ID, payor, DRGs) via `featurization_inference.py`
- Queries Clarity for clinical notes (14 note types) and structured data (labs, vitals, meds)
- Detects conflicts between physician notes and structured data
- Finds best matching gold letter via vector search
- Generates appeal using gold letter + notes + structured data
- Assesses appeal strength against Propel criteria
- Exports DOCX to `outputs/` folder (includes conflicts appendix if any)

## Configuration

### featurization_train.py (One-Time Setup)

| Flag | Default | Description |
|------|---------|-------------|
| `RUN_GOLD_INGESTION` | False | Process gold standard letter PDFs |
| `RUN_PROPEL_INGESTION` | False | Process Propel definition PDFs |

### inference.py (Per-Letter Processing)

| Setting | Default | Description |
|---------|---------|-------------|
| `DENIAL_PDF_PATH` | (required) | Path to denial letter PDF |
| `KNOWN_ACCOUNT_ID` | None | Account ID if known (production) |
| `SCOPE_FILTER` | "sepsis" | Which denial types to process |
| `MATCH_SCORE_THRESHOLD` | 0.7 | Minimum similarity to use gold letter |
| `NOTE_EXTRACTION_THRESHOLD` | 8000 | Char limit before LLM extraction |
| `EXPORT_TO_DOCX` | True | Export letters as Word documents |

### featurization_inference.py (Called by inference.py)

| Setting | Default | Description |
|---------|---------|-------------|
| `NOTE_EXTRACTION_THRESHOLD` | 8000 | Char limit before LLM extraction |
| `RUN_STRUCTURED_DATA` | True | Query and extract structured data |
| `RUN_CONFLICT_DETECTION` | True | Compare notes vs structured data |

## Technology Stack

| Component | Technology |
|-----------|------------|
| Platform | Databricks on Azure |
| LLM | Azure OpenAI GPT-4.1 |
| Embeddings | Azure OpenAI text-embedding-ada-002 (1536 dims) |
| Document OCR | Azure AI Document Intelligence |
| Storage | Delta Lake tables in Unity Catalog |
| Clinical Data | Epic Clarity |
| Runtime | Databricks Runtime 15.4 LTS ML |

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

**One-time setup (featurization_train.py):** <$1 for gold letter + Propel ingestion

**Infrastructure:** $0 incremental (uses existing Databricks/Azure)

## Extending to Other Conditions

The architecture supports any denial type. To add a new condition (e.g., pneumonia):

1. **Add clinical criteria**: Place `propel_pneumonia.pdf` in `utils/propel_data/`
2. **Add gold letters**: Add winning appeals to appropriate `gold_standard_appeals_*/` folder
3. **Update scope filter**: Modify `SCOPE_FILTER` logic in inference.py
4. **Run ingestion**: Re-run featurization_train.py with ingestion flags enabled

No architectural changes needed - the same pipeline handles any denial type.

## Team

**Financial Data Science** | Mercy Hospital

---

*Built with Azure OpenAI, Databricks, and Epic Clarity integration.*
