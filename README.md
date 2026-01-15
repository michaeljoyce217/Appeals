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
│                         KNOWLEDGE LAYER (One-Time Setup)                    │
│  ┌──────────────┐    ┌──────────────┐                                      │
│  │    Propel    │    │    Gold      │                                      │
│  │   Criteria   │    │   Letters    │                                      │
│  │  (propel_    │    │  (gold_      │                                      │
│  │   data)      │    │   letters)   │                                      │
│  └──────────────┘    └──────────────┘                                      │
│         ▲                   ▲                                               │
│         └─────────┬─────────┘                                               │
│                   │                                                         │
│          featurization.py (run once)                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                    PER-LETTER PROCESSING (inference.py)                     │
│                                                                             │
│  [Denial PDF] ──► inference.py ──► [DOCX Appeal Letter]                    │
│                                                                             │
│  Step 1: Parse PDF (Document Intelligence)                                  │
│  Step 2: Vector search for best gold letter (uses denial text only)        │
│  Step 3: Extract denial info (LLM: account ID, payor, DRGs)                │
│  Step 4: Query Clarity for this account's clinical notes                   │
│  Step 5: Extract clinical data from long notes (LLM)                       │
│  Step 6: Generate appeal letter (LLM)                                      │
│  Step 7: Export to DOCX                                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

[DOCX Letter] ──► CDI Review ──► Approved Letter ──► Send to Payor
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Single-Letter Processing** | One denial at a time - no batch processing, no memory issues |
| **Vector Search** | Embeddings-based similarity matching finds the most relevant past denial |
| **Gold Letter Learning** | Uses winning appeals as templates - proven arguments get reused |
| **Default Template Fallback** | When no good match found, uses default template as structural guide |
| **Propel Integration** | LLM extracts key criteria from Propel PDFs into concise summaries |
| **Comprehensive Clinical Notes** | Pulls 14 note types from Clarity (see below) |
| **Smart Note Extraction** | Long notes (>8k chars) auto-extracted with timestamps via LLM |
| **Simple Output** | Appeals saved to `outputs/` as `{account_id}_{patient}_appeal.docx` |
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
│   └── featurization.py      # ONE-TIME: Knowledge base ingestion
├── model/
│   └── inference.py          # MAIN: Single-letter processing
├── utils/
│   ├── gold_standard_appeals/  # Past winning appeal letters (PDFs) + default template
│   ├── sample_denial_letters/  # Denial letters to test with (PDFs)
│   ├── propel_data/            # Clinical criteria definitions (PDFs)
│   └── outputs/                # Generated appeal letters (DOCX)
├── docs/
│   └── appeal-assistant-guide.html  # Executive overview
├── compare_denials.py        # Utility: check for duplicate denials
├── test_queries.sql          # Validation queries for Unity Catalog
└── README.md
```

## Unity Catalog Tables

| Table | Purpose |
|-------|---------|
| `dev.fin_ds.fudgesicle_gold_letters` | Past winning appeals with denial embeddings |
| `dev.fin_ds.fudgesicle_propel_data` | Official clinical criteria (sepsis definition) |

Note: No intermediate inference tables needed - single-letter processing queries Clarity directly.

## Quick Start (Databricks)

### 1. Initial Setup

Copy files to Databricks notebooks and set the paths:
```python
GOLD_LETTERS_PATH = "/Workspace/Repos/your-user/fudgesicle/utils/gold_standard_appeals"
PROPEL_DATA_PATH = "/Workspace/Repos/your-user/fudgesicle/utils/propel_data"
```

### 2. Ingest Gold Standard Letters (one-time)

In `featurization.py`:
```python
RUN_GOLD_INGESTION = True
```
Run the notebook. This extracts appeals and denials from gold letter PDFs, generates embeddings, and stores in `fudgesicle_gold_letters`.

### 3. Ingest Propel Definitions (one-time)

In `featurization.py`:
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
- Parses PDF and finds best matching gold letter via vector search
- Extracts denial info (account ID, payor, DRGs) via LLM
- Queries Clarity for clinical notes (14 note types)
- Includes Propel sepsis definition
- Generates appeal using clinical notes
- Exports DOCX to `outputs/` folder

## Configuration

### featurization.py (One-Time Setup)

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

### Per Appeal Letter (~$0.20)

| Step | Input Tokens | Output Tokens | Cost |
|------|-------------|---------------|------|
| Denial info extraction | ~4,000 | ~100 | $0.01 |
| Note extraction (4 calls avg) | ~12,000 | ~3,200 | $0.05 |
| Appeal letter generation | ~50,000 | ~3,000 | $0.14 |
| **Total** | ~66,000 | ~6,300 | **~$0.20** |

### Monthly Projections

| Volume | LLM Cost |
|--------|----------|
| 100 appeals/month | ~$20 |
| 500 appeals/month | ~$100 |
| 1,000 appeals/month | ~$200 |

**One-time setup (featurization.py):** <$1 for gold letter + Propel ingestion

**Infrastructure:** $0 incremental (uses existing Databricks/Azure)

## Extending to Other Conditions

The architecture supports any denial type. To add a new condition (e.g., pneumonia):

1. **Add clinical criteria**: Place `propel_pneumonia.pdf` in `utils/propel_data/`
2. **Add gold letters**: Add winning pneumonia appeals to `gold_standard_appeals/`
3. **Update scope filter**: Modify `SCOPE_FILTER` logic in inference.py
4. **Run ingestion**: Re-run featurization.py with ingestion flags enabled

No architectural changes needed - the same pipeline handles any denial type.

## Team

**Financial Data Science** | Mercy Hospital

---

*Built with Azure OpenAI, Databricks, and Epic Clarity integration.*
