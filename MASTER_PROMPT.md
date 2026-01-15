# Sepsis Appeal Engine - Master Prompt

**Last Updated:** 2026-01-14
**Repo:** https://github.com/michaeljoyce217/SEPSIS

---

## Project Overview

**Goal:** Automated generation of DRG appeal letters for sepsis-related insurance denials (DRG 870/871/872).

**Architecture:** Multi-agent AI pipeline using Azure OpenAI GPT-4.1

**Platform:** Databricks on Azure with Unity Catalog

---

## Repository Structure

```
SEPSIS/
├── data/
│   └── featurization.py        # Data ingestion and preparation (Parser Agent)
├── model/
│   └── inference.py            # Letter generation (Writer Agent + vector search)
├── utils/
│   ├── gold_standard_appeals/  # Past winning appeal letters (PDFs) + default template
│   ├── sample_denial_letters/  # New denial letters to process (PDFs)
│   ├── propel_data/            # Clinical criteria definitions (PDFs)
│   └── outputs/                # Generated appeal letters (timestamped folders)
├── docs/
│   └── rebuttal-engine-overview.html  # Executive overview with Technical Architecture
├── compare_denials.py          # Utility: check for duplicate denials
├── test_queries.sql            # Validation queries for Unity Catalog
├── README.md                   # Project documentation
└── MASTER_PROMPT.md            # This file
```

---

## Multi-Agent Pipeline

### Stage 1: Featurization (data/featurization.py)

| Agent | Technology | Function |
|-------|------------|----------|
| Document Intel | Azure AI Document Intelligence | OCR extraction from PDF |
| Parser | GPT-4.1 | Extract: account_id, payor, DRGs, is_sepsis |
| Embedding | text-embedding-ada-002 | Generate 1536-dim denial vector |
| Clarity Join | Spark SQL | Attach 14 clinical note types |

### Stage 2: Inference (model/inference.py)

| Agent | Technology | Function |
|-------|------------|----------|
| Retrieval | Cosine Similarity | Find best-matching gold letter |
| Extraction (x14) | GPT-4.1 (parallel) | Extract clinical data with timestamps |
| Writer | GPT-4.1 | Generate appeal letter |

---

## Unity Catalog Tables

| Table | Purpose |
|-------|---------|
| `dev.fin_ds.fudgesicle_gold_letters` | Past winning appeals with denial embeddings |
| `dev.fin_ds.fudgesicle_propel_data` | Official clinical criteria (definition_summary for prompts) |
| `dev.fin_ds.fudgesicle_inference` | Denial cases with clinical notes |
| `dev.fin_ds.fudgesicle_inference_score` | Generated appeal letters |

---

## Key Features

### 14 Clinical Note Types (from Epic Clarity)
Progress Notes, Consults, H&P, Discharge Summary, ED Notes, Initial Assessments, ED Triage Notes, ED Provider Notes, Addendum Note, Hospital Course, Subjective & Objective, Assessment & Plan Note, Nursing Note, Code Documentation

### Smart Note Extraction
Notes >8,000 chars are automatically extracted via LLM to pull relevant clinical data WITH timestamps (e.g., "03/15/2024 08:00: Lactate 4.2, MAP 63").

### Propel Definition Summary
Full Propel PDFs are processed at ingestion time - LLM extracts key clinical criteria into `definition_summary` field for efficient prompt inclusion.

### Default Template Fallback
When vector search doesn't find a good match (score < 0.7), the system falls back to `default_sepsis_appeal_template.docx` as a structural guide.

### Validation Checkpoints
Both featurization.py and inference.py include checkpoint functions that validate:
- Path existence and file counts
- Table structure and row counts
- Generation results

### Timestamped Output Folders
Each inference run creates a new folder: `utils/outputs/output_YYYY-MM-DD_HHMMSS/`

---

## Configuration

### featurization.py Flags
| Flag | Default | Description |
|------|---------|-------------|
| `RUN_GOLD_INGESTION` | False | Process gold standard letter PDFs |
| `RUN_PROPEL_INGESTION` | False | Process Propel definition PDFs |
| `RUN_DENIAL_PROCESSING` | False | Process new denial letter PDFs |
| `RUN_DENIAL_FEATURIZATION` | False | Join with Clarity and write to table |
| `WRITE_TO_TABLE` | False | Write results to Unity Catalog |

### inference.py Settings
| Setting | Default | Description |
|---------|---------|-------------|
| `SCOPE_FILTER` | "sepsis" | Which denial types to process |
| `MATCH_SCORE_THRESHOLD` | 0.7 | Minimum similarity for gold letter match |
| `NOTE_EXTRACTION_THRESHOLD` | 8000 | Char limit before LLM extraction |
| `WRITE_TO_TABLE` | False | Persist to score table |
| `EXPORT_TO_DOCX` | True | Export as Word documents |

---

### POC Limitation
In the POC, the LLM extracts account ID and sepsis relevance from denial letter text alone. Some generic denials lack DRG codes or diagnosis information and cannot be classified. In production, Epic workqueue integration will provide this context directly, enabling 100% coverage.

---

## Quick Start (Databricks)

1. **Install dependencies** (run Cell 1 alone, then restart):
   ```python
   %pip install azure-ai-documentintelligence==1.0.2 openai python-docx
   dbutils.library.restartPython()
   ```

2. **Run featurization.py** with appropriate flags enabled

3. **Run inference.py** to generate appeal letters

4. **Review DOCX outputs** in `utils/outputs/output_<timestamp>/`

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
