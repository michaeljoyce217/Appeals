# poc/run_poc.py
# Rebuttal Engine v2 POC - Quick Reference
#
# MLFlow Structure:
#   1. featurization.py - All data gathering (denial letters + clinical data)
#   2. inference.py     - All inference and output (agents + letter generation)
#
# This file is a quick reference. The actual work happens in the above files.

"""
=============================================================================
HOW TO RUN THE POC
=============================================================================

STEP 1: Configure Target Accounts
----------------------------------
Edit featurization.py Cell 3 and set TARGET_ACCOUNTS:

    TARGET_ACCOUNTS = [
        ("123456789", "denial_letter_1.pdf"),
        ("987654321", "denial_letter_2.docx"),
        # ... your 10 test cases
    ]


STEP 2: Run Featurization
-------------------------
Open featurization.py in Databricks and run cells 1-9.

This will:
- Read denial letters using Azure AI Document Intelligence
- Query clinical data from Clarity (notes, demographics, claims)
- Write to fudgesicle_inference table


STEP 3: Run Inference
---------------------
Open inference.py in Databricks and run cells 1-11.

This will:
- Read from fudgesicle_inference table
- Run Parser Agent (LLM extracts denial info - no regex!)
- Run Research Agent (stub until Propel docs loaded)
- Run Reference Agent (stub until gold letters loaded)
- Run Writer Agent (generates rebuttal letter)
- Write to fudgesicle_inference_score table


=============================================================================
FILE STRUCTURE
=============================================================================

poc/
├── featurization.py     # DATA GATHERING
│   ├── Azure Doc Intelligence for PDFs/DOCX
│   ├── Clarity queries for clinical notes
│   └── Output: fudgesicle_inference table
│
├── inference.py         # INFERENCE & OUTPUT
│   ├── Parser Agent (LLM-based extraction)
│   ├── Research Agent (stub)
│   ├── Reference Agent (stub)
│   ├── Writer Agent (letter generation)
│   └── Output: fudgesicle_inference_score table
│
├── config.py            # Shared configuration
│   ├── SCOPE_FILTER = "sepsis"
│   ├── SEPSIS_DRG_CODES
│   ├── SEPSIS_ICD10_CODES
│   └── EVIDENCE_PRIORITY
│
├── document_reader.py   # Azure Doc Intelligence wrapper
│
├── agents/              # Agent modules (imported by inference.py)
│   ├── parser_agent.py
│   ├── research_agent.py
│   ├── reference_agent.py
│   └── writer_agent.py
│
└── pipeline.py          # Orchestrator (alternative to inference.py)


=============================================================================
KEY DESIGN DECISIONS
=============================================================================

1. LLM for extraction (not regex)
   - Parser Agent uses GPT-4.1 to extract claim refs, member IDs, etc.
   - Handles variations like "Claim Ref #", "Reference Number", "Claim ID"

2. Evidence priority
   - Provider notes (Discharge Summary, H&P) = BEST
   - Structured data (labs, vitals) = backup
   - Inference (our conclusions) = LEAST important

3. Scope filter
   - SCOPE_FILTER = "sepsis" limits to sepsis denials
   - Change to "all" to process any denial type

4. Stubs for future
   - Research Agent returns empty until Propel docs loaded
   - Reference Agent falls back to template until gold letters loaded


=============================================================================
DEPENDENCIES
=============================================================================

Run once per cluster:
    %pip install azure-ai-documentintelligence azure-core openai python-docx


=============================================================================
DATABRICKS SECRETS REQUIRED
=============================================================================

Scope: idp_etl
Keys:
  - az-openai-key1
  - az-openai-base
  - az-doc-intelligence-endpoint
  - az-doc-intelligence-key

"""

# For local testing / development only
if __name__ == "__main__":
    print(__doc__)
