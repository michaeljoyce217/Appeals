# Sepsis DRG Appeal Letter Generator (Rebuttal Engine v2)

Automated generation of DRG appeal letters for sepsis-related insurance denials.

## Overview

When insurance companies deny or downgrade sepsis DRG claims (870/871/872), this system generates professional rebuttal letters by:

1. **Parsing denial letters** - Extracts payor, DRG codes, and denial reasons
2. **Vector search** - Finds the most similar past denial from our gold standard letters
3. **Learning from winners** - Uses the winning rebuttal as a template/guide
4. **Generating rebuttals** - Creates patient-specific appeal letters using clinical notes

## Files

```
data/featurization.py   - Ingests gold letters, prepares denial cases
model/inference.py      - Generates rebuttal letters
utils/
  gold_standard_rebuttals/  - Past winning appeal letters (PDFs)
  Sample_Denial_Letters/    - Test denial letters
```

## Usage (Databricks)

1. Copy `data/featurization.py` to a Databricks notebook
2. Update paths to your workspace
3. Run with `RUN_GOLD_INGESTION = True` to ingest gold letters
4. Copy `model/inference.py` to another notebook
5. Run to generate rebuttal letters (outputs DOCX files)

## Requirements

- Databricks Runtime 15.4 LTS ML
- Azure OpenAI (GPT-4.1, text-embedding-ada-002)
- Azure Document Intelligence
