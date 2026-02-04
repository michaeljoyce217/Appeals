# Appeal Strength Assessment Design

**Date:** 2026-01-25
**Status:** Approved for implementation

---

## Overview

Add a strength assessment feature that evaluates generated appeal letters against Propel sepsis criteria, argument structure, and evidence quality. The assessment appears as a separate section in the DOCX output before the letter body, providing CDI reviewers with a quick signal about letter quality.

---

## Output Format

The assessment appears after the header metadata and before "Dear...":

```
Generated: 2026-01-15 21:12
Patient: John Doe
Payor: United Healthcare
DRG: 871 -> 872
Gold Letter: WON-Sepsis-UHC-2024-001

═══════════════════════════════════════════════════════
APPEAL STRENGTH ASSESSMENT (Internal Review Only)
═══════════════════════════════════════════════════════

Overall Strength: 7/10 - MODERATE

Summary: Strong infection documentation with clear antibiotic
timeline. Organ dysfunction evidence present but could cite
specific lactate values from Day 2 notes.

Detailed Breakdown:
┌─────────────────────────────────────────────────────────┐
│ PROPEL CRITERIA COVERAGE (from: Propel definitions)     │
│ Score: 8/10                                             │
│ ✓ Infection source identified (pneumonia)               │
│ ✓ SIRS criteria documented (temp, WBC, tachycardia)     │
│ ✓ Organ dysfunction present (respiratory, renal)        │
│ △ Could strengthen: Specific SOFA score calculation     │
├─────────────────────────────────────────────────────────┤
│ ARGUMENT STRUCTURE (from: denial letter, gold template) │
│ Score: 7/10                                             │
│ ✓ Directly addresses denial reason                      │
│ ✓ Follows proven gold letter structure                  │
│ △ Could strengthen: Explicit rebuttal of payor's claim  │
├─────────────────────────────────────────────────────────┤
│ EVIDENCE QUALITY                                        │
│                                                         │
│ From Clinical Notes (from: Epic Clarity notes): 6/10    │
│ ✓ Timestamps present for key events                     │
│ ✓ Antibiotic timing documented                          │
│ ✗ Missing: Lactate 4.2 from 03/15 08:00 notes          │
│ ✗ Missing: MAP 63 requiring vasopressors               │
│                                                         │
│ From Structured Data: (not yet available)               │
└─────────────────────────────────────────────────────────┘
═══════════════════════════════════════════════════════

Dear Medical Director,
...
```

All three formats (simple score, summary, detailed breakdown) are included for POC feedback gathering.

---

## Pipeline Integration

The assessment runs as **Step 6.5** in inference.py:

```
Step 1: PDF Parse (Document Intelligence)
Step 2: Vector Search for best gold letter
Step 3: Extract denial info (LLM)
Step 4: Query Clarity for clinical notes
Step 5: Extract clinical data from long notes (LLM)
Step 6: Generate appeal letter (LLM)
Step 6.5: ──► ASSESS APPEAL STRENGTH (LLM) ◄── NEW
Step 7: Export to DOCX (now includes assessment section)
```

### Data Flow In

| Input | Source | Purpose |
|-------|--------|---------|
| `generated_letter` | Step 6 output | The letter being evaluated |
| `propel_definition` | `fudgesicle_propel_data` table | Official sepsis criteria to check against |
| `denial_text` | Step 1 output | What payor claimed, so we can check if it's addressed |
| `extracted_clinical_data` | Step 5 output | Full clinical evidence available |
| `gold_letter_used` | Step 2 output | Template structure for argument comparison |

### Data Flow Out

| Output | Type | Used By |
|--------|------|---------|
| `overall_score` | int (1-10) | Header display |
| `overall_rating` | string (LOW/MODERATE/HIGH) | Header display |
| `summary` | string | Brief rationale |
| `propel_breakdown` | dict | Detailed section |
| `argument_breakdown` | dict | Detailed section |
| `evidence_breakdown` | dict | Detailed section with missed evidence |

---

## Assessment Prompt Structure

```
You are evaluating the strength of a sepsis DRG appeal letter.

═══ PROPEL SEPSIS CRITERIA (source: official definitions) ═══
{propel_definition_summary}

═══ DENIAL LETTER (source: payor's denial) ═══
{denial_text}

═══ GOLD LETTER TEMPLATE USED (source: past winning appeal) ═══
{gold_letter_appeal_text}

═══ AVAILABLE CLINICAL EVIDENCE ═══

── From Clinical Notes (source: Epic Clarity notes) ──
{extracted_clinical_data}

── From Structured Data (source: Epic Clarity tables) ──
{structured_data or "Not yet available - structured data integration pending"}

═══ GENERATED APPEAL LETTER (being evaluated) ═══
{generated_letter}

═══ EVALUATION INSTRUCTIONS ═══
Evaluate this appeal letter and provide:

1. OVERALL SCORE (1-10) and RATING (LOW/MODERATE/HIGH)

2. SUMMARY (2-3 sentences explaining the score)

3. DETAILED BREAKDOWN with scores and specific findings:

   a) PROPEL CRITERIA COVERAGE - Does the letter document:
      - Suspected or confirmed infection
      - Organ dysfunction (per SOFA criteria)
      - Clinical response to treatment
      Note what's present (✓), what could be stronger (△), what's missing (✗)

   b) ARGUMENT STRUCTURE - Does the letter:
      - Directly address the payor's stated denial reason
      - Follow the logical structure of the gold letter template
      - Provide clear clinical reasoning

   c) EVIDENCE QUALITY - Split into two parts:

      From Clinical Notes:
      - Are specific values cited (not just "elevated lactate")?
      - Are timestamps present for key events?
      - List any relevant evidence in the notes NOT cited in the letter

      From Structured Data:
      - [Evaluate when available, otherwise note "pending"]
      - Labs, vitals, medications with timestamps
      - List any relevant structured evidence NOT cited in the letter

Return as JSON: {output_schema}
```

---

## LLM Output Schema

```json
{
  "overall_score": 7,
  "overall_rating": "MODERATE",
  "summary": "Strong infection documentation with clear antibiotic timeline. Organ dysfunction evidence present but could cite specific lactate values from Day 2 notes.",

  "propel_criteria": {
    "score": 8,
    "source": "Propel definitions",
    "findings": [
      {"status": "present", "item": "Infection source identified (pneumonia)"},
      {"status": "present", "item": "SIRS criteria documented (temp, WBC, tachycardia)"},
      {"status": "present", "item": "Organ dysfunction present (respiratory, renal)"},
      {"status": "could_strengthen", "item": "Specific SOFA score calculation"}
    ]
  },

  "argument_structure": {
    "score": 7,
    "source": "denial letter, gold template",
    "findings": [
      {"status": "present", "item": "Directly addresses denial reason"},
      {"status": "present", "item": "Follows proven gold letter structure"},
      {"status": "could_strengthen", "item": "Explicit rebuttal of payor's claim"}
    ]
  },

  "evidence_quality": {
    "clinical_notes": {
      "score": 6,
      "source": "Epic Clarity notes",
      "findings": [
        {"status": "present", "item": "Timestamps present for key events"},
        {"status": "present", "item": "Antibiotic timing documented"},
        {"status": "missing", "item": "Lactate 4.2 from 03/15 08:00 notes"},
        {"status": "missing", "item": "MAP 63 requiring vasopressors"}
      ]
    },
    "structured_data": {
      "score": null,
      "source": "Epic Clarity tables",
      "findings": [],
      "status": "pending_integration"
    }
  }
}
```

**Status values:**
- `present` → ✓ in output
- `could_strengthen` → △ in output
- `missing` → ✗ in output

**Rating thresholds:**
- 1-4: LOW
- 5-7: MODERATE
- 8-10: HIGH

---

## Error Handling

### LLM Call Failures

| Scenario | Handling |
|----------|----------|
| Assessment LLM call times out | Proceed with letter export, add "Assessment unavailable - LLM timeout" |
| LLM returns malformed JSON | Retry once; if still fails, show "Assessment unavailable - parse error" |
| LLM returns scores outside 1-10 | Clamp to valid range, log warning |

### Data Availability Edge Cases

| Scenario | Handling |
|----------|----------|
| No Propel definition found | Skip Propel section, note "Propel criteria not available for this condition" |
| No gold letter matched | Note "Evaluated against default template (no gold letter match)" |
| Clinical notes extraction failed | Show "No extracted notes available" with score N/A |
| Structured data not integrated | Show "pending_integration" status |

### Fallback Display

If assessment fails entirely:

```
═══════════════════════════════════════════════════════
APPEAL STRENGTH ASSESSMENT (Internal Review Only)
═══════════════════════════════════════════════════════

Assessment unavailable: {error_reason}

Please review letter manually before sending.
═══════════════════════════════════════════════════════
```

The assessment never blocks letter generation.

---

## Cost & Performance

**Assessment call tokens:**
- Input: ~12,000 tokens
- Output: ~800 tokens
- Cost: ~$0.03 per assessment

**Updated per-letter cost:** ~$0.27 (was ~$0.20, includes structured data + assessment)

**Latency:** Adds ~3-5 seconds (not a concern given workflow)

**ROI context:** One successful appeal brings $5-30k; assessment cost is negligible.

---

## Open Items & To-Do List

### Data to Add (pending Clarity integration)

| Item | Source | Status |
|------|--------|--------|
| Labs with timestamps | Epic Clarity | To do |
| Medications with administration times | Epic Clarity | To do |
| Vitals with measurement times | Epic Clarity | To do |
| Procedures with timestamps | Epic Clarity | To do |
| ICD-10 codes (with diagnosis timing if available) | Epic Clarity | To do |
| Inpatient medication orders | Epic Clarity | To do |

### Information Needed from Writing Team

| Question | Why It Matters | Status |
|----------|----------------|--------|
| Which of the ~75 note types are most valuable for sepsis appeals? | Currently pulling 47 - may be missing key evidence or pulling noise | Pending |
| Are there specific phrases/arguments that win appeals? | Could weight these higher in argument structure scoring | Pending |
| What makes a "weak" vs "strong" appeal in their experience? | Calibrate assessment scoring to match expert judgment | Pending |

### Decisions to Make

| Decision | Options | Status |
|----------|---------|--------|
| SOFA score calculation | LLM estimates vs. explicit calculation from structured data | Pending structured data |
| Assessment threshold for flagging | What score triggers "needs extra review"? | Gather feedback first |
| Outcome tracking | How to capture won/lost status for correlation analysis | Pending |

---

## Future Enhancements

| When | Enhancement |
|------|-------------|
| Structured data added | Enable `evidence_quality.structured_data` scoring |
| Enough letters processed | Correlate scores with actual appeal outcomes (won/lost) |
| Patterns emerge | Add payor-specific scoring adjustments |

---

## Appendix: Structured Data Extraction Specification

### Overview

Structured data (labs, vitals, meds, procedures, ICD-10) will be:
1. Queried from separate Clarity tables
2. Merged into a single chronological timeline
3. Extracted by LLM for sepsis-relevant data (runs parallel to note extraction)
4. Fed to both the writer LLM and the strength assessment

```
Step 5a: Extract from clinical notes (LLM)  ─┐
                                             ├─► Step 6: Generate appeal
Step 5b: Extract from structured data (LLM) ─┘
```

---

### 1. SOFA Score Components (Organ Dysfunction Quantification)

| System | Data Points | Critical Thresholds | SOFA Points |
|--------|-------------|---------------------|-------------|
| **Respiratory** | PaO2, FiO2, PaO2/FiO2 ratio, SpO2/FiO2 | P/F ≥400 (0), <400 (1), <300 (2), <200 w/vent (3), <100 w/vent (4) | 0-4 |
| **Coagulation** | Platelet count | ≥150k (0), <150k (1), <100k (2), <50k (3), <20k (4) | 0-4 |
| **Liver** | Total bilirubin | <1.2 (0), 1.2-1.9 (1), 2.0-5.9 (2), 6.0-11.9 (3), >12.0 (4) mg/dL | 0-4 |
| **Cardiovascular** | MAP, vasopressor type/dose/duration | MAP ≥70 (0), <70 (1), dopa <5 (2), dopa 5-15 or norepi ≤0.1 (3), dopa >15 or norepi >0.1 (4) µg/kg/min | 0-4 |
| **CNS** | GCS | 15 (0), 13-14 (1), 10-12 (2), 6-9 (3), <6 (4) | 0-4 |
| **Renal** | Creatinine, urine output | Cr <1.2 (0), 1.2-1.9 (1), 2.0-3.4 (2), 3.5-4.9 (3), >5.0 or UO <200mL/day (4) mg/dL | 0-4 |

**Key threshold:** SOFA ≥2 = organ dysfunction = sepsis diagnosis per Sepsis-3

---

### 2. Lactate (Critical - All Values with Timestamps)

| Threshold | Clinical Significance |
|-----------|----------------------|
| >1.0 mmol/L | Tissue hypoperfusion marker |
| **>2.0 mmol/L** | Sepsis indicator, requires repeat measurement |
| **>4.0 mmol/L** | Severe - triggers aggressive fluid resuscitation |
| **Clearance ≥10-20%** per 2 hours | Target for successful resuscitation |
| Trend normalization | Predicts survival; failure to clear = reassess treatment |

**Extract:** Initial value, all subsequent values, timestamps, calculate clearance %

---

### 3. Infection Markers

| Marker | Normal | Sepsis Threshold | Notes |
|--------|--------|------------------|-------|
| **Procalcitonin (PCT)** | <0.05 ng/mL | **>0.5 ng/mL** (sens 75%); **>10 ng/mL** highly specific | More specific than CRP for bacterial infection |
| **CRP** | <3 mg/L | **>8 mg/L** | Less specific, elevated in any inflammation |
| **WBC** | 4.5-11k/µL | **<4k or >12k** | SIRS criterion |
| **Bands** | <5% | **>10%** | Left shift = active infection |
| **Blood cultures** | Negative | Positive | Confirms infection source, identifies organism |
| **Urine cultures** | Negative | Positive | Source identification |

---

### 4. Vital Signs (qSOFA + SIRS Criteria)

| Vital | SIRS Criterion | qSOFA Criterion | Septic Shock |
|-------|---------------|-----------------|--------------|
| **Temperature** | <36°C or >38°C | - | - |
| **Heart rate** | >90 bpm | - | - |
| **Respiratory rate** | >20/min | **≥22/min** | - |
| **Systolic BP** | - | **≤100 mmHg** | - |
| **MAP** | - | - | **<65 mmHg** requiring vasopressors |
| **Mental status** | - | GCS <15 | - |

**qSOFA ≥2** = high mortality risk
**SIRS ≥2 criteria** = systemic inflammatory response

---

### 5. SEP-1 Bundle Compliance (Timing Critical for Appeals)

#### 3-Hour Bundle (from severe sepsis recognition)

| Intervention | Requirement | Data to Extract |
|--------------|-------------|-----------------|
| **Lactate** | Measure within 3 hours | Lab timestamp, value |
| **Blood cultures** | Draw before antibiotics | Collection timestamp |
| **Broad-spectrum antibiotics** | Within 3 hours (1 hour for shock) | Med name, admin timestamp |
| **Crystalloid fluids** | 30 mL/kg if hypotensive OR lactate ≥4 | Fluid type, volume, start time |

#### 6-Hour Bundle

| Intervention | Requirement | Data to Extract |
|--------------|-------------|-----------------|
| **Repeat lactate** | If initial ≥2 mmol/L | Lab timestamp, value, clearance % |
| **Vasopressors** | If MAP <65 after fluids | Drug, dose, start time |
| **Volume status reassessment** | If persistent hypotension or lactate ≥4 | Documentation timestamp |

---

### 6. Vasopressor Details (For Septic Shock)

| Drug | Starting Dose | Add Vasopressin Threshold | High Dose (Refractory) |
|------|---------------|---------------------------|------------------------|
| **Norepinephrine** | 0.01-0.05 µg/kg/min | 0.25-0.5 µg/kg/min | **≥1 µg/kg/min** |
| **Vasopressin** | 0.01-0.03 U/min (fixed) | Added as adjunct | 0.04 U/min max |
| **Dopamine** | 5 µg/kg/min | - | >15 µg/kg/min |
| **Epinephrine** | 0.01 µg/kg/min | Rescue agent | >0.1 µg/kg/min |
| **Phenylephrine** | 0.5-2 µg/kg/min | Alternative | - |
| **Dobutamine** | 2.5 µg/kg/min | Cardiac dysfunction | 20 µg/kg/min |

**Extract:** Drug name, dose (µg/kg/min), start time, duration, dose changes with timestamps

---

### 7. Fluid Resuscitation

| Parameter | Target/Threshold |
|-----------|------------------|
| **Initial bolus** | 30 mL/kg crystalloid within 3 hours |
| **Fluid type** | Balanced crystalloids preferred (LR) over NS |
| **Total volume** | Track cumulative for fluid balance |
| **Response assessment** | Dynamic measures (SVV, passive leg raise) |

---

### 8. Renal Function / Urine Output (AKI Staging)

| Parameter | Stage 1 | Stage 2 | Stage 3 |
|-----------|---------|---------|---------|
| **Urine output** | <0.5 mL/kg/h for 6-12h | <0.5 mL/kg/h for ≥12h | <0.3 mL/kg/h for ≥24h OR anuria ≥12h |
| **Creatinine** | 1.5-1.9x baseline | 2.0-2.9x baseline | 3.0x baseline OR ≥4.0 mg/dL OR dialysis |

**Key:** UO <0.5 mL/kg/h for 3-5 consecutive hours predicts AKI progression

---

### 9. ICD-10 Codes to Flag

| Code | Description | DRG Impact |
|------|-------------|------------|
| **A41.9** | Sepsis, unspecified | Base sepsis |
| **A41.01/.02** | Sepsis due to MSSA/MRSA | Organism-specific |
| **R65.20** | Severe sepsis without shock | Higher DRG |
| **R65.21** | Severe sepsis with septic shock | Highest DRG (870) |
| **A40.x** | Streptococcal sepsis | - |
| **Infection codes** | Source identification | Supporting |

---

### 10. Additional Labs

| Lab | Why It Matters |
|-----|----------------|
| BUN | Renal function, hydration |
| Sodium, Potassium | Electrolyte derangement |
| Glucose | Stress hyperglycemia |
| PT/INR, PTT, Fibrinogen, D-dimer | Coagulopathy / DIC |
| Troponin, BNP | Cardiac dysfunction |
| Albumin | Nutritional status |
| AST, ALT, Ammonia | Hepatic dysfunction |

---

### Extraction Prompt Strategy

The structured data extractor LLM should:

1. **Organize chronologically** by timestamp
2. **Flag SOFA-relevant data** with calculated points
3. **Highlight bundle compliance** - 3-hour and 6-hour targets met?
4. **Calculate lactate clearance** if multiple values
5. **Note trends** - improving vs. worsening
6. **Flag missing data** - what wasn't documented?

---

### Sources

- [Sepsis-3 Definition (JAMA/PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4968574/)
- [SEP-1 Bundle Compliance](https://pmc.ncbi.nlm.nih.gov/articles/PMC5396984/)
- [Lactate Clearance Targets](https://pmc.ncbi.nlm.nih.gov/articles/PMC5496745/)
- [Procalcitonin/CRP Thresholds](https://pmc.ncbi.nlm.nih.gov/articles/PMC5564169/)
- [Vasopressor Dosing](https://pmc.ncbi.nlm.nih.gov/articles/PMC7333107/)
- [Antibiotic Timing](https://pmc.ncbi.nlm.nih.gov/articles/PMC5649973/)
- [Fluid Resuscitation](https://pmc.ncbi.nlm.nih.gov/articles/PMC7963440/)
- [Urine Output/AKI](https://pmc.ncbi.nlm.nih.gov/articles/PMC10228087/)
- [SEP-1 Overview (Cytovale)](https://cytovale.com/what-is-sep-1-essential-insights-for-improving-sepsis-care-outcomes/)
