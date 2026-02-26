# Summary of Changes — 2026-02-26

## Overview

Two major enhancements to the ARF (Acute Respiratory Failure) condition profile, plus one resolved SME feedback item.

---

## 1. Programmatic P/F Ratio Calculator

**Commit:** `1f148d9`
**Files:** `condition_profiles/respiratory_failure.py`, `data/featurization_inference.py`, `CLAUDE.md`

### What Changed

Added deterministic PaO2/FiO2 ratio calculation to the ARF condition profile, replacing the previous approach where the LLM was instructed to calculate P/F ratios from summarized data. The new calculator follows the same pattern as the existing SOFA scorer for sepsis.

### Key Discovery: Data Source

Through iterative querying of Mercy's Epic Clarity database, we discovered that:

- **FiO2 is NOT in the respiratory flowsheet** — not even for ventilated patients
- **FiO2 IS in the ABG lab panel** (component name: `FIO2`) alongside PaO2 (`PO2 ARTERIAL` / `PAO2 POC`)
- Since both values come from the same blood gas order, timestamps are inherently paired (within seconds)
- **PEEP IS in the ventilator flowsheet** (`FLO_MEAS_ID 1050046701` = VENTILATOR PEEP, `1050056801` = TOTAL PEEP)

### Technical Details

**New functions in `respiratory_failure.py`:**

| Function | Purpose |
|----------|---------|
| `_classify_berlin()` | Berlin ARDS classification (Severe/Moderate/Mild/Normal) |
| `_empty_pf_result()` | Zero-result dict for inference.py compatibility |
| `calculate_clinical_scores()` | Main calculator — pairs PaO2 with FiO2 (30-min window), pairs PEEP (2-hr window), computes ratio |
| `write_clinical_scores_table()` | Writes to `fudgesicle_case_pf_scores` Delta table |
| `format_scores_for_prompt()` | Formats P/F measurements as markdown table for LLM prompts |
| `render_scores_status_note()` | Internal review note when table is not rendered |
| `render_scores_in_docx()` | Appendix table with 6 columns: Timestamp, PaO2, FiO2, PEEP, P/F Ratio, Berlin Classification |

**Changes to `featurization_inference.py`:**
- Added PEEP FLO_MEAS_IDs (`1050046701`, `1050056801`) to the vitals query whitelist

**No changes to `inference.py`** — existing `hasattr(profile, ...)` guards pick up all new functions automatically.

### Design Decisions

- **Full encounter capture:** Unlike SOFA (single 24h window), P/F captures ALL ABG measurements across the encounter. The LLM writer uses presentation/admission values; the table provides full context for CDI reviewers.
- **30-minute PaO2/FiO2 pairing window:** Safety check since ABG panel values should be within seconds. Rejects spurious pairings from values hours apart.
- **2-hour PEEP pairing window:** Ventilator flowsheet entries have independent timing from ABG draws.
- **Render threshold:** Table appears in DOCX appendix when any P/F ratio < 300 (ARF-qualifying).
- **FiO2 normalization:** Handles both percentage format (40 -> 0.40) and fraction format (0.40 -> 0.40).

### Writer Prompt Updates

- `WRITER_SCORING_INSTRUCTIONS` updated: LLM now told to "cite those deterministic values directly" instead of calculating P/F ratio itself
- Added: "Do NOT recalculate P/F ratio from raw values"
- Added: "Do NOT include a P/F ratio table in the letter text" (table rendered separately)

---

## 2. Physical Manifestations of ARF

**Commit:** `3d3de01`
**File:** `condition_profiles/respiratory_failure.py`

### What Changed

Added 15 physical manifestations of acute respiratory failure to both the note extraction targets and the writer prompt. These are narrative clinical note findings (physician exams, nursing assessments), not structured flowsheet data — so the change is prompt-only with no SQL or pipeline modifications.

### Full Manifestation List

1. Use of accessory muscles to breathe
2. Tachypnea (respirations >30)
3. Cyanosis
4. Unable to speak in full sentences
5. Blue lips / pursed lips
6. Blue nail bed
7. Clubbing
8. Tripod position
9. Agonal respirations
10. Cheyne-Stokes respirations
11. Air hunger
12. Kussmaul's respiration
13. Hyperventilation
14. Hypoventilation
15. Apneic breathing

### Where Updated

- **`NOTE_EXTRACTION_TARGETS`** — New dedicated "Physical Manifestations" section added (fallback when Propel data unavailable)
- **`WRITER_SCORING_INSTRUCTIONS`** — SIGNS AND SYMPTOMS section expanded to enumerate all 15 manifestations

---

## 3. Resolved SME Feedback Item

**Deferred Item #3:** ARF respiratory rate threshold — resolved as **>20 breaths/min** per Propel/ACDIS criteria.

Added explicit `RESPIRATORY RATE THRESHOLD` line to `WRITER_SCORING_INSTRUCTIONS`.

### Two Distinct Thresholds Now in Effect

| Threshold | Purpose | Source |
|-----------|---------|--------|
| **>20 breaths/min** | ARF diagnostic criterion (tachypnea supporting ARF diagnosis) | Propel/ACDIS |
| **>30 breaths/min** | Physical manifestation of ARF (more severe, cited as clinical sign) | SME specification |

---

## Files Modified

| File | Lines Changed | Description |
|------|--------------|-------------|
| `condition_profiles/respiratory_failure.py` | +495/-4 | P/F calculator, physical manifestations, scoring instructions |
| `data/featurization_inference.py` | +4/-3 | PEEP FLO_MEAS_IDs in vitals query |
| `CLAUDE.md` | +1/-1 | Deferred item #3 resolved |
