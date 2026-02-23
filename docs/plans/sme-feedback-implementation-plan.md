# SME Feedback Implementation Plan

**Date:** 2026-02-23
**Source:** FEEDBACK/SME_FEEDBACK.rtfd (Mike Joyce + Diana, reviewed by CDI SMEs)
**Scope:** Absorb SME feedback into pipeline prompts, SOFA scoring, SOFA table rendering, and condition profiles

---

## Overview

14 actionable changes across 3 files, organized into 4 tasks. Each task is independent and can be implemented/verified in isolation.

**Files modified:**
- `model/inference.py` (writer prompt language rules)
- `condition_profiles/sepsis.py` (SOFA calculator, SOFA table, writer instructions)
- `condition_profiles/respiratory_failure.py` (writer instructions, conditional rebuttals)

**Deferred items (not in this plan):**
- Bundle compliance payor-awareness (#11 from brainstorm)
- Critical care charge verification (#12)
- RR threshold consensus — >20 vs >30 (#21)
- ARF case pre-screening (#22)
- Propel → "consensus-based guidelines" rename (TODO only)

---

## Task 1: Writer Prompt — Cross-Cutting Rules

**File:** `model/inference.py`
**Location:** Lines 359-400 (inside `WRITER_PROMPT` string, after `{scoring_instructions}` and before `# LANGUAGE RULES`)

### What to do

Insert 5 new rules between the existing instruction #6 (`Follow the Mercy Hospital template structure exactly`) and the `# LANGUAGE RULES` section. These apply to ALL conditions.

### Exact change

Find this block (lines 366-368):
```python
6. Follow the Mercy Hospital template structure exactly
{conditional_rebuttals_section}

# LANGUAGE RULES (MANDATORY)
```

Replace with:
```python
6. Follow the Mercy Hospital template structure exactly

# DATA SOURCE RULES (MANDATORY)
These rules govern which data sources to trust when writing the letter.

NOTE PRIORITY: When evidence appears in multiple note types, weight them in this order:
1. Discharge Summary  2. H&P  3. Progress Notes  4. Consult  5. ED Provider Note  6. Query  7. ED Notes  8. Nursing Note  9. All other note types (equal weight, lesser than above)

STRUCTURED DATA PRIMACY: For lab values and vital signs, ALWAYS use the values and timestamps from the Structured Data Summary (labs, vitals flowsheet) as the source of truth. Do NOT use lab/vital values or timestamps extracted from narrative physician notes when the same measurement exists in structured data. Physician notes may document approximate or recalled values — the lab system and vitals flowsheet are authoritative.

OMIT NORMAL VALUES: Do NOT cite lab values that fall within normal reference ranges as evidence of organ dysfunction. Normal values weaken the appeal. If a value is normal, omit it entirely — do not mention it. Examples: Lactic acid < 2.0 mmol/L, Platelet count 140-350 K/uL, Creatinine < 1.2 mg/dL.

USE WORST VALUES: Always cite the most severe/worst value within the appropriate clinical time window. Do not cite a lesser value when a more severe one exists in the data.

{conditional_rebuttals_section}

# LANGUAGE RULES (MANDATORY)
```

### Verification
- Read the modified `WRITER_PROMPT` string and confirm:
  - The 4 new rules appear between instruction #6 and `# LANGUAGE RULES`
  - `{conditional_rebuttals_section}` is preserved and moved after the new rules
  - No existing content was removed
  - The rules reference "Structured Data Summary" which matches the prompt section header on line 337

### TODO marker
Add this comment at the top of the `# SOURCE FIDELITY` section (line 395 area):
```python
# TODO: SME feedback — consider replacing "Propel" with "consensus-based guidelines" in letter text.
# Propel is hospital-facing; payors do not adhere to it. Deferred pending team decision.
```

---

## Task 2: Sepsis Profile — Writer Instructions + SOFA Table Rendering

**File:** `condition_profiles/sepsis.py`

### 2A: Add AKI/KDIGO terminology guard to WRITER_SCORING_INSTRUCTIONS

**Location:** Lines 141-145 (`WRITER_SCORING_INSTRUCTIONS`)

Find:
```python
WRITER_SCORING_INSTRUCTIONS = """5. SOFA SCORING:
   - If SOFA scores are provided above (total >= 2), reference them narratively when arguing organ dysfunction
   - Cite the individual organ scores and total score as clinical evidence of organ dysfunction severity
   - Do NOT include a SOFA table in the letter text — the table is rendered separately in the document
   - If no SOFA scores are provided, do not mention SOFA scoring"""
```

Replace with:
```python
WRITER_SCORING_INSTRUCTIONS = """5. SOFA SCORING:
   - If SOFA scores are provided above (total >= 2), reference them narratively when arguing organ dysfunction
   - Cite the individual organ scores and total score as clinical evidence of organ dysfunction severity
   - Do NOT include a SOFA table in the letter text — the table is rendered separately in the document
   - If no SOFA scores are provided, do not mention SOFA scoring

   RENAL TERMINOLOGY (CRITICAL): Do NOT use the term "acute kidney injury" or "AKI" unless the creatinine meets KDIGO criteria (creatinine >= 1.5x baseline). If the creatinine ratio is below 1.5x baseline, use "impaired renal function" instead. The creatinine value can still count toward the SOFA score even if it does not meet KDIGO AKI criteria. A physician may document "AKI" but the appeal letter must use the clinically supported terminology.

   POA-BASED TIMING: When citing the most severe clinical values:
   - If the sepsis diagnosis is POA "Y" (present on admission): use the most severe values within 24 hours of admission.
   - If the sepsis diagnosis is POA "N" (not present on admission): use the most severe values within 24 hours of the first sepsis documentation."""
```

### 2B: SOFA Table — Replace Timestamp column with Criteria column

**Location:** Lines 575-607 (`ORGAN_DISPLAY_NAMES` and `format_scores_for_prompt`)

Add a new constant after `ORGAN_DISPLAY_NAMES` (after line 582):

```python
# SOFA scoring criteria descriptions for each organ system (displayed in SOFA table)
SOFA_CRITERIA = {
    "respiratory": {
        0: "PaO2/FiO2 >= 400",
        1: "PaO2/FiO2 300-399",
        2: "PaO2/FiO2 200-299",
        3: "PaO2/FiO2 100-199",
        4: "PaO2/FiO2 < 100",
    },
    "coagulation": {
        0: "Platelets >= 150 K/uL",
        1: "Platelets 100-149 K/uL",
        2: "Platelets 50-99 K/uL",
        3: "Platelets 20-49 K/uL",
        4: "Platelets < 20 K/uL",
    },
    "liver": {
        0: "Bilirubin < 1.2 mg/dL",
        1: "Bilirubin 1.2-1.9 mg/dL",
        2: "Bilirubin 2.0-5.9 mg/dL",
        3: "Bilirubin 6.0-11.9 mg/dL",
        4: "Bilirubin >= 12.0 mg/dL",
    },
    "cardiovascular": {
        0: "MAP >= 70, no vasopressors",
        1: "MAP < 70",
        2: "Dopamine <= 5 or dobutamine",
        3: "Dopamine > 5 or epi/norepi <= 0.1",
        4: "Dopamine > 15 or epi/norepi > 0.1",
    },
    "cns": {
        0: "GCS = 15",
        1: "GCS 13-14",
        2: "GCS 10-12",
        3: "GCS 6-9",
        4: "GCS < 6",
    },
    "renal": {
        0: "Creatinine < 1.2 mg/dL",
        1: "Creatinine 1.2-1.9 mg/dL",
        2: "Creatinine 2.0-3.4 mg/dL",
        3: "Creatinine 3.5-4.9 mg/dL",
        4: "Creatinine >= 5.0 mg/dL",
    },
}
```

### 2C: Update `format_scores_for_prompt` — replace Timestamp with Criteria

**Location:** Lines 585-607

Find:
```python
def format_scores_for_prompt(scores_data):
    """Format SOFA scores as a markdown table for prompt inclusion."""
    if not scores_data or scores_data["total_score"] < 2:
        return "SOFA score < 2 or unavailable. Do not include a SOFA table in the letter."

    lines = []
    lines.append("SOFA SCORES (calculated from raw clinical data):")
    lines.append("")
    lines.append("| Organ System | Score | Value | Timestamp |")
    lines.append("|---|---|---|---|")
    for organ_key in ["respiratory", "coagulation", "liver", "cardiovascular", "cns", "renal"]:
        if organ_key in scores_data["organ_scores"]:
            data = scores_data["organ_scores"][organ_key]
            display = ORGAN_DISPLAY_NAMES.get(organ_key, organ_key)
            lines.append(f"| {display} | {data['score']} | {data['value']} | {data.get('timestamp', 'N/A')} |")
    lines.append(f"| **TOTAL** | **{scores_data['total_score']}** | {scores_data['organs_scored']} organs scored | |")
    lines.append("")

    if scores_data.get("vasopressor_detail"):
        drugs = set(v["drug"] for v in scores_data["vasopressor_detail"])
        lines.append(f"Vasopressors administered: {', '.join(sorted(drugs))}")

    return "\n".join(lines)
```

Replace with:
```python
def format_scores_for_prompt(scores_data):
    """Format SOFA scores as a markdown table for prompt inclusion."""
    if not scores_data or scores_data["total_score"] < 2:
        return "SOFA score < 2 or unavailable. Do not include a SOFA table in the letter."

    lines = []
    lines.append("SOFA SCORES (calculated from raw clinical data):")
    # Include the 24-hour window if available
    if scores_data.get("window_start") and scores_data.get("window_end"):
        lines.append(f"24-hour scoring window: {scores_data['window_start']} to {scores_data['window_end']}")
    lines.append("")
    lines.append("| Organ System | Score | Value | Criteria |")
    lines.append("|---|---|---|---|")
    for organ_key in ["respiratory", "coagulation", "liver", "cardiovascular", "cns", "renal"]:
        if organ_key in scores_data["organ_scores"]:
            data = scores_data["organ_scores"][organ_key]
            if data["score"] == 0:
                continue  # Omit zero-score rows
            display = ORGAN_DISPLAY_NAMES.get(organ_key, organ_key)
            criteria = SOFA_CRITERIA.get(organ_key, {}).get(data["score"], "")
            lines.append(f"| {display} | {data['score']} | {data['value']} | {criteria} |")
    lines.append(f"| **TOTAL** | **{scores_data['total_score']}** | {scores_data['organs_scored']} organs scored | |")
    lines.append("")

    if scores_data.get("vasopressor_detail"):
        drugs = set(v["drug"] for v in scores_data["vasopressor_detail"])
        lines.append(f"Vasopressors administered: {', '.join(sorted(drugs))}")

    return "\n".join(lines)
```

### 2D: Update `render_scores_in_docx` — Criteria column + omit zero-score rows

**Location:** Lines 625-671

Find:
```python
def render_scores_in_docx(doc, scores_data):
    """Render SOFA score table in DOCX (placed after letter body)."""
    from docx.shared import Pt

    if not scores_data or scores_data.get("total_score", 0) < 2:
        return

    doc.add_paragraph()
    sofa_header = doc.add_paragraph()
    sofa_header.add_run("Appendix: SOFA Score Summary").bold = True
    sofa_header.paragraph_format.space_after = Pt(4)

    organ_order = ["respiratory", "coagulation", "liver", "cardiovascular", "cns", "renal"]
    organs_with_data = [o for o in organ_order if o in scores_data["organ_scores"]]

    table = doc.add_table(rows=1 + len(organs_with_data) + 1, cols=4, style='Table Grid')
    # Header row
    hdr = table.rows[0]
    for i, text in enumerate(["Organ System", "Score", "Value", "Timestamp"]):
        hdr.cells[i].text = text
        for run in hdr.cells[i].paragraphs[0].runs:
            run.bold = True

    # Data rows
    row_idx = 1
    for organ_key in organ_order:
        if organ_key not in scores_data["organ_scores"]:
            continue
        data = scores_data["organ_scores"][organ_key]
        display = ORGAN_DISPLAY_NAMES.get(organ_key, organ_key)
        row = table.rows[row_idx]
        row.cells[0].text = display
        row.cells[1].text = str(data["score"])
        row.cells[2].text = str(data.get("value", ""))
        row.cells[3].text = str(data.get("timestamp", ""))[:19] if data.get("timestamp") else ""
        row_idx += 1

    # Total row
    total_row = table.rows[row_idx]
    total_row.cells[0].text = "TOTAL"
    total_row.cells[1].text = str(scores_data["total_score"])
    total_row.cells[2].text = f"{scores_data['organs_scored']} organs scored"
    total_row.cells[3].text = ""
    for cell in total_row.cells:
        for run in cell.paragraphs[0].runs:
            run.bold = True
```

Replace with:
```python
def render_scores_in_docx(doc, scores_data):
    """Render SOFA score table in DOCX (placed after letter body)."""
    from docx.shared import Pt

    if not scores_data or scores_data.get("total_score", 0) < 2:
        return

    doc.add_paragraph()
    sofa_header = doc.add_paragraph()
    sofa_header.add_run("Appendix: SOFA Score Summary").bold = True
    sofa_header.paragraph_format.space_after = Pt(4)

    # Show the 24-hour scoring window if available
    if scores_data.get("window_start") and scores_data.get("window_end"):
        window_note = doc.add_paragraph()
        window_note.add_run("Scoring Window: ").bold = True
        window_note.add_run(f"{scores_data['window_start']} to {scores_data['window_end']}")
        window_note.paragraph_format.space_after = Pt(4)

    organ_order = ["respiratory", "coagulation", "liver", "cardiovascular", "cns", "renal"]
    # Filter to non-zero scores only
    organs_with_data = [
        o for o in organ_order
        if o in scores_data["organ_scores"] and scores_data["organ_scores"][o]["score"] > 0
    ]

    table = doc.add_table(rows=1 + len(organs_with_data) + 1, cols=4, style='Table Grid')
    # Header row
    hdr = table.rows[0]
    for i, text in enumerate(["Organ System", "Score", "Value", "Criteria"]):
        hdr.cells[i].text = text
        for run in hdr.cells[i].paragraphs[0].runs:
            run.bold = True

    # Data rows (zero-score rows omitted)
    row_idx = 1
    for organ_key in organs_with_data:
        data = scores_data["organ_scores"][organ_key]
        display = ORGAN_DISPLAY_NAMES.get(organ_key, organ_key)
        criteria = SOFA_CRITERIA.get(organ_key, {}).get(data["score"], "")
        row = table.rows[row_idx]
        row.cells[0].text = display
        row.cells[1].text = str(data["score"])
        row.cells[2].text = str(data.get("value", ""))
        row.cells[3].text = criteria
        row_idx += 1

    # Total row
    total_row = table.rows[row_idx]
    total_row.cells[0].text = "TOTAL"
    total_row.cells[1].text = str(scores_data["total_score"])
    total_row.cells[2].text = f"{scores_data['organs_scored']} organs scored"
    total_row.cells[3].text = ""
    for cell in total_row.cells:
        for run in cell.paragraphs[0].runs:
            run.bold = True
```

### Verification
- `SOFA_CRITERIA` dict has entries for all 6 organ systems and all score levels 0-4
- `format_scores_for_prompt` outputs "Criteria" column header, skips zero-score rows, includes window info
- `render_scores_in_docx` outputs "Criteria" column header, skips zero-score rows, shows window dates
- `WRITER_SCORING_INSTRUCTIONS` includes AKI/KDIGO guard and POA timing
- Table row count matches `len(organs_with_data) + 2` (header + data + total)

---

## Task 3: Sepsis Profile — SOFA 24-Hour Sliding Window

**File:** `condition_profiles/sepsis.py`
**Location:** Lines 291-528 (`calculate_clinical_scores`)

This is the largest change. The current function takes the single worst value per organ across the entire encounter. The new version collects ALL timestamped values, then finds the 24-hour window that maximizes the total SOFA score.

### What to do

Rewrite `calculate_clinical_scores` to:

1. **Collect phase:** Gather ALL values with timestamps into per-organ lists (not just the worst)
2. **Window phase:** Generate candidate 24h windows anchored at each measurement timestamp, score all organs within each window, find the window with highest total SOFA
3. **Output phase:** Return the same output structure (organ_scores, total_score, organs_scored) plus new fields: `window_start`, `window_end`

### Exact replacement

Replace the entire `calculate_clinical_scores` function (lines 291-528) with:

```python
def calculate_clinical_scores(account_id, spark, tables):
    """
    Calculate SOFA scores from raw structured data tables using a 24-hour sliding window.

    Collects all timestamped lab/vital/med values, then finds the 24-hour window
    that produces the highest total SOFA score. This ensures all component values
    used in the final score fall within a clinically valid 24-hour period.

    Zero LLM calls — purely deterministic.

    Args:
        account_id: The hospital account ID
        spark: SparkSession instance
        tables: dict with keys "labs", "vitals", "meds" -> full table names

    Returns:
        dict with organ_scores, total_score, organs_scored, vasopressor_detail,
        window_start, window_end
    """
    from datetime import timedelta

    print("  Calculating SOFA scores (24h sliding window)...")

    labs_table = tables["labs"]
    vitals_table = tables["vitals"]
    meds_table = tables["meds"]

    # --- Gather all raw values ---
    try:
        labs_rows = spark.sql(f"""
            SELECT LAB_NAME, lab_value, EVENT_TIMESTAMP
            FROM {labs_table}
            WHERE lab_value IS NOT NULL AND EVENT_TIMESTAMP IS NOT NULL
            ORDER BY EVENT_TIMESTAMP ASC
        """).collect()
    except Exception as e:
        print(f"    Warning: Could not read labs: {e}")
        labs_rows = []

    try:
        vitals_rows = spark.sql(f"""
            SELECT VITAL_NAME, vital_value, EVENT_TIMESTAMP
            FROM {vitals_table}
            WHERE vital_value IS NOT NULL AND EVENT_TIMESTAMP IS NOT NULL
            ORDER BY EVENT_TIMESTAMP ASC
        """).collect()
    except Exception as e:
        print(f"    Warning: Could not read vitals: {e}")
        vitals_rows = []

    try:
        meds_rows = spark.sql(f"""
            SELECT MED_NAME, MED_DOSE, EVENT_TIMESTAMP
            FROM {meds_table}
            WHERE MED_NAME IS NOT NULL AND EVENT_TIMESTAMP IS NOT NULL
            ORDER BY EVENT_TIMESTAMP ASC
        """).collect()
    except Exception as e:
        print(f"    Warning: Could not read meds: {e}")
        meds_rows = []

    # ---- Collect all timestamped measurements per organ ----
    # Each entry: (value, timestamp, score, raw_display)
    organ_candidates = {
        "respiratory": [],
        "coagulation": [],
        "liver": [],
        "cardiovascular": [],
        "cns": [],
        "renal": [],
    }
    all_timestamps = []

    # PaO2 and FiO2 need pairing, so collect separately first
    pao2_values = []
    fio2_values = []

    for row in labs_rows:
        name = row["LAB_NAME"]
        val = safe_float(row["lab_value"])
        ts = row["EVENT_TIMESTAMP"]
        if val is None or ts is None:
            continue

        for category, matcher in LAB_VITAL_MATCHERS.items():
            if matcher["type"] != "lab":
                continue
            if not match_name(name, matcher):
                continue

            if category == "pao2":
                pao2_values.append((val, ts))
            elif category == "fio2":
                fio2_values.append((val, ts))
            elif category == "platelets":
                score = score_organ("coagulation", val)
                organ_candidates["coagulation"].append((val, ts, score, str(val)))
                all_timestamps.append(ts)
            elif category == "bilirubin":
                score = score_organ("liver", val)
                organ_candidates["liver"].append((val, ts, score, str(val)))
                all_timestamps.append(ts)
            elif category == "creatinine":
                score = score_organ("renal", val)
                organ_candidates["renal"].append((val, ts, score, str(val)))
                all_timestamps.append(ts)

    # Build respiratory candidates from PaO2/FiO2 pairs
    for pao2_val, pao2_ts in pao2_values:
        closest_fio2 = None
        closest_diff = None
        for fio2_val, fio2_ts in fio2_values:
            if fio2_val <= 0:
                continue
            actual_fio2 = fio2_val if fio2_val <= 1.0 else fio2_val / 100.0
            if actual_fio2 <= 0:
                continue
            diff = abs((pao2_ts - fio2_ts).total_seconds())
            if closest_diff is None or diff < closest_diff:
                closest_diff = diff
                closest_fio2 = actual_fio2
        if closest_fio2 and closest_fio2 > 0:
            ratio = pao2_val / closest_fio2
            score = score_organ("respiratory", ratio)
            organ_candidates["respiratory"].append(
                (ratio, pao2_ts, score, f"PaO2/FiO2 = {round(ratio, 1)}")
            )
            all_timestamps.append(pao2_ts)

    # Vitals: MAP and GCS
    for row in vitals_rows:
        name = row["VITAL_NAME"]
        val = safe_float(row["vital_value"])
        ts = row["EVENT_TIMESTAMP"]
        if val is None or ts is None:
            continue

        if match_name(name, LAB_VITAL_MATCHERS["map"]):
            score = score_organ("cardiovascular_map", val)
            organ_candidates["cardiovascular"].append(
                (val, ts, score, f"MAP = {val}")
            )
            all_timestamps.append(ts)

        if match_name(name, LAB_VITAL_MATCHERS["gcs"]):
            if val != val:  # NaN check
                continue
            int_val = int(round(val))
            score = score_organ("cns", int_val)
            organ_candidates["cns"].append(
                (int_val, ts, score, f"GCS = {int_val}")
            )
            all_timestamps.append(ts)

    # Meds: vasopressors
    vasopressor_entries = []
    for row in meds_rows:
        med_name = row["MED_NAME"]
        if not med_name:
            continue
        category = match_vasopressor(med_name)
        if category:
            dose = safe_float(row["MED_DOSE"])
            ts = row["EVENT_TIMESTAMP"]
            if ts is None:
                continue
            vasopressor_entries.append({
                "drug": category,
                "dose": dose,
                "timestamp": ts,
            })
            all_timestamps.append(ts)

    # ---- Edge case: no timestamps at all ----
    if not all_timestamps:
        print("    No timestamped values found — returning empty scores.")
        return {
            "organ_scores": {},
            "total_score": 0,
            "organs_scored": 0,
            "vasopressor_detail": [],
            "window_start": None,
            "window_end": None,
        }

    # ---- Sliding window: find the 24h window with highest total SOFA ----
    # Candidate windows are anchored at each measurement timestamp
    all_timestamps.sort()
    best_window = None
    best_total = -1
    best_organs = None
    best_vasopressor_detail = None

    for anchor_ts in all_timestamps:
        window_end = anchor_ts + timedelta(hours=24)

        # Score each organ using the WORST value within this window
        window_organs = {}

        for organ_key, candidates in organ_candidates.items():
            if organ_key == "cardiovascular":
                # Handle cardiovascular separately (MAP + vasopressor override)
                continue
            best_in_window = None
            for val, ts, score, display in candidates:
                if anchor_ts <= ts <= window_end:
                    if best_in_window is None or score > best_in_window[2]:
                        best_in_window = (val, ts, score, display)
            if best_in_window:
                window_organs[organ_key] = {
                    "score": best_in_window[2],
                    "value": best_in_window[3],
                    "timestamp": str(best_in_window[1]),
                }

        # Cardiovascular: MAP + vasopressor override within window
        map_candidates = organ_candidates["cardiovascular"]
        best_cv_score = 0
        best_cv_value = ""
        best_cv_ts = None

        for val, ts, score, display in map_candidates:
            if anchor_ts <= ts <= window_end:
                if score > best_cv_score:
                    best_cv_score = score
                    best_cv_value = display
                    best_cv_ts = ts

        window_vasopressor_detail = []
        window_vasopressors = {}
        for entry in vasopressor_entries:
            ts = entry["timestamp"]
            if anchor_ts <= ts <= window_end:
                window_vasopressor_detail.append({
                    "drug": entry["drug"],
                    "dose": entry["dose"],
                    "timestamp": str(ts),
                })
                drug = entry["drug"]
                dose = entry["dose"]
                if drug not in window_vasopressors:
                    window_vasopressors[drug] = (dose, ts)
                elif dose is not None:
                    existing_dose = window_vasopressors[drug][0]
                    if existing_dose is None or dose > existing_dose:
                        window_vasopressors[drug] = (dose, ts)

        for drug, (dose, ts) in window_vasopressors.items():
            cv_score = 2
            if drug == "dopamine" and dose is not None:
                if dose > 15:
                    cv_score = 4
                elif dose > 5:
                    cv_score = 3
                else:
                    cv_score = 2
            elif drug == "dobutamine":
                cv_score = 2
            elif drug in ("epinephrine", "norepinephrine") and dose is not None:
                if dose > 0.1:
                    cv_score = 4
                else:
                    cv_score = 3
            elif drug in ("epinephrine", "norepinephrine"):
                cv_score = 3

            if cv_score > best_cv_score:
                best_cv_score = cv_score
                dose_str = f" {dose}" if dose is not None else ""
                best_cv_value = f"{drug}{dose_str}"
                best_cv_ts = ts

        if best_cv_score > 0 or best_cv_ts is not None:
            window_organs["cardiovascular"] = {
                "score": best_cv_score,
                "value": best_cv_value,
                "timestamp": str(best_cv_ts) if best_cv_ts else None,
            }

        # Calculate total for this window
        window_total = sum(o["score"] for o in window_organs.values())

        if window_total > best_total:
            best_total = window_total
            best_organs = window_organs
            best_window = (anchor_ts, window_end)
            best_vasopressor_detail = window_vasopressor_detail

    # ---- Output ----
    organs_scored = len(best_organs) if best_organs else 0
    window_start_str = str(best_window[0])[:19] if best_window else None
    window_end_str = str(best_window[1])[:19] if best_window else None

    print(f"    SOFA Total: {best_total} ({organs_scored} organs scored)")
    if best_window:
        print(f"    24h window: {window_start_str} to {window_end_str}")
    if best_organs:
        for organ, data in best_organs.items():
            print(f"      {organ}: {data['score']} ({data['value']} at {data['timestamp']})")

    return {
        "organ_scores": best_organs or {},
        "total_score": best_total if best_total >= 0 else 0,
        "organs_scored": organs_scored,
        "vasopressor_detail": best_vasopressor_detail or [],
        "window_start": window_start_str,
        "window_end": window_end_str,
    }
```

### Verification
- Function signature and return type unchanged (backward compatible)
- New return fields `window_start` and `window_end` are additive (won't break existing callers)
- `write_clinical_scores_table` still works — it serializes `organ_scores` to JSON (structure unchanged)
- Vasopressor scoring logic is identical to original (dopamine/epi/norepi thresholds)
- Edge case: if no timestamps found, returns empty scores with `window_start=None`
- The sliding window iterates over ALL unique timestamps — O(n*m) where n=timestamps, m=organ candidates. For typical encounters (dozens of values) this is instant.

---

## Task 4: ARF Profile — Writer Instructions + Rebuttals

**File:** `condition_profiles/respiratory_failure.py`

### 4A: Expand WRITER_SCORING_INSTRUCTIONS

**Location:** Lines 217-222

Find:
```python
WRITER_SCORING_INSTRUCTIONS = """5. QUANTIFY ACUTE RESPIRATORY FAILURE using diagnostic criteria from the Propel guidelines when available:
   - Reference specific values: PaO2, SpO2, PaCO2, pH, FiO2, P/F ratio
   - Hypoxic criteria: PaO2 < 60 mmHg, SpO2 < 91% on room air, P/F ratio < 300
   - Hypercapnic criteria: PaCO2 > 50 mmHg with pH < 7.35
   - Acute-on-chronic indicators: >= 10 mmHg change from baseline PaO2 or PaCO2
   - Document oxygen delivery method and escalation of respiratory support"""
```

Replace with:
```python
WRITER_SCORING_INSTRUCTIONS = """5. QUANTIFY ACUTE RESPIRATORY FAILURE using diagnostic criteria from the Propel guidelines when available:
   - Reference specific values: PaO2, SpO2, PaCO2, pH, FiO2, P/F ratio
   - Hypoxic criteria: PaO2 < 60 mmHg, SpO2 < 91% on room air, P/F ratio < 300
   - Hypercapnic criteria: PaCO2 > 50 mmHg with pH < 7.35
   - Acute-on-chronic indicators: >= 10 mmHg change from baseline PaO2 or PaCO2
   - Document oxygen delivery method and escalation of respiratory support

   SPO2 THRESHOLD (CRITICAL): The hypoxic SpO2 criterion is STRICTLY LESS THAN 91%. An SpO2 of 91% is low but does NOT meet the hypoxic threshold. Do NOT cite SpO2 values of 91% or higher as evidence of hypoxemia. Only SpO2 values of 90% or below qualify.

   EVIDENCE TIMING (CRITICAL): Use clinical values from the presentation/admission timeframe ONLY. Values from 2 or more days after admission do not support an acute diagnosis at presentation. When the condition is POA, focus on the first 24 hours of admission.

   DOCUMENTATION CONSISTENCY: Do NOT use a clinical value as evidence of acute respiratory failure if the same documentation describes the patient as "stable" at that time. A value documented alongside "stable condition" cannot simultaneously support an acute diagnosis.

   P/F RATIO: If the P/F ratio is not documented by the provider, calculate it from available PaO2 and FiO2 values when both are available. P/F ratio < 300 supports ARF. Do NOT apply the P/F ratio criterion to acute-on-chronic respiratory failure patients.

   OXYGEN DELIVERY CONTEXT: When citing SpO2 readings, verify the oxygen delivery method. If the patient is on high-flow nasal cannula but FiO2 is set at 21% or less, this is essentially room air (~20% FiO2). Clarify whether the SpO2 was on room air or supplemental oxygen, as this affects the hypoxic criterion.

   SIGNS AND SYMPTOMS: Include ALL documented signs and symptoms of respiratory distress in the clinical presentation: dyspnea, difficulty breathing, increased work of breathing, use of accessory muscles, tachypnea, cyanosis, altered mental status related to respiratory compromise."""
```

### 4B: Remove Rebuttal C (Proprietary Clinical Criteria) from CONDITIONAL_REBUTTALS

**Location:** Lines 234-269

Find the entire `CONDITIONAL_REBUTTALS` list (lines 234-269) and replace with:

```python
CONDITIONAL_REBUTTALS = [
    {
        "name": "Consecutive/Sequential SpO2 Readings",
        "trigger": "Apply ONLY if the denial requires 'consecutive,' 'sequential,' or 'back-to-back' SpO2 readings.",
        "text": """Apply ONLY if the denial requires "consecutive," "sequential," or "back-to-back" SpO2 readings.

Rebuttal points:
1. No CMS, ICD-10-CM, or Coding Clinic requirement mandates multiple sequential SpO2 readings for acute hypoxemic respiratory failure.
2. Consecutive reading requirements are proprietary payor criteria -- not nationally recognized standards.
3. A documented SpO2 below 91% on room air with clinical context and treatment response supports the diagnosis.

Cite ALL relevant timestamped low SpO2 readings below 91%, even if not consecutive.""",
    },
    {
        "name": "Persistent/Continuous Symptoms Requirement",
        "trigger": "Apply ONLY if the denial argues symptoms were not 'persistent,' 'continuous,' or 'sustained.'",
        "text": """Apply ONLY if the denial argues symptoms were not "persistent," "continuous," or "sustained."

Rebuttal points:
1. Acute respiratory failure is defined by onset and severity at presentation -- not by unchanged symptoms throughout the stay.
2. Improvement with treatment confirms appropriate intervention, not absence of the condition.
3. Per CMS/Coding Clinic, a diagnosis is reportable when documented by the provider and supported by clinical indicators and treatment at presentation.

Cite documentation of respiratory distress at presentation and interventions required.""",
    },
]
```

**Changes from original:**
- Rebuttal A: Changed "A single documented SpO2 <91%" to "A documented SpO2 below 91%" (removed "single" absolutism). Changed final line to "below 91%" for consistency.
- Rebuttal B: Unchanged (SME praised this one).
- Rebuttal C (Proprietary Clinical Criteria): **Removed entirely** per SME feedback ("unfortunately they can have whatever criteria they want").

### Verification
- `CONDITIONAL_REBUTTALS` has exactly 2 entries (was 3)
- Rebuttal A no longer contains the word "single" before "documented SpO2"
- Rebuttal C (proprietary criteria) is completely gone
- `WRITER_SCORING_INSTRUCTIONS` now contains all 6 new sub-rules
- SpO2 threshold rule explicitly says "91% does NOT meet" the threshold

---

## Task 5: Note Deferred Items

**File:** `CLAUDE.md`
**Location:** End of file, before Technology Stack section (or after Verification Rules)

Add a new section:

```markdown
## Deferred SME Feedback (2026-02-23)

The following feedback items from SME review require external decisions or data before implementation:

| # | Item | Blocker | Status |
|---|------|---------|--------|
| 1 | CMS bundle compliance — don't use with non-CMS payors | Need payor classification data | Deferred |
| 2 | Critical care charges — verify CPT 99291 before citing | Need charge data access | Deferred |
| 3 | ARF respiratory rate threshold — >20 (Propel/ACDIS) vs >30 (Mercy Q-tip) | Needs internal consensus | Blocked |
| 4 | ARF case pre-screening — only appeal cases matching WON patterns | Needs outcome data + screening logic | Deferred |
| 5 | Propel terminology — replace "Propel criteria" with "consensus-based guidelines" in letter text | Pending team decision | TODO |
```

---

## Execution Order

Tasks 1-4 are independent and can be done in any order. Suggested sequence:

1. **Task 1** (writer prompt) — smallest, immediate impact on all conditions
2. **Task 4** (ARF rebuttals + instructions) — config/prompt only, no logic changes
3. **Task 2** (SOFA table rendering) — moderate, depends on Task 3's new output fields
4. **Task 3** (SOFA sliding window) — largest, most complex
5. **Task 5** (deferred items) — documentation only

**Note:** Tasks 2 and 3 should be done together since Task 2's rendering changes expect `window_start`/`window_end` and `SOFA_CRITERIA` which Task 3 produces. If doing Task 2 before Task 3, the new fields will simply be `None` (gracefully handled).

---

## Post-Implementation Verification

After all tasks complete, verify:
1. Read `model/inference.py` writer prompt — confirm 4 new DATA SOURCE RULES present
2. Read `condition_profiles/sepsis.py`:
   - `WRITER_SCORING_INSTRUCTIONS` has AKI/KDIGO + POA timing
   - `SOFA_CRITERIA` dict exists with all 6 organs
   - `format_scores_for_prompt` outputs "Criteria" column, skips zero scores, shows window
   - `render_scores_in_docx` outputs "Criteria" column, skips zero scores, shows window
   - `calculate_clinical_scores` uses sliding window, returns `window_start`/`window_end`
3. Read `condition_profiles/respiratory_failure.py`:
   - `WRITER_SCORING_INSTRUCTIONS` has 6 new sub-rules
   - `CONDITIONAL_REBUTTALS` has exactly 2 entries (C removed, A softened)
4. Read `CLAUDE.md` — deferred items section present
