# condition_profiles/sepsis.py
# Sepsis Condition Profile for the DRG Appeal Engine
#
# Contains all sepsis-specific configuration, prompts, scoring logic,
# and DOCX rendering. The base engine imports this module via:
#   profile = importlib.import_module(f"condition_profiles.{CONDITION_PROFILE}")

import json
import sys
from datetime import datetime

# =============================================================================
# Profile Validation
# =============================================================================
REQUIRED_ATTRIBUTES = [
    "CONDITION_NAME", "CONDITION_DISPLAY_NAME",
    "GOLD_LETTERS_PATH", "PROPEL_DATA_PATH", "DEFAULT_TEMPLATE_PATH",
    "CLINICAL_SCORES_TABLE_NAME",
    "DENIAL_CONDITION_QUESTION", "DENIAL_CONDITION_FIELD",
    "NOTE_EXTRACTION_TARGETS", "STRUCTURED_DATA_CONTEXT", "CONFLICT_EXAMPLES",
    "WRITER_SCORING_INSTRUCTIONS",
    "ASSESSMENT_CONDITION_LABEL", "ASSESSMENT_CRITERIA_LABEL",
]


def validate_profile(profile):
    """Check that a condition profile module has all required attributes."""
    missing = [attr for attr in REQUIRED_ATTRIBUTES if not hasattr(profile, attr)]
    if missing:
        name = getattr(profile, '__name__', str(profile))
        raise AttributeError(
            f"Condition profile '{name}' is missing required attributes:\n"
            + "\n".join(f"  - {attr}" for attr in missing)
            + "\n\nSee an existing profile (e.g., sepsis.py) for the full interface."
        )

# =============================================================================
# Identity & Paths
# =============================================================================
CONDITION_NAME = "sepsis"
CONDITION_DISPLAY_NAME = "Sepsis"
DRG_CODES = ["870", "871", "872"]

GOLD_LETTERS_PATH = "/Workspace/Repos/mijo8881@mercy.net/fudgesicle/utils/gold_standard_appeals.gold_standard_appeals__sepsis_only"
PROPEL_DATA_PATH = "/Workspace/Repos/mijo8881@mercy.net/fudgesicle/utils/propel_data"
DEFAULT_TEMPLATE_PATH = "/Workspace/Repos/mijo8881@mercy.net/fudgesicle/utils/gold_standard_appeals_sepsis_only/default_sepsis_appeal_template.docx"

# Clinical scores table suffix (base engine prepends catalog + schema)
CLINICAL_SCORES_TABLE_NAME = "fudgesicle_case_sofa_scores"

# =============================================================================
# Denial Parser
# =============================================================================
DENIAL_CONDITION_QUESTION = "5. IS SEPSIS RELATED - does this denial involve sepsis, severe sepsis, or septic shock?"
DENIAL_CONDITION_FIELD = "IS_SEPSIS"

# =============================================================================
# Note Extraction Targets (fallback when Propel definition_summary unavailable)
# =============================================================================
NOTE_EXTRACTION_TARGETS = """## SOFA Score Components (PRIORITY - extract ALL available)
- Respiration: PaO2/FiO2 ratio, SpO2/FiO2, oxygen requirements, ventilator settings
- Coagulation: Platelet count
- Liver: Bilirubin (total)
- Cardiovascular: MAP, hypotension, vasopressor use (drug, dose)
- CNS: GCS (Glasgow Coma Scale), mental status changes
- Renal: Creatinine, urine output

## Other Sepsis Markers
- Lactate levels (CRITICAL - include all values with times)
- WBC count, bands
- Temperature (fever, hypothermia)
- Heart rate, respiratory rate
- Blood culture results, infection source
- Antibiotic administration times

## Clinical Events
- Fluid resuscitation (volume, timing)
- ICU admission/transfer
- Physician assessments mentioning sepsis, SIRS, infection"""

# =============================================================================
# Structured Data Context (for structured data extraction prompt)
# =============================================================================
DIAGNOSIS_EXAMPLES = """For example:
- "Severe sepsis with septic shock due to Methicillin-susceptible Staphylococcus aureus"
- "Sepsis due to Escherichia coli"
"""

STRUCTURED_DATA_CONTEXT = """Extract a focused summary of sepsis-relevant clinical data from this timeline. Prioritize:

1. **SOFA Score Components** (with timestamps and trends):
   - Respiratory: PaO2/FiO2, SpO2, oxygen requirements
   - Coagulation: Platelet count
   - Liver: Bilirubin
   - Cardiovascular: MAP, vasopressor use with doses
   - CNS: GCS
   - Renal: Creatinine, urine output

2. **Sepsis Bundle Compliance**:
   - Time of suspected infection
   - Antibiotic administration (within 3 hours?)
   - Lactate measurement and remeasurement (within 6 hours if elevated?)
   - Fluid resuscitation (30 mL/kg within 3 hours if hypotensive/lactate ≥4?)
   - Vasopressor initiation (if MAP <65 after fluids?)

3. **Clinical Trajectory**:
   - When did patient meet sepsis criteria?
   - Worst values and when they occurred
   - Evidence of organ dysfunction

4. **Relevant Diagnoses** (with dates - note which are pre-existing vs new)"""

STRUCTURED_DATA_SYSTEM_MESSAGE = "You are a clinical data analyst specializing in sepsis cases."

# =============================================================================
# Conflict Detection Examples
# =============================================================================
CONFLICT_EXAMPLES = """- Note says "MAP maintained >65" but vitals show MAP values below 65
- Note says "lactate normalized" but labs show lactate still elevated (>2.0)
- Note says "no vasopressors needed" but meds show vasopressor administration
- Note says "patient alert and oriented" but GCS recorded as <15
- Note says "afebrile" but temps >38°C documented"""

# =============================================================================
# Numeric Cross-Check Parameter Mapping
# =============================================================================
PARAM_TO_CATEGORY = {
    "lactate": "lactate", "lactic acid": "lactate",
    "creatinine": "creatinine", "cr": "creatinine",
    "bilirubin": "bilirubin", "total bilirubin": "bilirubin",
    "platelets": "platelets", "platelet count": "platelets", "plt": "platelets",
    "map": "map", "mean arterial pressure": "map",
    "gcs": "gcs", "glasgow coma scale": "gcs", "glasgow": "gcs",
    "pao2": "pao2", "po2": "pao2",
    "fio2": "fio2",
}

# =============================================================================
# Writer Prompt — Scoring Instructions
# =============================================================================
WRITER_SCORING_INSTRUCTIONS = """5. SOFA SCORING:
   - If SOFA scores are provided above (total >= 2), reference them narratively when arguing organ dysfunction
   - Cite the individual organ scores and total score as clinical evidence of organ dysfunction severity
   - Do NOT include a SOFA table in the letter text — the table is rendered separately in the document
   - If no SOFA scores are provided, do not mention SOFA scoring"""

# =============================================================================
# Assessment Labels
# =============================================================================
ASSESSMENT_CONDITION_LABEL = "sepsis DRG appeal letter"
ASSESSMENT_CRITERIA_LABEL = "PROPEL SEPSIS CRITERIA"

# =============================================================================
# Clinical Scorer — SOFA
# =============================================================================

SOFA_THRESHOLDS = {
    "respiratory": {
        0: lambda ratio: ratio >= 400,
        1: lambda ratio: 300 <= ratio < 400,
        2: lambda ratio: 200 <= ratio < 300,
        3: lambda ratio: 100 <= ratio < 200,
        4: lambda ratio: ratio < 100,
    },
    "coagulation": {
        0: lambda v: v >= 150,
        1: lambda v: 100 <= v < 150,
        2: lambda v: 50 <= v < 100,
        3: lambda v: 20 <= v < 50,
        4: lambda v: v < 20,
    },
    "liver": {
        0: lambda v: v < 1.2,
        1: lambda v: 1.2 <= v < 2.0,
        2: lambda v: 2.0 <= v < 6.0,
        3: lambda v: 6.0 <= v < 12.0,
        4: lambda v: v >= 12.0,
    },
    "cardiovascular_map": {
        0: lambda v: v >= 70,
        1: lambda v: v < 70,
    },
    "cns": {
        0: lambda v: v == 15,
        1: lambda v: 13 <= v <= 14,
        2: lambda v: 10 <= v <= 12,
        3: lambda v: 6 <= v <= 9,
        4: lambda v: v < 6,
    },
    "renal": {
        0: lambda v: v < 1.2,
        1: lambda v: 1.2 <= v < 2.0,
        2: lambda v: 2.0 <= v < 3.5,
        3: lambda v: 3.5 <= v < 5.0,
        4: lambda v: v >= 5.0,
    },
}

LAB_VITAL_MATCHERS = {
    "platelets": {
        "keywords": ["platelet"],
        "exclude": ["platelet function", "platelet antibod", "platelet aggreg"],
        "type": "lab",
    },
    "bilirubin": {
        "keywords": ["bilirubin total", "total bilirubin", "bilirubin,total", "bilirubin, total"],
        "exclude": ["direct", "indirect", "conjugated", "neonatal", "urine"],
        "type": "lab",
    },
    "creatinine": {
        "keywords": ["creatinine"],
        "exclude": ["clearance", "urine", "ratio", "kinase", "random"],
        "type": "lab",
    },
    "pao2": {
        "keywords": ["po2", "pao2", "p02"],
        "exclude": ["pco2", "spco2", "venous"],
        "type": "lab",
    },
    "fio2": {
        "keywords": ["fio2", "fi02", "fraction of inspired"],
        "exclude": [],
        "type": "lab",
    },
    "lactate": {
        "keywords": ["lactate", "lactic acid"],
        "exclude": ["dehydrogenase", "ldh"],
        "type": "lab",
    },
    "map": {
        "keywords": ["map", "mean arterial"],
        "exclude": [],
        "type": "vital",
    },
    "gcs": {
        "keywords": ["gcs", "glasgow"],
        "exclude": ["component", "eye", "motor", "verbal"],
        "type": "vital",
    },
}

VASOPRESSOR_MATCHERS = {
    "norepinephrine": ["norepinephrine", "levophed"],
    "epinephrine": ["epinephrine", "adrenaline"],
    "dopamine": ["dopamine"],
    "dobutamine": ["dobutamine"],
    "vasopressin": ["vasopressin"],
    "phenylephrine": ["phenylephrine", "neosynephrine"],
}


def match_name(name, matcher):
    """Check if a lab/vital name matches a matcher pattern."""
    name_lower = name.lower()
    if not any(kw in name_lower for kw in matcher["keywords"]):
        return False
    if any(ex in name_lower for ex in matcher["exclude"]):
        return False
    return True


def match_vasopressor(med_name):
    """Return vasopressor category if med matches, else None."""
    med_lower = med_name.lower()
    for category, keywords in VASOPRESSOR_MATCHERS.items():
        if any(kw in med_lower for kw in keywords):
            return category
    return None


def safe_float(value):
    """Parse a numeric value, returning None if not parseable."""
    if value is None:
        return None
    try:
        cleaned = str(value).strip().replace(',', '').replace('>', '').replace('<', '')
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def score_organ(organ, value):
    """Score a single organ system. Returns score (0-4) or None if thresholds don't match."""
    thresholds = SOFA_THRESHOLDS.get(organ, {})
    for score in sorted(thresholds.keys(), reverse=True):
        if thresholds[score](value):
            return score
    return 0


def calculate_clinical_scores(account_id, spark, tables):
    """
    Calculate SOFA scores from raw structured data tables.
    Reads directly from labs, vitals, meds tables.
    Returns dict with organ_scores, total_score, organs_scored.
    Zero LLM calls — purely deterministic.

    Args:
        account_id: The hospital account ID
        spark: SparkSession instance
        tables: dict with keys "labs", "vitals", "meds" → full table names
    """
    print("  Calculating SOFA scores from raw data...")
    organ_scores = {}
    vasopressor_detail = []

    labs_table = tables["labs"]
    vitals_table = tables["vitals"]
    meds_table = tables["meds"]

    # --- Gather lab values ---
    try:
        labs_df = spark.sql(f"""
            SELECT LAB_NAME, lab_value, EVENT_TIMESTAMP
            FROM {labs_table}
            WHERE lab_value IS NOT NULL
            ORDER BY EVENT_TIMESTAMP ASC
        """)
        labs_rows = labs_df.collect()
    except Exception as e:
        print(f"    Warning: Could not read labs: {e}")
        labs_rows = []

    # --- Gather vital values ---
    try:
        vitals_df = spark.sql(f"""
            SELECT VITAL_NAME, vital_value, EVENT_TIMESTAMP
            FROM {vitals_table}
            WHERE vital_value IS NOT NULL
            ORDER BY EVENT_TIMESTAMP ASC
        """)
        vitals_rows = vitals_df.collect()
    except Exception as e:
        print(f"    Warning: Could not read vitals: {e}")
        vitals_rows = []

    # --- Gather med values ---
    try:
        meds_df = spark.sql(f"""
            SELECT MED_NAME, MED_DOSE, EVENT_TIMESTAMP
            FROM {meds_table}
            WHERE MED_NAME IS NOT NULL
            ORDER BY EVENT_TIMESTAMP ASC
        """)
        meds_rows = meds_df.collect()
    except Exception as e:
        print(f"    Warning: Could not read meds: {e}")
        meds_rows = []

    # --- Process labs for worst values ---
    lab_worsts = {}
    pao2_values = []
    fio2_values = []

    for row in labs_rows:
        name = row["LAB_NAME"]
        val = safe_float(row["lab_value"])
        ts = row["EVENT_TIMESTAMP"]
        if val is None:
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
                if "coagulation" not in lab_worsts or score > lab_worsts["coagulation"][2]:
                    lab_worsts["coagulation"] = (val, ts, score)
            elif category == "bilirubin":
                score = score_organ("liver", val)
                if "liver" not in lab_worsts or score > lab_worsts["liver"][2]:
                    lab_worsts["liver"] = (val, ts, score)
            elif category == "creatinine":
                score = score_organ("renal", val)
                if "renal" not in lab_worsts or score > lab_worsts["renal"][2]:
                    lab_worsts["renal"] = (val, ts, score)

    # --- Respiratory: PaO2/FiO2 ratio ---
    if pao2_values and fio2_values:
        best_resp_score = 0
        best_resp_val = None
        best_resp_ts = None
        for pao2_val, pao2_ts in pao2_values:
            closest_fio2 = None
            closest_diff = None
            for fio2_val, fio2_ts in fio2_values:
                if fio2_val <= 0:
                    continue
                actual_fio2 = fio2_val if fio2_val <= 1.0 else fio2_val / 100.0
                if actual_fio2 <= 0:
                    continue
                diff = abs((pao2_ts - fio2_ts).total_seconds()) if pao2_ts and fio2_ts else float('inf')
                if closest_diff is None or diff < closest_diff:
                    closest_diff = diff
                    closest_fio2 = actual_fio2
            if closest_fio2 and closest_fio2 > 0:
                ratio = pao2_val / closest_fio2
                score = score_organ("respiratory", ratio)
                if score > best_resp_score:
                    best_resp_score = score
                    best_resp_val = round(ratio, 1)
                    best_resp_ts = pao2_ts
        if best_resp_val is not None:
            organ_scores["respiratory"] = {
                "score": best_resp_score,
                "value": f"PaO2/FiO2 = {best_resp_val}",
                "timestamp": str(best_resp_ts) if best_resp_ts else None,
            }

    # --- Process vitals for worst values ---
    for row in vitals_rows:
        name = row["VITAL_NAME"]
        val = safe_float(row["vital_value"])
        ts = row["EVENT_TIMESTAMP"]
        if val is None:
            continue

        # MAP
        if match_name(name, LAB_VITAL_MATCHERS["map"]):
            score = score_organ("cardiovascular_map", val)
            if "cardiovascular" not in organ_scores or score > organ_scores.get("cardiovascular", {}).get("score", 0):
                organ_scores["cardiovascular"] = {
                    "score": score,
                    "value": f"MAP = {val}",
                    "timestamp": str(ts) if ts else None,
                }

        # GCS
        if match_name(name, LAB_VITAL_MATCHERS["gcs"]):
            if val != val:  # NaN check
                continue
            int_val = int(round(val))
            score = score_organ("cns", int_val)
            if "cns" not in organ_scores or score > organ_scores.get("cns", {}).get("score", 0):
                organ_scores["cns"] = {
                    "score": score,
                    "value": f"GCS = {int_val}",
                    "timestamp": str(ts) if ts else None,
                }

    # --- Process meds for vasopressor scoring ---
    vasopressor_found = {}
    for row in meds_rows:
        med_name = row["MED_NAME"]
        if not med_name:
            continue
        category = match_vasopressor(med_name)
        if category:
            dose = safe_float(row["MED_DOSE"])
            ts = row["EVENT_TIMESTAMP"]
            vasopressor_detail.append({
                "drug": category,
                "dose": dose,
                "timestamp": str(ts) if ts else None,
            })
            if category not in vasopressor_found:
                vasopressor_found[category] = (dose, ts)
            elif dose is not None:
                existing_dose = vasopressor_found[category][0]
                if existing_dose is None or dose > existing_dose:
                    vasopressor_found[category] = (dose, ts)

    # Override cardiovascular score if vasopressors found
    if vasopressor_found:
        max_cv_score = organ_scores.get("cardiovascular", {}).get("score", 0)
        cv_detail = organ_scores.get("cardiovascular", {}).get("value", "")
        cv_ts = organ_scores.get("cardiovascular", {}).get("timestamp")

        for drug, (dose, ts) in vasopressor_found.items():
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

            if cv_score > max_cv_score:
                max_cv_score = cv_score
                dose_str = f" {dose}" if dose is not None else ""
                cv_detail = f"{drug}{dose_str}"
                cv_ts = str(ts) if ts else None

        organ_scores["cardiovascular"] = {
            "score": max_cv_score,
            "value": cv_detail,
            "timestamp": cv_ts,
        }

    # --- Add lab-derived organ scores ---
    for organ_key, (val, ts, score) in lab_worsts.items():
        organ_scores[organ_key] = {
            "score": score,
            "value": str(val),
            "timestamp": str(ts) if ts else None,
        }

    # --- Compute totals ---
    total_score = sum(o["score"] for o in organ_scores.values())
    organs_scored = len(organ_scores)

    print(f"    SOFA Total: {total_score} ({organs_scored} organs scored)")
    for organ, data in organ_scores.items():
        print(f"      {organ}: {data['score']} ({data['value']} at {data['timestamp']})")

    return {
        "organ_scores": organ_scores,
        "total_score": total_score,
        "organs_scored": organs_scored,
        "vasopressor_detail": vasopressor_detail,
    }


def write_clinical_scores_table(account_id, scores_result, spark, table_name):
    """Write SOFA scores to case table."""
    from pyspark.sql.types import (
        StructType, StructField, StringType, IntegerType, TimestampType
    )

    spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        account_id STRING,
        total_score INT,
        organs_scored INT,
        organ_scores STRING,
        vasopressor_detail STRING,
        created_at TIMESTAMP
    ) USING DELTA
    """)

    record = [{
        "account_id": account_id,
        "total_score": scores_result["total_score"],
        "organs_scored": scores_result["organs_scored"],
        "organ_scores": json.dumps(scores_result["organ_scores"]),
        "vasopressor_detail": json.dumps(scores_result.get("vasopressor_detail", [])),
        "created_at": datetime.now()
    }]

    schema = StructType([
        StructField("account_id", StringType(), False),
        StructField("total_score", IntegerType(), True),
        StructField("organs_scored", IntegerType(), True),
        StructField("organ_scores", StringType(), True),
        StructField("vasopressor_detail", StringType(), True),
        StructField("created_at", TimestampType(), True)
    ])

    df = spark.createDataFrame(record, schema)
    df.write.format("delta").mode("overwrite").saveAsTable(table_name)
    print(f"  Written to {table_name}")


# =============================================================================
# DOCX Rendering — SOFA Scores
# =============================================================================

ORGAN_DISPLAY_NAMES = {
    "respiratory": "Respiratory (PaO2/FiO2)",
    "coagulation": "Coagulation (Platelets)",
    "liver": "Liver (Bilirubin)",
    "cardiovascular": "Cardiovascular",
    "cns": "CNS (GCS)",
    "renal": "Renal (Creatinine)",
}


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


def render_scores_status_note(doc, scores_data):
    """Render SOFA status note in the internal review section of DOCX."""
    from docx.shared import Pt

    if not scores_data:
        sofa_note = doc.add_paragraph()
        sofa_note.add_run("SOFA Table: ").bold = True
        sofa_note.add_run("Not included — insufficient structured data to calculate SOFA scores for this encounter.")
    elif scores_data.get("total_score", 0) < 2:
        sofa_note = doc.add_paragraph()
        sofa_note.add_run("SOFA Table: ").bold = True
        sofa_note.add_run(f"Not included — total SOFA score is {scores_data['total_score']} (below threshold of 2). "
                          f"{scores_data.get('organs_scored', 0)} organ system(s) scored.")


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
