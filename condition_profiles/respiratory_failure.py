# condition_profiles/respiratory_failure.py
# Acute Respiratory Failure Condition Profile for the DRG Appeal Engine
#
# Contains all ARF-specific configuration, prompts, and conditional rebuttals.
# No clinical scoring system (unlike sepsis which has SOFA).
#
# Conditional rebuttals are from Dr. Gharfeh and Dr. Bourland.
# The base engine imports this module via:
#   profile = importlib.import_module(f"condition_profiles.{CONDITION_PROFILE}")

# =============================================================================
# Identity & Paths
# =============================================================================
CONDITION_NAME = "respiratory_failure"
CONDITION_DISPLAY_NAME = "Acute Respiratory Failure"
DRG_CODES = ["189", "190", "191", "207", "208"]

GOLD_LETTERS_PATH = "/Workspace/Repos/mijo8881@mercy.net/fudgesicle/utils/gold_standard_appeals_arf_only"
PROPEL_DATA_PATH = "/Workspace/Repos/mijo8881@mercy.net/fudgesicle/utils/propel_data"
DEFAULT_TEMPLATE_PATH = "/Workspace/Repos/mijo8881@mercy.net/fudgesicle/utils/gold_standard_appeals_arf_only/default_respiratory_failure_template.docx"

# No clinical scoring system for ARF (unlike SOFA for sepsis)
CLINICAL_SCORES_TABLE_NAME = None

# =============================================================================
# Denial Parser
# =============================================================================
DENIAL_CONDITION_QUESTION = "5. IS ACUTE RESPIRATORY FAILURE RELATED - does this denial involve acute respiratory failure, hypoxic respiratory failure, or hypercapnic respiratory failure?"
DENIAL_CONDITION_FIELD = "IS_ARF"

# =============================================================================
# Note Extraction Targets (fallback when Propel definition_summary unavailable)
# =============================================================================
NOTE_EXTRACTION_TARGETS = """## Oxygenation Status (PRIORITY - extract ALL available with timestamps)
- SpO2 readings (especially <91% on room air)
- PaO2 values (especially <60 mmHg)
- FiO2 levels and changes
- P/F ratio (PaO2/FiO2)
- Oxygen delivery method and flow rates (nasal cannula, high-flow, BiPAP, CPAP, ventilator)
- Escalation/de-escalation of respiratory support

## Ventilation Parameters
- PaCO2 values (especially >50 mmHg)
- pH values (especially <7.35)
- Respiratory rate
- Tidal volume, minute ventilation
- Ventilator settings (mode, PEEP, FiO2)

## Clinical Presentation
- Dyspnea severity and onset
- Use of accessory muscles
- Work of breathing assessments
- Mental status changes related to respiratory failure
- Cyanosis

## Treatment Response
- Response to oxygen therapy
- Response to non-invasive ventilation (BiPAP/CPAP)
- Need for intubation/mechanical ventilation
- ABG trends over time

## Underlying Etiology
- Pneumonia, COPD exacerbation, pulmonary embolism, ARDS, CHF
- Chest X-ray / CT findings
- Culture results
- Antibiotic administration"""

# =============================================================================
# Structured Data Context (for structured data extraction prompt)
# =============================================================================
DIAGNOSIS_EXAMPLES = """For example:
- "Acute respiratory failure with hypoxia"
- "Acute hypoxic respiratory failure"
- "Acute on chronic respiratory failure with hypoxia"
- "Acute hypercapnic respiratory failure"
"""

STRUCTURED_DATA_CONTEXT = """Extract a focused summary of acute respiratory failure-relevant clinical data from this timeline. Prioritize:

1. **Oxygenation Data** (with timestamps and trends):
   - SpO2 readings (all values, especially <91% on room air)
   - PaO2 values from ABGs
   - FiO2 levels
   - P/F ratio calculations
   - Oxygen delivery method changes

2. **Ventilation Data**:
   - PaCO2 values from ABGs
   - pH values
   - Respiratory rate trends
   - Ventilator settings and mode changes

3. **Respiratory Support Escalation**:
   - Timeline of oxygen delivery: room air -> nasal cannula -> high-flow -> BiPAP/CPAP -> intubation
   - Flow rates and FiO2 at each step
   - Duration on each level of support

4. **Relevant Diagnoses** (with dates - note which are pre-existing vs new):
   - Acute respiratory failure type (hypoxic, hypercapnic, mixed)
   - Underlying cause (pneumonia, COPD, PE, ARDS, CHF, etc.)"""

STRUCTURED_DATA_SYSTEM_MESSAGE = "You are a clinical data analyst specializing in respiratory failure cases."

# =============================================================================
# Conflict Detection Examples
# =============================================================================
CONFLICT_EXAMPLES = """- Note says "SpO2 maintained >92% on room air" but vitals show SpO2 <91% on room air
- Note says "patient on room air" but respiratory flowsheet shows supplemental O2
- Note says "no respiratory distress" but respiratory rate >24 documented
- Note says "ABG normal" but labs show PaO2 <60 or PaCO2 >50
- Note says "weaned to room air" but vitals show continued supplemental oxygen
- Note says "FiO2 40%" but respiratory flowsheet shows FiO2 60%"""

# =============================================================================
# Numeric Cross-Check Parameter Mapping
# =============================================================================
PARAM_TO_CATEGORY = {
    "spo2": "spo2", "sp02": "spo2", "oxygen saturation": "spo2", "o2 sat": "spo2",
    "pao2": "pao2", "po2": "pao2", "partial pressure of oxygen": "pao2",
    "paco2": "paco2", "pco2": "paco2", "partial pressure of co2": "paco2",
    "fio2": "fio2", "fi02": "fio2", "fraction of inspired oxygen": "fio2",
    "ph": "ph", "arterial ph": "ph",
    "respiratory rate": "respiratory_rate", "rr": "respiratory_rate", "resp rate": "respiratory_rate",
    "peep": "peep",
    "tidal volume": "tidal_volume",
    "lactate": "lactate", "lactic acid": "lactate",
    "creatinine": "creatinine", "cr": "creatinine",
    "gcs": "gcs", "glasgow coma scale": "gcs",
}

LAB_VITAL_MATCHERS = {
    "spo2": {
        "keywords": ["spo2", "sp02", "pulse ox", "oxygen sat"],
        "exclude": [],
        "type": "vital",
    },
    "pao2": {
        "keywords": ["po2", "pao2", "p02"],
        "exclude": ["pco2", "spco2", "venous"],
        "type": "lab",
    },
    "paco2": {
        "keywords": ["pco2", "paco2"],
        "exclude": ["spco2"],
        "type": "lab",
    },
    "fio2": {
        "keywords": ["fio2", "fi02", "fraction of inspired"],
        "exclude": [],
        "type": "lab",
    },
    "ph": {
        "keywords": ["ph"],
        "exclude": ["phosph", "pharyn", "pheno", "photo", "urine ph"],
        "type": "lab",
    },
    "respiratory_rate": {
        "keywords": ["resp", "respiratory rate", "rr"],
        "exclude": ["response"],
        "type": "vital",
    },
    "lactate": {
        "keywords": ["lactate", "lactic acid"],
        "exclude": ["dehydrogenase", "ldh"],
        "type": "lab",
    },
    "creatinine": {
        "keywords": ["creatinine"],
        "exclude": ["clearance", "urine", "ratio", "kinase", "random"],
        "type": "lab",
    },
    "gcs": {
        "keywords": ["gcs", "glasgow"],
        "exclude": ["component", "eye", "motor", "verbal"],
        "type": "vital",
    },
    "peep": {
        "keywords": ["peep"],
        "exclude": [],
        "type": "vital",
    },
    "tidal_volume": {
        "keywords": ["tidal vol"],
        "exclude": [],
        "type": "vital",
    },
}

# =============================================================================
# Writer Prompt -- Scoring Instructions
# =============================================================================
WRITER_SCORING_INSTRUCTIONS = """5. QUANTIFY ACUTE RESPIRATORY FAILURE using diagnostic criteria from the Propel guidelines when available:
   - Reference specific values: PaO2, SpO2, PaCO2, pH, FiO2, P/F ratio
   - Hypoxic criteria: PaO2 < 60 mmHg, SpO2 < 91% on room air, P/F ratio < 300
   - Hypercapnic criteria: PaCO2 > 50 mmHg with pH < 7.35
   - Acute-on-chronic indicators: >= 10 mmHg change from baseline PaO2 or PaCO2
   - Document oxygen delivery method and escalation of respiratory support"""

# =============================================================================
# Assessment Labels
# =============================================================================
ASSESSMENT_CONDITION_LABEL = "acute respiratory failure DRG appeal letter"
ASSESSMENT_CRITERIA_LABEL = "PROPEL RESPIRATORY FAILURE CRITERIA (AUTHORITATIVE SOURCE - USE THIS ONLY)"

# =============================================================================
# Conditional Rebuttals (from Dr. Gharfeh and Dr. Bourland)
# =============================================================================

CONDITIONAL_REBUTTALS = [
    {
        "name": "Consecutive/Sequential SpO2 Readings",
        "trigger": "Apply ONLY if the denial requires 'consecutive,' 'sequential,' or 'back-to-back' SpO2 readings.",
        "text": """Apply ONLY if the denial requires "consecutive," "sequential," or "back-to-back" SpO2 readings.

Rebuttal points:
1. No CMS, ICD-10-CM, or Coding Clinic requirement mandates multiple sequential SpO2 readings for acute hypoxemic respiratory failure.
2. Consecutive reading requirements are proprietary payor criteria -- not nationally recognized standards.
3. A single documented SpO2 <91% on room air with clinical context and treatment response supports the diagnosis.

Cite ALL relevant timestamped low SpO2 readings, even if not consecutive.""",
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
    {
        "name": "Proprietary Clinical Criteria (General)",
        "trigger": "Apply if the denial imposes clinical thresholds beyond provider documentation and nationally recognized standards.",
        "text": """Apply if the denial imposes clinical thresholds beyond provider documentation and nationally recognized standards.

Rebuttal points:
1. DRG validation confirms clinical support for documented diagnoses -- it does not substitute proprietary payor thresholds for physician judgment.
2. Per CMS/AHIMA/Coding Clinic, a diagnosis is reportable when (a) documented by the provider and (b) clinically supported by patient-specific indicators and treatment.
3. Internal payor criteria exceeding CMS standards are not valid grounds for DRG reassignment.""",
    },
]

# =============================================================================
# Utility Functions (used by featurization_inference.py numeric cross-check)
# =============================================================================

def match_name(name, matcher):
    """Check if a lab/vital name matches a matcher pattern."""
    name_lower = name.lower()
    if not any(kw in name_lower for kw in matcher["keywords"]):
        return False
    if any(ex in name_lower for ex in matcher["exclude"]):
        return False
    return True


def safe_float(value):
    """Parse a numeric value, returning None if not parseable."""
    if value is None:
        return None
    try:
        cleaned = str(value).strip().replace(',', '').replace('>', '').replace('<', '')
        return float(cleaned)
    except (ValueError, TypeError):
        return None
