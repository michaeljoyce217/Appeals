# Respiratory Failure Integration Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate Diana's respiratory failure variant into the modular DRG Appeal Engine by creating a condition profile, adding three base engine improvements, and introducing a conditional rebuttals extension point.

**Architecture:** Extract all ARF-specific config from Diana's 4 inline files into `condition_profiles/respiratory_failure.py`. Enhance the base engine (`model/inference.py`) with three cross-condition improvements: source fidelity rules, no-summary-table formatting, and a conditional rebuttals mechanism loaded from profiles via `hasattr()`. Expand the assessment prompt with two new evaluation dimensions.

**Tech Stack:** Python, Azure OpenAI GPT-4.1, PySpark, python-docx, Delta Lake

---

## Task 1: Add Source Fidelity + No-Summary-Table Rules to WRITER_PROMPT

**Files:**
- Modify: `model/inference.py:329-394` (WRITER_PROMPT)

**Why:** Diana added two rules that improve ALL conditions: (1) don't cite payor's references unless directly refuting a specific claim, (2) don't include a "Summary Table of Key Clinical Data" at the end. Our current WRITER_PROMPT has neither.

**Step 1: Edit WRITER_PROMPT in inference.py**

In `model/inference.py`, find the WRITER_PROMPT (starts at line 329). Add two new sections.

**After line 362** (the current `2. ADDRESS EACH DENIAL ARGUMENT - quote the payer, then refute`), replace it with:

```python
2. ADDRESS EACH DENIAL ARGUMENT - quote the payer's argument, then refute using ONLY the Propel criteria. Do not list or critique the payor's references unless necessary to refute a specific clinical claim.
```

**After line 365** (`{scoring_instructions}`), and before line 366 (`6. Follow the Mercy Hospital template structure exactly`), add a new instruction line:

This changes the existing instruction #2 and keeps #6 intact. No new numbered items needed.

**After line 392** (the `EVIDENCE DENSITY` line), add before `Return ONLY the letter text.`:

```
# SOURCE FIDELITY
The ONLY authoritative source for clinical definitions, diagnostic criteria, and approved references is the PROPEL CRITERIA section above. You may cite references from the Propel document's reference list. DO NOT list or comment on the payor's cited references unless directly refuting a specific clinical claim.

# FORMATTING REQUIREMENTS
- DO NOT include a "Summary Table of Key Clinical Data" or any similar summary table at the end of the letter. All clinical data should be presented within the body of the letter where it is relevant to the argument. Do not repeat it in a table format at the end.
```

The full modified WRITER_PROMPT should now read (showing only the changed/added parts, with `...` for unchanged content):

```python
    WRITER_PROMPT = '''You are a clinical coding expert writing a DRG validation appeal letter for Mercy Hospital.

# Original Denial Letter
{denial_letter_text}

# Clinical Notes (PRIMARY EVIDENCE - from physician documentation)
{clinical_notes_section}

# Structured Data Summary (SUPPORTING EVIDENCE - from labs, vitals, meds)
{structured_data_summary}

# Computed Clinical Scores
{clinical_scores_section}

# Official Clinical Definition
{clinical_definition_section}

# Gold Standard Letter
{gold_letter_section}

# Patient Information
Name: {patient_name}
DOB: {patient_dob}
Hospital Account #: {hsp_account_id}
Date of Service: {date_of_service}
Facility: {facility_name}
Original DRG: {original_drg}
Proposed DRG: {proposed_drg}
Payor: {payor}

# Instructions
{gold_letter_instructions}
1. READ THE DENIAL LETTER - extract the payor address, reviewer name, claim numbers
2. ADDRESS EACH DENIAL ARGUMENT - quote the payer's argument, then refute using ONLY the Propel criteria. Do not list or critique the payor's references unless necessary to refute a specific clinical claim.
3. CITE CLINICAL EVIDENCE - cite physician notes FIRST as primary evidence, then supporting structured data values
4. INCLUDE TIMESTAMPS with every clinical value cited
{scoring_instructions}
6. Follow the Mercy Hospital template structure exactly
{conditional_rebuttals_section}

# LANGUAGE RULES (MANDATORY)
These rules are non-negotiable. Every sentence in the letter must comply.

CORE PRINCIPLE: The payor will scrutinize every word. Never give them anything to seize on. Only state facts that support the appeal.

ASSERTION-ONLY: Every sentence must advance the argument. State what IS documented, not what is missing.

FORBIDDEN PHRASES — do NOT use any of these patterns:
- "although" / "while" / "despite" / "however" / "nonetheless"
- "despite the lack of" / "despite the absence of"
- "while X was not documented" / "while X is not available"
- "unfortunately" / "regrettably"
- "only" / "merely" / "just" (minimizing qualifiers)
- "may" / "might" / "could possibly" / "potentially" (hedging verbs)
- Any sentence that acknowledges missing data, absent documentation, or unavailable information
- Any sentence that concedes a point the payor made in the denial
- Any qualifier that minimizes or softens a clinical finding

RULES:
1. If data for a parameter is absent, OMIT IT ENTIRELY. Never mention it. Never hedge.
2. Never concede any argument from the denial letter — refute or ignore, never agree.
3. Frame every clinical finding assertively: "Lactate was 4.2 mmol/L" not "Lactate was elevated at 4.2 mmol/L, though it later improved."
4. Do not qualify severity — let the clinical values speak for themselves.

EVIDENCE DENSITY: Every paragraph in the clinical argument must contain at least one specific clinical value with its timestamp.

# SOURCE FIDELITY
The ONLY authoritative source for clinical definitions, diagnostic criteria, and approved references is the PROPEL CRITERIA section above. You may cite references from the Propel document's reference list. DO NOT list or comment on the payor's cited references unless directly refuting a specific clinical claim.

# FORMATTING REQUIREMENTS
- DO NOT include a "Summary Table of Key Clinical Data" or any similar summary table at the end of the letter. All clinical data should be presented within the body of the letter where it is relevant to the argument. Do not repeat it in a table format at the end.

Return ONLY the letter text.'''
```

Note the new `{conditional_rebuttals_section}` placeholder — it goes after instruction #6 and before the LANGUAGE RULES section. This will be populated in Step 2 of this task.

**Step 2: Update the writer_prompt.format() call to include the new placeholder**

In `model/inference.py`, find the `writer_prompt = WRITER_PROMPT.format(` block (starts at line 636). Add the `conditional_rebuttals_section` parameter.

Before the `.format()` call (around line 635), add this code to build the conditional rebuttals section:

```python
    # Build conditional rebuttals section (from profile, if available)
    conditional_rebuttals = getattr(profile, 'CONDITIONAL_REBUTTALS', [])
    if conditional_rebuttals:
        rebuttals_parts = ["\n# CONDITIONAL REBUTTALS - Apply ONLY if the denial matches the specific scenario"]
        for rebuttal in conditional_rebuttals:
            rebuttals_parts.append(f"\n## {rebuttal['name']}")
            rebuttals_parts.append(rebuttal['text'])
        rebuttals_parts.append("\n**IMPORTANT**: Read the denial carefully. Apply ONLY the rebuttal(s) that match the payor's actual argument. Do not apply rebuttals for arguments the payor did not make.")
        conditional_rebuttals_section = "\n".join(rebuttals_parts)
    else:
        conditional_rebuttals_section = ""
```

Then add `conditional_rebuttals_section=conditional_rebuttals_section,` to the `.format()` call. The full updated block:

```python
    # Build conditional rebuttals section (from profile, if available)
    conditional_rebuttals = getattr(profile, 'CONDITIONAL_REBUTTALS', [])
    if conditional_rebuttals:
        rebuttals_parts = ["\n# CONDITIONAL REBUTTALS - Apply ONLY if the denial matches the specific scenario"]
        for rebuttal in conditional_rebuttals:
            rebuttals_parts.append(f"\n## {rebuttal['name']}")
            rebuttals_parts.append(rebuttal['text'])
        rebuttals_parts.append("\n**IMPORTANT**: Read the denial carefully. Apply ONLY the rebuttal(s) that match the payor's actual argument. Do not apply rebuttals for arguments the payor did not make.")
        conditional_rebuttals_section = "\n".join(rebuttals_parts)
    else:
        conditional_rebuttals_section = ""

    # Build prompt
    writer_prompt = WRITER_PROMPT.format(
        denial_letter_text=case_data["denial_text"],
        clinical_notes_section=clinical_notes_section,
        structured_data_summary=structured_summary,
        clinical_scores_section=clinical_scores_section,
        clinical_definition_section=clinical_definition_section,
        gold_letter_section=gold_letter_section,
        gold_letter_instructions=gold_letter_instructions,
        scoring_instructions=profile.WRITER_SCORING_INSTRUCTIONS,
        conditional_rebuttals_section=conditional_rebuttals_section,
        patient_name=case_data.get("patient_name", ""),
        patient_dob=case_data.get("patient_dob", ""),
        hsp_account_id=account_id,
        date_of_service=case_data.get("date_of_service", ""),
        facility_name=case_data.get("facility_name", "Mercy Hospital"),
        original_drg=case_data.get("original_drg") or "Unknown",
        proposed_drg=case_data.get("proposed_drg") or "Unknown",
        payor=case_data.get("payor", "Unknown"),
    )
```

**Step 3: Self-verify**

Re-read the modified WRITER_PROMPT and `.format()` call. Confirm:
- `{conditional_rebuttals_section}` appears in the prompt template
- `conditional_rebuttals_section=conditional_rebuttals_section` appears in the `.format()` call
- Source fidelity text appears after EVIDENCE DENSITY
- Formatting requirements text appears after source fidelity
- Instruction #2 now says "refute using ONLY the Propel criteria"
- All existing `{placeholder}` names are unchanged (no broken references)

**Step 4: Commit**

```bash
git add model/inference.py
git commit -m "Add source fidelity, no-summary-table, and conditional rebuttals to writer prompt

- Instruction #2 now restricts refutation to Propel criteria only
- SOURCE FIDELITY section: only cite Propel-approved references
- FORMATTING REQUIREMENTS: no summary table at end of letter
- {conditional_rebuttals_section} placeholder for profile-driven rebuttals
- Builds rebuttals from profile.CONDITIONAL_REBUTTALS (empty for sepsis)"
```

---

## Task 2: Add source_fidelity + formatting_compliance to ASSESSMENT_PROMPT

**Files:**
- Modify: `model/inference.py:400-459` (ASSESSMENT_PROMPT)
- Modify: `model/inference.py:462-523` (assess_appeal_strength function)
- Modify: `model/inference.py:526-569` (format_assessment_for_docx function)

**Why:** Diana's assessment evaluates two additional dimensions that catch important problems: (1) source_fidelity — did the letter cite only Propel-approved references? (2) formatting_compliance — is there a summary table that shouldn't be there? These dimensions also evaluate conditional rebuttals when present.

**Step 1: Edit ASSESSMENT_PROMPT**

Replace the entire ASSESSMENT_PROMPT (lines 400-459) with:

```python
    ASSESSMENT_PROMPT = '''You are evaluating the strength of a {condition_label}.

═══ {criteria_label} ═══'''.format(
        condition_label=profile.ASSESSMENT_CONDITION_LABEL,
        criteria_label=profile.ASSESSMENT_CRITERIA_LABEL,
    ) + '''
{{propel_definition}}

═══ DENIAL LETTER ═══
{{denial_text}}

═══ GOLD LETTER TEMPLATE USED ═══
{{gold_letter_text}}

═══ AVAILABLE CLINICAL EVIDENCE ═══
{{extracted_clinical_data}}

═══ STRUCTURED DATA SUMMARY ═══
{{structured_summary}}

═══ COMPUTED CLINICAL SCORES ═══
{{clinical_scores_text}}

═══ GENERATED APPEAL LETTER ═══
{{generated_letter}}

{{conditional_rebuttals_section}}

═══ EVALUATION INSTRUCTIONS ═══
IMPORTANT: The Propel criteria above is the ONLY authoritative source for clinical definitions and approved references.

- References cited in the appeal letter are acceptable ONLY if they appear in the Propel document's reference list.
- If the appeal letter cites a reference from the payor's denial letter that is NOT in the Propel document, flag this as a source fidelity error.
- If the appeal letter cites a reference that appears in BOTH the Propel document and the denial letter, this is acceptable.

Evaluate this appeal letter and provide:

1. OVERALL SCORE (1-10) and RATING (LOW for 1-4, MODERATE for 5-7, HIGH for 8-10)
2. SUMMARY (2-3 sentences)
3. DETAILED BREAKDOWN with scores and specific findings

Evaluation Criteria:
- **Source Fidelity**: Did the letter use ONLY Propel-approved criteria and references? Did it appropriately reject proprietary payor criteria when applicable? Did it avoid listing/critiquing payor references unnecessarily?
- **Propel Criteria Alignment**: ONLY criteria explicitly stated in the criteria definition section above (the first ═══ section). Do NOT infer additional criteria from clinical evidence, denial letters, gold letters, or general medical knowledge. If a criterion is not written in the Propel definition, it does not belong in this section.
- **Argument Structure**: Does the letter systematically address and refute each payor argument? Were rebuttals applied appropriately (only when matching the denial's actual claims)?
- **Evidence Quality**: Is clinical evidence from notes and structured data properly cited with timestamps?
- **Formatting Compliance**: The letter should NOT contain a "Summary Table of Key Clinical Data" or similar summary table at the end. If such a table is present, flag it as a formatting error.

Return ONLY valid JSON in this format:
{{{{
  "overall_score": <1-10>,
  "overall_rating": "<LOW|MODERATE|HIGH>",
  "summary": "<2-3 sentence summary>",
  "source_fidelity": {{{{
    "score": <1-10>,
    "findings": [
      {{{{"status": "<correct|incorrect>", "item": "<description>"}}}}
    ]
  }}}},
  "propel_criteria": {{{{
    "score": <1-10>,
    "findings": [
      {{{{"status": "<present|could_strengthen|missing>", "item": "<description>"}}}}
    ]
  }}}},
  "argument_structure": {{{{
    "score": <1-10>,
    "findings": [
      {{{{"status": "<present|could_strengthen|missing>", "item": "<description>"}}}}
    ]
  }}}},
  "evidence_quality": {{{{
    "clinical_notes": {{{{"score": <1-10>, "findings": [...]}}}},
    "structured_data": {{{{"score": <1-10>, "findings": [...]}}}}
  }}}},
  "formatting_compliance": {{{{
    "score": <1-10>,
    "findings": [
      {{{{"status": "<compliant|non_compliant>", "item": "<description>"}}}}
    ]
  }}}}
}}}}'''
```

**CRITICAL NOTE on escaping:** The ASSESSMENT_PROMPT uses two-phase string formatting. The first `.format()` (at definition time) injects `condition_label` and `criteria_label`. The second `.format()` (at call time in `assess_appeal_strength`) injects runtime data. Therefore:
- `{condition_label}` and `{criteria_label}` are single-braced (resolved at definition time)
- Runtime placeholders like `{propel_definition}` need double-braces `{{propel_definition}}` in the second half (after the first `.format()`)
- JSON braces in the output format need QUADRUPLE braces `{{{{` because they pass through TWO levels of `.format()`

**Step 2: Update assess_appeal_strength function**

Replace the function signature and the `.format()` call inside (lines 462-499). The function needs to accept and pass through `conditional_rebuttals`:

```python
    def assess_appeal_strength(generated_letter, propel_definition, denial_text,
                               extracted_notes, gold_letter_text, structured_summary,
                               scores_data=None, conditional_rebuttals=None):
        """Assess the strength of a generated appeal letter."""
        print("  Running strength assessment...")

        # Format extracted notes
        notes_summary = []
        for note_type, content in extracted_notes.items():
            if content and content != "Not available":
                truncated = content[:2000] + "..." if len(content) > 2000 else content
                notes_summary.append(f"## {note_type}\n{truncated}")
        extracted_clinical_data = "\n\n".join(notes_summary) if notes_summary else "No clinical notes available"

        # Format clinical scores for assessment
        if scores_data and hasattr(profile, 'format_scores_for_prompt'):
            clinical_scores_text = profile.format_scores_for_prompt(scores_data)
        else:
            clinical_scores_text = "Clinical scores not available"

        # Build conditional rebuttals section for assessment
        if conditional_rebuttals:
            rebuttals_parts = ["═══ CONDITIONAL REBUTTALS (for evaluation) ═══",
                               "The appeal letter should apply rebuttals ONLY when they match the payor's actual argument:"]
            for rebuttal in conditional_rebuttals:
                rebuttals_parts.append(f"\n## {rebuttal['name']}")
                rebuttals_parts.append(rebuttal['text'])
            rebuttals_parts.append("\nIf the denial includes one of these arguments and the appeal letter fails to rebut it appropriately, flag this as a deficiency. If the appeal letter applies a rebuttal for an argument the payor did NOT make, flag this as an error.")
            conditional_rebuttals_section = "\n".join(rebuttals_parts)
        else:
            conditional_rebuttals_section = ""

        try:
            response = openai_client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "You are a clinical appeal quality assessor. Return only valid JSON."},
                    {"role": "user", "content": ASSESSMENT_PROMPT.format(
                        propel_definition=propel_definition or "Propel criteria not available",
                        denial_text=denial_text[:5000] if denial_text else "Denial text not available",
                        gold_letter_text=gold_letter_text[:3000] if gold_letter_text else "No gold letter template used",
                        extracted_clinical_data=extracted_clinical_data,
                        structured_summary=structured_summary[:3000] if structured_summary else "No structured data",
                        clinical_scores_text=clinical_scores_text,
                        generated_letter=generated_letter,
                        conditional_rebuttals_section=conditional_rebuttals_section,
                    )}
                ],
                temperature=0,
                max_tokens=4000
            )

            raw_response = response.choices[0].message.content.strip()

            # Parse JSON
            if raw_response.startswith("```"):
                raw_response = raw_response.split("```")[1]
                if raw_response.startswith("json"):
                    raw_response = raw_response[4:]
                raw_response = raw_response.strip()

            assessment = json.loads(raw_response)

            if "overall_score" in assessment:
                assessment["overall_score"] = max(1, min(10, int(assessment["overall_score"])))

            print(f"  Assessment complete: {assessment.get('overall_score', '?')}/10 - {assessment.get('overall_rating', '?')}")
            return assessment

        except json.JSONDecodeError as e:
            print(f"  Warning: Assessment JSON parse error: {e}")
            return None
        except Exception as e:
            print(f"  Warning: Assessment failed: {e}")
            return None
```

**Step 3: Update format_assessment_for_docx function**

Replace the function (lines 526-569) to include the two new dimensions:

```python
    def format_assessment_for_docx(assessment):
        """Format assessment dict into text for DOCX output."""
        if not assessment:
            return "Assessment unavailable\n\nPlease review letter manually."

        status_symbols = {"present": "✓", "could_strengthen": "△", "missing": "✗",
                          "correct": "✓", "incorrect": "✗",
                          "compliant": "✓", "non_compliant": "✗"}
        lines = []

        lines.append(f"Overall Strength: {assessment.get('overall_score', '?')}/10 - {assessment.get('overall_rating', '?')}")
        lines.append("")
        lines.append(f"Summary: {assessment.get('summary', 'No summary available')}")
        lines.append("")
        lines.append("Detailed Breakdown:")
        lines.append("─" * 55)

        # Source Fidelity
        source = assessment.get("source_fidelity", {})
        if source:
            lines.append(f"SOURCE FIDELITY: {source.get('score', '?')}/10")
            for finding in source.get("findings", []):
                symbol = status_symbols.get(finding.get("status", ""), "?")
                lines.append(f"  {symbol} {finding.get('item', '')}")

        # Propel Criteria
        propel = assessment.get("propel_criteria", {})
        lines.append(f"\nPROPEL CRITERIA: {propel.get('score', '?')}/10")
        for finding in propel.get("findings", []):
            symbol = status_symbols.get(finding.get("status", ""), "?")
            lines.append(f"  {symbol} {finding.get('item', '')}")

        # Argument Structure
        argument = assessment.get("argument_structure", {})
        lines.append(f"\nARGUMENT STRUCTURE: {argument.get('score', '?')}/10")
        for finding in argument.get("findings", []):
            symbol = status_symbols.get(finding.get("status", ""), "?")
            lines.append(f"  {symbol} {finding.get('item', '')}")

        # Evidence Quality
        evidence = assessment.get("evidence_quality", {})
        clinical = evidence.get("clinical_notes", {})
        structured = evidence.get("structured_data", {})
        lines.append(f"\nEVIDENCE - Clinical Notes: {clinical.get('score', '?')}/10")
        for finding in clinical.get("findings", []):
            symbol = status_symbols.get(finding.get("status", ""), "?")
            lines.append(f"  {symbol} {finding.get('item', '')}")
        lines.append(f"\nEVIDENCE - Structured Data: {structured.get('score', '?')}/10")
        for finding in structured.get("findings", []):
            symbol = status_symbols.get(finding.get("status", ""), "?")
            lines.append(f"  {symbol} {finding.get('item', '')}")

        # Formatting Compliance
        formatting = assessment.get("formatting_compliance", {})
        if formatting:
            lines.append(f"\nFORMATTING COMPLIANCE: {formatting.get('score', '?')}/10")
            for finding in formatting.get("findings", []):
                symbol = status_symbols.get(finding.get("status", ""), "?")
                lines.append(f"  {symbol} {finding.get('item', '')}")

        lines.append("─" * 55)
        return "\n".join(lines)
```

**Step 4: Update the assess_appeal_strength() call site**

In the main pipeline (around line 676), update the call to pass `conditional_rebuttals`:

```python
    assessment = assess_appeal_strength(
        letter_text, propel_def, case_data["denial_text"],
        extracted_notes, gold_text, structured_summary,
        scores_data=case_data.get("clinical_scores"),
        conditional_rebuttals=conditional_rebuttals,
    )
```

Note: `conditional_rebuttals` is already computed in Task 1 (the variable built before the writer `.format()` call). It's a list from `getattr(profile, 'CONDITIONAL_REBUTTALS', [])`.

**Step 5: Self-verify**

Re-read the modified ASSESSMENT_PROMPT, assess_appeal_strength, and format_assessment_for_docx. Confirm:
- ASSESSMENT_PROMPT has `{conditional_rebuttals_section}` placeholder
- assess_appeal_strength passes `conditional_rebuttals_section` to `.format()`
- format_assessment_for_docx handles `source_fidelity` and `formatting_compliance`
- The call site passes `conditional_rebuttals=conditional_rebuttals`
- JSON escaping is correct (quadruple braces for JSON literal braces in the two-phase format)

**Step 6: Commit**

```bash
git add model/inference.py
git commit -m "Add source_fidelity and formatting_compliance to assessment

- Assessment now evaluates 5 dimensions: source_fidelity, propel_criteria,
  argument_structure, evidence_quality, formatting_compliance
- Conditional rebuttals passed to assessment for evaluation
- format_assessment_for_docx renders new dimensions in DOCX"
```

---

## Task 3: Create condition_profiles/respiratory_failure.py

**Files:**
- Create: `condition_profiles/respiratory_failure.py`

**Why:** This is the core deliverable. All ARF-specific configuration extracted from Diana's 4 inline files, organized into the profile interface contract. No clinical scorer (no SOFA equivalent for ARF).

**Step 1: Create the respiratory failure profile**

Create file `condition_profiles/respiratory_failure.py` with the following content. Every value is extracted from Diana's PDF (pages 1-117):

```python
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
   - Timeline of oxygen delivery: room air → nasal cannula → high-flow → BiPAP/CPAP → intubation
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
}

# =============================================================================
# Writer Prompt — Scoring Instructions
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
2. Consecutive reading requirements are proprietary payor criteria — not nationally recognized standards.
3. A single documented SpO2 <91% on room air with clinical context and treatment response supports the diagnosis.

Cite ALL relevant timestamped low SpO2 readings, even if not consecutive.""",
    },
    {
        "name": "Persistent/Continuous Symptoms Requirement",
        "trigger": "Apply ONLY if the denial argues symptoms were not 'persistent,' 'continuous,' or 'sustained.'",
        "text": """Apply ONLY if the denial argues symptoms were not "persistent," "continuous," or "sustained."

Rebuttal points:
1. Acute respiratory failure is defined by onset and severity at presentation — not by unchanged symptoms throughout the stay.
2. Improvement with treatment confirms appropriate intervention, not absence of the condition.
3. Per CMS/Coding Clinic, a diagnosis is reportable when documented by the provider and supported by clinical indicators and treatment at presentation.

Cite documentation of respiratory distress at presentation and interventions required.""",
    },
    {
        "name": "Proprietary Clinical Criteria (General)",
        "trigger": "Apply if the denial imposes clinical thresholds beyond provider documentation and nationally recognized standards.",
        "text": """Apply if the denial imposes clinical thresholds beyond provider documentation and nationally recognized standards.

Rebuttal points:
1. DRG validation confirms clinical support for documented diagnoses — it does not substitute proprietary payor thresholds for physician judgment.
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
```

**Step 2: Validate profile contract**

Check this file against `condition_profiles/__init__.py`'s REQUIRED_ATTRIBUTES list. Every required attribute must be present:

| Required Attribute | Present? | Value |
|---|---|---|
| CONDITION_NAME | Yes | "respiratory_failure" |
| CONDITION_DISPLAY_NAME | Yes | "Acute Respiratory Failure" |
| GOLD_LETTERS_PATH | Yes | path to arf_only |
| PROPEL_DATA_PATH | Yes | path to propel_data |
| DEFAULT_TEMPLATE_PATH | Yes | path to template |
| CLINICAL_SCORES_TABLE_NAME | Yes | None |
| DENIAL_CONDITION_QUESTION | Yes | IS_ARF question |
| DENIAL_CONDITION_FIELD | Yes | "IS_ARF" |
| NOTE_EXTRACTION_TARGETS | Yes | respiratory-specific |
| STRUCTURED_DATA_CONTEXT | Yes | respiratory-specific |
| CONFLICT_EXAMPLES | Yes | respiratory-specific |
| WRITER_SCORING_INSTRUCTIONS | Yes | ARF criteria |
| ASSESSMENT_CONDITION_LABEL | Yes | "acute respiratory failure..." |
| ASSESSMENT_CRITERIA_LABEL | Yes | "PROPEL RESPIRATORY..." |

Also verify optional attributes used via `hasattr()`:
- `PARAM_TO_CATEGORY` — present (for numeric cross-check)
- `LAB_VITAL_MATCHERS` — present (for numeric cross-check)
- `match_name()` — present (helper function)
- `safe_float()` — present (helper function)
- `DIAGNOSIS_EXAMPLES` — present
- `STRUCTURED_DATA_SYSTEM_MESSAGE` — present
- `CONDITIONAL_REBUTTALS` — present (NEW, the 3 rebuttal templates)
- `calculate_clinical_scores` — NOT present (no scorer for ARF, correct)
- `write_clinical_scores_table` — NOT present (correct)
- `format_scores_for_prompt` — NOT present (correct)
- `render_scores_status_note` — NOT present (correct)
- `render_scores_in_docx` — NOT present (correct)

**Step 3: Self-verify**

Re-read the file. Confirm:
- DRG codes match Diana's: ["189", "190", "191", "207", "208"]
- CONDITIONAL_REBUTTALS has 3 entries matching Diana's 3 rebuttal templates
- LAB_VITAL_MATCHERS includes respiratory-specific parameters (spo2, paco2, ph, respiratory_rate)
- PARAM_TO_CATEGORY covers Diana's respiratory parameters
- Paths use `fudgesicle` repo path (not Diana's `creamsicle`)
- No SOFA-related functions (ARF has no clinical scorer)

**Step 4: Commit**

```bash
git add condition_profiles/respiratory_failure.py
git commit -m "Add respiratory failure condition profile

- Extracted from Diana's 4 inline files into modular profile
- DRG codes: 189, 190, 191, 207, 208
- 3 conditional rebuttals from Dr. Gharfeh/Dr. Bourland
- Respiratory-specific LAB_VITAL_MATCHERS and PARAM_TO_CATEGORY
- No clinical scorer (no SOFA equivalent for ARF)"
```

---

## Task 4: Update TEMPLATE.py with CONDITIONAL_REBUTTALS Documentation

**Files:**
- Modify: `condition_profiles/TEMPLATE.py`

**Why:** New profiles need to know about the optional `CONDITIONAL_REBUTTALS` extension point. The template should document it so future conditions can add their own rebuttals.

**Step 1: Add CONDITIONAL_REBUTTALS section to TEMPLATE.py**

After the existing `OPTIONAL: DOCX Rendering` section (ends at line 157), add:

```python

# =============================================================================
# OPTIONAL: Conditional Rebuttals
# =============================================================================

# If this condition has specific rebuttal templates for common payor denial arguments,
# define them here. Each rebuttal is conditionally injected into the writer and assessment
# prompts — the LLM applies ONLY the rebuttals that match the payor's actual argument.
#
# Format: list of dicts, each with:
#   "name": Display name for the rebuttal (used as section header)
#   "trigger": One-sentence description of when to apply
#   "text": Full rebuttal text with points and instructions
#
# CONDITIONAL_REBUTTALS = [
#     {
#         "name": "Example Rebuttal Scenario",
#         "trigger": "Apply ONLY if the denial argues X.",
#         "text": """Apply ONLY if the denial argues X.
#
# Rebuttal points:
# 1. First rebuttal point with clinical/regulatory basis.
# 2. Second rebuttal point.
# 3. Third rebuttal point.
#
# Cite relevant clinical evidence.""",
#     },
# ]
```

**Step 2: Self-verify**

Re-read TEMPLATE.py. Confirm:
- New section is at the end after DOCX Rendering
- Format documentation matches what respiratory_failure.py actually uses
- It's clearly marked OPTIONAL

**Step 3: Commit**

```bash
git add condition_profiles/TEMPLATE.py
git commit -m "Add CONDITIONAL_REBUTTALS docs to profile template"
```

---

## Task 5: Update CLAUDE.md Documentation

**Files:**
- Modify: `CLAUDE.md` (project root)

**Why:** The CLAUDE.md is the master prompt. It needs to document the new conditional rebuttals extension point, the new assessment dimensions, and the respiratory failure profile.

**Step 1: Update the "What a Condition Profile Provides" table**

In the `### What a Condition Profile Provides` section, add a new row:

```
| **Conditional Rebuttals** (optional) | `CONDITIONAL_REBUTTALS` (list of rebuttal dicts) |
```

**Step 2: Update the Assessment section**

Find the `### Appeal Strength Assessment` section. Update the dimensions list:

From:
```
- **Propel Criteria Coverage** (from: Propel definitions)
- **Argument Structure** (from: denial letter, gold template)
- **Evidence Quality** (from: clinical notes AND structured data)
```

To:
```
- **Source Fidelity** (did the letter use ONLY Propel-approved criteria and references?)
- **Propel Criteria Coverage** (from: Propel definitions)
- **Argument Structure** (from: denial letter, gold template; rebuttals applied correctly?)
- **Evidence Quality** (from: clinical notes AND structured data)
- **Formatting Compliance** (no summary table at end of letter)
```

**Step 3: Add Assertion-Only Language update**

In the `### Assertion-Only Language` section, add:

```
- Only cite Propel-approved references — do not list or comment on the payor's cited references unless directly refuting a specific clinical claim
- Do not include a "Summary Table of Key Clinical Data" or similar table at the end of the letter
```

**Step 4: Add note about Conditional Rebuttals**

Add a new `### Conditional Rebuttals` subsection after `### Assertion-Only Language`:

```
### Conditional Rebuttals
Some conditions have specific rebuttal templates for common payor denial arguments (e.g., respiratory failure has SpO2 reading rebuttals from Dr. Gharfeh and Dr. Bourland). These are defined in the condition profile as `CONDITIONAL_REBUTTALS` and are conditionally injected into both the writer prompt and assessment prompt. The LLM applies ONLY the rebuttals that match the payor's actual argument — it does not apply rebuttals for arguments the payor did not make.
```

**Step 5: Self-verify and Commit**

```bash
git add CLAUDE.md
git commit -m "Update CLAUDE.md with respiratory failure and new assessment dimensions"
```

---

## Task 6: Final Integration Verification

**Files:**
- Read: All modified files for cross-reference check

**Step 1: Verify profile loads correctly**

Mentally trace the import path:
1. `inference.py` line 41: `profile = importlib.import_module(f"condition_profiles.respiratory_failure")`
2. `__init__.py` `validate_profile(profile)` — checks 14 REQUIRED_ATTRIBUTES — all present
3. `inference.py` line 73: `CASE_SCORES_TABLE = ... if profile.CLINICAL_SCORES_TABLE_NAME else None` → None (correct, no scorer)
4. `inference.py` line 630: `if case_data.get("clinical_scores") and hasattr(profile, 'format_scores_for_prompt'):` → False (correct)
5. `inference.py` line 734: `if hasattr(profile, 'render_scores_status_note'):` → False (correct)
6. `inference.py` line 765: `if hasattr(profile, 'render_scores_in_docx'):` → False (correct)

**Step 2: Verify conditional rebuttals flow**

1. `getattr(profile, 'CONDITIONAL_REBUTTALS', [])` → returns list of 3 dicts for ARF, empty list for sepsis
2. `conditional_rebuttals_section` built with 3 rebuttals for ARF, empty string for sepsis
3. `WRITER_PROMPT.format(..., conditional_rebuttals_section=...)` → injects rebuttals into writer for ARF, empty for sepsis
4. `assess_appeal_strength(..., conditional_rebuttals=conditional_rebuttals)` → passes list to assessment
5. Assessment `.format(..., conditional_rebuttals_section=...)` → injects into assessment for ARF, empty for sepsis

**Step 3: Verify featurization_inference.py compatibility**

The profile is used in featurization_inference.py at these points:
- `profile.CLINICAL_SCORES_TABLE_NAME` → None (handled)
- `profile.CONDITION_NAME` → "respiratory_failure" (correct for Propel lookup)
- `profile.DENIAL_CONDITION_QUESTION` / `DENIAL_CONDITION_FIELD` → IS_ARF (correct)
- `profile.NOTE_EXTRACTION_TARGETS` → respiratory-specific (correct)
- `profile.LAB_VITAL_MATCHERS` → respiratory-specific (correct)
- `profile.match_name()` / `profile.safe_float()` → defined (correct)
- `profile.PARAM_TO_CATEGORY` → respiratory-specific (correct)
- `profile.STRUCTURED_DATA_CONTEXT` → respiratory-specific (correct)
- `profile.CONFLICT_EXAMPLES` → respiratory-specific (correct)
- `hasattr(profile, 'calculate_clinical_scores')` → False (skips scorer, correct)

**Step 4: Verify backward compatibility with sepsis**

Running with `CONDITION_PROFILE = "sepsis"`:
- `getattr(profile, 'CONDITIONAL_REBUTTALS', [])` → `[]` (sepsis has no CONDITIONAL_REBUTTALS)
- `conditional_rebuttals_section` → empty string
- WRITER_PROMPT `{conditional_rebuttals_section}` → renders as blank line (harmless)
- Assessment `{conditional_rebuttals_section}` → renders as blank (harmless)
- New source_fidelity and formatting_compliance dimensions → will be evaluated (improvement for sepsis too)
- All existing hasattr() checks unchanged → same behavior

**Step 5: Final commit (squash if desired)**

```bash
git add -A
git commit -m "Complete respiratory failure integration into modular engine

Summary:
- condition_profiles/respiratory_failure.py: Full ARF profile with 3 conditional rebuttals
- model/inference.py: Source fidelity, no-summary-table, conditional rebuttals mechanism,
  expanded assessment (5 dimensions)
- condition_profiles/TEMPLATE.py: CONDITIONAL_REBUTTALS documentation
- CLAUDE.md: Updated with new features and ARF profile docs"
```
