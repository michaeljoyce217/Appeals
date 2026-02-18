# poc/agents/writer_agent.py
# Writer Agent: Generates the final rebuttal letter
#
# Uses LLM to compose letter from:
# - Denial info (from Parser)
# - Clinical criteria (from Research - when available)
# - Gold standard letter (from Reference - when available, else template)
# - Clinical notes (provider notes = best evidence)
# - Patient data (demographics, dates)

import json
import os
from typing import Dict, Any, List, Optional
from datetime import date

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EVIDENCE_PRIORITY


class WriterAgent:
    """
    Generates rebuttal letters following the Mercy Hospital template structure.

    Evidence priority:
    1. Provider notes (Discharge Summary, H&P) - BEST
    2. Structured data (labs, vitals from Clarity) - backup
    3. Inference (our conclusions from structured data) - LEAST important

    Output: Structured JSON that can be rendered to DOCX.
    """

    LETTER_PROMPT = '''You are a clinical coding expert writing a DRG validation appeal letter for Mercy Hospital.

# Task
Write a complete appeal letter using the Mercy Hospital template format.

# Denial Information
{denial_info_json}

# Clinical Notes (PRIMARY EVIDENCE - use these first)
## Discharge Summary
{discharge_summary}

## H&P Note
{hp_note}

# Structured Clinical Data (SECONDARY EVIDENCE - use if notes lack detail)
{structured_data}

# Reference Criteria (from Propel/guidelines)
{criteria_json}

# Gold Standard Letter (WINNING REBUTTAL - learn from this)
{gold_letter_section}

# Patient Information
{patient_info_json}

# Instructions
{gold_letter_instructions}
1. ADDRESS EACH DENIAL ARGUMENT specifically - quote the payer's argument, then refute it
2. CITE CLINICAL EVIDENCE from provider notes FIRST (best source)
3. Use structured data only when notes don't contain the information
4. Follow the Mercy template structure exactly
5. Include specific values (lactate 2.4, MAP 62, etc.) whenever available
6. DELETE sections that don't apply to this patient
7. Be thorough but concise - every statement should support the appeal

# Output Format
Return a JSON object with this structure:
{{
  "letter": {{
    "mercy_header": {{
      "address": "Mercy Hospital\\nPayor Audits & Denials Dept\\nATTN: Compliance Manager\\n2115 S Fremont Ave - Ste LL1\\nSpringfield, MO 65804",
      "date": "{current_date}"
    }},
    "payor_address": "Full payer address from denial letter",
    "appeal_level": "First Level Appeal",
    "patient_info": {{
      "beneficiary_name": "Patient name",
      "dob": "MM/DD/YYYY",
      "claim_reference": "From denial letter",
      "hospital_account": "HSP account ID",
      "date_of_service": "Admission - Discharge dates"
    }},
    "salutation": "Dear [Reviewer Name or Medical Review Team]:",
    "opening_paragraph": "Mercy Hospital [facility] is in receipt of your DRG Validation Review dated [date]...",
    "justification": "After review of the claim, we respectfully disagree with the determination...",
    "rationale": {{
      "payor_argument_quote": "Quote what the payer said",
      "infection_source": "Identified infection and how diagnosed",
      "organ_dysfunction": [
        {{"criterion": "Elevated Lactate", "value": "2.4 mmol/L", "threshold": ">2.0", "source": "provider_notes"}},
        ...
      ],
      "sofa_score": null or {{"score": 4, "components": [...]}},
      "blood_culture": "Organism if positive, or 'Negative' or null",
      "sirs_criteria": [
        {{"criterion": "Temperature", "value": "38.9°C", "threshold": ">38.3°C", "source": "provider_notes"}},
        ...
      ],
      "inflammatory_markers": [...],
      "other_conditions": ["Acute Metabolic Encephalopathy", ...]
    }},
    "hospital_course": "Narrative of the patient's hospitalization...",
    "summary_paragraph": "Our patient was admitted for a total of X days due to Sepsis secondary to [source]...",
    "conclusion": "We appreciate the opportunity to respond to your determination, and following review, we anticipate our original DRG of [X] will be approved.",
    "contact_info": "Fax: 314-364-6231\\nContact: Alyssa Whitlock at 417-885-5420",
    "signature_block": "Clinical Appeals Specialist\\nDRG Validation & Appeal Specialist\\nMercy Health System"
  }},
  "evidence_sources": {{
    "provider_notes_used": ["List of specific evidence from notes"],
    "structured_data_used": ["List of structured data points used"],
    "inference_used": ["Any inferences made - mark as lower confidence"]
  }},
  "citations_used": [{{"source": "...", "section": "..."}}],
  "gold_letter_used": "letter_id or null",
  "confidence_score": 0.0 to 1.0
}}

Return ONLY the JSON, no markdown or explanation.'''

    def __init__(self, client, model: str = "gpt-4.1"):
        """
        Initialize with Azure OpenAI client.

        Args:
            client: AzureOpenAI client instance
            model: Model deployment name
        """
        self.client = client
        self.model = model

    @classmethod
    def from_databricks_secrets(cls, scope: str = "idp_etl",
                                  api_key_secret: str = "az-openai-key1",
                                  endpoint_secret: str = "az-openai-base",
                                  model: str = "gpt-4.1"):
        """Create writer from Databricks secrets."""
        from pyspark.dbutils import DBUtils
        from pyspark.sql import SparkSession
        from openai import AzureOpenAI

        spark = SparkSession.builder.getOrCreate()
        dbutils = DBUtils(spark)

        api_key = dbutils.secrets.get(scope=scope, key=api_key_secret)
        endpoint = dbutils.secrets.get(scope=scope, key=endpoint_secret)

        client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version="2024-10-21"
        )

        return cls(client=client, model=model)

    @classmethod
    def from_env(cls, model: str = "gpt-4.1"):
        """Create writer from environment variables."""
        from openai import AzureOpenAI

        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")

        if not api_key or not endpoint:
            raise ValueError(
                "Missing environment variables. Set:\n"
                "  AZURE_OPENAI_API_KEY\n"
                "  AZURE_OPENAI_ENDPOINT"
            )

        client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version="2024-10-21"
        )

        return cls(client=client, model=model)

    def generate(self,
                 denial_info: Dict[str, Any],
                 clinical_notes: Dict[str, str],
                 patient_data: Dict[str, Any],
                 relevant_criteria: Optional[List[Dict]] = None,
                 gold_letter: Optional[Dict[str, Any]] = None,
                 structured_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a rebuttal letter.

        Args:
            denial_info: Parsed denial information from ParserAgent
            clinical_notes: Dict with 'discharge_summary' and 'hp_note' keys
            patient_data: Patient demographics and admin data
            relevant_criteria: Optional criteria from ResearchAgent
            gold_letter: Optional gold standard letter from ReferenceAgent
            structured_data: Optional structured clinical data (labs, vitals)

        Returns:
            Dictionary with:
            - status: "success" | "error"
            - letter: Structured letter JSON (if success)
            - evidence_sources: What evidence was used
            - confidence_score: Model's confidence in the letter
            - _generation_metadata: Debug info
        """
        # Prepare prompt inputs
        current_date = date.today().strftime("%m/%d/%Y")

        denial_info_json = json.dumps(denial_info, indent=2, default=str)
        patient_info_json = json.dumps(patient_data, indent=2, default=str)

        discharge_summary = clinical_notes.get("discharge_summary", "Not available")
        hp_note = clinical_notes.get("hp_note", "Not available")

        criteria_json = json.dumps(relevant_criteria or [], indent=2)
        structured_data_str = json.dumps(structured_data or {}, indent=2)

        # Build gold letter section - this is key for learning from past successes
        if gold_letter and gold_letter.get("letter_text"):
            gold_letter_section = f"""## THIS LETTER WON A SIMILAR APPEAL - LEARN FROM IT
Source: {gold_letter.get('source_file', 'Unknown')}
Match Score: {gold_letter.get('match_score', 'N/A')}

### Winning Rebuttal Text:
{gold_letter['letter_text']}
"""
            gold_letter_instructions = """**CRITICAL: A gold standard letter that won a similar appeal is provided above.**
- Study how it structures arguments and presents clinical evidence
- Emulate its persuasive techniques and medical reasoning
- Adapt its successful patterns to this patient's specific situation
- Do NOT copy verbatim - adapt the approach with this patient's actual clinical data

"""
        else:
            gold_letter_section = "No similar winning rebuttal available. Use the Mercy template structure."
            gold_letter_instructions = ""

        # Build the prompt
        prompt = self.LETTER_PROMPT.format(
            denial_info_json=denial_info_json,
            discharge_summary=discharge_summary,
            hp_note=hp_note,
            structured_data=structured_data_str,
            criteria_json=criteria_json,
            gold_letter_section=gold_letter_section,
            gold_letter_instructions=gold_letter_instructions,
            patient_info_json=patient_info_json,
            current_date=current_date,
        )

        # Call LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a clinical coding expert writing DRG validation appeal letters. "
                        "Prioritize evidence from provider notes over structured data. "
                        "Be thorough, specific, and cite clinical values. "
                        "Return only valid JSON."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,  # Slightly creative for better writing
            max_tokens=4000
        )

        # Parse response
        raw_response = response.choices[0].message.content.strip()

        # Extract JSON
        json_str = raw_response
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()

        try:
            result = json.loads(json_str)
        except json.JSONDecodeError as e:
            return {
                "status": "error",
                "error": f"JSON parsing failed: {str(e)}",
                "raw_response": raw_response,
            }

        # Add generation metadata
        result["status"] = "success"
        result["_generation_metadata"] = {
            "model": self.model,
            "tokens_used": response.usage.total_tokens if response.usage else None,
            "evidence_priority": EVIDENCE_PRIORITY,
            "gold_letter_used": gold_letter.get("letter_id") if gold_letter else None,
        }

        return result

    def generate_letter_text(self, letter_json: Dict[str, Any]) -> str:
        """
        Convert structured letter JSON to plain text.

        Args:
            letter_json: The 'letter' portion of generate() output

        Returns:
            Formatted letter as plain text
        """
        letter = letter_json.get("letter", letter_json)

        parts = []

        # Header
        header = letter.get("mercy_header", {})
        parts.append(header.get("address", ""))
        parts.append("")
        parts.append(header.get("date", ""))
        parts.append("")

        # Payor address
        parts.append(letter.get("payor_address", ""))
        parts.append("")

        # Appeal level
        parts.append(letter.get("appeal_level", "First Level Appeal"))
        parts.append("")

        # Patient info
        patient = letter.get("patient_info", {})
        parts.append(f"Beneficiary Name: {patient.get('beneficiary_name', '')}")
        parts.append(f"DOB: {patient.get('dob', '')}")
        parts.append(f"Claim reference #: {patient.get('claim_reference', '')}")
        parts.append(f"Hospital Account #: {patient.get('hospital_account', '')}")
        parts.append(f"Date of Service: {patient.get('date_of_service', '')}")
        parts.append("")

        # Salutation
        parts.append(letter.get("salutation", ""))
        parts.append("")

        # Opening
        parts.append(letter.get("opening_paragraph", ""))
        parts.append("")

        # Justification
        parts.append("Justification for Appeal:")
        parts.append("")
        parts.append(letter.get("justification", ""))
        parts.append("")

        # Rationale
        parts.append("Rationale:")
        parts.append("")
        rationale = letter.get("rationale", {})
        if rationale.get("payor_argument_quote"):
            parts.append(rationale["payor_argument_quote"])
            parts.append("")

        # Clinical evidence sections
        if rationale.get("infection_source"):
            parts.append(f"Infection Source: {rationale['infection_source']}")
            parts.append("")

        for section_name, section_key in [
            ("Organ Dysfunction", "organ_dysfunction"),
            ("SIRS Criteria Met", "sirs_criteria"),
            ("Inflammatory Markers", "inflammatory_markers"),
        ]:
            section_data = rationale.get(section_key, [])
            if section_data:
                parts.append(f"{section_name}:")
                for item in section_data:
                    if isinstance(item, dict):
                        parts.append(f"  - {item.get('criterion', '')}: {item.get('value', '')} (threshold: {item.get('threshold', '')})")
                    else:
                        parts.append(f"  - {item}")
                parts.append("")

        if rationale.get("blood_culture"):
            parts.append(f"Blood Culture: {rationale['blood_culture']}")
            parts.append("")

        if rationale.get("other_conditions"):
            parts.append("Other Conditions:")
            for cond in rationale["other_conditions"]:
                parts.append(f"  - {cond}")
            parts.append("")

        # Hospital course
        parts.append("Hospital Course:")
        parts.append(letter.get("hospital_course", ""))
        parts.append("")

        # Summary
        parts.append(letter.get("summary_paragraph", ""))
        parts.append("")

        # Conclusion
        parts.append("Conclusion:")
        parts.append("")
        parts.append(letter.get("conclusion", ""))
        parts.append("")

        # Contact
        parts.append("Please direct all written correspondence to:")
        parts.append("")
        parts.append(letter.get("contact_info", ""))
        parts.append("")

        # Signature
        parts.append("Sincerely,")
        parts.append("")
        parts.append(letter.get("signature_block", ""))

        return "\n".join(parts)


# =============================================================================
# DATABRICKS NOTEBOOK USAGE
# =============================================================================
"""
# Cell: Generate a rebuttal letter
from agents.writer_agent import WriterAgent

writer = WriterAgent.from_databricks_secrets()

result = writer.generate(
    denial_info=parsed_denial,
    clinical_notes={
        "discharge_summary": row["discharge_summary_text"],
        "hp_note": row["hp_note_text"],
    },
    patient_data={
        "formatted_name": row["formatted_name"],
        "formatted_birthdate": row["formatted_birthdate"],
        "hsp_account_id": row["hsp_account_id"],
        "claim_number": row["claim_number"],
        "formatted_date_of_service": row["formatted_date_of_service"],
        "facility_name": row["facility_name"],
    },
    relevant_criteria=research_result.get("relevant_criteria"),
    gold_letter=reference_result.get("matched_letters", [None])[0],
)

if result["status"] == "success":
    letter_text = writer.generate_letter_text(result)
    print(letter_text)
"""
