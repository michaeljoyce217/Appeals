# poc/agents/parser_agent.py
# Parser Agent: Extracts structured denial information from denial letters using LLM
#
# Key principle: Use LLM for extraction, NOT regex.
# LLM handles variations like "Claim Ref #", "Reference Number", "Claim ID", etc.

import json
import os
from typing import Dict, Any, Optional

# Import config for scope filter
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SCOPE_FILTER, SEPSIS_DRG_CODES, SEPSIS_ICD10_CODES


class ParserAgent:
    """
    Parses denial letters to extract structured information.

    Uses LLM (Azure OpenAI GPT-4.1) to extract:
    - Denial date, payer, reviewer
    - Original and proposed DRGs
    - All administrative identifiers (claim ref, member ID, auth number, etc.)
    - Denial reasons with specific arguments
    - Sepsis-related flag for scope filtering

    Usage:
        parser = ParserAgent.from_databricks_secrets()
        result = parser.parse(denial_letter_text)
    """

    EXTRACTION_PROMPT = '''You are a medical billing expert extracting information from denial letters.

# Task
Extract ALL relevant information from this denial letter into structured JSON.

# Denial Letter
{denial_letter_text}

# Output Format
Return ONLY valid JSON with this structure (no markdown, no explanation):
{{
  "denial_date": "YYYY-MM-DD or null if not found",
  "payer_name": "Insurance company name",
  "payer_address": "Full mailing address if present, else null",
  "reviewer_name": "Name and credentials of reviewer, else null",
  "original_drg": "The DRG the hospital billed (e.g., '871')",
  "proposed_drg": "The DRG the payer wants to change to (e.g., '872')",

  "administrative_data": {{
    "claim_reference_number": "Primary claim/reference number",
    "member_id": "Patient member/subscriber ID",
    "authorization_number": "Prior auth number if present",
    "date_of_service": "Admission date or date range",
    "date_of_discharge": "Discharge date if present",
    "facility_name": "Hospital name if mentioned",
    "patient_name": "Patient name if present",
    "patient_dob": "Patient DOB if present",
    "other_identifiers": {{
      "key": "value for any other IDs found (case number, appeal ID, etc.)"
    }}
  }},

  "denial_reasons": [
    {{
      "type": "clinical_validation | medical_necessity | level_of_care | coding | other",
      "summary": "Brief summary of this denial reason",
      "specific_arguments": [
        "Exact quote or paraphrase of each specific argument made by the payer"
      ],
      "payer_quote": "Direct quote from the letter if available"
    }}
  ],

  "is_sepsis_related": true/false,
  "is_single_issue": true/false,
  "sepsis_indicators": ["List any sepsis-related terms found: sepsis, septic, A41, R65, DRG 870/871/872, etc."]
}}

# Instructions
1. Extract ALL administrative identifiers you can find - claim numbers, member IDs, auth numbers, case numbers, etc.
2. For denial_reasons, identify EACH distinct reason given for the denial
3. Set is_sepsis_related to true if the denial involves sepsis, severe sepsis, septic shock, DRG 870/871/872, or ICD codes A41.x/R65.x
4. Set is_single_issue to true if there is only ONE denial reason, false if multiple
5. Use null for any field you cannot find
6. Return ONLY the JSON, no other text'''

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
        """Create parser from Databricks secrets."""
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
        """Create parser from environment variables."""
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

    def parse(self, denial_letter_text: str) -> Dict[str, Any]:
        """
        Parse a denial letter and extract structured information.

        Args:
            denial_letter_text: Raw text from the denial letter

        Returns:
            Dictionary with extracted denial information including:
            - denial_date, payer_name, reviewer_name
            - original_drg, proposed_drg
            - administrative_data (all identifiers found)
            - denial_reasons (list of reasons with arguments)
            - is_sepsis_related, is_single_issue
            - _parsing_metadata (for debugging)
        """
        # Build the prompt
        prompt = self.EXTRACTION_PROMPT.format(denial_letter_text=denial_letter_text)

        # Call LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical billing expert. Extract information accurately. Return only valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,  # Deterministic extraction
            max_tokens=2000
        )

        # Parse response
        raw_response = response.choices[0].message.content.strip()

        # Try to extract JSON (handle potential markdown code blocks)
        json_str = raw_response
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()

        try:
            result = json.loads(json_str)
        except json.JSONDecodeError as e:
            # Return error info if JSON parsing fails
            return {
                "error": f"JSON parsing failed: {str(e)}",
                "raw_response": raw_response,
                "is_sepsis_related": None,
                "is_single_issue": None,
            }

        # Add parsing metadata
        result["_parsing_metadata"] = {
            "model": self.model,
            "tokens_used": response.usage.total_tokens if response.usage else None,
            "scope_filter": SCOPE_FILTER,
        }

        return result

    def parse_with_scope_check(self, denial_letter_text: str) -> Dict[str, Any]:
        """
        Parse denial letter and apply scope filter.

        If SCOPE_FILTER is "sepsis" and the denial is not sepsis-related,
        returns early with status "out_of_scope".

        Args:
            denial_letter_text: Raw text from the denial letter

        Returns:
            Dictionary with:
            - status: "success" | "out_of_scope" | "error"
            - denial_info: Extracted information (if success)
            - reason: Explanation (if out_of_scope or error)
        """
        # Parse the denial letter
        result = self.parse(denial_letter_text)

        # Check for parsing errors
        if "error" in result:
            return {
                "status": "error",
                "reason": result["error"],
                "raw_response": result.get("raw_response"),
            }

        # Apply scope filter
        if SCOPE_FILTER == "sepsis":
            if not result.get("is_sepsis_related", False):
                return {
                    "status": "out_of_scope",
                    "reason": "Denial is not sepsis-related (scope_filter='sepsis')",
                    "denial_info": result,  # Still include for debugging
                }

            # Validate DRG codes if present
            original_drg = result.get("original_drg", "")
            proposed_drg = result.get("proposed_drg", "")
            if original_drg and original_drg not in SEPSIS_DRG_CODES:
                # Log warning but don't reject - LLM might have found sepsis indicators
                result["_parsing_metadata"]["drg_warning"] = (
                    f"original_drg '{original_drg}' not in SEPSIS_DRG_CODES"
                )

        # Check for multi-issue denials (warn but don't reject in POC)
        if not result.get("is_single_issue", True):
            result["_parsing_metadata"]["multi_issue_warning"] = (
                "Multiple denial reasons found. POC processes primary issue only."
            )

        return {
            "status": "success",
            "denial_info": result,
        }


# =============================================================================
# DATABRICKS NOTEBOOK USAGE
# =============================================================================
"""
# Cell: Parse a denial letter
from agents.parser_agent import ParserAgent
from document_reader import DocumentReader

# Read the denial letter
reader = DocumentReader.from_databricks_secrets()
denial_text = reader.read_document("/path/to/denial.pdf")

# Parse it
parser = ParserAgent.from_databricks_secrets()
result = parser.parse_with_scope_check(denial_text)

if result["status"] == "success":
    denial_info = result["denial_info"]
    print(f"Payer: {denial_info['payer_name']}")
    print(f"Original DRG: {denial_info['original_drg']}")
    print(f"Proposed DRG: {denial_info['proposed_drg']}")
    print(f"Claim Ref: {denial_info['administrative_data']['claim_reference_number']}")
else:
    print(f"Status: {result['status']}")
    print(f"Reason: {result['reason']}")
"""
