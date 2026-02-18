# poc/pipeline.py
# Rebuttal Engine Pipeline: Orchestrates the four-agent workflow
#
# Pipeline: Parser → Research → Reference → Writer
#
# Each agent's output feeds into the next.

import os
import sys
from typing import Dict, Any, Optional
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import SCOPE_FILTER
from document_reader import DocumentReader
from agents.parser_agent import ParserAgent
from agents.research_agent import ResearchAgent
from agents.reference_agent import ReferenceAgent
from agents.writer_agent import WriterAgent


class RebuttalPipeline:
    """
    Orchestrates the four-agent rebuttal letter generation pipeline.

    Pipeline:
    1. Parser Agent - Extract denial info from letter (LLM-based)
    2. Research Agent - Find relevant criteria (stub until Propel loaded)
    3. Reference Agent - Find gold standard letters (stub until letters loaded)
    4. Writer Agent - Generate the rebuttal letter

    Usage:
        pipeline = RebuttalPipeline.from_databricks_secrets()
        result = pipeline.process(
            denial_letter_path="/path/to/denial.pdf",
            clinical_notes={"discharge_summary": "...", "hp_note": "..."},
            patient_data={"formatted_name": "...", ...}
        )
    """

    def __init__(self,
                 document_reader: DocumentReader,
                 parser: ParserAgent,
                 research: ResearchAgent,
                 reference: ReferenceAgent,
                 writer: WriterAgent):
        """
        Initialize pipeline with all agents.

        Args:
            document_reader: For reading denial letters
            parser: Parser Agent
            research: Research Agent
            reference: Reference Agent
            writer: Writer Agent
        """
        self.document_reader = document_reader
        self.parser = parser
        self.research = research
        self.reference = reference
        self.writer = writer

    @classmethod
    def from_databricks_secrets(cls, catalog: str = "dev"):
        """Create pipeline from Databricks secrets."""
        document_reader = DocumentReader.from_databricks_secrets()
        parser = ParserAgent.from_databricks_secrets()
        research = ResearchAgent.from_databricks(catalog=catalog)
        reference = ReferenceAgent.from_databricks_secrets(catalog=catalog)
        writer = WriterAgent.from_databricks_secrets()

        return cls(
            document_reader=document_reader,
            parser=parser,
            research=research,
            reference=reference,
            writer=writer,
        )

    @classmethod
    def from_env(cls, catalog: str = "dev"):
        """Create pipeline from environment variables."""
        document_reader = DocumentReader.from_env()
        parser = ParserAgent.from_env()
        research = ResearchAgent.from_databricks(catalog=catalog)
        reference = ReferenceAgent.from_env(catalog=catalog)
        writer = WriterAgent.from_env()

        return cls(
            document_reader=document_reader,
            parser=parser,
            research=research,
            reference=reference,
            writer=writer,
        )

    def process(self,
                denial_letter_path: Optional[str] = None,
                denial_letter_text: Optional[str] = None,
                clinical_notes: Dict[str, str] = None,
                patient_data: Dict[str, Any] = None,
                structured_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a denial and generate a rebuttal letter.

        Args:
            denial_letter_path: Path to denial letter file (PDF, DOCX, etc.)
            denial_letter_text: Or provide raw text directly
            clinical_notes: Dict with 'discharge_summary' and 'hp_note'
            patient_data: Patient demographics and admin info
            structured_data: Optional structured clinical data

        Returns:
            Dictionary with:
            - status: "success" | "out_of_scope" | "error"
            - letter: Generated letter (if success)
            - letter_text: Plain text version (if success)
            - pipeline_results: Results from each agent
            - _pipeline_metadata: Timing and debug info
        """
        start_time = datetime.now()
        pipeline_results = {}

        # =====================================================================
        # Step 0: Read denial letter if path provided
        # =====================================================================
        if denial_letter_path and not denial_letter_text:
            try:
                denial_letter_text = self.document_reader.read_document(denial_letter_path)
                pipeline_results["document_read"] = {
                    "status": "success",
                    "path": denial_letter_path,
                    "text_length": len(denial_letter_text),
                }
            except Exception as e:
                return {
                    "status": "error",
                    "error": f"Failed to read denial letter: {str(e)}",
                    "stage": "document_read",
                }

        if not denial_letter_text:
            return {
                "status": "error",
                "error": "No denial letter provided (path or text)",
                "stage": "input_validation",
            }

        # =====================================================================
        # Step 1: Parse denial letter
        # =====================================================================
        parse_result = self.parser.parse_with_scope_check(denial_letter_text)
        pipeline_results["parser"] = parse_result

        if parse_result["status"] == "out_of_scope":
            return {
                "status": "out_of_scope",
                "reason": parse_result["reason"],
                "denial_info": parse_result.get("denial_info"),
                "pipeline_results": pipeline_results,
                "_pipeline_metadata": self._get_metadata(start_time, "parser"),
            }

        if parse_result["status"] == "error":
            return {
                "status": "error",
                "error": parse_result["reason"],
                "stage": "parser",
                "pipeline_results": pipeline_results,
            }

        denial_info = parse_result["denial_info"]

        # =====================================================================
        # Step 2: Research relevant criteria
        # =====================================================================
        research_result = self.research.search(
            denial_info=denial_info,
            clinical_notes=clinical_notes.get("discharge_summary") if clinical_notes else None,
        )
        pipeline_results["research"] = research_result

        # =====================================================================
        # Step 3: Find reference letters (vector search against gold standards)
        # =====================================================================
        reference_result = self.reference.find_matches(
            denial_text=denial_letter_text,
            denial_info=denial_info,
        )
        pipeline_results["reference"] = reference_result

        # Extract best match in format expected by WriterAgent
        gold_letter = None
        if not reference_result.get("use_template_fallback") and reference_result.get("best_match"):
            best = reference_result["best_match"]
            gold_letter = {
                "letter_id": best["letter_id"],
                "letter_text": best["rebuttal_text"],
                "match_score": best["match_score"],
                "source_file": best["source_file"],
            }

        # =====================================================================
        # Step 4: Generate rebuttal letter
        # =====================================================================
        if not clinical_notes:
            clinical_notes = {}
        if not patient_data:
            patient_data = {}

        writer_result = self.writer.generate(
            denial_info=denial_info,
            clinical_notes=clinical_notes,
            patient_data=patient_data,
            relevant_criteria=research_result.get("relevant_criteria"),
            gold_letter=gold_letter,
            structured_data=structured_data,
        )
        pipeline_results["writer"] = writer_result

        if writer_result["status"] == "error":
            return {
                "status": "error",
                "error": writer_result.get("error", "Unknown error in writer"),
                "stage": "writer",
                "pipeline_results": pipeline_results,
            }

        # =====================================================================
        # Generate plain text version
        # =====================================================================
        letter_text = self.writer.generate_letter_text(writer_result)

        return {
            "status": "success",
            "letter": writer_result.get("letter"),
            "letter_text": letter_text,
            "evidence_sources": writer_result.get("evidence_sources"),
            "confidence_score": writer_result.get("confidence_score"),
            "pipeline_results": pipeline_results,
            "_pipeline_metadata": self._get_metadata(start_time, "complete"),
        }

    def _get_metadata(self, start_time: datetime, stage: str) -> Dict[str, Any]:
        """Generate pipeline metadata."""
        end_time = datetime.now()
        return {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "completed_stage": stage,
            "scope_filter": SCOPE_FILTER,
        }


# =============================================================================
# DATABRICKS NOTEBOOK USAGE
# =============================================================================
"""
# Cell 1: Install dependencies
%pip install azure-ai-documentintelligence azure-core openai

# Cell 2: Run the pipeline
from pipeline import RebuttalPipeline

# Create pipeline
pipeline = RebuttalPipeline.from_databricks_secrets(catalog="dev")

# Process a denial
result = pipeline.process(
    denial_letter_path="/Workspace/path/to/denial.pdf",
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
        "number_of_midnights": row["number_of_midnights"],
    }
)

# Check result
if result["status"] == "success":
    print("=== GENERATED LETTER ===")
    print(result["letter_text"])
    print(f"\\nConfidence: {result['confidence_score']}")
elif result["status"] == "out_of_scope":
    print(f"Skipped: {result['reason']}")
else:
    print(f"Error: {result['error']}")
"""
