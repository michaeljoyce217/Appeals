# poc/agents/research_agent.py
# Research Agent: Finds relevant clinical criteria from Propel/reference documents
#
# STUB: Returns empty results until Propel documents are ingested.
# When ready, this will use hybrid retrieval (structured filter + vector search).

from typing import Dict, Any, List, Optional


class ResearchAgent:
    """
    Searches reference documents (Propel, CMS, etc.) for relevant clinical criteria.

    Current status: STUB - returns empty results.
    Future: Will use hybrid retrieval from fudgesicle_reference_documents table.

    Retrieval strategy:
    1. Structured filter: WHERE tags contains relevant terms (sepsis, DRG, etc.)
    2. Vector search: Semantic similarity within filtered set
    3. LLM synthesis: Extract relevant criteria from top matches
    """

    def __init__(self, catalog: str = "dev"):
        """
        Initialize Research Agent.

        Args:
            catalog: Databricks catalog (dev, test, prod)
        """
        self.catalog = catalog
        self.table_name = f"{catalog}.fin_ds.fudgesicle_reference_documents"

    @classmethod
    def from_databricks(cls, catalog: str = "dev"):
        """Create agent configured for Databricks."""
        return cls(catalog=catalog)

    def search(self, denial_info: Dict[str, Any],
               clinical_notes: Optional[str] = None) -> Dict[str, Any]:
        """
        Search for relevant clinical criteria.

        Args:
            denial_info: Parsed denial information from ParserAgent
            clinical_notes: Optional clinical notes for context

        Returns:
            Dictionary with:
            - relevant_criteria: List of matching criteria documents
            - suggested_arguments: List of suggested arguments based on criteria
            - _search_metadata: Debug info
        """
        # STUB: Return empty results until Propel docs are loaded
        return {
            "relevant_criteria": [],
            "suggested_arguments": [],
            "_search_metadata": {
                "status": "stub",
                "message": "Research Agent is a stub. No reference documents loaded yet.",
                "table": self.table_name,
                "denial_type": self._get_denial_types(denial_info),
            }
        }

    def _get_denial_types(self, denial_info: Dict[str, Any]) -> List[str]:
        """Extract denial types from parsed denial info."""
        denial_reasons = denial_info.get("denial_reasons", [])
        return [r.get("type", "unknown") for r in denial_reasons]

    def _build_search_query(self, denial_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build search query from denial info.

        Future implementation will use this to construct:
        - Structured filters (tags, DRGs, ICD codes)
        - Vector search query text
        """
        return {
            "tags_filter": self._get_search_tags(denial_info),
            "query_text": self._get_query_text(denial_info),
        }

    def _get_search_tags(self, denial_info: Dict[str, Any]) -> List[str]:
        """Generate tags for structured filtering."""
        tags = []

        # Add DRG codes
        original_drg = denial_info.get("original_drg")
        proposed_drg = denial_info.get("proposed_drg")
        if original_drg:
            tags.append(f"drg_{original_drg}")
        if proposed_drg:
            tags.append(f"drg_{proposed_drg}")

        # Add sepsis if relevant
        if denial_info.get("is_sepsis_related"):
            tags.extend(["sepsis", "severe_sepsis", "septic_shock"])

        # Add denial types
        for reason in denial_info.get("denial_reasons", []):
            tags.append(reason.get("type", "unknown"))

        return tags

    def _get_query_text(self, denial_info: Dict[str, Any]) -> str:
        """Generate query text for vector search."""
        parts = []

        # Add denial reason summaries
        for reason in denial_info.get("denial_reasons", []):
            if reason.get("summary"):
                parts.append(reason["summary"])
            for arg in reason.get("specific_arguments", []):
                parts.append(arg)

        return " ".join(parts)


# =============================================================================
# FUTURE IMPLEMENTATION (when Propel docs are loaded)
# =============================================================================
"""
def search(self, denial_info, clinical_notes=None):
    # 1. Build search parameters
    search_query = self._build_search_query(denial_info)

    # 2. Structured filter
    filtered_docs = spark.sql(f'''
        SELECT doc_id, section_title, section_text, embedding_vector
        FROM {self.table_name}
        WHERE array_contains(tags, 'sepsis')  -- or other relevant tags
    ''')

    # 3. Vector search within filtered results
    query_embedding = get_embedding(search_query["query_text"])
    top_matches = vector_similarity_search(filtered_docs, query_embedding, top_k=10)

    # 4. LLM synthesis - extract relevant criteria
    criteria = []
    for match in top_matches:
        criteria.append({
            "doc_id": match.doc_id,
            "section": match.section_title,
            "criteria_text": match.section_text,
            "relevance_score": match.similarity_score,
        })

    # 5. Generate suggested arguments
    suggested_arguments = self._generate_arguments(criteria, clinical_notes)

    return {
        "relevant_criteria": criteria,
        "suggested_arguments": suggested_arguments,
    }
"""
