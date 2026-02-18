# poc/agents/reference_agent.py
# Reference Agent: Finds best matching gold standard rebuttal letters
#
# Uses vector search (cosine similarity) to find the most similar past denial,
# then returns that case's successful rebuttal as context for the Writer Agent.

from typing import Dict, Any, List, Optional
import os
import sys
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MATCH_SCORE_THRESHOLD, VECTOR_SEARCH_TOP_K


class ReferenceAgent:
    """
    Searches gold standard letter library for best matching rebuttals.

    Retrieval strategy:
    1. Embed the incoming denial text
    2. Compute cosine similarity against all stored denial embeddings
    3. Return the best match's successful rebuttal
    4. Fallback to template if no good match (score < threshold)
    """

    def __init__(self, openai_client, catalog: str = "dev",
                 template_path: Optional[str] = None,
                 embedding_model: str = "text-embedding-ada-002"):
        """
        Initialize Reference Agent.

        Args:
            openai_client: AzureOpenAI client instance for embeddings
            catalog: Databricks catalog (dev, test, prod)
            template_path: Path to fallback template (Mercy sepsis template)
            embedding_model: Model to use for embeddings
        """
        self.openai_client = openai_client
        self.catalog = catalog
        self.table_name = f"{catalog}.fin_ds.fudgesicle_gold_letters"
        self.template_path = template_path or "utils/sepsis_template.docx"
        self.match_threshold = MATCH_SCORE_THRESHOLD
        self.top_k = VECTOR_SEARCH_TOP_K
        self.embedding_model = embedding_model
        self._gold_letters_cache = None

    @classmethod
    def from_databricks_secrets(cls, catalog: str = "dev",
                                 scope: str = "idp_etl",
                                 api_key_secret: str = "az-openai-key1",
                                 endpoint_secret: str = "az-openai-base"):
        """Create agent from Databricks secrets."""
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

        return cls(openai_client=client, catalog=catalog)

    @classmethod
    def from_env(cls, catalog: str = "dev"):
        """Create agent from environment variables."""
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

        return cls(openai_client=client, catalog=catalog)

    def _load_gold_letters(self) -> List[Dict[str, Any]]:
        """Load gold letters from Delta table into memory."""
        if self._gold_letters_cache is not None:
            return self._gold_letters_cache

        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()

        # Check if table exists and has data
        try:
            df = spark.sql(f"""
                SELECT letter_id, source_file, denial_text, rebuttal_text,
                       denial_embedding, metadata
                FROM {self.table_name}
            """)
            rows = df.collect()
        except Exception as e:
            print(f"Warning: Could not load gold letters table: {e}")
            self._gold_letters_cache = []
            return []

        self._gold_letters_cache = [
            {
                "letter_id": row["letter_id"],
                "source_file": row["source_file"],
                "denial_text": row["denial_text"],
                "rebuttal_text": row["rebuttal_text"],
                "denial_embedding": list(row["denial_embedding"]) if row["denial_embedding"] else None,
                "metadata": dict(row["metadata"]) if row["metadata"] else {},
            }
            for row in rows
        ]

        print(f"Loaded {len(self._gold_letters_cache)} gold standard letters")
        return self._gold_letters_cache

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Azure OpenAI."""
        # Truncate if too long
        if len(text) > 30000:
            text = text[:30000]

        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text
        )

        return response.data[0].embedding

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not vec1 or not vec2:
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def find_matches(self, denial_text: str,
                     denial_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Find best matching gold standard letters for a denial.

        Args:
            denial_text: The full text of the denial letter to match
            denial_info: Optional parsed denial info (for logging/debugging)

        Returns:
            Dictionary with:
            - matched_letters: List of matching letters with scores (best first)
            - best_match: The top matching rebuttal (convenience accessor)
            - use_template_fallback: Whether to use template instead
            - fallback_reason: Why fallback is needed (if applicable)
            - _search_metadata: Debug info
        """
        # Load gold letters
        gold_letters = self._load_gold_letters()

        if not gold_letters:
            return {
                "matched_letters": [],
                "best_match": None,
                "use_template_fallback": True,
                "fallback_reason": "No gold standard letters in table. Using template.",
                "template_path": self.template_path,
                "_search_metadata": {
                    "status": "empty_table",
                    "table": self.table_name,
                    "letters_checked": 0,
                }
            }

        # Generate embedding for the incoming denial
        query_embedding = self._generate_embedding(denial_text)

        # Compute similarity against all gold letters
        scored_matches = []
        for letter in gold_letters:
            if letter["denial_embedding"]:
                similarity = self._cosine_similarity(query_embedding, letter["denial_embedding"])
                scored_matches.append({
                    "letter_id": letter["letter_id"],
                    "source_file": letter["source_file"],
                    "match_score": similarity,
                    "rebuttal_text": letter["rebuttal_text"],
                    "denial_text": letter["denial_text"],
                    "metadata": letter["metadata"],
                })

        # Sort by similarity (highest first)
        scored_matches.sort(key=lambda x: x["match_score"], reverse=True)

        # Take top_k
        top_matches = scored_matches[:self.top_k]

        # Determine if best match is good enough
        best_match = top_matches[0] if top_matches else None
        use_fallback = best_match is None or best_match["match_score"] < self.match_threshold

        return {
            "matched_letters": top_matches,
            "best_match": best_match if not use_fallback else None,
            "use_template_fallback": use_fallback,
            "fallback_reason": self._get_fallback_reason(best_match),
            "template_path": self.template_path if use_fallback else None,
            "_search_metadata": {
                "status": "searched",
                "table": self.table_name,
                "letters_checked": len(gold_letters),
                "top_score": best_match["match_score"] if best_match else 0.0,
                "threshold": self.match_threshold,
            }
        }

    def _get_fallback_reason(self, best_match: Optional[Dict[str, Any]]) -> Optional[str]:
        """Get human-readable reason for fallback."""
        if best_match is None:
            return "No gold letters have embeddings"
        if best_match["match_score"] < self.match_threshold:
            return f"Best match score ({best_match['match_score']:.2f}) below threshold ({self.match_threshold})"
        return None

    def get_gold_letter_for_writer(self, denial_text: str,
                                    denial_info: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Convenience method: Get the best gold letter in format expected by WriterAgent.

        Args:
            denial_text: The denial letter text to match
            denial_info: Optional parsed denial info

        Returns:
            Dict with 'letter_id' and 'letter_text' keys, or None if fallback
        """
        result = self.find_matches(denial_text, denial_info)

        if result["use_template_fallback"]:
            return None

        best = result["best_match"]
        return {
            "letter_id": best["letter_id"],
            "letter_text": best["rebuttal_text"],
            "match_score": best["match_score"],
            "source_file": best["source_file"],
        }
