# poc/document_reader.py
# Document reading using Azure AI Document Intelligence
#
# Handles: PDF, DOCX, images (PNG, JPG, TIFF), and .txt files
# .txt files use a simple wrapper since Document Intelligence doesn't support them.

import os
from typing import Optional

# =============================================================================
# DOCUMENT READER CLASS
# =============================================================================

class DocumentReader:
    """
    Reads documents using Azure AI Document Intelligence.
    Falls back to simple text reading for .txt files.

    Usage (Databricks):
        reader = DocumentReader.from_databricks_secrets()
        text = reader.read_document("/path/to/denial_letter.pdf")

    Usage (Local with env vars):
        reader = DocumentReader.from_env()
        text = reader.read_document("/path/to/denial_letter.pdf")
    """

    def __init__(self, endpoint: str, api_key: str):
        """
        Initialize with Azure AI Document Intelligence credentials.

        Args:
            endpoint: Azure Document Intelligence endpoint URL
            api_key: Azure Document Intelligence API key
        """
        self.endpoint = endpoint
        self.api_key = api_key
        self._client = None

    @classmethod
    def from_databricks_secrets(cls, scope: str = "idp_etl",
                                  endpoint_key: str = "az-doc-intelligence-endpoint",
                                  api_key_key: str = "az-doc-intelligence-key"):
        """
        Create reader from Databricks secrets.

        Args:
            scope: Databricks secret scope
            endpoint_key: Secret key for endpoint URL
            api_key_key: Secret key for API key
        """
        # This import only works in Databricks
        from pyspark.dbutils import DBUtils
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        dbutils = DBUtils(spark)

        endpoint = dbutils.secrets.get(scope=scope, key=endpoint_key)
        api_key = dbutils.secrets.get(scope=scope, key=api_key_key)

        return cls(endpoint=endpoint, api_key=api_key)

    @classmethod
    def from_env(cls):
        """
        Create reader from environment variables.

        Expected env vars:
            AZURE_DOC_INTELLIGENCE_ENDPOINT
            AZURE_DOC_INTELLIGENCE_KEY
        """
        endpoint = os.environ.get("AZURE_DOC_INTELLIGENCE_ENDPOINT")
        api_key = os.environ.get("AZURE_DOC_INTELLIGENCE_KEY")

        if not endpoint or not api_key:
            raise ValueError(
                "Missing environment variables. Set:\n"
                "  AZURE_DOC_INTELLIGENCE_ENDPOINT\n"
                "  AZURE_DOC_INTELLIGENCE_KEY"
            )

        return cls(endpoint=endpoint, api_key=api_key)

    @property
    def client(self):
        """Lazy initialization of Document Intelligence client."""
        if self._client is None:
            from azure.ai.documentintelligence import DocumentIntelligenceClient
            from azure.core.credentials import AzureKeyCredential

            self._client = DocumentIntelligenceClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.api_key)
            )
        return self._client

    def read_document(self, file_path: str) -> str:
        """
        Read a document and extract text content.

        Args:
            file_path: Path to the document (PDF, DOCX, image, or TXT)

        Returns:
            Extracted text content as a string
        """
        # Get file extension
        _, ext = os.path.splitext(file_path.lower())

        # .txt files: simple read (Document Intelligence doesn't support them)
        if ext == ".txt":
            return self._read_txt(file_path)

        # All other formats: use Azure AI Document Intelligence
        return self._read_with_doc_intelligence(file_path)

    def _read_txt(self, file_path: str) -> str:
        """Read plain text file directly."""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _read_with_doc_intelligence(self, file_path: str) -> str:
        """
        Read document using Azure AI Document Intelligence.

        Uses the 'prebuilt-read' model which extracts text from:
        - PDFs
        - Images (JPEG, PNG, BMP, TIFF, HEIF)
        - Microsoft Office files (DOCX, XLSX, PPTX)
        """
        from azure.ai.documentintelligence.models import AnalyzeDocumentRequest

        # Read file bytes
        with open(file_path, "rb") as f:
            document_bytes = f.read()

        # Analyze document
        poller = self.client.begin_analyze_document(
            model_id="prebuilt-read",  # General text extraction model
            body=AnalyzeDocumentRequest(bytes_source=document_bytes),
        )

        result = poller.result()

        # Extract text from all pages
        text_parts = []
        for page in result.pages:
            for line in page.lines:
                text_parts.append(line.content)

        return "\n".join(text_parts)

    def read_document_from_bytes(self, document_bytes: bytes,
                                   file_type: str = "pdf") -> str:
        """
        Read document from bytes (useful for Databricks file system).

        Args:
            document_bytes: Raw bytes of the document
            file_type: File type hint ("pdf", "docx", "png", etc.)

        Returns:
            Extracted text content
        """
        # .txt: decode directly
        if file_type.lower() == "txt":
            return document_bytes.decode("utf-8")

        # All other formats: use Document Intelligence
        from azure.ai.documentintelligence.models import AnalyzeDocumentRequest

        poller = self.client.begin_analyze_document(
            model_id="prebuilt-read",
            body=AnalyzeDocumentRequest(bytes_source=document_bytes),
        )

        result = poller.result()

        text_parts = []
        for page in result.pages:
            for line in page.lines:
                text_parts.append(line.content)

        return "\n".join(text_parts)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def read_denial_letter(file_path: str, reader: Optional[DocumentReader] = None) -> str:
    """
    Read a denial letter from file.

    Args:
        file_path: Path to the denial letter
        reader: Optional DocumentReader instance. If not provided,
                creates one from Databricks secrets.

    Returns:
        Extracted text from the denial letter
    """
    if reader is None:
        try:
            reader = DocumentReader.from_databricks_secrets()
        except Exception:
            # Fall back to environment variables if not in Databricks
            reader = DocumentReader.from_env()

    return reader.read_document(file_path)


# =============================================================================
# DATABRICKS NOTEBOOK USAGE
# =============================================================================
"""
# Cell 1: Install dependencies (run once)
%pip install azure-ai-documentintelligence azure-core

# Cell 2: Read a denial letter
from document_reader import DocumentReader

reader = DocumentReader.from_databricks_secrets()
denial_text = reader.read_document("/Workspace/path/to/denial_letter.pdf")
print(denial_text[:500])  # Preview first 500 chars
"""
