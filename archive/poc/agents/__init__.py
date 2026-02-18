# poc/agents/__init__.py
# Agent modules for Rebuttal Engine v2

from .parser_agent import ParserAgent
from .research_agent import ResearchAgent
from .reference_agent import ReferenceAgent
from .writer_agent import WriterAgent

__all__ = [
    "ParserAgent",
    "ResearchAgent",
    "ReferenceAgent",
    "WriterAgent",
]
