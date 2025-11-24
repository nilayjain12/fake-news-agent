# backend/mcp/__init__.py
"""MCP (Model Context Protocol) integration for Fact-Checking Agent.

This module provides a custom MCP server implementation that doesn't conflict
with the external 'mcp' library.
"""

from mcp.server import FactCheckingMCPServer

__all__ = ["FactCheckingMCPServer"]

__version__ = "1.0.0"