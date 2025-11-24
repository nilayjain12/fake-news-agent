# backend/mcp/server.py
"""Custom MCP Server for Fact-Checking Agent (avoiding conflicts with mcp library)."""
from typing import Any, Optional, Callable, Dict, List
from config import get_logger

logger = get_logger(__name__)

class FactCheckingMCPServer:
    """Custom MCP-compatible server for fact-checking tools."""
    
    def __init__(self):
        self.tools: Dict[str, dict] = {}
        self.resources: Dict[str, dict] = {}
        logger.warning("🖥️  Fact-Checking MCP Server initialized")
    
    def register_tool(self, name: str, description: str, 
                     input_schema: dict, handler: Callable) -> None:
        """Register a tool with the MCP server."""
        self.tools[name] = {
            "name": name,
            "description": description,
            "input_schema": input_schema,
            "handler": handler
        }
        logger.warning(f"📍 MCP Tool registered: {name}")
    
    def list_tools(self) -> List[dict]:
        """Return list of available tools."""
        return [
            {
                "name": name,
                "description": data["description"],
                "input_schema": data["input_schema"]
            }
            for name, data in self.tools.items()
        ]
    
    def get_tool(self, tool_name: str) -> Optional[dict]:
        """Get a specific tool by name."""
        return self.tools.get(tool_name)
    
    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Call a tool synchronously (non-async version for Gradio compatibility)."""
        if tool_name not in self.tools:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found",
                "available_tools": list(self.tools.keys())
            }
        
        tool_info = self.tools[tool_name]
        handler = tool_info["handler"]
        
        try:
            logger.warning(f"🔨 Calling MCP tool: {tool_name} with args: {list(arguments.keys())}")
            result = handler(**arguments)
            return {
                "success": True,
                "tool": tool_name,
                "result": result
            }
        except Exception as e:
            logger.warning(f"❌ Tool execution failed: {str(e)[:100]}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "tool": tool_name,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def get_mcp_schema(self) -> dict:
        """Return full MCP schema for client discovery."""
        return {
            "version": "1.0",
            "server_name": "fake-news-detection-mcp",
            "tools": self.list_tools(),
            "resources": list(self.resources.keys())
        }
    
    def to_json(self) -> str:
        """Export schema as JSON."""
        import json
        return json.dumps(self.get_mcp_schema(), indent=2)