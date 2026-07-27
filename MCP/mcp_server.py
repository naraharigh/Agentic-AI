from mcp.server.fastmcp import FastMCP

# Initialize the MCP Server
mcp = FastMCP("Log Analysis Server")

# Expose a tool that the AI model can call
@mcp.tool()
def classify_log_entry(log_line: str) -> str:
    """Analyzes a log line and assigns a criticality severity level."""
    if "FATAL" in log_line or "CRITICAL" in log_line:
        return "SEV-1: High Criticality"
    elif "ERROR" in log_line:
        return "SEV-2: Medium Criticality"
    return "SEV-3: Info/Low"

# Expose a read-only resource
@mcp.resource("config://app-settings")
def get_config() -> str:
    """Exposes application context to the LLM."""
    return "Environment: Production | Region: us-east-1 | Version: 2.4.0"

if __name__ == "__main__":
    mcp.run(transport="stdio")