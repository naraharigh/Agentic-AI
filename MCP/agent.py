# agent.py
import asyncio
import os
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_groq import ChatGroq
from langchain.agents import create_agent  # Updated import to address warning

async def main():
    server_path = os.path.abspath("mcp_server.py")

    client = MultiServerMCPClient({
        "system_tools": {
            "transport": "stdio",
            "command": "python",
            "args": [server_path]
        }
    })

    tools = await client.get_tools()
    print(f"✅ Discovered {len(tools)} tool(s) from MCP Server:")

    # Switched to ChatGroq
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    # Note: Address deprecation by using updated agent setup
    from langgraph.prebuilt import create_react_agent
    agent = create_react_agent(llm, tools)

    query = (
        "Please calculate 345 + 678. "
        "Also, analyze this log entry: 'FATAL: Neo4j database connection refused on port 7687'."
    )
    
    response = await agent.ainvoke({"messages": [("user", query)]})
    print(f"\n🤖 Response:\n{response['messages'][-1].content}")

if __name__ == "__main__":
    asyncio.run(main())