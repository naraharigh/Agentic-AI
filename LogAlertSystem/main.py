import os
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import List, Dict
import requests
import re
from langchain.tools import tool
from pydantic import BaseModel, Field


class ClassificationOutput(BaseModel):
    category: str = Field(description="Must be INFO, WARNING, or CRITICAL")
    reason: str = Field(description="Brief justification")

class GuardrailVerifier:
    """Validates structural integrity and guards against prompt injection / invalid output."""
    
    ALLOWED_CATEGORIES = {"INFO", "WARNING", "CRITICAL"}

    @classmethod
    def sanitize_input(cls, log_line: str) -> str:
        # Prevent prompt injection attacks within log payloads
        prohibited_phrases = ["IGNORE PREVIOUS INSTRUCTIONS", "SYSTEM PROMPT", "DROP TABLE"]
        clean_log = log_line
        for phrase in prohibited_phrases:
            clean_log = re.sub(re.escape(phrase), "[REDACTED_SECURITY]", clean_log, flags=re.IGNORECASE)
        return clean_log

    @classmethod
    def validate_output(cls, raw_output: str) -> dict:
        category = "UNKNOWN"
        for level in cls.ALLOWED_CATEGORIES:
            if f"Classification: {level}" in raw_output or level in raw_output.split("\n")[0]:
                category = level
                break
        
        if category not in cls.ALLOWED_CATEGORIES:
            raise ValueError(f"Guardrail Alert: Invalid output classification detected -> {category}")

        return {"valid": True, "category": category, "raw": raw_output}



def create_mcp_tools(vector_db: FAISS):
    """Creates tool abstractions for Model Context Protocol (MCP) interfaces."""

    @tool
    def search_similar_logs(query_log: str) -> str:
        """Retrieves similar past log entries from the FAISS database to provide context."""
        docs = vector_db.similarity_search(query_log, k=3)
        context = "\n---\n".join([d.page_content for d in docs])
        return f"Historical Context Matches:\n{context}"

    @tool
    def verify_classification_rules(level: str) -> str:
        """Verifies definition rules for log severity categories."""
        rules = {
            "INFO": "Routine operations, normal system state changes, standard status messages.",
            "WARNING": "Potential issues, non-fatal errors, resource threshold limits, auto-recovered retries.",
            "CRITICAL": "Unrecoverable failures, data loss risk, service crashes, disk corruption, hardware faults."
        }
        return rules.get(level.upper(), "Unknown level specified.")

    return [search_similar_logs, verify_classification_rules]


def fetch_sample_logs() -> str:
    """Fetches a public sample log dataset from GitHub (HDFS log sample)."""
    url = "https://raw.githubusercontent.com/logpai/loghub/master/HDFS/HDFS_2k.log"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        # Fallback inline sample logs if network fails
        return """
2026-08-01 10:00:01 INFO [dfs.DataNode$DataXceiver] Receiving block blk_-1073741829 src: /10.251.195.70:54106
2026-08-01 10:00:02 WARN [dfs.FSNamesystem] Address re-registration detected for node 10.251.195.70
2026-08-01 10:00:03 CRITICAL [dfs.DataBlockScanner] Verification failed for Block blk_-1073741829 on disk /data/drive1
2026-08-01 10:00:05 ERROR [dfs.DataNode$PacketResponder] Exception in PacketResponder for block blk_-1073741829
2026-08-01 10:00:06 INFO [dfs.FSNamesystem] BLOCK* NameSystem.allocateBlock: /user/hadoop/file1.txt. blk_1001
        """

def evaluate_classifier_agent(agent_executor: AgentExecutor, test_dataset: List[Dict[str, str]]) -> Dict[str, float]:
    """Runs an evaluation dataset through the agent and calculates classification accuracy."""
    correct = 0
    total = len(test_dataset)

    print("\n--- Running Evaluation Suite ---")
    for item in test_dataset:
        log_input = item["log"]
        expected = item["expected"]
        
        sanitized = GuardrailVerifier.sanitize_input(log_input)
        response = agent_executor.invoke({"input": sanitized})
        output_text = response["output"]

        try:
            validated = GuardrailVerifier.validate_output(output_text)
            pred_category = validated["category"]
        except ValueError:
            pred_category = "INVALID"

        is_correct = (pred_category == expected)
        if is_correct:
            correct += 1
        
        print(f"Log: {log_input[:50]}... | Expected: {expected} | Pred: {pred_category} | Result: {'✅' if is_correct else '❌'}")

    accuracy = (correct / total) * 100 if total > 0 else 0.0
    print(f"\nEvaluation Complete. Accuracy: {accuracy:.2f}%")
    return {"accuracy": accuracy, "total_evals": total}

def build_log_classifier_agent(tools: list, api_key: str) -> AgentExecutor:
    """Configures the main classification agent using LangChain tool calling."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=api_key)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert Reliability Engineer and Log Security Analyst.
Your task is to classify incoming log entries into exactly one of these categories:
- INFO
- WARNING
- CRITICAL

Use the `search_similar_logs` tool to retrieve historical context from RAG before deciding.
Output your final answer strictly in this format:
Classification: <INFO|WARNING|CRITICAL>
Reason: <Brief technical justification>
"""),
        ("user", "Classify the following target log:\n\n{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False)


def build_faiss_rag_db(documents: List[Document], api_key: str) -> FAISS:
      """Embeds chunked log documents and initializes a FAISS vector store."""
      embeddings = OpenAIEmbeddings(openai_api_key=api_key)
      vector_db = FAISS.from_documents(documents, embeddings)
      return vector_db    
class LogSemanticChunker:
    """Splits raw log files into semantically meaningful chunks based on timestamps,

    log levels, and session patterns.
    """
    
    def __init__(self, max_chunk_lines: int = 5):
        self.max_chunk_lines = max_chunk_lines
        # RegEx matching standard ISO/Hadoop timestamp formats
        self.log_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}|\d{6} \d{6}')

    def split_text(self, log_text: str) -> List[Document]:
        lines = log_text.strip().split('\n')
        chunks = []
        current_chunk = []

        for line in lines:
            if not line.strip():
                continue
            
            # Start new semantic boundary when standard log start is detected
            if self.log_pattern.match(line) and len(current_chunk) >= self.max_chunk_lines:
                chunk_str = "\n".join(current_chunk)
                chunks.append(Document(
                    page_content=chunk_str,
                    metadata={"line_count": len(current_chunk)}
                ))
                current_chunk = []
                
            current_chunk.append(line)

        if current_chunk:
            chunks.append(Document(
                page_content="\n".join(current_chunk),
                metadata={"line_count": len(current_chunk)}
            ))

        return chunks

def main():
    # Set your OpenAI API key
    OPENAI_API_KEY = ""

    # 1. Fetch online logs
    print("[1/6] Fetching online log data...")
    raw_log_data = fetch_sample_logs()

    # 2. Perform semantic log chunking
    print("[2/6] Chunking logs semantically...")
    chunker = LogSemanticChunker(max_chunk_lines=3)
    documents = chunker.split_text(raw_log_data)
    print(f"Created {len(documents)} semantic document chunks.")

    # 3. Create FAISS RAG database
    print("[3/6] Indexing vector database with FAISS...")
    vector_db = build_faiss_rag_db(documents, api_key=OPENAI_API_KEY)

    # 4. Setup MCP Tools and LangChain Agent
    print("[4/6] Initializing MCP tools and LangChain agent...")
    mcp_tools = create_mcp_tools(vector_db)
    agent_executor = build_log_classifier_agent(mcp_tools, api_key=OPENAI_API_KEY)

    # 5. Classify a Single Incoming Log Line with Guardrails
    print("\n[5/6] Testing Single Log Classification...")
    sample_log = "2026-08-01 12:14:02 ERROR [dfs.DataNode] Hardware fault: Disk /dev/sdb unreachable."
    
    sanitized_log = GuardrailVerifier.sanitize_input(sample_log)
    result = agent_executor.invoke({"input": sanitized_log})
    
    # Guardrail check
    verification = GuardrailVerifier.validate_output(result["output"])
    print("\n--- Agent Response ---")
    print(result["output"])

    # 6. Run Evaluation Module
    test_eval_dataset = [
        {
            "log": "2026-08-01 14:00:00 INFO [Server] Session created for user_102", 
            "expected": "INFO"
        },
        {
            "log": "2026-08-01 14:01:05 WARN [DiskMonitor] Free disk space is below 10%", 
            "expected": "WARNING"
        },
        {
            "log": "2026-08-01 14:02:10 CRITICAL [Kernel] Memory parity error detected. System halting.", 
            "expected": "CRITICAL"
        }
    ]
    evaluate_classifier_agent(agent_executor, test_eval_dataset)

if __name__ == "__main__":
    main()