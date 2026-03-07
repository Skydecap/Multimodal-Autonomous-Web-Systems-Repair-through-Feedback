import os
import json
import glob
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from core.state import AgentState

# File extensions to index from the website source
SOURCE_EXTENSIONS = {".html", ".css", ".js", ".jsx", ".ts", ".tsx", ".vue", ".svelte", ".php", ".py"}


def _load_website_sources(source_dir: str) -> list[Document]:
    """Recursively load all website source files as LangChain Documents."""
    documents = []
    for ext in SOURCE_EXTENSIONS:
        for filepath in glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True):
            # Skip node_modules, .venv, __pycache__
            if any(skip in filepath for skip in ["node_modules", ".venv", "__pycache__", ".git"]):
                continue
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                rel_path = os.path.relpath(filepath, source_dir)
                documents.append(Document(
                    page_content=f"### FILE: {rel_path}\n\n{content}",
                    metadata={"source": rel_path, "type": "source_code"},
                ))
            except Exception as e:
                print(f"  [RAG] Could not read {filepath}: {e}")
    return documents


def _build_trace_documents(trace_summary: dict, bug_report: str) -> list[Document]:
    """Convert trace artifacts into LangChain Documents for RAG indexing."""
    docs = []

    # 1. Bug report
    docs.append(Document(
        page_content=f"### BUG REPORT\n\n{bug_report}",
        metadata={"source": "bug_report", "type": "trace"},
    ))

    # 2. Agent summary (the MCP agent's findings)
    agent_summary = trace_summary.get("agent_summary", "")
    if agent_summary:
        docs.append(Document(
            page_content=f"### AGENT ANALYSIS SUMMARY\n\n{agent_summary}",
            metadata={"source": "agent_summary", "type": "trace"},
        ))

    # 3. Console errors
    console_errors = trace_summary.get("console_errors", [])
    if console_errors:
        errors_text = "\n".join(
            f"- [{e.get('type', 'error')}] {e.get('text', '')}" for e in console_errors
        )
        docs.append(Document(
            page_content=f"### CONSOLE ERRORS CAPTURED DURING REPRODUCTION\n\n{errors_text}",
            metadata={"source": "console_errors", "type": "trace"},
        ))

    # 4. Network errors
    network_errors = trace_summary.get("failed_network_requests", [])
    if network_errors:
        net_text = "\n".join(
            f"- HTTP {e.get('status', '?')}: {e.get('url', e.get('text', ''))}" for e in network_errors
        )
        docs.append(Document(
            page_content=f"### FAILED NETWORK REQUESTS\n\n{net_text}",
            metadata={"source": "network_errors", "type": "trace"},
        ))

    # 5. Action log (what the agent did step by step)
    action_log = trace_summary.get("action_log", "[]")
    try:
        actions = json.loads(action_log)
        if actions:
            steps_text = "\n".join(
                f"- {a.get('name', '?')}({json.dumps(a.get('args', {}))[:200]})"
                for a in actions
            )
            docs.append(Document(
                page_content=f"### REPRODUCTION STEPS (Actions taken by the agent)\n\n{steps_text}",
                metadata={"source": "action_log", "type": "trace"},
            ))
    except json.JSONDecodeError:
        pass

    return docs


async def rag_analyzer_node(state: AgentState):
    """
    RAG-based code analysis node.
    1. Indexes the website source files + trace artifacts into a FAISS vector store
    2. Retrieves the most relevant chunks for the bug context
    3. Sends them to the LLM to generate a root cause analysis and code fix
    """
    print(f"\n{'='*60}")
    print(f"[RAG Analyzer] Starting RAG-based root cause analysis...")
    print(f"{'='*60}")

    trace_summary = state.get("trace_summary", {})
    bug_report = state.get("bug_report", "")

    # --- 1. Collect documents ---

    # Website source files
    source_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test")
    source_docs = _load_website_sources(source_dir)
    print(f"[RAG] Loaded {len(source_docs)} source file(s) from '{source_dir}'")

    # Trace artifacts
    trace_docs = _build_trace_documents(trace_summary, bug_report)
    print(f"[RAG] Created {len(trace_docs)} trace document(s)")

    all_docs = source_docs + trace_docs

    if not all_docs:
        print("[RAG] No documents to index. Skipping analysis.")
        return {"root_cause_analysis": "No documents available for analysis."}

    # --- 2. Chunk and index ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n### ", "\n## ", "\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(all_docs)
    print(f"[RAG] Split into {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    print(f"[RAG] FAISS vector store built successfully")

    # --- 3. Retrieve relevant context ---
    # Build a rich query combining bug report + errors
    console_errors = trace_summary.get("console_errors", [])
    error_texts = " | ".join(e.get("text", "")[:100] for e in console_errors[:5])

    retrieval_query = (
        f"Bug report: {bug_report}\n"
        f"Errors found: {error_texts}\n"
        f"Find the source code causing these errors and suggest a fix."
    )

    retrieved_docs = vectorstore.similarity_search(retrieval_query, k=8)
    print(f"[RAG] Retrieved {len(retrieved_docs)} relevant chunks")

    # --- 4. Build context for the LLM ---
    context_text = "\n\n---\n\n".join(doc.page_content for doc in retrieved_docs)

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        base_url="https://models.inference.ai.azure.com",
        api_key=os.getenv("GITHUB_TOKEN"),
    )

    analysis_prompt = f"""You are a senior web developer and bug-fixing expert.

You are given a bug report, the trace data from reproducing the bug in a browser, and the relevant source code of the website.

Your task:
1. Identify the ROOT CAUSE of the bug in the source code.
2. Explain what is wrong and why.
3. Provide the EXACT code fix as a diff (showing the old code and new code).
4. If there are multiple bugs, provide a SEPARATE diff block for each fix.
5. Make sure your fixes are CONSISTENT across all files — e.g. if one file writes data to localStorage in a certain format, any other file reading it must use the same format.

CRITICAL DIFF FORMAT RULES:
- Each diff block MUST start with: --- a/FILENAME (the filename relative to the project)
- Each line to REMOVE starts with - (this line must EXACTLY match a real line in the source file)
- Each line to ADD starts with + (the replacement line)
- Pair each - line with its corresponding + line (one-to-one replacement)
- Do NOT include unchanged context lines
- Do NOT include @@ hunk headers or +++ lines
- Each - line and + line should contain the code WITHOUT leading whitespace (indentation is auto-detected)

Example of a CORRECT diff:
```diff
--- a/index.html
- let x = null;
+ let x = {{}};
```

Format your response as:

## Root Cause Analysis
[Explain what's wrong]

## Bug Details
[For each bug found, explain the issue in a separate subsection]

## Suggested Fix
```diff
--- a/index.html
- old line of code
+ new corrected line of code
```

```diff
--- a/index.html
- another old line
+ another new line
```

## Additional Notes
[Any other recommendations]

---

### BUG REPORT
{bug_report}

### RETRIEVED CONTEXT (Source code + Trace data)
{context_text}
"""

    print(f"[RAG] Sending analysis request to LLM...")
    response = await llm.ainvoke([HumanMessage(content=analysis_prompt)])
    analysis = response.content

    print(f"\n[RAG Analyzer] Analysis complete.")
    print(f"\n{'='*60}")
    print(analysis)
    print(f"{'='*60}")

    # Save analysis to file
    os.makedirs("artifacts", exist_ok=True)
    analysis_path = "artifacts/rag_analysis.md"
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write(analysis)
    print(f"[RAG] Analysis saved to {analysis_path}")

    return {
        "root_cause_analysis": analysis,
        "relevant_files": [doc.metadata.get("source", "") for doc in retrieved_docs if doc.metadata.get("type") == "source_code"],
    }


async def rag_reanalyze_with_feedback(state: dict, feedback: str) -> str:
    """
    Re-run RAG analysis incorporating human feedback.
    Uses the same RAG pipeline but adds the previous analysis + feedback to the prompt.
    """
    print(f"\n{'='*60}")
    print(f"[RAG Re-analyzer] Re-analyzing with feedback...")
    print(f"[RAG Re-analyzer] Feedback: {feedback}")
    print(f"{'='*60}")

    trace_summary = state.get("trace_summary", {})
    bug_report = state.get("bug_report", "")
    previous_analysis = state.get("root_cause_analysis", "")

    # --- 1. Collect documents (same as before) ---
    source_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test")
    source_docs = _load_website_sources(source_dir)
    trace_docs = _build_trace_documents(trace_summary, bug_report)
    all_docs = source_docs + trace_docs

    if not all_docs:
        return "No documents available for re-analysis."

    # --- 2. Chunk and index ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n### ", "\n## ", "\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    # --- 3. Retrieve with feedback-enriched query ---
    console_errors = trace_summary.get("console_errors", [])
    error_texts = " | ".join(e.get("text", "")[:100] for e in console_errors[:5])

    retrieval_query = (
        f"Bug report: {bug_report}\n"
        f"Errors: {error_texts}\n"
        f"Human feedback on previous fix: {feedback}\n"
        f"Find the source code and suggest a better fix."
    )

    retrieved_docs = vectorstore.similarity_search(retrieval_query, k=8)
    context_text = "\n\n---\n\n".join(doc.page_content for doc in retrieved_docs)

    # --- 4. LLM with feedback context ---
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        base_url="https://models.inference.ai.azure.com",
        api_key=os.getenv("GITHUB_TOKEN"),
    )

    feedback_prompt = f"""You are a senior web developer and bug-fixing expert.

You previously suggested a fix for a bug, but the human reviewer has provided feedback.
You must revise your analysis and provide an UPDATED fix based on the feedback.

## Previous Analysis
{previous_analysis}

## Human Feedback
{feedback}

## Instructions
1. Consider the human's feedback carefully.
2. Re-examine the source code and trace data.
3. Provide a REVISED root cause analysis and code fix.
4. Make sure your fixes are CONSISTENT across all files — e.g. if one file writes data to localStorage in a certain format, any other file reading it must use the same format.

CRITICAL DIFF FORMAT RULES:
- Each diff block MUST start with: --- a/FILENAME (the filename relative to the project)
- Each line to REMOVE starts with - (this line must EXACTLY match a real line in the source file)
- Each line to ADD starts with + (the replacement line)
- Pair each - line with its corresponding + line (one-to-one replacement)
- Do NOT include unchanged context lines
- Do NOT include @@ hunk headers or +++ lines
- Make sure to stick with the domain of web server while performing changes in end-points
- Provide a SEPARATE diff block for each independent fix
- Each - line and + line should contain the code WITHOUT leading whitespace (indentation is auto-detected)

Format your response as:

## Root Cause Analysis (Revised)
[Updated explanation]

## Bug Details
[Updated details]

## Suggested Fix
```diff
--- a/index.html
- old line of code
+ new corrected line of code
```

```diff
--- a/index.html
- another old line
+ another new line
```

## Additional Notes
[Any other recommendations]

---

### BUG REPORT
{bug_report}

### RETRIEVED CONTEXT (Source code + Trace data)
{context_text}
"""

    print(f"[RAG] Sending re-analysis request to LLM...")
    response = await llm.ainvoke([HumanMessage(content=feedback_prompt)])
    analysis = response.content

    print(f"\n[RAG Re-analyzer] Re-analysis complete.")
    print(f"\n{'='*60}")
    print(analysis)
    print(f"{'='*60}")

    # Save updated analysis
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/rag_analysis.md", "w", encoding="utf-8") as f:
        f.write(analysis)
    print(f"[RAG] Updated analysis saved to artifacts/rag_analysis.md")

    return analysis
