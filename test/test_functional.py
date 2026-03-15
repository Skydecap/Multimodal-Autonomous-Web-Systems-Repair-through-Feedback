"""
Functional tests for the Multimodal Autonomous Web Systems Repair project.
Tests cover: patch engine, Flask routes, error parsers, RAG helpers,
executor deduplication, graph construction, and state management.
"""

import os
import json
import pytest
import tempfile
import shutil
from unittest.mock import patch, MagicMock, AsyncMock

# ── Module imports ──────────────────────────────────────────────
from utils.patch_engine import (
    parse_diffs_from_analysis,
    _find_block_in_lines,
    _apply_diff_in_memory,
    _apply_diff_to_file,
    preview_patch,
    apply_patch,
    revert_patch,
    get_patch_state,
    add_iteration,
    _load_patch_state,
    _save_patch_state,
    PATCH_STATE_FILE,
)
from agents.planner import _parse_console_errors, _parse_network_errors, format_mcp_tool
from agents.rag_analyzer import _load_website_sources, _build_trace_documents
from agents.executor import browser_execution_node
from core.state import AgentState
from graph import build_graph
from main import flask_app, _extract_real_error


# ================================================================
# Fixtures
# ================================================================

@pytest.fixture
def sample_analysis():
    """A realistic LLM analysis markdown with two diff blocks."""
    return """## Root Cause Analysis
The cart initialization does not handle null localStorage.

## Suggested Fix
```diff
--- a/index.html
- let cartData = JSON.parse(localStorage.getItem('cart'));
+ let cartData = JSON.parse(localStorage.getItem('cart')) || [];
```

```diff
--- a/index.html
- localStorage.setItem('cart', JSON.stringify(cartData));
+ cartData.push(newItem);
+ localStorage.setItem('cart', JSON.stringify(cartData));
```
"""


@pytest.fixture
def sample_source():
    """Simulated source file content matching the diff targets."""
    return (
        "function addToCart(name, price) {\n"
        "    let cartData = JSON.parse(localStorage.getItem('cart'));\n"
        "    const newItem = { name: name, price: price, qty: 1 };\n"
        "    localStorage.setItem('cart', JSON.stringify(cartData));\n"
        "    window.location.href = '';\n"
        "}"
    )


@pytest.fixture
def tmp_source_dir(sample_source):
    """Create a temporary directory with a source file for disk-based tests."""
    tmpdir = tempfile.mkdtemp()
    with open(os.path.join(tmpdir, "index.html"), "w", encoding="utf-8") as f:
        f.write(sample_source)
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def clean_patch_state():
    """Ensure patch state file is reset before and after each test."""
    if os.path.exists(PATCH_STATE_FILE):
        os.remove(PATCH_STATE_FILE)
    yield
    if os.path.exists(PATCH_STATE_FILE):
        os.remove(PATCH_STATE_FILE)


@pytest.fixture
def flask_client():
    """Flask test client."""
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        yield client


# ================================================================
# 1. Diff Parsing Tests
# ================================================================

class TestDiffParsing:
    """Tests for parse_diffs_from_analysis — the core diff parser."""

    def test_parse_single_hunk_replacement(self):
        """TC-01: Parse a simple single-line replacement diff."""
        analysis = """```diff
--- a/index.html
- let x = null;
+ let x = [];
```"""
        diffs = parse_diffs_from_analysis(analysis)
        assert len(diffs) == 1
        assert diffs[0]["target_file"] == "index.html"
        assert len(diffs[0]["hunks"]) == 1
        hunk = diffs[0]["hunks"][0]
        assert hunk["removed"] == [" let x = null;"]
        assert hunk["added"] == [" let x = [];"]

    def test_parse_multi_line_hunk(self):
        """TC-02: Parse a hunk where 1 removed line maps to 2 added lines."""
        analysis = """```diff
--- a/app.js
- save(data);
+ data.push(item);
+ save(data);
```"""
        diffs = parse_diffs_from_analysis(analysis)
        assert len(diffs) == 1
        hunk = diffs[0]["hunks"][0]
        assert len(hunk["removed"]) == 1
        assert len(hunk["added"]) == 2

    def test_parse_multiple_diff_blocks(self, sample_analysis):
        """TC-03: Parse multiple separate diff blocks from one analysis."""
        diffs = parse_diffs_from_analysis(sample_analysis)
        assert len(diffs) == 2
        assert all(d["target_file"] == "index.html" for d in diffs)

    def test_parse_empty_analysis(self):
        """TC-04: Return empty list when no diff blocks exist."""
        diffs = parse_diffs_from_analysis("No code changes needed.")
        assert diffs == []

    def test_parse_context_lines_tracked(self):
        """TC-05: Context lines before a hunk are recorded in context_before."""
        analysis = """```diff
--- a/style.css
 body {
   color: black;
 }
- font-size: 12px;
+ font-size: 16px;
```"""
        diffs = parse_diffs_from_analysis(analysis)
        hunk = diffs[0]["hunks"][0]
        assert len(hunk["context_before"]) > 0
        assert any("color: black" in c for c in hunk["context_before"])


# ================================================================
# 2. Block Finding Tests
# ================================================================

class TestBlockFinding:
    """Tests for _find_block_in_lines — consecutive line matching."""

    def test_find_single_line(self):
        """TC-06: Find a single-line block in a list of lines."""
        lines = ["  let a = 1;", "  let b = 2;", "  let c = 3;"]
        idx = _find_block_in_lines(lines, ["let b = 2;"])
        assert idx == 1

    def test_find_multi_line_block(self):
        """TC-07: Find a multi-line consecutive block."""
        lines = ["a", "b", "c", "d", "e"]
        idx = _find_block_in_lines(lines, ["b", "c", "d"])
        assert idx == 1

    def test_block_not_found(self):
        """TC-08: Return -1 when block doesn't exist."""
        lines = ["a", "b", "c"]
        idx = _find_block_in_lines(lines, ["x", "y"])
        assert idx == -1


# ================================================================
# 3. In-Memory Diff Application Tests
# ================================================================

class TestInMemoryDiffApplication:
    """Tests for _apply_diff_in_memory — applying patches to strings."""

    def test_single_line_replacement(self, sample_source):
        """TC-09: Replace one line in memory, preserving indentation."""
        diff = {
            "hunks": [{
                "removed": ["let cartData = JSON.parse(localStorage.getItem('cart'));"],
                "added": ["let cartData = JSON.parse(localStorage.getItem('cart')) || [];"],
                "context_before": [],
            }]
        }
        result = _apply_diff_in_memory(sample_source, diff)
        assert result is not None
        assert "|| []" in result

    def test_one_to_many_replacement(self, sample_source):
        """TC-10: Replace 1 line with 2 lines (block expansion)."""
        diff = {
            "hunks": [{
                "removed": ["localStorage.setItem('cart', JSON.stringify(cartData));"],
                "added": ["cartData.push(newItem);", "localStorage.setItem('cart', JSON.stringify(cartData));"],
                "context_before": [],
            }]
        }
        result = _apply_diff_in_memory(sample_source, diff)
        assert result is not None
        assert "cartData.push(newItem);" in result
        assert "localStorage.setItem" in result

    def test_sequential_hunks(self, sample_source):
        """TC-11: Apply two hunks from the same diff sequentially."""
        diff = {
            "hunks": [
                {
                    "removed": ["let cartData = JSON.parse(localStorage.getItem('cart'));"],
                    "added": ["let cartData = JSON.parse(localStorage.getItem('cart')) || [];"],
                    "context_before": [],
                },
                {
                    "removed": ["localStorage.setItem('cart', JSON.stringify(cartData));"],
                    "added": ["cartData.push(newItem);", "localStorage.setItem('cart', JSON.stringify(cartData));"],
                    "context_before": [],
                },
            ]
        }
        result = _apply_diff_in_memory(sample_source, diff)
        assert result is not None
        assert "|| []" in result
        assert "cartData.push(newItem);" in result

    def test_no_match_returns_none(self):
        """TC-12: Return None when removed lines don't match any content."""
        diff = {
            "hunks": [{
                "removed": ["this line does not exist anywhere"],
                "added": ["replacement"],
                "context_before": [],
            }]
        }
        result = _apply_diff_in_memory("some unrelated content", diff)
        assert result is None


# ================================================================
# 4. Disk-Based Patch Application Test
# ================================================================

class TestDiskPatchApplication:
    """Tests for _apply_diff_to_file — applying patches to actual files."""

    def test_apply_diff_to_file(self, tmp_source_dir):
        """TC-13: Apply a diff to a real file on disk and verify contents."""
        filepath = os.path.join(tmp_source_dir, "index.html")
        diff = {
            "hunks": [{
                "removed": ["let cartData = JSON.parse(localStorage.getItem('cart'));"],
                "added": ["let cartData = JSON.parse(localStorage.getItem('cart')) || [];"],
                "context_before": [],
            }]
        }
        result = _apply_diff_to_file(filepath, diff)
        assert result is True

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        assert "|| []" in content


# ================================================================
# 5. Patch State Management Tests
# ================================================================

class TestPatchStateManagement:
    """Tests for patch state persistence (save/load/iterations)."""

    def test_load_empty_state(self, clean_patch_state):
        """TC-14: Loading state when no file exists returns defaults."""
        state = _load_patch_state()
        assert state["applied"] is False
        assert state["iterations"] == []

    def test_add_iteration(self, clean_patch_state):
        """TC-15: add_iteration records feedback with a timestamp."""
        iterations = add_iteration(feedback="Fix the null check", analysis="Some analysis text")
        assert len(iterations) == 1
        assert iterations[0]["feedback"] == "Fix the null check"
        assert "timestamp" in iterations[0]
        assert iterations[0]["analysis_preview"] == "Some analysis text"


# ================================================================
# 6. Console & Network Error Parsers
# ================================================================

class TestErrorParsers:
    """Tests for _parse_console_errors and _parse_network_errors."""

    def test_parse_console_errors(self):
        """TC-16: Extract errors and warnings from console output text."""
        text = (
            "info: page loaded\n"
            "TypeError: Cannot read property of null\n"
            "warning: deprecated API usage\n"
            "log: all good\n"
        )
        errors = []
        _parse_console_errors(text, errors)
        assert len(errors) == 2
        assert errors[0]["type"] == "error"
        assert "TypeError" in errors[0]["text"]
        assert errors[1]["type"] == "warning"

    def test_parse_network_errors(self):
        """TC-17: Extract HTTP error codes from network request text."""
        text = (
            "GET https://api.example.com/data 200 OK\n"
            "POST https://api.example.com/submit 500 Internal Server Error\n"
            "GET https://api.example.com/missing 404 Not Found\n"
        )
        errors = []
        _parse_network_errors(text, errors)
        assert len(errors) == 2
        status_codes = {e["status"] for e in errors}
        assert 500 in status_codes
        assert 404 in status_codes


# ================================================================
# 7. Flask Route Tests
# ================================================================

class TestFlaskRoutes:
    """Tests for the Flask API endpoints."""

    def test_report_missing_body(self, flask_client):
        """TC-18: POST /report with empty bug_report returns 400."""
        resp = flask_client.post("/report", json={"bug_report": ""})
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data

    def test_get_analysis_empty(self, flask_client):
        """TC-19: GET /review/analysis returns empty defaults before any workflow."""
        resp = flask_client.get("/review/analysis")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "analysis" in data
        assert "bug_report" in data

    def test_push_no_analysis(self, flask_client):
        """TC-20: POST /review/push with no prior analysis returns 400."""
        import main
        original = main._latest_state
        main._latest_state = {}
        try:
            resp = flask_client.post("/review/push")
            assert resp.status_code == 400
            data = resp.get_json()
            assert "error" in data
        finally:
            main._latest_state = original

    def test_feedback_empty(self, flask_client):
        """TC-21: POST /review/feedback with empty text returns 400."""
        resp = flask_client.post("/review/feedback", json={"feedback": ""})
        assert resp.status_code == 400

    def test_preview_file_not_found(self, flask_client):
        """TC-22: GET /review/preview for a nonexistent file returns 404."""
        resp = flask_client.get("/review/preview/nonexistent_file.html")
        assert resp.status_code == 404


# ================================================================
# 8. RAG Helper Tests
# ================================================================

class TestRAGHelpers:
    """Tests for RAG document loading and trace document construction."""

    def test_load_website_sources(self, tmp_source_dir):
        """TC-23: _load_website_sources finds .html files in a directory."""
        docs = _load_website_sources(tmp_source_dir)
        assert len(docs) >= 1
        assert any("index.html" in d.metadata["source"] for d in docs)
        assert docs[0].metadata["type"] == "source_code"

    def test_build_trace_documents(self):
        """TC-24: _build_trace_documents creates Documents from trace artifacts."""
        trace = {
            "agent_summary": "Found a null reference error",
            "console_errors": [{"type": "error", "text": "TypeError: null"}],
            "failed_network_requests": [{"status": 500, "url": "http://api/data"}],
            "action_log": json.dumps([{"name": "browser_navigate", "args": {"url": "http://localhost"}}]),
        }
        docs = _build_trace_documents(trace, "Cart button is broken")
        # Should produce: bug_report + agent_summary + console_errors + network_errors + action_log = 5
        assert len(docs) == 5
        types = [d.metadata["type"] for d in docs]
        assert all(t == "trace" for t in types)

    def test_build_trace_documents_minimal(self):
        """TC-25: Minimal trace with no errors still produces bug report doc."""
        docs = _build_trace_documents({}, "Some bug")
        assert len(docs) >= 1
        assert docs[0].metadata["source"] == "bug_report"


# ================================================================
# 9. Graph Construction Test
# ================================================================

class TestGraphConstruction:
    """Tests for the LangGraph workflow assembly."""

    def test_build_graph_compiles(self):
        """TC-26: build_graph() produces a compiled runnable graph."""
        graph = build_graph()
        assert graph is not None
        # A compiled LangGraph has an invoke/ainvoke method
        assert hasattr(graph, "ainvoke")


# ================================================================
# 10. Error Extraction Test
# ================================================================

class TestErrorExtraction:
    """Tests for _extract_real_error in main.py."""

    def test_extract_plain_exception(self):
        """TC-27: Plain exceptions are returned as-is."""
        e = ValueError("bad value")
        result = _extract_real_error(e)
        assert "ValueError" in result
        assert "bad value" in result

    def test_extract_from_exception_group(self):
        """TC-28: Unwraps BaseExceptionGroup to find the real cause."""
        inner = RuntimeError("real cause")
        group = BaseExceptionGroup("wrapper", [inner])
        result = _extract_real_error(group)
        assert "real cause" in result


# ================================================================
# 11. Executor Deduplication Test
# ================================================================

class TestExecutorDeduplication:
    """Tests for error dedup logic in browser_execution_node."""

    @pytest.mark.asyncio
    async def test_deduplicates_errors(self):
        """TC-29: Executor merges and deduplicates console/network errors."""
        state = {
            "trace_summary": {
                "action_log": "[]",
                "failed_network_requests": [{"url": "http://a.com", "status": 404}],
                "console_errors": [{"type": "error", "text": "TypeError: null"}],
                "final_screenshot": "",
                "agent_summary": "test",
            },
            "iteration_count": 0,
        }

        mock_replay = {
            "failed_network_requests": [
                {"url": "http://a.com", "status": 404},  # duplicate
                {"url": "http://b.com", "status": 500},  # new
            ],
            "console_errors": [
                {"type": "error", "text": "TypeError: null"},  # duplicate
                {"type": "error", "text": "ReferenceError: x"},  # new
            ],
            "trace_file": "trace.zip",
            "final_screenshot": "",
        }

        with patch("agents.executor.replay_and_capture_trace", return_value=mock_replay):
            result = await browser_execution_node(state)

        net_errors = result["trace_summary"]["failed_network_requests"]
        con_errors = result["trace_summary"]["console_errors"]
        assert len(net_errors) == 2  # deduplicated
        assert len(con_errors) == 2  # deduplicated


# ================================================================
# 12. MCP Tool Formatting Test
# ================================================================

class TestMCPToolFormatting:
    """Tests for format_mcp_tool in planner.py."""

    def test_format_mcp_tool(self):
        """TC-30: Converts an MCP tool object to OpenAI-compatible dict."""
        mock_tool = MagicMock()
        mock_tool.name = "browser_click"
        mock_tool.description = "Click an element"
        mock_tool.inputSchema = {"type": "object", "properties": {"ref": {"type": "string"}}}

        result = format_mcp_tool(mock_tool)
        assert result["name"] == "browser_click"
        assert result["description"] == "Click an element"
        assert "properties" in result["parameters"]
