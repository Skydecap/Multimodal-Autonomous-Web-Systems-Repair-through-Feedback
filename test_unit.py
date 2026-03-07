"""
Unit tests for the Multimodal Autonomous Web Systems Repair project.
Organised by module pipeline: Playwright → LLM / RAG → Feedback → Patch Engine → Graph & State.

Run:  python -m pytest test_unit.py -v
"""

import os
import json
import tempfile
import shutil
import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock, PropertyMock

# ── Project imports ─────────────────────────────────────────────
from core.state import AgentState
from graph import build_graph
from agents.planner import (
    planner_node, format_mcp_tool,
    _parse_console_errors, _parse_network_errors, MAX_TURNS,
)
from agents.executor import browser_execution_node
from agents.rag_analyzer import (
    rag_analyzer_node, rag_reanalyze_with_feedback,
    _load_website_sources, _build_trace_documents, SOURCE_EXTENSIONS,
)
from utils.patch_engine import (
    parse_diffs_from_analysis, _find_block_in_lines,
    _apply_diff_in_memory, _apply_diff_to_file,
    apply_patch, revert_patch, preview_patch,
    get_patch_state, add_iteration,
    _load_patch_state, _save_patch_state,
    PATCH_STATE_FILE,
)
from main import flask_app, _extract_real_error


# ================================================================
#  SHARED FIXTURES
# ================================================================

@pytest.fixture
def clean_patch_state():
    """Reset artifacts/patch_state.json before and after every test that touches it."""
    if os.path.exists(PATCH_STATE_FILE):
        os.remove(PATCH_STATE_FILE)
    yield
    if os.path.exists(PATCH_STATE_FILE):
        os.remove(PATCH_STATE_FILE)


@pytest.fixture
def tmp_source_dir():
    """A temp directory containing a small HTML file for disk‑based patch tests."""
    tmpdir = tempfile.mkdtemp()
    html = (
        "<html>\n"
        "<body>\n"
        "  <script>\n"
        "    function addToCart(name, price) {\n"
        "        let cartData = JSON.parse(localStorage.getItem('cart'));\n"
        "        const newItem = { name: name, price: price, qty: 1 };\n"
        "        localStorage.setItem('cart', JSON.stringify(cartData));\n"
        "        window.location.href = '';\n"
        "    }\n"
        "  </script>\n"
        "</body>\n"
        "</html>"
    )
    with open(os.path.join(tmpdir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def flask_client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


# ================================================================
#  MODULE 1 — PLAYWRIGHT / BROWSER TOOLS  (browser_tools + executor + planner helpers)
# ================================================================

class TestPlaywrightModule:
    """Unit tests for the Playwright browser‑interaction layer."""

    # ── planner: format_mcp_tool ────────────────────────────────

    def test_format_mcp_tool_basic(self):
        """format_mcp_tool converts an MCP tool object into an OpenAI‑schema dict."""
        tool = MagicMock()
        tool.name = "browser_click"
        tool.description = "Click an element on the page"
        tool.inputSchema = {"type": "object", "properties": {"ref": {"type": "string"}}}

        result = format_mcp_tool(tool)

        assert result["name"] == "browser_click"
        assert result["description"] == "Click an element on the page"
        assert result["parameters"]["type"] == "object"

    def test_format_mcp_tool_empty_schema(self):
        """format_mcp_tool works when inputSchema is an empty dict."""
        tool = MagicMock()
        tool.name = "browser_snapshot"
        tool.description = "Take a snapshot"
        tool.inputSchema = {}

        result = format_mcp_tool(tool)
        assert result["parameters"] == {}

    # ── planner: _parse_console_errors ──────────────────────────

    def test_parse_console_errors_captures_error(self):
        """Lines containing 'error' are classified as type='error'."""
        errors = []
        _parse_console_errors("TypeError: Cannot read property 'push' of null", errors)
        assert len(errors) == 1
        assert errors[0]["type"] == "error"
        assert "TypeError" in errors[0]["text"]

    def test_parse_console_errors_captures_warning(self):
        """Lines with 'warning'/'warn' but not 'error' are classified as type='warning'."""
        errors = []
        _parse_console_errors("Deprecation warning: use the new API instead", errors)
        assert len(errors) == 1
        assert errors[0]["type"] == "warning"

    def test_parse_console_errors_ignores_clean(self):
        """Normal log lines without error keywords are ignored."""
        errors = []
        _parse_console_errors("info: page fully loaded\nlog: rendering complete", errors)
        assert errors == []

    def test_parse_console_errors_multiple(self):
        """Multiple error/warning lines are all captured."""
        text = (
            "TypeError: null is not an object\n"
            "info: loaded\n"
            "ReferenceError: x is not defined\n"
            "warn: slow render\n"
        )
        errors = []
        _parse_console_errors(text, errors)
        assert len(errors) == 3

    # ── planner: _parse_network_errors ──────────────────────────

    def test_parse_network_errors_captures_500(self):
        """Lines containing '500' produce a network error with status 500."""
        errors = []
        _parse_network_errors("POST https://api.example.com/submit 500 Internal Server Error", errors)
        assert len(errors) == 1
        assert errors[0]["status"] == 500

    def test_parse_network_errors_captures_404(self):
        errors = []
        _parse_network_errors("GET /missing 404 Not Found", errors)
        assert len(errors) == 1
        assert errors[0]["status"] == 404

    def test_parse_network_errors_ignores_200(self):
        """Successful responses (200) are NOT captured."""
        errors = []
        _parse_network_errors("GET https://api.example.com/data 200 OK", errors)
        assert errors == []

    def test_parse_network_errors_multiple_codes(self):
        """Each distinct error status code in multi‑line output is captured."""
        text = (
            "GET /a 401 Unauthorized\n"
            "GET /b 403 Forbidden\n"
            "GET /c 502 Bad Gateway\n"
        )
        errors = []
        _parse_network_errors(text, errors)
        codes = {e["status"] for e in errors}
        assert codes == {401, 403, 502}

    # ── planner: planner_node error fallback ────────────────────

    @pytest.mark.asyncio
    async def test_planner_node_mcp_failure_returns_fallback(self):
        """When MCP connection fails, planner_node returns a safe fallback state."""
        state = {"bug_report": "button broken", "iteration_count": 0}

        with patch("agents.planner.stdio_client", side_effect=RuntimeError("npx not found")):
            result = await planner_node(state)

        assert result["reproduction_script"] == "[]"
        assert "MCP planner failed" in result["trace_summary"]["console_errors"][0]["text"]
        assert result["iteration_count"] == 1

    # ── executor: deduplication logic ───────────────────────────

    @pytest.mark.asyncio
    async def test_executor_deduplicates_console_errors(self):
        """Duplicate console errors (same text) from MCP + replay are merged into one."""
        state = {
            "trace_summary": {
                "action_log": "[]",
                "failed_network_requests": [],
                "console_errors": [{"type": "error", "text": "TypeError: null"}],
                "final_screenshot": "",
                "agent_summary": "",
            },
            "iteration_count": 0,
        }
        replay = {
            "failed_network_requests": [],
            "console_errors": [
                {"type": "error", "text": "TypeError: null"},      # dup
                {"type": "error", "text": "ReferenceError: x"},     # new
            ],
            "trace_file": "",
            "final_screenshot": "",
        }
        with patch("agents.executor.replay_and_capture_trace", return_value=replay):
            result = await browser_execution_node(state)

        assert len(result["trace_summary"]["console_errors"]) == 2

    @pytest.mark.asyncio
    async def test_executor_deduplicates_network_errors(self):
        """Duplicate network errors (same status+url) are merged."""
        state = {
            "trace_summary": {
                "action_log": "[]",
                "failed_network_requests": [{"url": "http://a.com", "status": 404}],
                "console_errors": [],
                "final_screenshot": "",
                "agent_summary": "",
            },
            "iteration_count": 0,
        }
        replay = {
            "failed_network_requests": [
                {"url": "http://a.com", "status": 404},  # dup
                {"url": "http://b.com", "status": 500},  # new
            ],
            "console_errors": [],
            "trace_file": "t.zip",
            "final_screenshot": "",
        }
        with patch("agents.executor.replay_and_capture_trace", return_value=replay):
            result = await browser_execution_node(state)

        assert len(result["trace_summary"]["failed_network_requests"]) == 2

    @pytest.mark.asyncio
    async def test_executor_handles_empty_actions(self):
        """Executor handles an empty action log without crashing."""
        state = {
            "trace_summary": {
                "action_log": "[]",
                "failed_network_requests": [],
                "console_errors": [],
                "final_screenshot": "",
                "agent_summary": "Nothing found",
            },
            "iteration_count": 0,
        }
        replay = {
            "failed_network_requests": [],
            "console_errors": [],
            "trace_file": "",
            "final_screenshot": "",
        }
        with patch("agents.executor.replay_and_capture_trace", return_value=replay):
            result = await browser_execution_node(state)

        assert result["trace_summary"]["console_errors"] == []
        assert result["screenshots"] == []

    @pytest.mark.asyncio
    async def test_executor_preserves_screenshot(self):
        """If replay produces a screenshot path, it is propagated into trace_summary."""
        state = {
            "trace_summary": {
                "action_log": "[]",
                "failed_network_requests": [],
                "console_errors": [],
                "final_screenshot": "",
                "agent_summary": "",
            },
            "iteration_count": 0,
        }
        replay = {
            "failed_network_requests": [],
            "console_errors": [],
            "trace_file": "artifacts/traces/t.zip",
            "final_screenshot": "artifacts/final_state.png",
        }
        with patch("agents.executor.replay_and_capture_trace", return_value=replay):
            result = await browser_execution_node(state)

        assert result["trace_summary"]["final_screenshot"] == "artifacts/final_state.png"
        assert result["screenshots"] == ["artifacts/final_state.png"]


# ================================================================
#  MODULE 2 — LLM / RAG ANALYSIS
# ================================================================

class TestLLMRagModule:
    """Unit tests for the RAG indexing, retrieval, and LLM analysis pipeline."""

    # ── _load_website_sources ───────────────────────────────────

    def test_load_sources_finds_html(self, tmp_source_dir):
        """_load_website_sources discovers .html files."""
        docs = _load_website_sources(tmp_source_dir)
        assert len(docs) >= 1
        assert docs[0].metadata["type"] == "source_code"

    def test_load_sources_empty_dir(self):
        """Returns an empty list for a directory with no source files."""
        tmpdir = tempfile.mkdtemp()
        try:
            docs = _load_website_sources(tmpdir)
            assert docs == []
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_load_sources_skips_pycache(self, tmp_source_dir):
        """Files inside __pycache__ are excluded."""
        pycache = os.path.join(tmp_source_dir, "__pycache__")
        os.makedirs(pycache)
        with open(os.path.join(pycache, "cache.py"), "w") as f:
            f.write("# cached")
        docs = _load_website_sources(tmp_source_dir)
        sources = [d.metadata["source"] for d in docs]
        assert not any("__pycache__" in s for s in sources)

    def test_load_sources_respects_extensions(self, tmp_source_dir):
        """Only files matching SOURCE_EXTENSIONS are loaded."""
        with open(os.path.join(tmp_source_dir, "readme.txt"), "w") as f:
            f.write("not a source file")
        with open(os.path.join(tmp_source_dir, "style.css"), "w") as f:
            f.write("body { color: red; }")
        docs = _load_website_sources(tmp_source_dir)
        sources = [d.metadata["source"] for d in docs]
        assert not any(s.endswith(".txt") for s in sources)
        assert any(s.endswith(".css") for s in sources)

    # ── _build_trace_documents ──────────────────────────────────

    def test_build_trace_docs_full(self):
        """Full trace summary with all fields produces 5 documents."""
        trace = {
            "agent_summary": "Found a crash",
            "console_errors": [{"type": "error", "text": "TypeError"}],
            "failed_network_requests": [{"status": 500, "url": "http://api/data"}],
            "action_log": json.dumps([{"name": "browser_navigate", "args": {"url": "http://localhost"}}]),
        }
        docs = _build_trace_documents(trace, "Cart is broken")
        assert len(docs) == 5  # bug_report + summary + console + network + actions
        assert all(d.metadata["type"] == "trace" for d in docs)

    def test_build_trace_docs_minimal(self):
        """Empty trace still produces the bug report document."""
        docs = _build_trace_documents({}, "Some bug")
        assert len(docs) >= 1
        assert docs[0].metadata["source"] == "bug_report"
        assert "Some bug" in docs[0].page_content

    def test_build_trace_docs_no_action_log(self):
        """Missing action_log is handled without error."""
        docs = _build_trace_documents({"agent_summary": "x"}, "bug")
        sources = [d.metadata["source"] for d in docs]
        assert "action_log" not in sources

    def test_build_trace_docs_bad_json_action_log(self):
        """Invalid JSON in action_log is silently skipped."""
        trace = {"action_log": "{{invalid json}}"}
        docs = _build_trace_documents(trace, "bug")
        sources = [d.metadata["source"] for d in docs]
        assert "action_log" not in sources

    # ── rag_analyzer_node (mocked LLM) ──────────────────────────

    @pytest.mark.asyncio
    async def test_rag_analyzer_node_returns_analysis(self):
        """rag_analyzer_node returns root_cause_analysis and relevant_files when LLM responds."""
        state = {
            "bug_report": "Add-to-cart button crashes",
            "trace_summary": {
                "agent_summary": "Crash on click",
                "console_errors": [{"type": "error", "text": "TypeError"}],
                "failed_network_requests": [],
                "action_log": "[]",
            },
        }
        mock_response = MagicMock()
        mock_response.content = "## Root Cause\nNull dereference\n```diff\n--- a/index.html\n- old\n+ new\n```"

        mock_llm_instance = AsyncMock()
        mock_llm_instance.ainvoke = AsyncMock(return_value=mock_response)

        with patch("agents.rag_analyzer.ChatOpenAI", return_value=mock_llm_instance), \
             patch("agents.rag_analyzer.HuggingFaceEmbeddings") as mock_emb, \
             patch("agents.rag_analyzer.FAISS") as mock_faiss:

            # Make FAISS.from_documents return a mock vectorstore
            mock_vs = MagicMock()
            mock_vs.similarity_search.return_value = []
            mock_faiss.from_documents.return_value = mock_vs

            result = await rag_analyzer_node(state)

        assert "root_cause_analysis" in result
        assert "Null dereference" in result["root_cause_analysis"]

    # ── rag_reanalyze_with_feedback (mocked LLM) ────────────────

    @pytest.mark.asyncio
    async def test_rag_reanalyze_with_feedback(self):
        """Re-analysis returns updated markdown incorporating feedback."""
        state = {
            "bug_report": "Button broken",
            "trace_summary": {"agent_summary": "", "console_errors": [], "failed_network_requests": [], "action_log": "[]"},
            "root_cause_analysis": "Previous analysis here",
        }
        mock_response = MagicMock()
        mock_response.content = "## Revised\nFixed based on feedback"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        with patch("agents.rag_analyzer.ChatOpenAI", return_value=mock_llm), \
             patch("agents.rag_analyzer.HuggingFaceEmbeddings"), \
             patch("agents.rag_analyzer.FAISS") as mock_faiss:

            mock_vs = MagicMock()
            mock_vs.similarity_search.return_value = []
            mock_faiss.from_documents.return_value = mock_vs

            result = await rag_reanalyze_with_feedback(state, "Check the push call too")

        assert "Revised" in result


# ================================================================
#  MODULE 3 — FEEDBACK LOOP (Flask routes + review workflow)
# ================================================================

class TestFeedbackModule:
    """Unit tests for the Flask feedback / review dashboard routes."""

    # ── POST /report validation ─────────────────────────────────

    def test_report_empty_body_returns_400(self, flask_client):
        """POST /report with empty bug_report is rejected."""
        resp = flask_client.post("/report", json={"bug_report": ""})
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_report_missing_field_returns_400(self, flask_client):
        """POST /report without the key at all is rejected."""
        resp = flask_client.post("/report", json={})
        assert resp.status_code == 400

    # ── GET /review/analysis ────────────────────────────────────

    def test_get_analysis_defaults(self, flask_client):
        """Returns empty analysis and bug_report before any workflow."""
        resp = flask_client.get("/review/analysis")
        data = resp.get_json()
        assert resp.status_code == 200
        assert "analysis" in data
        assert "patch_applied" in data

    # ── POST /review/push validation ────────────────────────────

    def test_push_no_analysis_returns_400(self, flask_client):
        """Push fails when no workflow has been run."""
        import main
        orig = main._latest_state
        main._latest_state = {}
        try:
            resp = flask_client.post("/review/push")
            assert resp.status_code == 400
            assert "error" in resp.get_json()
        finally:
            main._latest_state = orig

    # ── POST /review/apply validation ───────────────────────────

    def test_apply_no_analysis_returns_400(self, flask_client):
        """Apply (preview) fails when there's no analysis."""
        import main
        orig = main._latest_state
        main._latest_state = {}
        try:
            resp = flask_client.post("/review/apply")
            assert resp.status_code == 400
        finally:
            main._latest_state = orig

    # ── POST /review/feedback validation ────────────────────────

    def test_feedback_empty_body(self, flask_client):
        """Feedback with empty text returns 400."""
        resp = flask_client.post("/review/feedback", json={"feedback": ""})
        assert resp.status_code == 400

    def test_feedback_no_prior_analysis(self, flask_client):
        """Feedback without a prior workflow returns 400."""
        import main
        orig = main._latest_state
        main._latest_state = {}
        try:
            resp = flask_client.post("/review/feedback", json={"feedback": "try again"})
            assert resp.status_code == 400
            assert "No prior analysis" in resp.get_json()["error"]
        finally:
            main._latest_state = orig

    # ── POST /review/revert when nothing applied ────────────────

    def test_revert_no_patch_returns_400(self, flask_client, clean_patch_state):
        """Revert with no applied patch returns 400."""
        resp = flask_client.post("/review/revert")
        assert resp.status_code == 400

    # ── GET /review/preview ─────────────────────────────────────

    def test_preview_returns_in_memory_content(self, flask_client):
        """When _preview_data contains a file, it is served with correct content-type."""
        import main
        main._preview_data = {"index.html": "<html>patched</html>"}
        try:
            resp = flask_client.get("/review/preview/index.html")
            assert resp.status_code == 200
            assert b"patched" in resp.data
            assert "text/html" in resp.content_type
        finally:
            main._preview_data = {}

    def test_preview_css_content_type(self, flask_client):
        """CSS files are served with text/css content-type."""
        import main
        main._preview_data = {"style.css": "body { color: red; }"}
        try:
            resp = flask_client.get("/review/preview/style.css")
            assert resp.status_code == 200
            assert "text/css" in resp.content_type
        finally:
            main._preview_data = {}

    def test_preview_nonexistent_returns_404(self, flask_client):
        """Preview for a file neither in memory nor on disk returns 404."""
        resp = flask_client.get("/review/preview/nonexistent_xyz.html")
        assert resp.status_code == 404

    # ── _extract_real_error ─────────────────────────────────────

    def test_extract_error_plain(self):
        """Plain exceptions are formatted as 'Type: message'."""
        result = _extract_real_error(ValueError("bad"))
        assert "ValueError" in result and "bad" in result

    def test_extract_error_from_group(self):
        """BaseExceptionGroup is unwrapped to the inner cause."""
        inner = RuntimeError("root cause")
        group = BaseExceptionGroup("wrapper", [inner])
        result = _extract_real_error(group)
        assert "root cause" in result

    def test_extract_error_nested_group(self):
        """Doubly nested ExceptionGroup still finds the leaf error."""
        leaf = TypeError("leaf")
        inner_group = BaseExceptionGroup("inner", [leaf])
        outer_group = BaseExceptionGroup("outer", [inner_group])
        result = _extract_real_error(outer_group)
        assert "leaf" in result


# ================================================================
#  MODULE 4 — PATCH ENGINE
# ================================================================

class TestPatchEngineModule:
    """Unit tests for the diff parsing, hunk matching, and patch apply/revert engine."""

    # ── parse_diffs_from_analysis ───────────────────────────────

    def test_parse_single_replacement(self):
        """Single -/+ pair produces one diff with one hunk."""
        md = "```diff\n--- a/index.html\n- old line\n+ new line\n```"
        diffs = parse_diffs_from_analysis(md)
        assert len(diffs) == 1
        assert diffs[0]["target_file"] == "index.html"
        assert len(diffs[0]["hunks"]) == 1

    def test_parse_one_removed_two_added(self):
        """1 removed → 2 added lines form a single hunk."""
        md = "```diff\n--- a/app.js\n- save(data);\n+ data.push(item);\n+ save(data);\n```"
        diffs = parse_diffs_from_analysis(md)
        h = diffs[0]["hunks"][0]
        assert len(h["removed"]) == 1
        assert len(h["added"]) == 2

    def test_parse_multiple_blocks(self):
        """Two separate ```diff blocks produce two diff entries."""
        md = (
            "Fix 1:\n```diff\n--- a/a.html\n- x\n+ y\n```\n"
            "Fix 2:\n```diff\n--- a/b.css\n- m\n+ n\n```"
        )
        assert len(parse_diffs_from_analysis(md)) == 2

    def test_parse_no_diff_returns_empty(self):
        """Plain text without ```diff returns []."""
        assert parse_diffs_from_analysis("Everything looks fine.") == []

    def test_parse_context_lines_recorded(self):
        """Unchanged context lines before a hunk are captured in context_before."""
        md = "```diff\n--- a/f.html\n body {\n   color: black;\n- font: 12px;\n+ font: 16px;\n```"
        h = parse_diffs_from_analysis(md)[0]["hunks"][0]
        assert len(h["context_before"]) > 0

    def test_parse_skips_hunk_headers(self):
        """@@ lines are silently skipped."""
        md = "```diff\n--- a/f.js\n@@ -1,3 +1,3 @@\n- old\n+ new\n```"
        diffs = parse_diffs_from_analysis(md)
        assert len(diffs) == 1
        assert diffs[0]["hunks"][0]["removed"] == [" old"]

    def test_parse_diff_git_header_skipped(self):
        """'diff --git' lines are ignored."""
        md = "```diff\ndiff --git a/f.js b/f.js\n--- a/f.js\n- a\n+ b\n```"
        diffs = parse_diffs_from_analysis(md)
        assert diffs[0]["target_file"] == "f.js"

    # ── _find_block_in_lines ────────────────────────────────────

    def test_find_block_single(self):
        idx = _find_block_in_lines(["  a", "  b", "  c"], ["b"])
        assert idx == 1

    def test_find_block_multi(self):
        idx = _find_block_in_lines(["a", "b", "c", "d"], ["b", "c"])
        assert idx == 1

    def test_find_block_not_found(self):
        assert _find_block_in_lines(["a", "b"], ["x"]) == -1

    def test_find_block_empty_block(self):
        assert _find_block_in_lines(["a"], []) == -1

    def test_find_block_start_from(self):
        """start_from skips earlier matches."""
        lines = ["x", "a", "x", "a"]
        # First 'a' is at idx 1; start_from=2 should find idx 3
        assert _find_block_in_lines(lines, ["a"], start_from=2) == 3

    # ── _apply_diff_in_memory ───────────────────────────────────

    def test_in_memory_single_replace(self):
        src = "    let x = null;"
        diff = {"hunks": [{"removed": ["let x = null;"], "added": ["let x = [];"], "context_before": []}]}
        result = _apply_diff_in_memory(src, diff)
        assert result is not None
        assert "let x = [];" in result

    def test_in_memory_expand_block(self):
        src = "    save(data);\n    return;"
        diff = {"hunks": [{"removed": ["save(data);"], "added": ["data.push(x);", "save(data);"], "context_before": []}]}
        result = _apply_diff_in_memory(src, diff)
        assert "data.push(x);" in result
        assert "save(data);" in result

    def test_in_memory_no_match_returns_none(self):
        diff = {"hunks": [{"removed": ["nonexistent line"], "added": ["x"], "context_before": []}]}
        assert _apply_diff_in_memory("unrelated content", diff) is None

    def test_in_memory_preserves_indentation(self):
        src = "        let x = 1;"
        diff = {"hunks": [{"removed": ["let x = 1;"], "added": ["let x = 2;"], "context_before": []}]}
        result = _apply_diff_in_memory(src, diff)
        assert result.startswith("        ")

    def test_in_memory_pure_addition_with_context(self):
        """Pure addition (no removed) uses context_before to locate insert point."""
        src = "line1\nline2\nline3"
        diff = {"hunks": [{"removed": [], "added": ["inserted"], "context_before": ["line2"]}]}
        result = _apply_diff_in_memory(src, diff)
        assert result is not None
        lines = result.splitlines()
        assert "inserted" in lines
        assert lines.index("inserted") == 2  # after line2

    # ── _apply_diff_to_file ─────────────────────────────────────

    def test_apply_to_file_replaces(self, tmp_source_dir):
        fp = os.path.join(tmp_source_dir, "index.html")
        diff = {"hunks": [{"removed": ["let cartData = JSON.parse(localStorage.getItem('cart'));"],
                           "added": ["let cartData = JSON.parse(localStorage.getItem('cart')) || [];"],
                           "context_before": []}]}
        assert _apply_diff_to_file(fp, diff) is True
        with open(fp, "r") as f:
            assert "|| []" in f.read()

    def test_apply_to_file_no_hunks(self, tmp_source_dir):
        fp = os.path.join(tmp_source_dir, "index.html")
        assert _apply_diff_to_file(fp, {"hunks": []}) is False

    # ── Patch state management ──────────────────────────────────

    def test_load_empty_state(self, clean_patch_state):
        state = _load_patch_state()
        assert state["applied"] is False

    def test_save_and_load_roundtrip(self, clean_patch_state):
        _save_patch_state({"applied": True, "iterations": [{"feedback": "ok"}]})
        state = _load_patch_state()
        assert state["applied"] is True
        assert state["iterations"][0]["feedback"] == "ok"

    def test_add_iteration_stores_feedback(self, clean_patch_state):
        iters = add_iteration(feedback="Fix null", analysis="analysis text")
        assert len(iters) == 1
        assert iters[0]["feedback"] == "Fix null"
        assert "timestamp" in iters[0]
        assert iters[0]["analysis_preview"] == "analysis text"

    def test_add_iteration_appends(self, clean_patch_state):
        add_iteration(feedback="first")
        iters = add_iteration(feedback="second")
        assert len(iters) == 2

    def test_get_patch_state(self, clean_patch_state):
        _save_patch_state({"applied": True, "iterations": []})
        state = get_patch_state()
        assert state["applied"] is True

    # ── preview_patch (integration‑ish, mocked source dir) ──────

    def test_preview_patch_no_diffs(self):
        result = preview_patch("no diffs here")
        assert result["status"] == "error"

    # ── apply_patch blocks double apply ─────────────────────────

    def test_apply_patch_rejects_when_already_applied(self, clean_patch_state):
        _save_patch_state({"applied": True, "iterations": []})
        result = apply_patch("```diff\n--- a/x\n- a\n+ b\n```")
        assert result["status"] == "error"
        assert "Revert first" in result["error"]

    # ── revert_patch when nothing applied ───────────────────────

    def test_revert_no_patch(self, clean_patch_state):
        result = revert_patch()
        assert result["status"] == "error"
        assert "No patch" in result["error"]


# ================================================================
#  MODULE 5 — GRAPH & STATE
# ================================================================

class TestGraphAndStateModule:
    """Unit tests for the LangGraph workflow graph and AgentState type."""

    def test_build_graph_compiles(self):
        """build_graph produces a compiled graph with ainvoke."""
        graph = build_graph()
        assert graph is not None
        assert hasattr(graph, "ainvoke")

    def test_agent_state_accepts_minimal_keys(self):
        """AgentState can be instantiated with only required keys."""
        state = AgentState(bug_report="test", iteration_count=0)
        assert state["bug_report"] == "test"
        assert state["iteration_count"] == 0

    def test_agent_state_all_keys(self):
        """AgentState accepts all defined keys without error."""
        state = AgentState(
            bug_report="test",
            reproduction_script="[]",
            trace_summary={},
            screenshots=[],
            console_logs="[]",
            root_cause_analysis="",
            relevant_files=[],
            candidate_patch="",
            diff_hunk="",
            iteration_count=1,
            test_results="",
            human_feedback="some feedback",
        )
        assert state["human_feedback"] == "some feedback"

    def test_max_turns_constant(self):
        """MAX_TURNS safety constant is a positive integer."""
        assert isinstance(MAX_TURNS, int)
        assert MAX_TURNS > 0

    def test_source_extensions_contains_html(self):
        """SOURCE_EXTENSIONS includes core web file types."""
        assert ".html" in SOURCE_EXTENSIONS
        assert ".css" in SOURCE_EXTENSIONS
        assert ".js" in SOURCE_EXTENSIONS
