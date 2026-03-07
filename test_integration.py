"""
Integration tests for the Multimodal Autonomous Web Systems Repair project.
Tests cross-module interactions:
  1. Planner → Executor pipeline
  2. Executor → RAG Analyzer pipeline
  3. RAG → Patch Engine pipeline (parse → preview → apply → revert)
  4. Flask API end-to-end workflows (report → review → apply → push → revert → feedback)
  5. Full graph with mocked nodes

Run:  python -m pytest test_integration.py -v
"""

import os
import json
import shutil
import tempfile
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from core.state import AgentState
from graph import build_graph
from agents.planner import _parse_console_errors, _parse_network_errors
from agents.executor import browser_execution_node
from agents.rag_analyzer import (
    _load_website_sources, _build_trace_documents, rag_analyzer_node,
)
from utils.patch_engine import (
    parse_diffs_from_analysis, _apply_diff_in_memory, _apply_diff_to_file,
    apply_patch, revert_patch, preview_patch, get_patch_state, add_iteration,
    _load_patch_state, _save_patch_state, PATCH_STATE_FILE,
)
from main import flask_app, _extract_real_error


# ================================================================
#  Shared fixtures
# ================================================================

BUGGY_HTML = """\
<html>
<body>
  <script>
    function addToCart(name, price) {
        let cartData = JSON.parse(localStorage.getItem('cart'));
        const newItem = { name: name, price: price, qty: 1 };
        localStorage.setItem('cart', JSON.stringify(cartData));
        window.location.href = '';
    }
  </script>
</body>
</html>"""

LLM_ANALYSIS = """\
## Root Cause Analysis
`cartData` is null when localStorage is empty.

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

REVISED_ANALYSIS = """\
## Revised
Also fix navigation.

## Suggested Fix
```diff
--- a/index.html
- window.location.href = '';
+ window.location.href = 'cart.html';
```
"""


@pytest.fixture
def clean_patch_state():
    if os.path.exists(PATCH_STATE_FILE):
        os.remove(PATCH_STATE_FILE)
    yield
    if os.path.exists(PATCH_STATE_FILE):
        os.remove(PATCH_STATE_FILE)


@pytest.fixture
def tmp_source_dir():
    """Temp directory with a buggy index.html for disk-based tests."""
    tmpdir = tempfile.mkdtemp()
    with open(os.path.join(tmpdir, "index.html"), "w", encoding="utf-8") as f:
        f.write(BUGGY_HTML)
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def flask_client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


# ================================================================
#  1. PLANNER → EXECUTOR PIPELINE
# ================================================================

class TestPlannerToExecutor:
    """
    The planner produces a trace_summary with console/network errors and an
    action_log.  The executor replays the actions and merges + deduplicates
    errors from both stages.
    """

    @pytest.mark.asyncio
    async def test_planner_output_feeds_executor(self):
        """Planner-style trace_summary is correctly consumed by the executor."""
        planner_output = {
            "trace_summary": {
                "action_log": json.dumps([
                    {"name": "browser_navigate", "args": {"url": "http://127.0.0.1:3000"}},
                    {"name": "browser_click", "args": {"selector": "#add-to-cart"}},
                ]),
                "failed_network_requests": [{"url": "http://api/submit", "status": 500}],
                "console_errors": [{"type": "error", "text": "TypeError: null ref"}],
                "final_screenshot": "",
                "agent_summary": "Found a crash on add-to-cart click.",
            },
            "iteration_count": 1,
        }

        replay_result = {
            "failed_network_requests": [
                {"url": "http://api/submit", "status": 500},   # duplicate
                {"url": "http://api/image", "status": 404},     # new
            ],
            "console_errors": [
                {"type": "error", "text": "TypeError: null ref"},  # duplicate
                {"type": "warning", "text": "deprecated API"},     # new
            ],
            "trace_file": "artifacts/traces/trace_test.zip",
            "final_screenshot": "artifacts/final_state.png",
        }

        with patch("agents.executor.replay_and_capture_trace", return_value=replay_result):
            result = await browser_execution_node(planner_output)

        # Duplicates removed
        assert len(result["trace_summary"]["console_errors"]) == 2
        assert len(result["trace_summary"]["failed_network_requests"]) == 2
        # Trace file from replay propagated
        assert result["trace_summary"]["trace_file"] == "artifacts/traces/trace_test.zip"
        assert result["screenshots"] == ["artifacts/final_state.png"]

    @pytest.mark.asyncio
    async def test_executor_handles_empty_planner_trace(self):
        """If planner captured nothing, executor still runs cleanly."""
        planner_output = {
            "trace_summary": {
                "action_log": "[]",
                "failed_network_requests": [],
                "console_errors": [],
                "final_screenshot": "",
                "agent_summary": "",
            },
            "iteration_count": 0,
        }
        replay_result = {
            "failed_network_requests": [],
            "console_errors": [],
            "trace_file": "",
            "final_screenshot": "",
        }
        with patch("agents.executor.replay_and_capture_trace", return_value=replay_result):
            result = await browser_execution_node(planner_output)

        assert result["trace_summary"]["console_errors"] == []
        assert result["trace_summary"]["failed_network_requests"] == []
        assert result["screenshots"] == []


# ================================================================
#  2. EXECUTOR → RAG ANALYZER PIPELINE
# ================================================================

class TestExecutorToRAG:
    """
    The executor produces a trace_summary which gets fed into the RAG
    analyzer.  Test that trace artifacts are correctly indexed and that
    the LLM (mocked) receives the right context.
    """

    @pytest.mark.asyncio
    async def test_trace_docs_reach_rag_analyzer(self):
        """Executor trace_summary is converted to Documents and indexed before LLM call."""
        state = {
            "bug_report": "Add-to-cart button does nothing",
            "trace_summary": {
                "agent_summary": "Clicked button, JS error thrown",
                "console_errors": [{"type": "error", "text": "TypeError: null is not an object"}],
                "failed_network_requests": [{"status": 500, "url": "http://api/cart"}],
                "action_log": json.dumps([
                    {"name": "browser_navigate", "args": {"url": "http://localhost:3000"}},
                    {"name": "browser_click", "args": {"selector": "#add-to-cart"}},
                ]),
            },
        }

        mock_resp = MagicMock()
        mock_resp.content = LLM_ANALYSIS
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_resp)

        captured_chunks = []

        def fake_from_docs(chunks, embeddings):
            captured_chunks.extend(chunks)
            mock_vs = MagicMock()
            mock_vs.similarity_search.return_value = chunks[:3]
            return mock_vs

        with patch("agents.rag_analyzer.ChatOpenAI", return_value=mock_llm), \
             patch("agents.rag_analyzer.HuggingFaceEmbeddings"), \
             patch("agents.rag_analyzer.FAISS") as mock_faiss:
            mock_faiss.from_documents.side_effect = fake_from_docs
            result = await rag_analyzer_node(state)

        # Verify trace artifacts were indexed
        chunk_texts = " ".join(c.page_content for c in captured_chunks)
        assert "TypeError: null" in chunk_texts
        assert "browser_navigate" in chunk_texts
        assert "root_cause_analysis" in result

    @pytest.mark.asyncio
    async def test_rag_returns_parseable_diffs(self):
        """The analysis returned by the mocked LLM is parseable by the patch engine."""
        state = {
            "bug_report": "Cart button broken",
            "trace_summary": {
                "agent_summary": "Error on click",
                "console_errors": [],
                "failed_network_requests": [],
                "action_log": "[]",
            },
        }
        mock_resp = MagicMock()
        mock_resp.content = LLM_ANALYSIS
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_resp)

        with patch("agents.rag_analyzer.ChatOpenAI", return_value=mock_llm), \
             patch("agents.rag_analyzer.HuggingFaceEmbeddings"), \
             patch("agents.rag_analyzer.FAISS") as mock_faiss:
            mock_vs = MagicMock()
            mock_vs.similarity_search.return_value = []
            mock_faiss.from_documents.return_value = mock_vs
            result = await rag_analyzer_node(state)

        # Now pipe the analysis into the patch engine
        diffs = parse_diffs_from_analysis(result["root_cause_analysis"])
        assert len(diffs) == 2
        assert diffs[0]["target_file"] == "index.html"


# ================================================================
#  3. RAG → PATCH ENGINE PIPELINE (parse → preview → apply → revert)
# ================================================================

class TestRAGToPatchEngine:
    """
    Tests the full patch lifecycle:
    LLM analysis markdown → parse diffs → preview in memory → apply to disk → revert.
    """

    def test_parse_then_preview_in_memory(self):
        """Parsed diffs can be applied in-memory to produce a correct preview."""
        diffs = parse_diffs_from_analysis(LLM_ANALYSIS)
        content = BUGGY_HTML
        for d in diffs:
            result = _apply_diff_in_memory(content, d)
            if result is not None:
                content = result

        assert "|| []" in content
        assert "cartData.push(newItem);" in content

    def test_apply_then_revert_roundtrip(self, tmp_source_dir, clean_patch_state):
        """Apply patch to disk, then revert restores the original content."""
        filepath = os.path.join(tmp_source_dir, "index.html")
        original = open(filepath, "r", encoding="utf-8").read()

        with patch("utils.patch_engine._get_source_dir", return_value=tmp_source_dir):
            result = apply_patch(LLM_ANALYSIS)

        assert result["status"] == "success"
        assert result["files_modified"] >= 1
        patched = open(filepath, "r", encoding="utf-8").read()
        assert "|| []" in patched

        # Revert
        with patch("utils.patch_engine._get_source_dir", return_value=tmp_source_dir):
            rev = revert_patch()
        assert rev["status"] == "success"

        restored = open(filepath, "r", encoding="utf-8").read()
        assert "|| []" not in restored

    def test_double_apply_blocked(self, tmp_source_dir, clean_patch_state):
        """Applying again without revert is rejected."""
        with patch("utils.patch_engine._get_source_dir", return_value=tmp_source_dir):
            apply_patch(LLM_ANALYSIS)
            result = apply_patch(LLM_ANALYSIS)

        assert result["status"] == "error"
        assert "Revert first" in result["error"]

    def test_preview_matches_disk_apply(self, tmp_source_dir, clean_patch_state):
        """In-memory preview and disk apply produce the same patched content."""
        with patch("utils.patch_engine._get_source_dir", return_value=tmp_source_dir):
            preview = preview_patch(LLM_ANALYSIS)
        assert preview["status"] == "success"
        preview_content = preview["previews"].get("index.html", "")

        with patch("utils.patch_engine._get_source_dir", return_value=tmp_source_dir):
            apply_patch(LLM_ANALYSIS)
        disk_content = open(os.path.join(tmp_source_dir, "index.html"), "r").read()

        assert preview_content.strip() == disk_content.strip()

    def test_iteration_tracking_across_apply_revert(self, tmp_source_dir, clean_patch_state):
        """Iterations are recorded through apply/revert cycles."""
        with patch("utils.patch_engine._get_source_dir", return_value=tmp_source_dir):
            apply_patch(LLM_ANALYSIS)

        add_iteration(feedback="", analysis=LLM_ANALYSIS)
        state = get_patch_state()
        assert state["applied"] is True
        assert len(state["iterations"]) == 1

        with patch("utils.patch_engine._get_source_dir", return_value=tmp_source_dir):
            revert_patch()

        add_iteration(feedback="Try fixing the nav too", analysis=REVISED_ANALYSIS)
        state = get_patch_state()
        assert state["applied"] is False
        assert len(state["iterations"]) == 2
        assert state["iterations"][1]["feedback"] == "Try fixing the nav too"


# ================================================================
#  4. FLASK API END-TO-END WORKFLOWS
# ================================================================

class TestFlaskAPIWorkflows:
    """
    Integration tests for multi-step API interactions simulating
    the review dashboard workflow.
    """

    def test_report_then_analysis_available(self, flask_client, clean_patch_state):
        """After POST /report, GET /review/analysis returns the result."""
        import main

        mock_state = {
            "bug_report": "Cart button broken",
            "root_cause_analysis": LLM_ANALYSIS,
            "trace_summary": {
                "failed_network_requests": [],
                "console_errors": [{"type": "error", "text": "TypeError"}],
                "agent_summary": "Bug reproduced",
            },
            "relevant_files": ["index.html"],
        }

        with patch("main.run_workflow", new_callable=AsyncMock, return_value=mock_state), \
             patch("main.asyncio") as mock_asyncio:
            mock_asyncio.run.return_value = mock_state
            main._latest_state = mock_state

        resp = flask_client.get("/review/analysis")
        data = resp.get_json()
        assert resp.status_code == 200
        assert "cartData" in data["analysis"]
        assert data["bug_report"] == "Cart button broken"

    def test_apply_then_preview_serves_patched(self, flask_client, clean_patch_state):
        """POST /review/apply populates preview; GET /review/preview serves it."""
        import main
        main._latest_state = {"root_cause_analysis": LLM_ANALYSIS}

        # Preview uses the real test/ directory — mock _get_source_dir
        with patch("utils.patch_engine._get_source_dir") as mock_dir:
            tmpdir = tempfile.mkdtemp()
            with open(os.path.join(tmpdir, "index.html"), "w") as f:
                f.write(BUGGY_HTML)
            mock_dir.return_value = tmpdir

            try:
                resp = flask_client.post("/review/apply")
                assert resp.status_code == 200
                data = resp.get_json()
                assert data["status"] == "success"
                assert data["files_modified"] >= 1

                # Now fetch the preview
                resp2 = flask_client.get("/review/preview/index.html")
                assert resp2.status_code == 200
                assert b"|| []" in resp2.data
            finally:
                main._preview_data = {}
                main._latest_state = {}
                shutil.rmtree(tmpdir, ignore_errors=True)

    def test_push_applies_to_disk(self, flask_client, clean_patch_state):
        """POST /review/push writes the patched file to disk."""
        import main
        main._latest_state = {"root_cause_analysis": LLM_ANALYSIS}

        tmpdir = tempfile.mkdtemp()
        with open(os.path.join(tmpdir, "index.html"), "w") as f:
            f.write(BUGGY_HTML)

        try:
            with patch("utils.patch_engine._get_source_dir", return_value=tmpdir):
                resp = flask_client.post("/review/push")

            assert resp.status_code == 200
            data = resp.get_json()
            assert data["status"] == "success"

            content = open(os.path.join(tmpdir, "index.html")).read()
            assert "|| []" in content
            assert "cartData.push(newItem);" in content
        finally:
            main._latest_state = {}
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_push_then_revert_via_api(self, flask_client, clean_patch_state):
        """POST /review/push then POST /review/revert restores the original."""
        import main
        main._latest_state = {"root_cause_analysis": LLM_ANALYSIS}

        tmpdir = tempfile.mkdtemp()
        filepath = os.path.join(tmpdir, "index.html")
        with open(filepath, "w") as f:
            f.write(BUGGY_HTML)

        try:
            with patch("utils.patch_engine._get_source_dir", return_value=tmpdir):
                flask_client.post("/review/push")

            patched = open(filepath).read()
            assert "|| []" in patched

            with patch("utils.patch_engine._get_source_dir", return_value=tmpdir):
                resp = flask_client.post("/review/revert")

            assert resp.status_code == 200
            restored = open(filepath).read()
            assert "|| []" not in restored
        finally:
            main._latest_state = {}
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_feedback_reruns_rag_and_updates_state(self, flask_client, clean_patch_state):
        """POST /review/feedback calls re-analysis and updates the stored analysis."""
        import main
        main._latest_state = {
            "bug_report": "Cart broken",
            "root_cause_analysis": LLM_ANALYSIS,
            "trace_summary": {
                "agent_summary": "", "console_errors": [],
                "failed_network_requests": [], "action_log": "[]",
            },
        }

        mock_resp = MagicMock()
        mock_resp.content = REVISED_ANALYSIS
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_resp)

        with patch("agents.rag_analyzer.ChatOpenAI", return_value=mock_llm), \
             patch("agents.rag_analyzer.HuggingFaceEmbeddings"), \
             patch("agents.rag_analyzer.FAISS") as mock_faiss:
            mock_vs = MagicMock()
            mock_vs.similarity_search.return_value = []
            mock_faiss.from_documents.return_value = mock_vs

            resp = flask_client.post("/review/feedback", json={"feedback": "Fix the navigation too"})

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert "Revised" in data["analysis"]
        assert len(data["iterations"]) >= 1

        # _latest_state is updated
        assert "Revised" in main._latest_state["root_cause_analysis"]
        main._latest_state = {}

    def test_feedback_auto_reverts_before_rerun(self, flask_client, clean_patch_state):
        """If a patch is applied, feedback auto-reverts it before re-analyzing."""
        import main

        tmpdir = tempfile.mkdtemp()
        filepath = os.path.join(tmpdir, "index.html")
        with open(filepath, "w") as f:
            f.write(BUGGY_HTML)

        main._latest_state = {
            "bug_report": "Bug",
            "root_cause_analysis": LLM_ANALYSIS,
            "trace_summary": {
                "agent_summary": "", "console_errors": [],
                "failed_network_requests": [], "action_log": "[]",
            },
        }

        try:
            # Apply patch first
            with patch("utils.patch_engine._get_source_dir", return_value=tmpdir):
                apply_patch(LLM_ANALYSIS)

            assert get_patch_state()["applied"] is True

            mock_resp = MagicMock()
            mock_resp.content = REVISED_ANALYSIS
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_resp)

            with patch("agents.rag_analyzer.ChatOpenAI", return_value=mock_llm), \
                 patch("agents.rag_analyzer.HuggingFaceEmbeddings"), \
                 patch("agents.rag_analyzer.FAISS") as mock_faiss, \
                 patch("utils.patch_engine._get_source_dir", return_value=tmpdir):
                mock_vs = MagicMock()
                mock_vs.similarity_search.return_value = []
                mock_faiss.from_documents.return_value = mock_vs

                resp = flask_client.post("/review/feedback", json={"feedback": "try again"})

            assert resp.status_code == 200
            # Patch was auto-reverted
            assert get_patch_state()["applied"] is False
        finally:
            main._latest_state = {}
            shutil.rmtree(tmpdir, ignore_errors=True)


# ================================================================
#  5. FULL GRAPH WITH MOCKED NODES
# ================================================================

class TestFullGraphIntegration:
    """
    Test the compiled LangGraph executing planner → executor → rag_analyzer
    end-to-end with all external calls mocked.
    """

    @pytest.mark.asyncio
    async def test_graph_runs_all_nodes(self):
        """Graph invokes planner → executor → rag_analyzer in sequence."""
        planner_return = {
            "reproduction_script": json.dumps([{"name": "browser_navigate", "args": {"url": "http://localhost:3000"}}]),
            "trace_summary": {
                "action_log": "[]",
                "failed_network_requests": [],
                "console_errors": [{"type": "error", "text": "TypeError"}],
                "final_screenshot": "",
                "agent_summary": "Crash found",
            },
            "screenshots": [],
            "console_logs": "[]",
            "iteration_count": 1,
        }

        executor_return = {
            "trace_summary": {
                "action_log": "[]",
                "failed_network_requests": [],
                "console_errors": [{"type": "error", "text": "TypeError"}],
                "final_screenshot": "",
                "agent_summary": "Crash found",
                "trace_file": "trace.zip",
            },
            "screenshots": [],
            "console_logs": "[]",
        }

        rag_return = {
            "root_cause_analysis": LLM_ANALYSIS,
            "relevant_files": ["index.html"],
        }

        with patch("graph.planner_node", new_callable=AsyncMock, return_value=planner_return) as mock_p, \
             patch("graph.browser_execution_node", new_callable=AsyncMock, return_value=executor_return) as mock_e, \
             patch("graph.rag_analyzer_node", new_callable=AsyncMock, return_value=rag_return) as mock_r:

            graph = build_graph()
            initial = {"bug_report": "Add-to-cart broken", "iteration_count": 0}
            final = await graph.ainvoke(initial)

        mock_p.assert_awaited_once()
        mock_e.assert_awaited_once()
        mock_r.assert_awaited_once()
        assert final["root_cause_analysis"] == LLM_ANALYSIS
        assert final["relevant_files"] == ["index.html"]

    @pytest.mark.asyncio
    async def test_graph_state_propagates_through_nodes(self):
        """State fields set by planner are visible to executor and RAG."""
        call_args = {}

        async def fake_planner(state):
            return {
                "reproduction_script": "[]",
                "trace_summary": {
                    "action_log": "[]",
                    "failed_network_requests": [],
                    "console_errors": [{"type": "error", "text": "test error"}],
                    "final_screenshot": "",
                    "agent_summary": "planner says hello",
                },
                "screenshots": [],
                "console_logs": "[]",
                "iteration_count": 1,
            }

        async def fake_executor(state):
            call_args["executor_summary"] = state.get("trace_summary", {}).get("agent_summary", "")
            return {
                "trace_summary": state.get("trace_summary", {}),
                "screenshots": [],
                "console_logs": "[]",
            }

        async def fake_rag(state):
            call_args["rag_summary"] = state.get("trace_summary", {}).get("agent_summary", "")
            return {
                "root_cause_analysis": "fixed",
                "relevant_files": [],
            }

        with patch("graph.planner_node", side_effect=fake_planner), \
             patch("graph.browser_execution_node", side_effect=fake_executor), \
             patch("graph.rag_analyzer_node", side_effect=fake_rag):

            graph = build_graph()
            await graph.ainvoke({"bug_report": "test", "iteration_count": 0})

        assert call_args["executor_summary"] == "planner says hello"
        assert call_args["rag_summary"] == "planner says hello"


# ================================================================
#  6. ERROR PARSER → TRACE DOCUMENT PIPELINE
# ================================================================

class TestErrorParserToTraceDocuments:
    """
    Tests that error strings parsed by planner helpers produce correct
    trace documents for RAG indexing.
    """

    def test_parsed_console_errors_become_trace_docs(self):
        """Console errors parsed from MCP text appear in trace documents."""
        text = "TypeError: Cannot read property 'push' of null\nwarning: deprecated API"
        errors = []
        _parse_console_errors(text, errors)

        trace = {"console_errors": errors, "failed_network_requests": [], "action_log": "[]"}
        docs = _build_trace_documents(trace, "Cart broken")

        console_doc = [d for d in docs if d.metadata["source"] == "console_errors"]
        assert len(console_doc) == 1
        assert "TypeError" in console_doc[0].page_content
        assert "deprecated" in console_doc[0].page_content

    def test_parsed_network_errors_become_trace_docs(self):
        """Network errors parsed from MCP text appear in trace documents."""
        text = "POST /api/cart 500 Internal Server Error\nGET /api/health 200 OK"
        errors = []
        _parse_network_errors(text, errors)

        trace = {"failed_network_requests": errors, "console_errors": [], "action_log": "[]"}
        docs = _build_trace_documents(trace, "API error")

        net_doc = [d for d in docs if d.metadata["source"] == "network_errors"]
        assert len(net_doc) == 1
        assert "500" in net_doc[0].page_content

    def test_combined_errors_produce_all_trace_sections(self):
        """Both console and network errors together produce separate trace documents."""
        console_errs = [{"type": "error", "text": "ReferenceError"}]
        net_errs = [{"status": 404, "url": "/missing"}]
        trace = {
            "console_errors": console_errs,
            "failed_network_requests": net_errs,
            "agent_summary": "Found issues",
            "action_log": json.dumps([{"name": "browser_click", "args": {}}]),
        }
        docs = _build_trace_documents(trace, "Bug report")

        sources = {d.metadata["source"] for d in docs}
        assert sources == {"bug_report", "agent_summary", "console_errors", "network_errors", "action_log"}


# ================================================================
#  7. MULTI-DIFF PATCH LIFECYCLE
# ================================================================

class TestMultiDiffPatchLifecycle:
    """
    Tests applying multiple diff blocks from a single analysis,
    verifying each is applied, and the combined revert restores everything.
    """

    def test_two_diffs_both_applied(self, tmp_source_dir, clean_patch_state):
        """Both diff blocks from the analysis are applied to the file."""
        filepath = os.path.join(tmp_source_dir, "index.html")

        with patch("utils.patch_engine._get_source_dir", return_value=tmp_source_dir):
            result = apply_patch(LLM_ANALYSIS)

        assert result["status"] == "success"
        content = open(filepath).read()
        assert "|| []" in content
        assert "cartData.push(newItem);" in content

    def test_revert_undoes_both_diffs(self, tmp_source_dir, clean_patch_state):
        """Reverting after two diffs removes both changes."""
        filepath = os.path.join(tmp_source_dir, "index.html")
        original = open(filepath).read()

        with patch("utils.patch_engine._get_source_dir", return_value=tmp_source_dir):
            apply_patch(LLM_ANALYSIS)
            revert_patch()

        restored = open(filepath).read()
        assert "|| []" not in restored
        assert "cartData.push(newItem);" not in restored

    def test_state_reflects_applied_diffs_count(self, tmp_source_dir, clean_patch_state):
        """Patch state stores the correct number of applied hunks."""
        with patch("utils.patch_engine._get_source_dir", return_value=tmp_source_dir):
            apply_patch(LLM_ANALYSIS)

        state = get_patch_state()
        assert state["applied"] is True
        assert len(state["applied_diffs"]) == 2  # two diff blocks
        total_hunks = sum(len(d.get("hunks", [])) for d in state["applied_diffs"])
        assert total_hunks == 2
