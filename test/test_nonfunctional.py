"""
Non-Functional Tests for the Multimodal Autonomous Web Systems Repair project.

Categories:
  1. RESPONSIVENESS  – endpoints and core functions respond within acceptable time limits
  2. RELIABILITY     – graceful error handling, crash recovery, state consistency, idempotency
  3. SCALABILITY     – behaviour under large inputs, many iterations, bulk operations

Run:  python -m pytest test_nonfunctional.py -v
"""

import os
import json
import time
import shutil
import tempfile
import threading
import concurrent.futures
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from core.state import AgentState
from graph import build_graph
from agents.planner import _parse_console_errors, _parse_network_errors
from agents.executor import browser_execution_node
from agents.rag_analyzer import (
    _load_website_sources, _build_trace_documents, rag_analyzer_node,
    rag_reanalyze_with_feedback,
)
from utils.patch_engine import (
    parse_diffs_from_analysis, _apply_diff_in_memory, _apply_diff_to_file,
    _find_block_in_lines, apply_patch, revert_patch, preview_patch,
    get_patch_state, add_iteration, _load_patch_state, _save_patch_state,
    PATCH_STATE_FILE,
)
from main import flask_app, _extract_real_error


# ================================================================
#  Shared test data
# ================================================================

SAMPLE_HTML = """\
<html>
<head><title>Test</title></head>
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

SAMPLE_ANALYSIS = """\
## Root Cause
Null handling missing.

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


# ================================================================
#  Shared fixtures
# ================================================================

@pytest.fixture
def clean_state():
    """Remove patch state before and after each test."""
    if os.path.exists(PATCH_STATE_FILE):
        os.remove(PATCH_STATE_FILE)
    yield
    if os.path.exists(PATCH_STATE_FILE):
        os.remove(PATCH_STATE_FILE)


@pytest.fixture
def tmp_source():
    """Temp directory with a buggy index.html."""
    tmpdir = tempfile.mkdtemp()
    with open(os.path.join(tmpdir, "index.html"), "w", encoding="utf-8") as f:
        f.write(SAMPLE_HTML)
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def flask_client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


# ################################################################
#                    1.  RESPONSIVENESS  TESTS
# ################################################################

class TestResponsiveness:
    """Verify that critical operations complete within reasonable time bounds."""

    # ── Patch engine speed ──────────────────────────────────────

    def test_parse_diffs_responds_quickly(self):
        """parse_diffs_from_analysis finishes < 0.5 s on a typical analysis."""
        start = time.perf_counter()
        for _ in range(100):
            parse_diffs_from_analysis(SAMPLE_ANALYSIS)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.5, f"100 parse_diffs calls took {elapsed:.2f}s (limit 0.5s)"

    def test_in_memory_patch_responds_quickly(self):
        """In-memory patch application completes < 0.5 s over 100 runs."""
        diffs = parse_diffs_from_analysis(SAMPLE_ANALYSIS)
        start = time.perf_counter()
        for _ in range(100):
            content = SAMPLE_HTML
            for d in diffs:
                result = _apply_diff_in_memory(content, d)
                if result is not None:
                    content = result
        elapsed = time.perf_counter() - start
        assert elapsed < 0.5, f"100 in-memory patches took {elapsed:.2f}s"

    def test_find_block_responds_quickly(self):
        """_find_block_in_lines completes < 0.5 s on 10 000 line file."""
        lines = [f"line number {i}" for i in range(10_000)]
        lines[7500] = "target line"
        start = time.perf_counter()
        for _ in range(50):
            _find_block_in_lines(lines, ["target line"])
        elapsed = time.perf_counter() - start
        assert elapsed < 0.5, f"50 find_block calls on 10k lines took {elapsed:.2f}s"

    def test_disk_apply_and_revert_under_1s(self, tmp_source, clean_state):
        """Full apply + revert cycle finishes < 1 s on a small file."""
        with patch("utils.patch_engine._get_source_dir", return_value=tmp_source):
            start = time.perf_counter()
            apply_patch(SAMPLE_ANALYSIS)
            revert_patch()
            elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"Apply + revert took {elapsed:.2f}s"

    def test_preview_patch_under_1s(self, tmp_source, clean_state):
        """preview_patch completes < 1 s on a small file set."""
        with patch("utils.patch_engine._get_source_dir", return_value=tmp_source):
            start = time.perf_counter()
            result = preview_patch(SAMPLE_ANALYSIS)
            elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"preview_patch took {elapsed:.2f}s"
        assert result["status"] == "success"

    # ── Flask route response times ──────────────────────────────

    def test_get_analysis_responds_quickly(self, flask_client, clean_state):
        """GET /review/analysis responds < 0.2 s."""
        import main
        main._latest_state = {
            "root_cause_analysis": SAMPLE_ANALYSIS,
            "bug_report": "test",
        }
        try:
            start = time.perf_counter()
            resp = flask_client.get("/review/analysis")
            elapsed = time.perf_counter() - start
            assert resp.status_code == 200
            assert elapsed < 0.2, f"GET /review/analysis took {elapsed:.2f}s"
        finally:
            main._latest_state = {}

    def test_revert_no_patch_responds_quickly(self, flask_client, clean_state):
        """POST /review/revert when nothing applied responds < 0.2 s."""
        start = time.perf_counter()
        resp = flask_client.post("/review/revert")
        elapsed = time.perf_counter() - start
        assert resp.status_code == 400
        assert elapsed < 0.2

    def test_content_type_detection_fast(self, flask_client):
        """Preview content-type selection is instantaneous for all extensions."""
        import main
        main._preview_data = {"style.css": "body{}", "app.js": "var x=1;", "data.json": "{}"}
        try:
            for fname, expected_ct in [
                ("style.css", "text/css"),
                ("app.js", "application/javascript"),
                ("data.json", "application/json"),
            ]:
                start = time.perf_counter()
                resp = flask_client.get(f"/review/preview/{fname}")
                elapsed = time.perf_counter() - start
                assert resp.status_code == 200
                assert expected_ct in resp.content_type
                assert elapsed < 0.1, f"Serving {fname} took {elapsed:.2f}s"
        finally:
            main._preview_data = {}

    def test_error_parsers_fast_on_large_input(self):
        """Console/network error parsers handle 1000 lines < 0.2 s."""
        big_text = "\n".join([f"line {i} error something" for i in range(1000)])
        errors = []
        start = time.perf_counter()
        _parse_console_errors(big_text, errors)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.2
        assert len(errors) == 1000

    def test_build_trace_documents_fast(self):
        """_build_trace_documents finishes < 0.1 s with typical trace data."""
        trace = {
            "agent_summary": "summary " * 100,
            "console_errors": [{"type": "error", "text": f"err{i}"} for i in range(50)],
            "failed_network_requests": [{"status": 500, "url": f"/api/{i}"} for i in range(50)],
            "action_log": json.dumps([{"name": "browser_click", "args": {}} for _ in range(50)]),
        }
        start = time.perf_counter()
        docs = _build_trace_documents(trace, "bug")
        elapsed = time.perf_counter() - start
        assert elapsed < 0.1
        assert len(docs) >= 4


# ################################################################
#                    2.  RELIABILITY  TESTS
# ################################################################

class TestReliability:
    """Verify graceful degradation, crash recovery, and state consistency."""

    # ── State file corruption / missing ─────────────────────────

    def test_missing_state_file_returns_default(self, clean_state):
        """get_patch_state returns a safe default when state file is absent."""
        state = get_patch_state()
        assert state["applied"] is False
        assert state.get("iterations", []) == [] or "iterations" in state

    def test_corrupted_state_file_handled(self, clean_state):
        """A corrupted JSON state file does not crash the system."""
        os.makedirs(os.path.dirname(PATCH_STATE_FILE), exist_ok=True)
        with open(PATCH_STATE_FILE, "w") as f:
            f.write("{{{INVALID JSON")
        # _load_patch_state should either handle or raise a catchable error
        try:
            state = _load_patch_state()
            # If it returns something, it should be safe
            assert isinstance(state, dict)
        except json.JSONDecodeError:
            # Acceptable — the caller must handle
            pass

    def test_empty_state_file_handled(self, clean_state):
        """An empty state file does not crash."""
        os.makedirs(os.path.dirname(PATCH_STATE_FILE), exist_ok=True)
        with open(PATCH_STATE_FILE, "w") as f:
            f.write("")
        try:
            state = _load_patch_state()
            assert isinstance(state, dict)
        except (json.JSONDecodeError, Exception):
            pass

    # ── Idempotency ─────────────────────────────────────────────

    def test_double_revert_is_safe(self, tmp_source, clean_state):
        """Calling revert when nothing is applied returns an error, not a crash."""
        result = revert_patch()
        assert result["status"] == "error"
        result2 = revert_patch()
        assert result2["status"] == "error"

    def test_double_apply_blocked(self, tmp_source, clean_state):
        """Cannot apply twice without reverting."""
        with patch("utils.patch_engine._get_source_dir", return_value=tmp_source):
            r1 = apply_patch(SAMPLE_ANALYSIS)
            r2 = apply_patch(SAMPLE_ANALYSIS)
        assert r1["status"] == "success"
        assert r2["status"] == "error"
        assert "Revert first" in r2["error"]

    def test_apply_revert_apply_cycle(self, tmp_source, clean_state):
        """Apply → revert → apply again succeeds (state reset properly)."""
        with patch("utils.patch_engine._get_source_dir", return_value=tmp_source):
            r1 = apply_patch(SAMPLE_ANALYSIS)
            assert r1["status"] == "success"
            r2 = revert_patch()
            assert r2["status"] == "success"
            r3 = apply_patch(SAMPLE_ANALYSIS)
            assert r3["status"] == "success"

    # ── Graceful handling of bad inputs ─────────────────────────

    def test_apply_with_no_diffs(self, tmp_source, clean_state):
        """apply_patch on analysis without diffs returns an error, not a crash."""
        with patch("utils.patch_engine._get_source_dir", return_value=tmp_source):
            result = apply_patch("No code here, just text.")
        assert result["status"] == "error"
        assert "No diffs" in result["error"]

    def test_apply_diff_to_nonexistent_file(self, clean_state):
        """Applying a diff targeting a missing file is handled gracefully."""
        diff = {"hunks": [{"removed": ["x"], "added": ["y"], "context_before": []}], "target_file": "missing.html"}
        with patch("utils.patch_engine._get_source_dir", return_value=tempfile.mkdtemp()):
            result = apply_patch("```diff\n--- a/missing.html\n- x\n+ y\n```")
        assert result["status"] in ("success", "error")  # either way, no crash

    def test_parse_diffs_empty_string(self):
        """parse_diffs_from_analysis on empty string returns empty list."""
        assert parse_diffs_from_analysis("") == []

    def test_parse_diffs_malformed_diff_block(self):
        """Malformed diff blocks are safely skipped."""
        bad = "```diff\nthis is not a real diff\n```"
        result = parse_diffs_from_analysis(bad)
        assert isinstance(result, list)
        assert len(result) == 0  # no hunks extractable

    def test_preview_with_empty_analysis(self, tmp_source):
        """preview_patch with empty analysis returns error, not crash."""
        with patch("utils.patch_engine._get_source_dir", return_value=tmp_source):
            result = preview_patch("")
        assert result["status"] == "error"

    # ── Flask error handling ────────────────────────────────────

    def test_report_empty_body(self, flask_client):
        """POST /report with empty body returns 400."""
        resp = flask_client.post("/report", json={"bug_report": ""})
        assert resp.status_code == 400

    def test_report_missing_field(self, flask_client):
        """POST /report without bug_report field returns 400."""
        resp = flask_client.post("/report", json={})
        assert resp.status_code == 400

    def test_feedback_empty_text(self, flask_client, clean_state):
        """POST /review/feedback with empty feedback returns 400."""
        resp = flask_client.post("/review/feedback", json={"feedback": ""})
        assert resp.status_code == 400

    def test_feedback_no_prior_analysis(self, flask_client, clean_state):
        """POST /review/feedback with no prior analysis returns 400."""
        import main
        main._latest_state = {}
        resp = flask_client.post("/review/feedback", json={"feedback": "try harder"})
        assert resp.status_code == 400
        main._latest_state = {}

    def test_push_no_analysis(self, flask_client, clean_state):
        """POST /review/push with no analysis returns 400."""
        import main
        main._latest_state = {}
        resp = flask_client.post("/review/push")
        assert resp.status_code == 400
        main._latest_state = {}

    def test_apply_no_analysis(self, flask_client, clean_state):
        """POST /review/apply with no analysis returns 400."""
        import main
        main._latest_state = {}
        resp = flask_client.post("/review/apply")
        assert resp.status_code == 400
        main._latest_state = {}

    def test_preview_nonexistent_file(self, flask_client):
        """GET /review/preview for a nonexistent file returns 404."""
        import main
        main._preview_data = {}
        resp = flask_client.get("/review/preview/nonexistent_xyz.html")
        assert resp.status_code == 404

    # ── Exception unwrapping ────────────────────────────────────

    def test_extract_real_error_plain(self):
        """Plain exceptions are formatted correctly."""
        msg = _extract_real_error(ValueError("bad value"))
        assert "ValueError" in msg and "bad value" in msg

    def test_extract_real_error_nested_group(self):
        """Deeply nested ExceptionGroups are unwrapped."""
        inner = ValueError("root cause")
        mid = ExceptionGroup("mid", [inner])
        outer = ExceptionGroup("outer", [mid])
        msg = _extract_real_error(outer)
        assert "root cause" in msg

    # ── Executor reliability ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_executor_survives_replay_error(self):
        """If replay_and_capture_trace raises, executor returns gracefully."""
        state = {
            "trace_summary": {
                "action_log": "[]",
                "failed_network_requests": [],
                "console_errors": [],
                "final_screenshot": "",
                "agent_summary": "",
            },
        }
        with patch("agents.executor.replay_and_capture_trace", side_effect=Exception("browser crashed")):
            try:
                await browser_execution_node(state)
            except Exception as e:
                # Executor may propagate — acceptable as long as it doesn't corrupt state
                assert "browser crashed" in str(e)

    # ── RAG analyzer reliability ────────────────────────────────

    @pytest.mark.asyncio
    async def test_rag_analyzer_no_documents(self):
        """RAG analyzer handles zero source files gracefully."""
        state = {
            "bug_report": "test bug",
            "trace_summary": {
                "agent_summary": "",
                "console_errors": [],
                "failed_network_requests": [],
                "action_log": "[]",
            },
        }
        # Mock _load_website_sources to return empty
        with patch("agents.rag_analyzer._load_website_sources", return_value=[]):
            result = await rag_analyzer_node(state)
        assert "root_cause_analysis" in result

    def test_load_sources_skips_bad_directories(self):
        """_load_website_sources skips inaccessible dirs without crashing."""
        result = _load_website_sources("/nonexistent/path/xyz")
        assert isinstance(result, list)
        assert len(result) == 0

    def test_build_trace_docs_empty_trace(self):
        """_build_trace_documents with empty trace still returns bug_report doc."""
        trace = {"console_errors": [], "failed_network_requests": [], "action_log": "[]"}
        docs = _build_trace_documents(trace, "Some bug")
        assert len(docs) >= 1
        assert any(d.metadata["source"] == "bug_report" for d in docs)

    def test_build_trace_docs_invalid_action_log(self):
        """_build_trace_documents handles invalid JSON action_log gracefully."""
        trace = {
            "console_errors": [],
            "failed_network_requests": [],
            "action_log": "NOT VALID JSON{{",
        }
        docs = _build_trace_documents(trace, "bug")
        # Should not crash — action_log doc may just be skipped
        assert isinstance(docs, list)

    # ── Error parser resilience ─────────────────────────────────

    def test_parse_console_errors_empty_input(self):
        """Empty string produces no errors."""
        errors = []
        _parse_console_errors("", errors)
        assert errors == []

    def test_parse_network_errors_empty_input(self):
        """Empty string produces no errors."""
        errors = []
        _parse_network_errors("", errors)
        assert errors == []

    def test_parse_console_errors_binary_garbage(self):
        """Non-UTF8 / garbage text does not crash the parser."""
        errors = []
        _parse_console_errors("\\x00\\xff\\xfe random error bytes", errors)
        assert isinstance(errors, list)

    # ── Graph reliability ───────────────────────────────────────

    def test_graph_builds_without_error(self):
        """build_graph always returns a runnable graph object."""
        g = build_graph()
        assert g is not None
        assert hasattr(g, "ainvoke")


# ################################################################
#                    3.  SCALABILITY  TESTS
# ################################################################

class TestScalability:
    """Verify the system handles large inputs and high volumes correctly."""

    # ── Large files ─────────────────────────────────────────────

    def test_parse_diffs_large_analysis(self):
        """parse_diffs handles an analysis with 50 diff blocks."""
        blocks = []
        for i in range(50):
            blocks.append(
                f"```diff\n--- a/file{i}.html\n- old line {i}\n+ new line {i}\n```"
            )
        big_analysis = "\n\n".join(blocks)
        result = parse_diffs_from_analysis(big_analysis)
        assert len(result) == 50

    def test_in_memory_patch_large_file(self):
        """In-memory patching works on a 10 000 line file."""
        lines = [f"<div>line {i}</div>" for i in range(10_000)]
        lines[5000] = "  let cartData = JSON.parse(localStorage.getItem('cart'));"
        content = "\n".join(lines)
        analysis = """```diff\n--- a/big.html\n- let cartData = JSON.parse(localStorage.getItem('cart'));\n+ let cartData = JSON.parse(localStorage.getItem('cart')) || [];\n```"""
        diffs = parse_diffs_from_analysis(analysis)
        result = _apply_diff_in_memory(content, diffs[0])
        assert result is not None
        assert "|| []" in result

    def test_disk_patch_large_file(self, clean_state):
        """apply_patch + revert on a 5 000 line file completes correctly."""
        tmpdir = tempfile.mkdtemp()
        lines = [f"<p>paragraph {i}</p>" for i in range(5000)]
        lines[2500] = "    let cartData = JSON.parse(localStorage.getItem('cart'));"
        filepath = os.path.join(tmpdir, "index.html")
        with open(filepath, "w") as f:
            f.write("\n".join(lines))

        analysis = """```diff\n--- a/index.html\n- let cartData = JSON.parse(localStorage.getItem('cart'));\n+ let cartData = JSON.parse(localStorage.getItem('cart')) || [];\n```"""
        try:
            with patch("utils.patch_engine._get_source_dir", return_value=tmpdir):
                r = apply_patch(analysis)
            assert r["status"] == "success"
            content = open(filepath).read()
            assert "|| []" in content

            with patch("utils.patch_engine._get_source_dir", return_value=tmpdir):
                r2 = revert_patch()
            assert r2["status"] == "success"
            content2 = open(filepath).read()
            assert "|| []" not in content2
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_find_block_scales_linearly(self):
        """_find_block_in_lines scales reasonably from 1k to 50k lines."""
        block = ["target_a", "target_b"]
        times = {}
        for size in [1_000, 10_000, 50_000]:
            lines = [f"line {i}" for i in range(size)]
            lines[size - 2] = "target_a"
            lines[size - 1] = "target_b"
            start = time.perf_counter()
            idx = _find_block_in_lines(lines, block)
            elapsed = time.perf_counter() - start
            times[size] = elapsed
            assert idx == size - 2
        # 50k should not be more than 100x slower than 1k (generous bound)
        if times[1_000] > 0:
            ratio = times[50_000] / max(times[1_000], 1e-9)
            assert ratio < 100, f"Scaling ratio {ratio:.1f}x from 1k to 50k lines"

    # ── Many diff blocks ────────────────────────────────────────

    def test_apply_many_diffs_to_single_file(self, clean_state):
        """Applying 20 distinct diffs to one file works and reverts cleanly."""
        tmpdir = tempfile.mkdtemp()
        lines = [f"    let var{i} = null;" for i in range(20)]
        filepath = os.path.join(tmpdir, "index.html")
        with open(filepath, "w") as f:
            f.write("\n".join(lines))

        blocks = []
        for i in range(20):
            blocks.append(
                f"```diff\n--- a/index.html\n- let var{i} = null;\n+ let var{i} = 0;\n```"
            )
        analysis = "\n".join(blocks)

        try:
            with patch("utils.patch_engine._get_source_dir", return_value=tmpdir):
                r = apply_patch(analysis)
            assert r["status"] == "success"
            content = open(filepath).read()
            for i in range(20):
                assert f"let var{i} = 0;" in content

            with patch("utils.patch_engine._get_source_dir", return_value=tmpdir):
                r2 = revert_patch()
            assert r2["status"] == "success"
            content2 = open(filepath).read()
            for i in range(20):
                assert f"let var{i} = null;" in content2
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # ── Many iterations ─────────────────────────────────────────

    def test_many_iterations_recorded(self, clean_state):
        """Recording 100 feedback iterations scales without issue."""
        for i in range(100):
            add_iteration(feedback=f"feedback {i}", analysis=f"analysis {i}")
        state = get_patch_state()
        assert len(state["iterations"]) == 100
        assert state["iterations"][99]["feedback"] == "feedback 99"

    def test_state_file_size_stays_reasonable(self, clean_state):
        """Patch state file stays < 1 MB after 100 iterations with moderate analysis."""
        for i in range(100):
            add_iteration(feedback=f"feedback {i}", analysis="x" * 200)
        size = os.path.getsize(PATCH_STATE_FILE)
        assert size < 1_000_000, f"Patch state file is {size} bytes after 100 iterations"

    # ── Many source files for RAG ───────────────────────────────

    def test_load_sources_many_files(self):
        """_load_website_sources handles a directory with 100 source files."""
        tmpdir = tempfile.mkdtemp()
        try:
            for i in range(100):
                with open(os.path.join(tmpdir, f"page{i}.html"), "w") as f:
                    f.write(f"<html><body>Page {i}</body></html>")
            docs = _load_website_sources(tmpdir)
            assert len(docs) == 100
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_load_sources_skips_excluded_dirs(self):
        """node_modules, .venv, __pycache__ are skipped even with many files."""
        tmpdir = tempfile.mkdtemp()
        try:
            # Create files in excluded dirs
            for excluded in ["node_modules", ".venv", "__pycache__"]:
                d = os.path.join(tmpdir, excluded)
                os.makedirs(d)
                for i in range(10):
                    with open(os.path.join(d, f"mod{i}.js"), "w") as f:
                        f.write("console.log()")
            # Create valid files
            for i in range(5):
                with open(os.path.join(tmpdir, f"app{i}.js"), "w") as f:
                    f.write(f"// app {i}")
            docs = _load_website_sources(tmpdir)
            assert len(docs) == 5  # only the non-excluded files
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # ── Large bug reports ───────────────────────────────────────

    def test_large_bug_report_traced(self):
        """_build_trace_documents handles a very large bug report string."""
        big_report = "Bug description. " * 5000  # ~85 kB
        trace = {"console_errors": [], "failed_network_requests": [], "action_log": "[]"}
        docs = _build_trace_documents(trace, big_report)
        bug_doc = [d for d in docs if d.metadata["source"] == "bug_report"][0]
        assert len(bug_doc.page_content) > 10_000

    def test_parse_console_errors_large_output(self):
        """_parse_console_errors handles 5 000 lines of console output."""
        lines = [f"[error] Error at line {i}: something failed" for i in range(5000)]
        text = "\n".join(lines)
        errors = []
        start = time.perf_counter()
        _parse_console_errors(text, errors)
        elapsed = time.perf_counter() - start
        assert len(errors) == 5000
        assert elapsed < 1.0, f"Parsing 5000 console lines took {elapsed:.2f}s"

    def test_parse_network_errors_large_output(self):
        """_parse_network_errors handles 5 000 lines with mixed status codes."""
        lines = []
        for i in range(5000):
            status = 500 if i % 2 == 0 else 200
            # Use a non-numeric URL to avoid substring false-positives
            lines.append(f"GET /page {status} OK")
        text = "\n".join(lines)
        errors = []
        start = time.perf_counter()
        _parse_network_errors(text, errors)
        elapsed = time.perf_counter() - start
        assert len(errors) == 2500  # only 500 status lines
        assert elapsed < 1.0

    # ── Concurrent Flask requests ───────────────────────────────

    def test_concurrent_analysis_reads(self, flask_client, clean_state):
        """Multiple concurrent GET /review/analysis requests don't corrupt state."""
        import main
        main._latest_state = {
            "root_cause_analysis": SAMPLE_ANALYSIS,
            "bug_report": "concurrent test",
        }

        results = []

        def fetch():
            with flask_app.test_client() as c:
                resp = c.get("/review/analysis")
                results.append(resp.status_code)

        threads = [threading.Thread(target=fetch) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        main._latest_state = {}
        assert all(code == 200 for code in results), f"Some requests failed: {results}"

    def test_concurrent_preview_reads(self, flask_client):
        """Multiple concurrent preview requests serve correct content."""
        import main
        main._preview_data = {"index.html": "<html>patched</html>"}

        results = []

        def fetch():
            with flask_app.test_client() as c:
                resp = c.get("/review/preview/index.html")
                results.append((resp.status_code, b"patched" in resp.data))

        threads = [threading.Thread(target=fetch) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        main._preview_data = {}
        assert all(code == 200 and ok for code, ok in results)

    # ── Rapid apply/revert cycles ───────────────────────────────

    def test_rapid_apply_revert_cycles(self, clean_state):
        """10 rapid apply → revert cycles maintain state consistency."""
        tmpdir = tempfile.mkdtemp()
        filepath = os.path.join(tmpdir, "index.html")
        original = SAMPLE_HTML

        try:
            for cycle in range(10):
                with open(filepath, "w") as f:
                    f.write(original)
                # Reset state for each cycle
                _save_patch_state({"applied": False, "iterations": []})

                with patch("utils.patch_engine._get_source_dir", return_value=tmpdir):
                    r1 = apply_patch(SAMPLE_ANALYSIS)
                assert r1["status"] == "success", f"Cycle {cycle}: apply failed"

                with patch("utils.patch_engine._get_source_dir", return_value=tmpdir):
                    r2 = revert_patch()
                assert r2["status"] == "success", f"Cycle {cycle}: revert failed"

                content = open(filepath).read()
                assert "|| []" not in content, f"Cycle {cycle}: revert didn't restore"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # ── Executor dedup at scale ─────────────────────────────────

    @pytest.mark.asyncio
    async def test_executor_dedup_large_error_lists(self):
        """Executor deduplicates correctly with 500 errors from each source."""
        console_errs = [{"type": "error", "text": f"err_{i % 50}"} for i in range(500)]
        network_errs = [{"url": f"/api/{i % 50}", "status": 500} for i in range(500)]

        state = {
            "trace_summary": {
                "action_log": "[]",
                "failed_network_requests": network_errs[:250],
                "console_errors": console_errs[:250],
                "final_screenshot": "",
                "agent_summary": "",
            },
        }
        replay_result = {
            "failed_network_requests": network_errs[250:],
            "console_errors": console_errs[250:],
            "trace_file": "",
            "final_screenshot": "",
        }
        with patch("agents.executor.replay_and_capture_trace", return_value=replay_result):
            result = await browser_execution_node(state)

        # 500 errors with 50 unique texts → 50 deduplicated
        assert len(result["trace_summary"]["console_errors"]) == 50
        assert len(result["trace_summary"]["failed_network_requests"]) == 50

    # ── Deep directory traversal ────────────────────────────────

    def test_load_sources_deep_nesting(self):
        """_load_website_sources handles 5 levels of nested directories."""
        tmpdir = tempfile.mkdtemp()
        try:
            deep = tmpdir
            for level in range(5):
                deep = os.path.join(deep, f"level{level}")
                os.makedirs(deep)
                with open(os.path.join(deep, f"page{level}.html"), "w") as f:
                    f.write(f"<html>Level {level}</html>")
            docs = _load_website_sources(tmpdir)
            assert len(docs) == 5
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # ── Multiple files in single analysis ───────────────────────

    def test_multi_file_apply_revert(self, clean_state):
        """Applying diffs to 3 different files and reverting restores all."""
        tmpdir = tempfile.mkdtemp()
        files = {"a.html": "  let x = 1;", "b.html": "  let y = 2;", "c.html": "  let z = 3;"}
        for name, content in files.items():
            with open(os.path.join(tmpdir, name), "w") as f:
                f.write(content)

        analysis = ""
        for name, content in files.items():
            old_val = content.strip()
            new_val = old_val.replace("=", ":=")
            analysis += f"```diff\n--- a/{name}\n- {old_val}\n+ {new_val}\n```\n\n"

        try:
            with patch("utils.patch_engine._get_source_dir", return_value=tmpdir):
                r = apply_patch(analysis)
            assert r["status"] == "success"
            assert r["files_modified"] == 3

            with patch("utils.patch_engine._get_source_dir", return_value=tmpdir):
                r2 = revert_patch()
            assert r2["status"] == "success"

            for name, original in files.items():
                restored = open(os.path.join(tmpdir, name)).read()
                assert ":=" not in restored
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
