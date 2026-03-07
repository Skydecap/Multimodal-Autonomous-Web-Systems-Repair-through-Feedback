import asyncio
import os
import traceback
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify
from flask_cors import CORS
from core.state import AgentState
from graph import build_graph
from utils.patch_engine import (
    apply_patch, revert_patch, get_patch_state, add_iteration, preview_patch
)
from agents.rag_analyzer import rag_reanalyze_with_feedback

flask_app = Flask(__name__)
CORS(flask_app)  # Allow requests from the test page on port 3000

# In-memory store for the latest workflow result
_latest_state = {}

# In-memory store for preview (patched content without modifying files)
_preview_data = {}


async def run_workflow(bug_report: str) -> dict:
    """Run the full analysis workflow for a given bug report."""
    global _latest_state
    app = build_graph()

    initial_state = AgentState(
        bug_report=bug_report,
        iteration_count=0,
    )

    print(f"\n{'='*60}")
    print(f"Starting Multimodal Autonomous Repair Workflow...")
    print(f"Bug Report: {bug_report}")
    print(f"{'='*60}")

    final_state = await app.ainvoke(initial_state)
    _latest_state = dict(final_state)

    # Record first iteration
    add_iteration(feedback="", analysis=_latest_state.get("root_cause_analysis", ""))

    print(f"\nWorkflow completed. Final State logged.")
    return final_state


def _extract_real_error(e: Exception) -> str:
    """Extract the actual error from ExceptionGroups / TaskGroup wrappers."""
    if isinstance(e, BaseExceptionGroup):
        # Unwrap the group to find the real cause
        for sub in e.exceptions:
            return _extract_real_error(sub)
    return f"{type(e).__name__}: {e}"


@flask_app.route("/report", methods=["POST"])
def handle_report():
    """Receive a bug report from the frontend and run the analysis."""
    data = request.get_json()
    bug_report = data.get("bug_report", "").strip()
    print("hello")
    if not bug_report:
        return jsonify({"error": "Bug report text is required."}), 400

    try:
        final_state = asyncio.run(run_workflow(bug_report))

        trace = final_state.get("trace_summary", {})
        return jsonify({
            "status": "success",
            "network_errors": len(trace.get("failed_network_requests", [])),
            "console_errors": len(trace.get("console_errors", [])),
            "summary": trace.get("agent_summary", ""),
            "root_cause_analysis": final_state.get("root_cause_analysis", ""),
            "relevant_files": final_state.get("relevant_files", []),
            "details": {
                "failed_network_requests": trace.get("failed_network_requests", []),
                "console_errors": trace.get("console_errors", []),
            },
        })
    except BaseException as e:
        real_error = _extract_real_error(e) if isinstance(e, BaseExceptionGroup) else str(e)
        print(f"[Server Error] {real_error}")
        traceback.print_exc()
        return jsonify({"error": real_error}), 500


# ============================================================
# Review Dashboard APIs
# ============================================================

@flask_app.route("/review/analysis", methods=["GET"])
def get_analysis():
    """Get the latest RAG analysis and iteration history."""
    patch_state = get_patch_state()
    return jsonify({
        "analysis": _latest_state.get("root_cause_analysis", ""),
        "bug_report": _latest_state.get("bug_report", ""),
        "iterations": patch_state.get("iterations", []),
        "patch_applied": patch_state.get("applied", False),
    })


@flask_app.route("/review/apply", methods=["POST"])
def apply_patch_route():
    """Preview the LLM-suggested patch without modifying source files."""
    global _preview_data
    analysis = _latest_state.get("root_cause_analysis", "")
    if not analysis:
        return jsonify({"status": "error", "error": "No analysis available. Run workflow first."}), 400

    result = preview_patch(analysis)
    if result["status"] == "success":
        _preview_data = result.get("previews", {})
        return jsonify({
            "status": "success",
            "files_modified": result["files_modified"],
            "files": list(_preview_data.keys()),
        })
    else:
        return jsonify(result), 400


@flask_app.route("/review/preview/<path:filename>", methods=["GET"])
def serve_preview(filename):
    """Serve the in-memory patched preview of a file, or fall back to the original on disk."""
    if filename in _preview_data:
        content = _preview_data[filename]
    else:
        # Serve the original unmodified file from disk
        source_dir = os.path.join(os.path.dirname(__file__), "test")
        filepath = os.path.join(source_dir, filename)
        if not os.path.isfile(filepath):
            return jsonify({"error": "File not found."}), 404
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

    # Determine content type
    ext = os.path.splitext(filename)[1].lower()
    content_types = {
        ".html": "text/html; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".js": "application/javascript; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".jpg": "image/jpeg",
    }
    ct = content_types.get(ext, "text/html; charset=utf-8")
    return content, 200, {"Content-Type": ct}


@flask_app.route("/review/revert", methods=["POST"])
def revert_patch_route():
    """Revert the applied patch to restore original files."""
    result = revert_patch()
    if result["status"] == "success":
        return jsonify(result)
    else:
        return jsonify(result), 400


@flask_app.route("/review/push", methods=["POST"])
def push_code():
    """Apply the patch to the actual source files in test/."""
    analysis = _latest_state.get("root_cause_analysis", "")
    if not analysis:
        return jsonify({"status": "error", "error": "No analysis available. Run workflow first."}), 400

    result = apply_patch(analysis)
    if result["status"] == "success":
        return jsonify(result)
    else:
        return jsonify(result), 400


@flask_app.route("/review/feedback", methods=["POST"])
def handle_feedback():
    """Re-run RAG analysis with human feedback."""
    global _latest_state
    data = request.get_json()
    feedback = data.get("feedback", "").strip()

    if not feedback:
        return jsonify({"status": "error", "error": "Feedback text is required."}), 400

    if not _latest_state:
        return jsonify({"status": "error", "error": "No prior analysis. Run workflow first."}), 400

    try:
        # If patch is applied, revert first
        patch_state = get_patch_state()
        if patch_state.get("applied"):
            revert_patch()

        # Re-run RAG with feedback
        new_analysis = asyncio.run(
            rag_reanalyze_with_feedback(_latest_state, feedback)
        )

        _latest_state["root_cause_analysis"] = new_analysis
        iterations = add_iteration(feedback=feedback, analysis=new_analysis)

        return jsonify({
            "status": "success",
            "analysis": new_analysis,
            "iterations": iterations,
        })
    except Exception as e:
        print(f"[Feedback Error] {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


if __name__ == "__main__":
    print("\n🚀 Bug Report Server running on http://127.0.0.1:5000")
    print("   📺 Review Dashboard: http://127.0.0.1:3000/review.html")
    print("   Submit reports via POST /report or from the web UI.\n")
    flask_app.run(host="127.0.0.1", port=5000, debug=False)