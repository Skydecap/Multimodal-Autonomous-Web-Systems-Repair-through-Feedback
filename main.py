import asyncio
import subprocess
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify
from flask_cors import CORS
from core.state import AgentState
from graph import build_graph
from utils.patch_engine import (
    apply_patch, revert_patch, get_patch_state, add_iteration
)
from agents.rag_analyzer import rag_reanalyze_with_feedback

flask_app = Flask(__name__)
CORS(flask_app)  # Allow requests from the test page on port 3000

# In-memory store for the latest workflow result
_latest_state = {}


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


@flask_app.route("/report", methods=["POST"])
def handle_report():
    """Receive a bug report from the frontend and run the analysis."""
    data = request.get_json()
    bug_report = data.get("bug_report", "").strip()

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
    except Exception as e:
        print(f"[Server Error] {e}")
        return jsonify({"error": str(e)}), 500


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
    """Apply the LLM-suggested patch to the source files."""
    analysis = _latest_state.get("root_cause_analysis", "")
    if not analysis:
        return jsonify({"status": "error", "error": "No analysis available. Run workflow first."}), 400

    result = apply_patch(analysis)
    if result["status"] == "success":
        return jsonify(result)
    else:
        return jsonify(result), 400


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
    """Git commit and push the patched code."""
    patch_state = get_patch_state()
    if not patch_state.get("applied"):
        return jsonify({"status": "error", "error": "Apply the patch first before pushing."}), 400

    try:
        project_root = os.path.dirname(__file__)
        commit_msg = f"fix: auto-patch from RAG analysis (iteration {len(patch_state.get('iterations', []))})"

        subprocess.run(["git", "add", "."], cwd=project_root, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", commit_msg], cwd=project_root, check=True, capture_output=True)
        subprocess.run(["git", "push"], cwd=project_root, check=True, capture_output=True)

        return jsonify({"status": "success", "commit_message": commit_msg})
    except subprocess.CalledProcessError as e:
        return jsonify({"status": "error", "error": f"Git error: {e.stderr.decode()[:300]}"}), 500
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


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
    import os
    print("\n🚀 Bug Report Server running on http://127.0.0.1:5000")
    print("   📺 Review Dashboard: http://127.0.0.1:3000/review.html")
    print("   Submit reports via POST /report or from the web UI.\n")
    flask_app.run(host="127.0.0.1", port=5000, debug=False)