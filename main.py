import asyncio
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify
from flask_cors import CORS
from core.state import AgentState
from graph import build_graph

flask_app = Flask(__name__)
CORS(flask_app)  # Allow requests from the test page on port 3000


async def run_workflow(bug_report: str) -> dict:
    """Run the full analysis workflow for a given bug report."""
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
        # Run the async workflow in a new event loop
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


if __name__ == "__main__":
    print("\n🚀 Bug Report Server running on http://127.0.0.1:5000")
    print("   Submit reports via POST /report or from the web UI.\n")
    flask_app.run(host="127.0.0.1", port=5000, debug=False)