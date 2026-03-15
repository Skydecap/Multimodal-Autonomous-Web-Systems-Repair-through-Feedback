import asyncio
import os
import traceback

from flask import Blueprint, Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from agents.rag_analyzer import rag_reanalyze_with_feedback
from utils.patch_engine import apply_patch, get_patch_state, preview_patch, revert_patch

from .service import WebRepairService


def _extract_real_error(exc: Exception) -> str:
    if isinstance(exc, BaseExceptionGroup):
        for sub in exc.exceptions:
            return _extract_real_error(sub)
    return f"{type(exc).__name__}: {exc}"


def create_dashboard_blueprint(service: WebRepairService, prefix: str = "") -> Blueprint:
    bp = Blueprint("mawsr_dashboard", __name__, url_prefix=prefix)

    @bp.route("/dashboard", methods=["GET"])
    def dashboard_ui():
        static_dir = os.path.join(os.path.dirname(__file__), "static")
        return send_from_directory(static_dir, "dashboard.html")

    @bp.route("/report", methods=["POST"])
    def handle_report():
        data = request.get_json(silent=True) or {}
        bug_report = str(data.get("bug_report", "")).strip()
        if not bug_report:
            return jsonify({"error": "Bug report text is required."}), 400

        if "target_url" in data and data["target_url"]:
            service.target_url = str(data["target_url"])
        if "source_dir" in data and data["source_dir"]:
            service.set_source_dir(str(data["source_dir"]))

        try:
            final_state = asyncio.run(service.run_workflow(bug_report))
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
        except BaseException as exc:
            real_error = _extract_real_error(exc) if isinstance(exc, BaseExceptionGroup) else str(exc)
            traceback.print_exc()
            return jsonify({"error": real_error}), 500

    @bp.route("/review/analysis", methods=["GET"])
    def get_analysis():
        patch_state = get_patch_state()
        return jsonify({
            "analysis": service.latest_state.get("root_cause_analysis", ""),
            "bug_report": service.latest_state.get("bug_report", ""),
            "iterations": patch_state.get("iterations", []),
            "patch_applied": patch_state.get("applied", False),
        })

    @bp.route("/review/apply", methods=["POST"])
    def apply_patch_route():
        analysis = service.latest_state.get("root_cause_analysis", "")
        if not analysis:
            return jsonify({"status": "error", "error": "No analysis available. Run workflow first."}), 400

        result = preview_patch(analysis, source_dir=service.source_dir)
        if result["status"] == "success":
            service.preview_data = result.get("previews", {})
            return jsonify({
                "status": "success",
                "files_modified": result["files_modified"],
                "files": list(service.preview_data.keys()),
            })
        return jsonify(result), 400

    @bp.route("/review/preview/<path:filename>", methods=["GET"])
    def serve_preview(filename: str):
        if filename in service.preview_data:
            content = service.preview_data[filename]
        else:
            source_dir = os.path.abspath(service.source_dir)
            filepath = os.path.join(source_dir, filename)
            if not os.path.isfile(filepath):
                return jsonify({"error": "File not found."}), 404
            with open(filepath, "r", encoding="utf-8", errors="ignore") as file_obj:
                content = file_obj.read()

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
        return content, 200, {"Content-Type": content_types.get(ext, "text/html; charset=utf-8")}

    @bp.route("/review/revert", methods=["POST"])
    def revert_patch_route():
        result = revert_patch(source_dir=service.source_dir)
        if result["status"] == "success":
            return jsonify(result)
        return jsonify(result), 400

    @bp.route("/review/push", methods=["POST"])
    def push_code():
        analysis = service.latest_state.get("root_cause_analysis", "")
        if not analysis:
            return jsonify({"status": "error", "error": "No analysis available. Run workflow first."}), 400

        result = apply_patch(analysis, source_dir=service.source_dir)
        if result["status"] == "success":
            return jsonify(result)
        return jsonify(result), 400

    @bp.route("/review/feedback", methods=["POST"])
    def handle_feedback():
        data = request.get_json(silent=True) or {}
        feedback = str(data.get("feedback", "")).strip()

        if not feedback:
            return jsonify({"status": "error", "error": "Feedback text is required."}), 400

        if not service.latest_state:
            return jsonify({"status": "error", "error": "No prior analysis. Run workflow first."}), 400

        try:
            patch_state = get_patch_state()
            if patch_state.get("applied"):
                revert_patch(source_dir=service.source_dir)

            new_analysis = asyncio.run(rag_reanalyze_with_feedback(service.latest_state, feedback))
            service.latest_state["root_cause_analysis"] = new_analysis

            from utils.patch_engine import add_iteration
            iterations = add_iteration(feedback=feedback, analysis=new_analysis)
            return jsonify({"status": "success", "analysis": new_analysis, "iterations": iterations})
        except Exception as exc:
            return jsonify({"status": "error", "error": str(exc)}), 500

    return bp


def create_dashboard_app(
    target_url: str | None = None,
    source_dir: str | None = None,
    route_prefix: str = "",
) -> Flask:
    app = Flask(__name__)
    CORS(app)
    service = WebRepairService(target_url=target_url, source_dir=source_dir)
    app.register_blueprint(create_dashboard_blueprint(service, prefix=route_prefix))
    return app
