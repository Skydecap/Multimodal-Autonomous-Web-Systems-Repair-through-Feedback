import asyncio
import os
from dataclasses import dataclass
from typing import Any

from core.state import AgentState
from graph import build_graph
from utils.patch_engine import add_iteration


@dataclass
class WorkflowResult:
    status: str
    network_errors: int
    console_errors: int
    summary: str
    root_cause_analysis: str
    relevant_files: list[str]
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "network_errors": self.network_errors,
            "console_errors": self.console_errors,
            "summary": self.summary,
            "root_cause_analysis": self.root_cause_analysis,
            "relevant_files": self.relevant_files,
            "details": self.details,
        }


class WebRepairService:
    """Reusable service that runs the agentic workflow and stores latest result."""

    def __init__(self, target_url: str = "http://127.0.0.1:3000", source_dir: str = "test"):
        self.target_url = target_url
        self.source_dir = self._resolve_source_dir(source_dir)
        self.latest_state: dict[str, Any] = {}
        self.preview_data: dict[str, str] = {}

    @staticmethod
    def _resolve_source_dir(source_dir: str) -> str:
        """Resolve source directory relative to current working directory when needed."""
        return source_dir if os.path.isabs(source_dir) else os.path.abspath(source_dir)

    def set_source_dir(self, source_dir: str) -> None:
        self.source_dir = self._resolve_source_dir(source_dir)

    async def run_workflow(self, bug_report: str) -> dict[str, Any]:
        app = build_graph()
        initial_state = AgentState(
            bug_report=bug_report,
            iteration_count=0,
            target_url=self.target_url,
            source_dir=self.source_dir,
        )
        final_state = await app.ainvoke(initial_state)
        self.latest_state = dict(final_state)
        self.latest_state.setdefault("target_url", self.target_url)
        self.latest_state.setdefault("source_dir", self.source_dir)
        add_iteration(feedback="", analysis=self.latest_state.get("root_cause_analysis", ""))
        return self.latest_state

    def analyze_message(self, message: str) -> dict[str, Any]:
        final_state = asyncio.run(self.run_workflow(message))
        trace = final_state.get("trace_summary", {})
        return WorkflowResult(
            status="success",
            network_errors=len(trace.get("failed_network_requests", [])),
            console_errors=len(trace.get("console_errors", [])),
            summary=trace.get("agent_summary", ""),
            root_cause_analysis=final_state.get("root_cause_analysis", ""),
            relevant_files=final_state.get("relevant_files", []),
            details={
                "failed_network_requests": trace.get("failed_network_requests", []),
                "console_errors": trace.get("console_errors", []),
            },
        ).to_dict()
