from typing import TypedDict, List, Optional

class AgentState(TypedDict):
    # Inputs
    bug_report: str
    
    # Artifacts
    reproduction_script: str
    trace_summary: dict
    screenshots: List[str] 
    console_logs: str
    
    # Reasoning
    root_cause_analysis: str
    relevant_files: List[str]
    
    # Outputs
    candidate_patch: str
    diff_hunk: str
    
    # Control Flow
    iteration_count: int
    test_results: str 
    human_feedback: Optional[str]