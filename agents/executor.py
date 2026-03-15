import json
from core.state import AgentState
from utils.browser_tools import replay_and_capture_trace


async def browser_execution_node(state: AgentState):
    """
    Replays the actions from the planner's MCP session in a fresh Playwright
    browser with tracing enabled, capturing a .zip trace file.
    """
    trace_summary = state.get("trace_summary", {})
    action_log = trace_summary.get("action_log", "[]")
    network_errors = trace_summary.get("failed_network_requests", [])
    console_errors = trace_summary.get("console_errors", [])

    print(f"\n[MAWSR] Step 2/3 — Executor (Playwright trace replay)")
    print(f"\n[Executor] MCP detected {len(network_errors)} network error(s), "
          f"{len(console_errors)} console error(s).")

    # Replay actions with Playwright tracing
    print(f"[Executor] Replaying actions with Playwright tracing...")
    replay_result = replay_and_capture_trace(action_log)
    # Handle both sync and async
    if hasattr(replay_result, '__await__'):
        replay_result = await replay_result

    # Merge errors: keep MCP-detected ones + any new ones from replay
    all_network = network_errors + replay_result.get("failed_network_requests", [])
    all_console = console_errors + replay_result.get("console_errors", [])

    # Deduplicate errors by text
    seen = set()
    deduped_console = []
    for e in all_console:
        key = e.get("text", "")
        if key not in seen:
            seen.add(key)
            deduped_console.append(e)

    seen_net = set()
    deduped_network = []
    for e in all_network:
        key = f"{e.get('status', '')}_{e.get('url', '')}"
        if key not in seen_net:
            seen_net.add(key)
            deduped_network.append(e)

    # Update trace summary
    trace_summary["failed_network_requests"] = deduped_network
    trace_summary["console_errors"] = deduped_console
    trace_summary["trace_file"] = replay_result.get("trace_file", "")
    if replay_result.get("final_screenshot"):
        trace_summary["final_screenshot"] = replay_result["final_screenshot"]

    print(f"\n[Executor] Final results:")
    print(f"  Network errors: {len(deduped_network)}")
    print(f"  Console errors: {len(deduped_console)}")
    print(f"  Trace file: {trace_summary.get('trace_file', 'N/A')}")

    if trace_summary.get("agent_summary"):
        print(f"\n[Agent Summary]\n{trace_summary['agent_summary']}")

    return {
        "trace_summary": trace_summary,
        "screenshots": [trace_summary["final_screenshot"]] if trace_summary.get("final_screenshot") else [],
        "console_logs": json.dumps(deduped_console, indent=2),
    }