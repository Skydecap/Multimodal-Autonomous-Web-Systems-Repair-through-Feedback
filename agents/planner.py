import os
import json
import base64
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv
from core.state import AgentState

MAX_TURNS = 15  # Safety limit for the agentic loop


def format_mcp_tool(mcp_tool):
    """Converts an MCP tool object into an OpenAI-compatible dictionary."""
    return {
        "name": mcp_tool.name,
        "description": mcp_tool.description,
        "parameters": mcp_tool.inputSchema,
    }


def _resolve_llm_api_key() -> str | None:
    load_dotenv(override=False)
    return os.getenv("OPENAI_API_KEY") or os.getenv("GITHUB_TOKEN")


async def planner_node(state: AgentState):
    """
    Multi-turn MCP agentic loop.
    The LLM navigates, takes snapshots, interacts with elements, and
    collects console/network errors — all through the Playwright MCP server.
    Works generically on any website.
    """
    target_url = state.get("target_url") or os.getenv("MAWSR_TARGET_URL", "http://127.0.0.1:3000")
    print(f"\n[Planner] Analyzing bug report: '{state['bug_report']}'")
    print(f"[Planner] Target URL: {target_url}")

    api_key = _resolve_llm_api_key()
    if not api_key:
        error_msg = (
            "Missing API key. Set OPENAI_API_KEY (recommended) or GITHUB_TOKEN in your environment/.env."
        )
        print(f"[Planner] {error_msg}")
        return {
            "reproduction_script": "[]",
            "trace_summary": {
                "failed_network_requests": [],
                "console_errors": [{"type": "error", "text": error_msg}],
                "final_screenshot": "",
                "action_log": "[]",
                "agent_summary": f"Could not run planner: {error_msg}",
            },
            "screenshots": [],
            "console_logs": "[]",
            "iteration_count": state.get("iteration_count", 0) + 1,
        }

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        base_url="https://models.inference.ai.azure.com",
        api_key=api_key,
    )

    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@playwright/mcp@latest"],
        env={**os.environ.copy(), "PLAYWRIGHT_BROWSERS_PATH": "0"},
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # 1. Discover MCP tools and bind them to the LLM
                tools_response = await session.list_tools()
                formatted_tools = [format_mcp_tool(t) for t in tools_response.tools]

                print(f"[Planner] Bound {len(formatted_tools)} MCP tools to LLM.")

                llm_with_tools = llm.bind_tools(formatted_tools)

                # 2. Build the initial prompt
                system_prompt = (
                    "You are an autonomous web testing agent. Your goal is to reproduce a bug report "
                    "by interacting with a real browser through the provided Playwright MCP tools.\n\n"
                    "IMPORTANT WORKFLOW — follow these steps IN ORDER:\n"
                    "  1. Use 'browser_navigate' to go to the target URL.\n"
                    "  2. Use 'browser_snapshot' to get the accessibility tree of the page.\n"
                    "  3. Identify the element(s) relevant to the bug and interact with them "
                    "     (e.g. 'browser_click' using the 'ref' from the snapshot).\n"
                    "  4. After interacting, use 'browser_snapshot' again to observe the result.\n"
                    "  5. Use 'browser_console_messages' to check for JS errors.\n"
                    "  6. Use 'browser_network_requests' to check for failed network calls.\n"
                    "  7. When you have collected enough evidence, respond with a plain text summary "
                    "     of all bugs found. Do NOT call any more tools at that point.\n\n"
                    "Call ONE tool at a time. Wait for the result before deciding the next step.\n"
                    "Be thorough — click buttons, check console, check network.\n\n"
                    f"Bug report: \"{state['bug_report']}\"\n"
                    f"Target URL: {target_url}\n"
                )

                messages = [HumanMessage(content=system_prompt)]

                # 3. Agentic loop — LLM calls tools, we execute via MCP, feed results back
                action_log = []
                all_console_errors = []
                all_network_errors = []
                final_summary = ""

                for turn in range(MAX_TURNS):
                    response = await llm_with_tools.ainvoke(messages)
                    messages.append(response)

                    # If LLM has no tool calls, it's done — extract its summary
                    if not response.tool_calls:
                        final_summary = response.content
                        print(f"\n[Planner] Agent finished after {turn + 1} turns.")
                        break

                    # Execute each tool call via the real MCP session
                    for tool_call in response.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]
                        tool_id = tool_call["id"]

                        print(f"  [Turn {turn + 1}] MCP call: {tool_name}({json.dumps(tool_args)[:100]})")

                        action_log.append({"name": tool_name, "args": tool_args})

                        try:
                            result = await session.call_tool(tool_name, tool_args)

                            # Extract text content from MCP result
                            result_text = ""
                            if result.content:
                                for block in result.content:
                                    if hasattr(block, "text"):
                                        result_text += block.text

                            # Parse console/network results to capture errors
                            if tool_name == "browser_console_messages":
                                _parse_console_errors(result_text, all_console_errors)
                            elif tool_name == "browser_network_requests":
                                _parse_network_errors(result_text, all_network_errors)

                        except Exception as e:
                            result_text = f"Error executing {tool_name}: {str(e)}"
                            print(f"    [MCP Error] {result_text[:120]}")

                        # Feed the tool result back to LLM
                        messages.append(
                            ToolMessage(content=result_text[:4000], tool_call_id=tool_id)
                        )
                else:
                    print(f"\n[Planner] Reached max turns ({MAX_TURNS}), stopping.")

                # 4. Build the trace summary
                trace_summary = {
                    "failed_network_requests": all_network_errors,
                    "console_errors": all_console_errors,
                    "final_screenshot": "",
                    "action_log": json.dumps(action_log),
                    "agent_summary": final_summary,
                }

                # Try to take a final screenshot
                try:
                    screenshot_result = await session.call_tool("browser_screenshot", {})
                    os.makedirs("artifacts", exist_ok=True)
                    for block in screenshot_result.content:
                        if hasattr(block, "data"):
                            with open("artifacts/final_state.png", "wb") as f:
                                f.write(base64.b64decode(block.data))
                            trace_summary["final_screenshot"] = "artifacts/final_state.png"
                            break
                except Exception as e:
                    print(f"  [Screenshot] Could not capture: {e}")

                # Print summary
                print(f"\n[Planner] Captured {len(all_network_errors)} network error(s), "
                      f"{len(all_console_errors)} console error(s).")

                if all_network_errors:
                    for req in all_network_errors:
                        print(f"  [Network Error] {req.get('status', '?')} -> {req.get('url', '?')}")

                if all_console_errors:
                    for err in all_console_errors:
                        print(f"  [Console {err['type']}] {err['text'][:120]}")

                return {
                    "reproduction_script": json.dumps(action_log),
                    "trace_summary": trace_summary,
                    "screenshots": [trace_summary["final_screenshot"]] if trace_summary["final_screenshot"] else [],
                    "console_logs": json.dumps(all_console_errors, indent=2),
                    "iteration_count": state.get("iteration_count", 0) + 1,
                }

    except Exception as e:
        error_msg = str(e)
        # Unwrap ExceptionGroup to get real error
        if isinstance(e, BaseExceptionGroup):
            for sub in e.exceptions:
                error_msg = f"{type(sub).__name__}: {sub}"
                break
        print(f"\n[Planner] MCP connection failed: {error_msg}")
        import traceback
        traceback.print_exc()
        return {
            "reproduction_script": "[]",
            "trace_summary": {
                "failed_network_requests": [],
                "console_errors": [{"type": "error", "text": f"MCP planner failed: {error_msg}"}],
                "final_screenshot": "",
                "action_log": "[]",
                "agent_summary": f"Could not reproduce the bug — MCP connection error: {error_msg}",
            },
            "screenshots": [],
            "console_logs": "[]",
            "iteration_count": state.get("iteration_count", 0) + 1,
        }


def _parse_console_errors(text: str, errors_list: list):
    """Parse console messages text from MCP and extract errors/warnings."""
    for line in text.strip().splitlines():
        line_lower = line.lower()
        if any(kw in line_lower for kw in ["error", "warning", "warn", "typeerror", "referenceerror", "failed"]):
            msg_type = "error" if "error" in line_lower else "warning"
            errors_list.append({"type": msg_type, "text": line.strip()})


def _parse_network_errors(text: str, errors_list: list):
    """Parse network requests text from MCP and extract failed requests (4xx/5xx)."""
    for line in text.strip().splitlines():
        for status_code in ["400", "401", "403", "404", "405", "500", "502", "503", "504"]:
            if status_code in line:
                errors_list.append({
                    "url": line.strip(),
                    "status": int(status_code),
                    "text": line.strip(),
                })
                break