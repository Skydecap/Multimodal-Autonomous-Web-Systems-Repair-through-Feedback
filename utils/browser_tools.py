import json
import os
import asyncio
from datetime import datetime
from playwright.async_api import async_playwright, Response, ConsoleMessage

TRACE_DIR = "artifacts/traces"


async def replay_and_capture_trace(action_log: str) -> dict:
    """
    Replays the actions captured by the MCP agentic loop in a fresh
    Playwright browser with tracing enabled.
    Saves a .zip trace file that can be viewed at https://trace.playwright.dev
    """
    os.makedirs(TRACE_DIR, exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_path = os.path.join(TRACE_DIR, f"trace_{timestamp}.zip")

    summary = {
        "failed_network_requests": [],
        "console_errors": [],
        "final_screenshot": "",
        "trace_file": "",
        "replayed_actions": 0,
    }

    actions = json.loads(action_log) if action_log else []
    if not actions:
        print("[Trace] No actions to replay.")
        return summary

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()

        # --- Start Playwright Tracing ---
        await context.tracing.start(screenshots=True, snapshots=True, sources=True)
        print(f"[Trace] Recording started...")

        page = await context.new_page()

        # --- Hooks for Network and Console ---
        async def handle_response(response: Response):
            if response.status >= 400:
                summary["failed_network_requests"].append({
                    "url": response.url,
                    "status": response.status,
                })
        page.on("response", handle_response)

        async def handle_console(msg: ConsoleMessage):
            if msg.type in ["error", "warning"]:
                summary["console_errors"].append({
                    "type": msg.type,
                    "text": msg.text,
                })
        page.on("console", handle_console)

        def handle_page_error(error):
            summary["console_errors"].append({
                "type": "exception",
                "text": str(error),
            })
        page.on("pageerror", handle_page_error)

        # --- Replay each action ---
        replayed = 0
        try:
            for action in actions:
                tool_name = action.get("name", "")
                args = action.get("args", {})

                # Only replay browser interaction actions
                if "navigate" in tool_name:
                    url = args.get("url", "")
                    print(f"  [Trace Replay] navigate -> {url}")
                    await page.goto(url, timeout=15000)
                    await page.wait_for_timeout(2000)
                    replayed += 1

                elif "click" in tool_name:
                    ref = args.get("ref")
                    selector = args.get("selector") or args.get("element")
                    text = args.get("text")

                    if selector:
                        print(f"  [Trace Replay] click -> {selector}")
                        try:
                            await page.click(selector, timeout=5000)
                        except Exception:
                            await page.click(f"text={selector}", timeout=5000)
                    elif text:
                        print(f"  [Trace Replay] click -> text={text}")
                        await page.click(f"text={text}", timeout=5000)
                    elif ref:
                        # Attempt to find by accessible name / role
                        print(f"  [Trace Replay] click -> ref={ref} (trying button role)")
                        try:
                            await page.get_by_role("button").first.click(timeout=5000)
                        except Exception:
                            print(f"    [Trace Replay] Could not resolve ref={ref}, clicking first button")
                    else:
                        print(f"  [Trace Replay] click -> no selector, skipping")
                        continue

                    await page.wait_for_timeout(3000)
                    replayed += 1

                elif "fill" in tool_name or "type" in tool_name:
                    selector = args.get("selector") or args.get("element", "input")
                    value = args.get("value") or args.get("text", "")
                    print(f"  [Trace Replay] fill -> {selector} = '{value}'")
                    await page.fill(selector, value, timeout=5000)
                    replayed += 1

                elif "select" in tool_name:
                    selector = args.get("selector") or args.get("element", "select")
                    values = args.get("values", [])
                    print(f"  [Trace Replay] select -> {selector}")
                    await page.select_option(selector, values, timeout=5000)
                    replayed += 1

                elif "hover" in tool_name:
                    selector = args.get("selector") or args.get("element", "")
                    print(f"  [Trace Replay] hover -> {selector}")
                    await page.hover(selector, timeout=5000)
                    replayed += 1

                # Skip non-interaction tools (snapshot, console_messages, etc.)

        except Exception as e:
            print(f"  [Trace Replay] Interrupted: {e}")
            summary["console_errors"].append({"type": "exception", "text": str(e)})

        finally:
            # Take final screenshot
            screenshot_path = "artifacts/final_state.png"
            await page.screenshot(path=screenshot_path)
            summary["final_screenshot"] = screenshot_path

            # Stop tracing and save
            await context.tracing.stop(path=trace_path)
            summary["trace_file"] = trace_path
            summary["replayed_actions"] = replayed

            print(f"[Trace] Recording saved: {trace_path}")
            print(f"[Trace] Replayed {replayed} actions, "
                  f"{len(summary['failed_network_requests'])} network errors, "
                  f"{len(summary['console_errors'])} console errors.")
            print(f"[Trace] View at: https://trace.playwright.dev  (open the .zip file)")

            await browser.close()

    return summary