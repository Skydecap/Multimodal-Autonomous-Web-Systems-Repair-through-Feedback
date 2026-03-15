"""
Shared pytest configuration — prints a test summary table at the end of each test run.
"""

import time
from collections import defaultdict

# ── Storage for per-test results ────────────────────────────────

_results = defaultdict(lambda: {"passed": 0, "failed": 0, "skipped": 0, "total_time": 0.0})
_start_times = {}
_overall_start = None


# ── Hooks ───────────────────────────────────────────────────────

def pytest_sessionstart(session):
    global _overall_start
    _overall_start = time.time()


def pytest_runtest_setup(item):
    _start_times[item.nodeid] = time.time()


def pytest_runtest_makereport(item, call):
    if call.when != "call":
        return

    # Build a consistent key from the nodeid
    nodeid = item.nodeid
    parts = nodeid.split("::")
    module = parts[0]
    cls = parts[1] if len(parts) >= 3 else "<module>"
    key = f"{module}::{cls}"

    duration = time.time() - _start_times.get(item.nodeid, time.time())
    _results[key]["total_time"] += duration

    if call.excinfo is None:
        _results[key]["passed"] += 1
    elif call.excinfo.typename == "Skipped":
        _results[key]["skipped"] += 1
    else:
        _results[key]["failed"] += 1


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not _results:
        return

    overall_elapsed = time.time() - _overall_start if _overall_start else 0

    terminalreporter.write_line("")
    terminalreporter.write_line("=" * 78)
    terminalreporter.write_line("  TEST SUMMARY REPORT")
    terminalreporter.write_line("=" * 78)

    total_passed = total_failed = total_skipped = 0
    col_w = 40

    header = f"  {'Test Class':<{col_w}} {'Passed':>8} {'Failed':>8} {'Skipped':>8} {'Time':>8}"
    terminalreporter.write_line(header)
    terminalreporter.write_line("  " + "-" * (col_w + 36))

    for key in sorted(_results.keys()):
        data = _results[key]
        p, f, s = data["passed"], data["failed"], data["skipped"]
        t = data["total_time"]
        total_passed += p
        total_failed += f
        total_skipped += s

        # Shorten display: just class name
        display_name = key.split("::")[-1]
        if len(display_name) > col_w:
            display_name = display_name[: col_w - 3] + "..."

        line = f"  {display_name:<{col_w}} {p:>8} {f:>8} {s:>8} {t:>7.2f}s"
        terminalreporter.write_line(line)

    terminalreporter.write_line("  " + "-" * (col_w + 36))

    total_tests = total_passed + total_failed + total_skipped
    status = "PASSED" if total_failed == 0 else "FAILED"

    terminalreporter.write_line(
        f"  {'TOTAL':<{col_w}} {total_passed:>8} {total_failed:>8} {total_skipped:>8} {overall_elapsed:>7.2f}s"
    )
    terminalreporter.write_line("")
    terminalreporter.write_line(f"  Result: {status}  |  {total_tests} tests  |  "
                                f"{total_passed} passed, {total_failed} failed, {total_skipped} skipped")
    terminalreporter.write_line("=" * 78)
