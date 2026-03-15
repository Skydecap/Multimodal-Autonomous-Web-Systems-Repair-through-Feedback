"""
Patch Engine — Parses diffs from RAG analysis and applies/reverts them on source files.
Reverts by reversing the applied diffs (no file backups needed).
"""

import os
import re
import json
from datetime import datetime

PATCH_STATE_FILE = "artifacts/patch_state.json"


def _get_source_dir(source_dir: str | None = None):
    """Get the test/ source directory."""
    if source_dir:
        if os.path.isabs(source_dir):
            return source_dir
        return os.path.abspath(source_dir)
    return os.path.abspath("test")


def _load_patch_state() -> dict:
    """Load the current patch state (backups, applied status, etc.)."""
    if os.path.exists(PATCH_STATE_FILE):
        with open(PATCH_STATE_FILE, "r") as f:
            return json.load(f)
    return {"applied": False, "backups": {}, "iterations": []}


def _save_patch_state(state: dict):
    """Persist patch state."""
    os.makedirs(os.path.dirname(PATCH_STATE_FILE), exist_ok=True)
    with open(PATCH_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def parse_diffs_from_analysis(analysis: str) -> list[dict]:
    """
    Parse diff blocks from the RAG analysis markdown into hunks.
    Each hunk has a list of removed lines and added lines that form a logical replacement block.
    Uses context lines (no +/-) to help locate changes in the source.
    Returns a list of {hunks: [{removed, added, context_before}], target_file, raw} dicts.
    """
    diffs = []
    diff_pattern = re.compile(r"```diff\s*\n([\s\S]*?)```", re.MULTILINE)
    for match in diff_pattern.finditer(analysis):
        diff_text = match.group(1)
        target_file = None
        hunks = []
        current_removed = []
        current_added = []
        context_before = []
        in_change = False

        for line in diff_text.strip().splitlines():
            stripped = line.rstrip()

            # Extract filename from metadata lines
            if stripped.startswith("--- "):
                fname_match = re.search(r"---\s+[ab]/?([\S]+)", stripped)
                if fname_match:
                    target_file = fname_match.group(1)
                continue
            if stripped.startswith("+++ "):
                fname_match = re.search(r"\+\+\+\s+[ab]/?([\S]+)", stripped)
                if fname_match:
                    target_file = fname_match.group(1)
                continue
            if stripped.startswith("@@"):
                continue
            if stripped.startswith("diff --git"):
                continue

            if stripped.startswith("-"):
                in_change = True
                current_removed.append(stripped[1:])
            elif stripped.startswith("+"):
                in_change = True
                current_added.append(stripped[1:])
            else:
                # Context line — flush any pending hunk
                if in_change and (current_removed or current_added):
                    hunks.append({
                        "removed": current_removed,
                        "added": current_added,
                        "context_before": list(context_before),
                    })
                    current_removed = []
                    current_added = []
                    in_change = False
                # Keep last few context lines to help locate changes
                context_before.append(stripped)
                if len(context_before) > 3:
                    context_before.pop(0)

        # Flush final hunk
        if current_removed or current_added:
            hunks.append({
                "removed": current_removed,
                "added": current_added,
                "context_before": list(context_before),
            })

        if hunks:
            diffs.append({
                "hunks": hunks,
                "target_file": target_file,
                "raw": diff_text.strip(),
            })
    return diffs


def _find_block_in_lines(lines: list[str], block: list[str], start_from: int = 0) -> int:
    """
    Find a consecutive block of lines in the file.
    Matches by stripped content. Returns the index of the first matching line, or -1.
    """
    if not block:
        return -1
    block_stripped = [b.strip() for b in block]
    for i in range(start_from, len(lines) - len(block) + 1):
        if all(lines[i + j].strip() == block_stripped[j] for j in range(len(block))):
            return i
    return -1


def _apply_diff_to_file(filepath: str, diff: dict) -> bool:
    """
    Apply a diff (with hunks) to a file.
    For each hunk: find the removed lines as a consecutive block, replace with added lines.
    Preserves the indentation of the first matched line.
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    hunks = diff.get("hunks", [])
    if not hunks:
        return False

    modified = False

    for hunk in hunks:
        removed = hunk.get("removed", [])
        added = hunk.get("added", [])

        if not removed and not added:
            continue

        if removed:
            # Find the block of removed lines in the file
            match_idx = _find_block_in_lines(lines, removed)
            if match_idx < 0:
                # Try matching individual lines as fallback
                for r_line in removed:
                    r_stripped = r_line.strip()
                    if not r_stripped:
                        continue
                    for li, fl in enumerate(lines):
                        if fl.strip() == r_stripped:
                            match_idx = li
                            break
                    if match_idx >= 0:
                        break

            if match_idx >= 0:
                # Detect indentation from the first matched line
                first_line = lines[match_idx]
                indent = first_line[:len(first_line) - len(first_line.lstrip())]

                # Try to replace as a consecutive block first
                block_idx = _find_block_in_lines(lines, removed)
                if block_idx >= 0:
                    # Replace the entire block
                    new_lines = [indent + a.strip() for a in added]
                    lines[block_idx:block_idx + len(removed)] = new_lines
                    modified = True
                    removed_preview = removed[0].strip()[:50]
                    print(f"    [Patch] Replaced block ({len(removed)} lines → {len(added)} lines) starting with: '{removed_preview}'")
                else:
                    # Fallback: replace lines individually where found
                    for r_line in removed:
                        r_stripped = r_line.strip()
                        if not r_stripped:
                            continue
                        for li, fl in enumerate(lines):
                            if fl.strip() == r_stripped:
                                lines.pop(li)
                                modified = True
                                print(f"    [Patch] Deleted: '{r_stripped}'")
                                break
                    # Insert added lines at the match position
                    if added:
                        insert_at = min(match_idx, len(lines))
                        for ai, a_line in enumerate(added):
                            lines.insert(insert_at + ai, indent + a_line.strip())
                            modified = True
                        print(f"    [Patch] Inserted {len(added)} line(s) at position {insert_at}")
            else:
                print(f"    [Patch] ❌ Could not find removed lines: {[r.strip()[:50] for r in removed[:3]]}")

        elif added and not removed:
            # Pure addition — use context_before to find insertion point
            context = hunk.get("context_before", [])
            insert_idx = len(lines)  # default: end of file
            if context:
                last_ctx = context[-1].strip()
                for li, fl in enumerate(lines):
                    if fl.strip() == last_ctx:
                        insert_idx = li + 1
                        break
            indent = ""
            if insert_idx > 0 and insert_idx <= len(lines):
                prev = lines[insert_idx - 1]
                indent = prev[:len(prev) - len(prev.lstrip())]
            for ai, a_line in enumerate(added):
                lines.insert(insert_idx + ai, indent + a_line.strip())
                modified = True
            print(f"    [Patch] Inserted {len(added)} new line(s) at position {insert_idx}")

    if modified:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    return modified


def apply_patch(analysis: str, source_dir: str | None = None) -> dict:
    """
    Parse diffs from RAG analysis and apply them to the source files.
    Stores the applied diffs so they can be reversed to revert.
    Returns {status, files_modified, details}.
    """
    patch_state = _load_patch_state()
    if patch_state.get("applied"):
        return {"status": "error", "error": "Patch already applied. Revert first."}

    source_dir = _get_source_dir(source_dir)
    diffs = parse_diffs_from_analysis(analysis)

    if not diffs:
        return {"status": "error", "error": "No diffs found in the analysis."}

    print(f"[Patch Engine] Found {len(diffs)} diff block(s) to apply.")

    # Apply each diff
    files_modified = set()
    details = []
    for i, diff in enumerate(diffs):
        applied = False
        target_file = diff.get("target_file")

        print(f"\n  [Diff #{i+1}] target_file={target_file}, "
              f"hunks={len(diff.get('hunks', []))}")

        # If the diff specifies a target file, try that file first
        if target_file:
            possible = [
                os.path.join(source_dir, target_file),
                os.path.join(source_dir, os.path.basename(target_file)),
            ]
            if target_file.startswith("test/") or target_file.startswith("test\\"):
                possible.append(os.path.join(source_dir, target_file[5:]))

            for fpath in possible:
                if os.path.isfile(fpath):
                    rel = os.path.relpath(fpath, source_dir)
                    if _apply_diff_to_file(fpath, diff):
                        files_modified.add(rel)
                        details.append(f"Diff #{i+1} applied to {rel}")
                        applied = True
                        break

        # Fallback: try every source file
        if not applied:
            for root, dirs, files in os.walk(source_dir):
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                for fname in files:
                    fpath = os.path.join(root, fname)
                    rel = os.path.relpath(fpath, source_dir)
                    if _apply_diff_to_file(fpath, diff):
                        files_modified.add(rel)
                        details.append(f"Diff #{i+1} applied to {rel}")
                        applied = True
                        break
                if applied:
                    break

        if not applied:
            details.append(f"Diff #{i+1} could not be matched to any file")
            print(f"  [Diff #{i+1}] ❌ Could not match to any file")

    # Store the diffs so we can reverse them on revert
    applied_diffs = []
    for diff in diffs:
        applied_diffs.append({
            "hunks": diff.get("hunks", []),
            "target_file": diff.get("target_file"),
        })

    patch_state["applied"] = True
    patch_state["applied_diffs"] = applied_diffs
    patch_state["files_modified"] = list(files_modified)
    _save_patch_state(patch_state)

    print(f"\n[Patch Engine] Result: {len(files_modified)} file(s) modified. Details: {details}")

    return {
        "status": "success",
        "files_modified": len(files_modified),
        "files": list(files_modified),
        "details": details,
    }


def _apply_diff_in_memory(content: str, diff: dict) -> str | None:
    """Apply a diff (with hunks) to content in memory. Returns new content or None if no match."""
    hunks = diff.get("hunks", [])
    if not hunks:
        return None

    lines = content.splitlines()
    modified = False

    for hunk in hunks:
        removed = hunk.get("removed", [])
        added = hunk.get("added", [])

        if not removed and not added:
            continue

        if removed:
            block_idx = _find_block_in_lines(lines, removed)
            if block_idx >= 0:
                indent = lines[block_idx][:len(lines[block_idx]) - len(lines[block_idx].lstrip())]
                new_lines = [indent + a.strip() for a in added]
                lines[block_idx:block_idx + len(removed)] = new_lines
                modified = True
            else:
                # Fallback: find individual lines
                match_idx = -1
                for r_line in removed:
                    r_stripped = r_line.strip()
                    if not r_stripped:
                        continue
                    for li, fl in enumerate(lines):
                        if fl.strip() == r_stripped:
                            match_idx = li
                            break
                    if match_idx >= 0:
                        break
                if match_idx >= 0:
                    indent = lines[match_idx][:len(lines[match_idx]) - len(lines[match_idx].lstrip())]
                    for r_line in removed:
                        r_stripped = r_line.strip()
                        if not r_stripped:
                            continue
                        for li, fl in enumerate(lines):
                            if fl.strip() == r_stripped:
                                lines.pop(li)
                                modified = True
                                break
                    insert_at = min(match_idx, len(lines))
                    for ai, a_line in enumerate(added):
                        lines.insert(insert_at + ai, indent + a_line.strip())
                        modified = True

        elif added and not removed:
            context = hunk.get("context_before", [])
            insert_idx = len(lines)
            if context:
                last_ctx = context[-1].strip()
                for li, fl in enumerate(lines):
                    if fl.strip() == last_ctx:
                        insert_idx = li + 1
                        break
            indent = ""
            if insert_idx > 0 and insert_idx <= len(lines):
                prev = lines[insert_idx - 1]
                indent = prev[:len(prev) - len(prev.lstrip())]
            for ai, a_line in enumerate(added):
                lines.insert(insert_idx + ai, indent + a_line.strip())
                modified = True

    if modified:
        return "\n".join(lines)
    return None


def preview_patch(analysis: str, source_dir: str | None = None) -> dict:
    """
    Parse diffs from RAG analysis and apply them in memory without modifying files.
    Returns {status, previews: {relative_path: patched_content}, files_modified}.
    """
    source_dir = _get_source_dir(source_dir)
    diffs = parse_diffs_from_analysis(analysis)

    if not diffs:
        return {"status": "error", "error": "No diffs found in the analysis."}

    # Read all source files into memory
    file_contents = {}
    for root, dirs, files in os.walk(source_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for fname in files:
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, source_dir)
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                file_contents[rel] = f.read()

    previews = dict(file_contents)
    files_modified = set()

    for i, diff in enumerate(diffs):
        target_file = diff.get("target_file")
        applied = False

        if target_file:
            candidates = [target_file, os.path.basename(target_file)]
            if target_file.startswith("test/") or target_file.startswith("test\\"):
                candidates.append(target_file[5:])
            for rel in candidates:
                if rel in previews:
                    new_content = _apply_diff_in_memory(previews[rel], diff)
                    if new_content is not None:
                        previews[rel] = new_content
                        files_modified.add(rel)
                        applied = True
                        break

        if not applied:
            for rel in list(previews.keys()):
                new_content = _apply_diff_in_memory(previews[rel], diff)
                if new_content is not None:
                    previews[rel] = new_content
                    files_modified.add(rel)
                    break

    modified_previews = {rel: previews[rel] for rel in files_modified}

    return {
        "status": "success",
        "previews": modified_previews,
        "files_modified": len(files_modified),
    }


def revert_patch(source_dir: str | None = None) -> dict:
    """Revert applied patch by reversing the stored diffs (swap removed/added)."""
    patch_state = _load_patch_state()
    if not patch_state.get("applied"):
        return {"status": "error", "error": "No patch to revert."}

    source_dir = _get_source_dir(source_dir)
    applied_diffs = patch_state.get("applied_diffs", [])
    restored_files = set()

    # Reverse each diff: swap removed <-> added in each hunk
    for diff in applied_diffs:
        reversed_hunks = []
        for hunk in diff.get("hunks", []):
            reversed_hunks.append({
                "removed": hunk.get("added", []),
                "added": hunk.get("removed", []),
                "context_before": hunk.get("context_before", []),
            })
        reversed_diff = {
            "hunks": reversed_hunks,
            "target_file": diff.get("target_file"),
        }
        target_file = reversed_diff.get("target_file")
        reverted = False

        if target_file:
            candidates = [
                os.path.join(source_dir, target_file),
                os.path.join(source_dir, os.path.basename(target_file)),
            ]
            if target_file.startswith("test/") or target_file.startswith("test\\"):
                candidates.append(os.path.join(source_dir, target_file[5:]))
            for fpath in candidates:
                if os.path.isfile(fpath):
                    if _apply_diff_to_file(fpath, reversed_diff):
                        restored_files.add(os.path.relpath(fpath, source_dir))
                        reverted = True
                        break

        if not reverted:
            for root, dirs, files in os.walk(source_dir):
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                for fname in files:
                    fpath = os.path.join(root, fname)
                    if _apply_diff_to_file(fpath, reversed_diff):
                        restored_files.add(os.path.relpath(fpath, source_dir))
                        reverted = True
                        break
                if reverted:
                    break

    patch_state["applied"] = False
    patch_state.pop("applied_diffs", None)
    patch_state.pop("files_modified", None)
    _save_patch_state(patch_state)

    return {"status": "success", "files_restored": len(restored_files), "files": list(restored_files)}


def get_patch_state() -> dict:
    """Get current patch state."""
    return _load_patch_state()


def add_iteration(feedback: str = "", analysis: str = ""):
    """Record a feedback iteration."""
    patch_state = _load_patch_state()
    patch_state.setdefault("iterations", []).append({
        "feedback": feedback,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "analysis_preview": analysis[:200] if analysis else "",
    })
    _save_patch_state(patch_state)
    return patch_state["iterations"]
