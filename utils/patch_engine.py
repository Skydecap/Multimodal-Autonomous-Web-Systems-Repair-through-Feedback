"""
Patch Engine — Parses diffs from RAG analysis and applies/reverts them on source files.
Reverts by reversing the applied diffs (no file backups needed).
"""

import os
import re
import json
from datetime import datetime

PATCH_STATE_FILE = "artifacts/patch_state.json"


def _get_source_dir():
    """Get the test/ source directory."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "test")


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
    Parse diff blocks from the RAG analysis markdown.
    Handles unified diff headers (--- / +++ / @@) gracefully.
    Returns a list of {removed, added, target_file, raw} dicts.
    Each removed[i] -> added[i] is a single line replacement pair.
    """
    diffs = []
    # Match ```diff ... ``` blocks
    diff_pattern = re.compile(r"```diff\s*\n([\s\S]*?)```", re.MULTILINE)
    for match in diff_pattern.finditer(analysis):
        diff_text = match.group(1)
        target_file = None
        # Collect pairs: group consecutive -/+ lines into replacement pairs
        removed = []
        added = []

        for line in diff_text.strip().splitlines():
            stripped = line.rstrip()

            # Skip unified diff metadata lines, extract filename
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

            # Parse diff lines
            if stripped.startswith("-"):
                content = stripped[1:]  # Keep original spacing after the -
                removed.append(content.strip())
            elif stripped.startswith("+"):
                content = stripped[1:]
                added.append(content.strip())
            # Context lines are ignored — we only care about changes

        if removed or added:
            diffs.append({
                "removed": removed,
                "added": added,
                "target_file": target_file,
                "raw": diff_text.strip(),
            })
    return diffs


def _apply_diff_to_file(filepath: str, diff: dict) -> bool:
    """
    Apply a single diff to a file.
    Strategy: for each removed line, find it in the file and replace with
    the corresponding added line. Works even when removed lines are
    non-consecutive in the source file.
    Returns True if at least one replacement was made.
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    removed = diff.get("removed", [])
    added = diff.get("added", [])
    modified = False

    if not removed and not added:
        return False

    # --- Handle pure additions (no removed lines) ---
    if not removed and added:
        # Can't apply additions without knowing where to insert
        return False

    lines = content.splitlines()

    # --- Strategy: pair up removed→added and replace line by line ---
    # Pair them: if there are more removed than added, extra removed lines get deleted.
    # If more added than removed, extra added lines get appended after the last match.
    max_pairs = max(len(removed), len(added))

    for idx in range(min(len(removed), len(added))):
        old_s = removed[idx].strip()
        new_s = added[idx].strip()

        if not old_s:
            continue

        # Skip if old == new (no change needed)
        if old_s == new_s:
            continue

        # Find and replace in lines
        for line_idx, file_line in enumerate(lines):
            if file_line.strip() == old_s:
                indent = file_line[:len(file_line) - len(file_line.lstrip())]
                lines[line_idx] = indent + new_s
                modified = True
                print(f"    [Patch] Replaced: '{old_s}' → '{new_s}'")
                break

    # Handle extra removed lines (delete them)
    if len(removed) > len(added):
        for idx in range(len(added), len(removed)):
            old_s = removed[idx].strip()
            if not old_s:
                continue
            for line_idx, file_line in enumerate(lines):
                if file_line.strip() == old_s:
                    lines.pop(line_idx)
                    modified = True
                    print(f"    [Patch] Deleted: '{old_s}'")
                    break

    # Handle extra added lines (insert after last modified line)
    if len(added) > len(removed):
        # Find the position of the last removed line to insert after
        last_match_idx = -1
        if removed:
            last_old = removed[-1].strip()
            for line_idx, file_line in enumerate(lines):
                if file_line.strip() == last_old or (modified and line_idx > 0):
                    last_match_idx = line_idx
        
        if last_match_idx >= 0:
            indent = lines[last_match_idx][:len(lines[last_match_idx]) - len(lines[last_match_idx].lstrip())]
            for idx in range(len(removed), len(added)):
                new_s = added[idx].strip()
                if new_s:
                    lines.insert(last_match_idx + 1 + (idx - len(removed)), indent + new_s)
                    modified = True
                    print(f"    [Patch] Inserted: '{new_s}'")

    if modified:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    return modified


def apply_patch(analysis: str) -> dict:
    """
    Parse diffs from RAG analysis and apply them to the source files.
    Stores the applied diffs so they can be reversed to revert.
    Returns {status, files_modified, details}.
    """
    patch_state = _load_patch_state()
    if patch_state.get("applied"):
        return {"status": "error", "error": "Patch already applied. Revert first."}

    source_dir = _get_source_dir()
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
              f"removed={diff.get('removed', [])[:2]}, "
              f"added={diff.get('added', [])[:2]}")

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
            details.append(f"Diff #{i+1} could not be matched (removed={diff.get('removed', [])[:3]})")
            print(f"  [Diff #{i+1}] ❌ Could not match to any file")

    # Store the diffs so we can reverse them on revert
    applied_diffs = []
    for diff in diffs:
        applied_diffs.append({
            "removed": diff.get("removed", []),
            "added": diff.get("added", []),
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
    """Apply a diff to content in memory. Returns new content or None if no match."""
    removed = diff.get("removed", [])
    added = diff.get("added", [])

    if not removed and not added:
        return None
    if not removed and added:
        return None

    lines = content.splitlines()
    modified = False

    for idx in range(min(len(removed), len(added))):
        old_s = removed[idx].strip()
        new_s = added[idx].strip()
        if not old_s or old_s == new_s:
            continue
        for line_idx, file_line in enumerate(lines):
            if file_line.strip() == old_s:
                indent = file_line[:len(file_line) - len(file_line.lstrip())]
                lines[line_idx] = indent + new_s
                modified = True
                break

    if len(removed) > len(added):
        for idx in range(len(added), len(removed)):
            old_s = removed[idx].strip()
            if not old_s:
                continue
            for line_idx, file_line in enumerate(lines):
                if file_line.strip() == old_s:
                    lines.pop(line_idx)
                    modified = True
                    break

    if len(added) > len(removed):
        last_match_idx = -1
        if removed:
            last_old = removed[-1].strip()
            for line_idx, file_line in enumerate(lines):
                if file_line.strip() == last_old or (modified and line_idx > 0):
                    last_match_idx = line_idx
        if last_match_idx >= 0:
            indent = lines[last_match_idx][:len(lines[last_match_idx]) - len(lines[last_match_idx].lstrip())]
            for idx in range(len(removed), len(added)):
                new_s = added[idx].strip()
                if new_s:
                    lines.insert(last_match_idx + 1 + (idx - len(removed)), indent + new_s)
                    modified = True

    if modified:
        return "\n".join(lines)
    return None


def preview_patch(analysis: str) -> dict:
    """
    Parse diffs from RAG analysis and apply them in memory without modifying files.
    Returns {status, previews: {relative_path: patched_content}, files_modified}.
    """
    source_dir = _get_source_dir()
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


def revert_patch() -> dict:
    """Revert applied patch by reversing the stored diffs (swap removed/added)."""
    patch_state = _load_patch_state()
    if not patch_state.get("applied"):
        return {"status": "error", "error": "No patch to revert."}

    source_dir = _get_source_dir()
    applied_diffs = patch_state.get("applied_diffs", [])
    restored_files = set()

    # Reverse each diff: swap removed <-> added
    for diff in applied_diffs:
        reversed_diff = {
            "removed": diff.get("added", []),
            "added": diff.get("removed", []),
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
