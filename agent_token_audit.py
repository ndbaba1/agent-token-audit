#!/usr/bin/env python3
"""
agent_token_audit.py

Estimates how much of your Claude Code token spend went to repeated tool steps
whose inputs had not changed since the last time the agent ran them, i.e. the
upper bound on what a dependency-aware cache could have saved.

Reads ~/.claude/projects/**/*.jsonl on your machine. Read-only. Nothing is sent anywhere.

Usage:
  python3 agent_token_audit.py                 # all projects
  python3 agent_token_audit.py --project vaultkit
  python3 agent_token_audit.py --git           # also check git history between repeats (slower)
  python3 agent_token_audit.py --csv steps.csv # dump every step for your own analysis
"""
import argparse
import bisect
import csv
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

FILE_EDIT_TOOLS = {"Edit": "file_path", "Write": "file_path",
                   "MultiEdit": "file_path", "NotebookEdit": "notebook_path"}
READ_ONLY_CMDS = {"ls", "cat", "head", "tail", "grep", "rg", "find", "wc", "tree", "pwd",
                  "which", "file", "stat", "du", "sed", "awk", "jq", "echo", "sort", "uniq", "cut"}
GIT_READ = {"log", "show", "diff", "status", "blame", "ls-files", "rev-parse", "grep",
            "describe", "shortlog"}
WEB_FRESH_WINDOW = timedelta(hours=24)  # assumption: web results stay valid for a day
DELTA_CONTEXT_CHARS = 800  # ~10 lines either side of each changed region in a delta read

EDIT_DETAILS = []            # every file edit: ts, path, tool, chars written, hunks
TURNS = defaultdict(list)    # session id -> timestamps of each assistant turn


def edit_size(name, inp):
    """(chars of new content, number of changed regions); chars=None means a full rewrite."""
    if name == "Edit":
        return len(inp.get("new_string") or ""), 1
    if name == "MultiEdit":
        es = inp.get("edits") or []
        return sum(len(e.get("new_string") or "") for e in es), max(len(es), 1)
    if name == "NotebookEdit":
        return len(inp.get("new_source") or ""), 1
    return None, 1  # Write replaces the whole file


def turns_after(session, ts):
    t = TURNS.get(session, [])
    return len(t) - bisect.bisect_right(t, ts)


def delta_report(pairs, total_res, cache_reads):
    if not pairs:
        print("\nDelta reads: no re-reads of a file after the agent edited it.")
        return
    read_tok = delta_tok = ctx_saved = 0
    full_reads = 0
    sizes = []
    per_file = defaultdict(lambda: Counter())
    for prev, cur in pairs:
        target = cur["deps"][0][1]
        r = cur["result_tokens"]
        es = [e for e in EDIT_DETAILS if e["path"] == target and prev["ts"] < e["ts"] < cur["ts"]]
        if not es or any(e["chars"] is None for e in es):
            d = r  # full rewrite or unknown: a delta read saves nothing
        else:
            d = min(r, sum(e["chars"] + DELTA_CONTEXT_CHARS * e["hunks"] for e in es) / 4)
        saved = r - d
        read_tok += r
        delta_tok += d
        ctx_saved += saved * turns_after(cur["session"], cur["ts"])
        sizes.append(r)
        if cur["input"].get("offset") is None and cur["input"].get("limit") is None:
            full_reads += 1
        per_file[target]["n"] += 1
        per_file[target]["saved"] += saved
    sizes.sort()
    med = sizes[len(sizes) // 2]
    print(f"\nDelta reads: re-reads of a file after the agent itself edited it")
    print(f"  re-reads: {len(pairs)}   whole-file: {full_reads}   median size: {med:,.0f} tok   largest: {sizes[-1]:,.0f} tok")
    print(f"  tokens read: {read_tok:,.0f}   if only the changed regions were returned: {delta_tok:,.0f}")
    print(f"  direct saving: {read_tok - delta_tok:,.0f} tok  ({pct(read_tok - delta_tok, total_res).strip()} of all tool-result tokens)")
    print(f"  saving incl. re-reading on later turns: {ctx_saved:,.0f} tok  "
          f"({pct(ctx_saved, cache_reads).strip()} of cache-read tokens; upper bound, ignores compaction)")
    print("  Files with the biggest savings:")
    for path, c in sorted(per_file.items(), key=lambda x: -x[1]["saved"])[:8]:
        print(f"    {c['n']:>3}x  {c['saved']:>8,.0f} tok saved  {path[-90:]}")


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def result_chars(content):
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(len(c.get("text", "")) for c in content if isinstance(c, dict))
    return 0


def is_read_only(cmd):
    cleaned = re.sub(r"\d?>&\d|\d?>\s*/dev/null", "", cmd)
    if ">" in cleaned or "`" in cleaned or "$(" in cleaned:
        return False
    for seg in re.split(r"&&|\|\||;|\|", cleaned):
        words = seg.strip().split()
        if not words:
            continue
        cmd0 = os.path.basename(words[0])
        if cmd0 == "cd":
            continue
        if cmd0 == "git":
            sub = next((w for w in words[1:] if not w.startswith("-")), "")
            if sub not in GIT_READ:
                return False
        elif cmd0 not in READ_ONLY_CMDS:
            return False
        if cmd0 == "sed" and any(w.startswith("-i") for w in words):
            return False
    return True


def step_key(name, inp, cwd):
    """Returns (key, dependencies) for a cacheable step, or (None, None)."""
    def norm(p):
        return os.path.normpath(os.path.join(cwd or "/", p)) if p else os.path.normpath(cwd or "/")

    if name == "Read":
        p = norm(inp.get("file_path"))
        return f"Read:{p}:{inp.get('offset')}:{inp.get('limit')}", [("file", p)]
    if name == "Grep":
        d = norm(inp.get("path"))
        opts = {k: inp.get(k) for k in ("pattern", "glob", "type", "output_mode",
                                        "-i", "-n", "-A", "-B", "-C", "multiline")}
        return f"Grep:{d}:{json.dumps(opts, sort_keys=True)}", [("dir", d)]
    if name == "Glob":
        d = norm(inp.get("path"))
        return f"Glob:{d}:{inp.get('pattern')}", [("dir", d)]
    if name == "Bash":
        cmd = (inp.get("command") or "").strip()
        if cmd and is_read_only(cmd):
            d = norm(None)
            return f"Bash:{d}:{cmd}", [("dir", d)]
        return None, None
    if name in ("WebFetch", "WebSearch"):
        return f"{name}:{json.dumps(inp, sort_keys=True)}", [("web", None)]
    if name.startswith("mcp__"):
        return f"{name}:{json.dumps(inp, sort_keys=True)}", [("opaque", None)]
    return None, None


def under(path, root):
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


class GitChecker:
    def __init__(self):
        self.tops = {}

    def toplevel(self, path):
        d = path if os.path.isdir(path) else os.path.dirname(path)
        if d not in self.tops:
            try:
                out = subprocess.run(["git", "-C", d, "rev-parse", "--show-toplevel"],
                                     capture_output=True, text=True, timeout=10)
                self.tops[d] = out.stdout.strip() if out.returncode == 0 else None
            except Exception:
                self.tops[d] = None
        return self.tops[d]

    def changed(self, path, t0, t1):
        top = self.toplevel(path)
        if not top:
            return False
        try:
            out = subprocess.run(["git", "-C", top, "log", f"--since={t0.isoformat()}",
                                  f"--until={t1.isoformat()}", "--format=%H", "--", path],
                                 capture_output=True, text=True, timeout=20)
            return bool(out.stdout.strip())
        except Exception:
            return False


def load(root, project_filter):
    steps, edits, mutations = [], [], []
    pending, usage_by_msg, tools_per_msg = {}, {}, Counter()
    all_tool_calls = 0
    sessions = set()
    seen_msgs = set()

    for f in sorted(Path(root).rglob("*.jsonl")):
        project = f.relative_to(root).parts[0]
        if project_filter and project_filter.lower() not in project.lower():
            continue
        with f.open(errors="replace") as fh:
            for line in fh:
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = ev.get("message")
                if not isinstance(msg, dict) or not ev.get("timestamp"):
                    continue
                ts = parse_ts(ev["timestamp"])
                cwd = ev.get("cwd") or ""
                sid = ev.get("sessionId") or f.stem
                content = msg.get("content")

                if ev.get("type") == "assistant":
                    mid = msg.get("id") or ev.get("uuid")
                    if mid not in seen_msgs:
                        seen_msgs.add(mid)
                        TURNS[sid].append(ts)
                    if msg.get("usage"):
                        usage_by_msg[mid] = msg["usage"]  # streamed chunks repeat usage; keep last
                    if isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict) and c.get("type") == "tool_use":
                                pending[c.get("id")] = dict(name=c.get("name", ""), input=c.get("input") or {},
                                                            ts=ts, cwd=cwd, project=project, session=sid, mid=mid)
                elif ev.get("type") == "user" and isinstance(content, list):
                    for c in content:
                        if not (isinstance(c, dict) and c.get("type") == "tool_result"):
                            continue
                        tu = pending.pop(c.get("tool_use_id"), None)
                        if not tu:
                            continue
                        all_tool_calls += 1
                        sessions.add(tu["session"])
                        tools_per_msg[tu["mid"]] += 1
                        name, inp = tu["name"], tu["input"]
                        if name in FILE_EDIT_TOOLS:
                            p = inp.get(FILE_EDIT_TOOLS[name])
                            if p:
                                full = os.path.normpath(os.path.join(tu["cwd"] or "/", p))
                                edits.append((tu["ts"], full))
                                chars, hunks = edit_size(name, inp)
                                EDIT_DETAILS.append(dict(ts=tu["ts"], path=full, tool=name,
                                                         chars=chars, hunks=hunks))
                            continue
                        key, deps = step_key(name, inp, tu["cwd"])
                        if key is None:
                            if name == "Bash":
                                mutations.append((tu["ts"], os.path.normpath(tu["cwd"] or "/")))
                            continue
                        tu.update(key=key, deps=deps, result_tokens=result_chars(c.get("content")) / 4)
                        steps.append(tu)

    for s in steps:
        out = (usage_by_msg.get(s["mid"]) or {}).get("output_tokens", 0) or 0
        s["out_tokens"] = out / max(tools_per_msg[s["mid"]], 1)
    steps.sort(key=lambda s: s["ts"])
    edits.sort()
    mutations.sort()
    for v in TURNS.values():
        v.sort()
    return steps, edits, mutations, usage_by_msg, all_tool_calls, sessions


def stale_reason(prev, cur, edits, mutations):
    """Why a repeat counted as stale: 'edit' (its own file/dir was edited) or 'bash' (only a shell command ran nearby)."""
    t0, t1 = prev["ts"], cur["ts"]
    for kind, target in cur["deps"]:
        if kind in ("file", "dir"):
            for ts, p in edits:
                if t0 < ts < t1 and (p == target if kind == "file" else under(p, target)):
                    return "edit"
    return "bash"


def is_fresh(prev, cur, edits, mutations, git):
    t0, t1 = prev["ts"], cur["ts"]
    for kind, target in cur["deps"]:
        if kind == "opaque":
            return None  # unknown: no way to tell if an MCP tool's source changed
        if kind == "web":
            if t1 - t0 > WEB_FRESH_WINDOW:
                return False
            continue
        for ts, p in edits:
            if t0 < ts < t1 and (p == target if kind == "file" else under(p, target)):
                return False
        for ts, d in mutations:
            if t0 < ts < t1 and (under(target, d) or under(d, target)):
                return False
        if git is not None and git.changed(target, t0, t1):
            return False
    return True


def pct(a, b):
    return f"{100 * a / b:5.1f}%" if b else "  n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path.home() / ".claude" / "projects"))
    ap.add_argument("--project", help="substring filter on project folder name")
    ap.add_argument("--git", action="store_true", help="treat git commits between repeats as changes")
    ap.add_argument("--csv", help="write every step to this CSV file")
    args = ap.parse_args()

    steps, edits, mutations, usage, all_calls, sessions = load(args.root, args.project)
    git = GitChecker() if args.git else None

    last_any, last_in_session = {}, {}
    buckets = defaultdict(lambda: Counter())
    by_tool = defaultdict(Counter)
    top = Counter()
    top_tokens = Counter()
    delta_pairs = []

    for s in steps:
        tool = s["name"] if not s["name"].startswith("mcp__") else "mcp"
        tokens = s["result_tokens"] + s["out_tokens"]
        s["status"] = "first"
        prev = last_any.get(s["key"])
        if prev:
            buckets["naive"]["n"] += 1
            buckets["naive"]["res"] += s["result_tokens"]
            buckets["naive"]["out"] += s["out_tokens"]
            fresh = is_fresh(prev, s, edits, mutations, git)
            if fresh is None:
                s["status"] = "repeat_unknown"
            elif fresh:
                same = prev["session"] == s["session"]
                s["status"] = "fresh_same_session" if same else "fresh_cross_session"
                for b in (["fresh_session"] if same else []) + ["fresh_any"]:
                    buckets[b]["n"] += 1
                    buckets[b]["res"] += s["result_tokens"]
                    buckets[b]["out"] += s["out_tokens"]
                by_tool[tool]["fresh"] += 1
                by_tool[tool]["fresh_tokens"] += tokens
                top[s["key"]] += 1
                top_tokens[s["key"]] += tokens
            else:
                reason = stale_reason(prev, s, edits, mutations)
                s["status"] = f"stale_{reason}"
                if reason == "edit" and s["name"] == "Read":
                    delta_pairs.append((prev, s))
                buckets[f"stale_{reason}"]["n"] += 1
                buckets[f"stale_{reason}"]["res"] += s["result_tokens"]
                buckets[f"stale_{reason}"]["out"] += s["out_tokens"]
        by_tool[tool]["calls"] += 1
        by_tool[tool]["tokens"] += tokens
        last_any[s["key"]] = s
        last_in_session[(s["session"], s["key"])] = s

    total_res = sum(s["result_tokens"] for s in steps)
    totals = Counter()
    for u in usage.values():
        for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens"):
            totals[k] += u.get(k, 0) or 0

    print(f"\nSessions: {len(sessions)}   tool calls: {all_calls}   cacheable steps: {len(steps)}")
    print("\nAPI usage recorded in transcripts:")
    for k, v in totals.items():
        print(f"  {k:32s} {v:>14,.0f}")
    print(f"  {'tool result tokens (est. chars/4)':32s} {total_res:>14,.0f}")

    label = {"naive": "Any repeat (overcounts)",
             "fresh_session": "Fresh repeat, same session",
             "fresh_any": "Fresh repeat, any session" + (" (git-checked)" if git else ""),
             "stale_edit": "Stale: its own file/dir was edited",
             "stale_bash": "Stale: only a shell cmd ran nearby"}
    print("\nRepeats                              steps    % steps   % result toks   % output toks")
    for b in ("naive", "fresh_session", "fresh_any", "stale_edit", "stale_bash"):
        c = buckets[b]
        print(f"  {label[b]:34s} {c['n']:>6}    {pct(c['n'], len(steps))}       "
              f"{pct(c['res'], total_res)}          {pct(c['out'], totals['output_tokens'])}")

    print("\nBy tool                calls    fresh repeats    share of that tool's tokens")
    for tool, c in sorted(by_tool.items(), key=lambda x: -x[1]["tokens"]):
        print(f"  {tool:20s} {c['calls']:>6}    {c['fresh']:>6}           {pct(c['fresh_tokens'], c['tokens'])}")

    print("\nMost repeated fresh steps:")
    for key, n in top.most_common(15):
        print(f"  {n:>4}x  {top_tokens[key]:>9,.0f} tok  {key[:110]}")

    # Path overlap: 3-step sequences of the same steps appearing in more than one session
    seqs = defaultdict(set)
    per_session = defaultdict(list)
    for s in steps:
        per_session[s["session"]].append(s["key"])
    for sid, keys in per_session.items():
        for i in range(len(keys) - 2):
            seqs[tuple(keys[i:i + 3])].add(sid)
    shared = {k: v for k, v in seqs.items() if len(v) > 1}
    print(f"\nRepeated 3-step paths across sessions: {len(shared)} distinct "
          f"(appearing in {sum(len(v) for v in shared.values())} session-occurrences)")
    for k, v in sorted(shared.items(), key=lambda x: -len(x[1]))[:5]:
        print(f"  in {len(v)} sessions: " + "  ->  ".join(x[:45] for x in k))

    delta_report(delta_pairs, total_res, totals["cache_read_input_tokens"])

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["ts", "project", "session", "tool", "status", "result_tokens", "out_tokens", "key"])
            for s in steps:
                w.writerow([s["ts"].isoformat(), s["project"], s["session"], s["name"], s["status"],
                            round(s["result_tokens"]), round(s["out_tokens"]), s["key"]])
        print(f"\nWrote {len(steps)} steps to {args.csv}")

    print("\nCaveats: edits you made outside Claude Code aren't visible unless committed (--git helps);"
          "\nMCP tool repeats are counted only as 'any repeat' since their sources can't be checked;"
          "\nweb results are assumed fresh for 24h. Treat 'fresh, any session' as an upper bound.\n")


if __name__ == "__main__":
    main()
