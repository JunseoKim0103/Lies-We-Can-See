"""Multi-LLM 1v1 matchup runner for the Aria 2-imposter / 6-crewmate experiment.

Enumerates every (imposter_model, crewmate_model) pair from an N-model pool.
Both imposter slots receive the same `imposter_model`; all 6 crewmate slots
receive the same `crewmate_model`. With N models there are N*N matchups
per architecture case (currently N=10 → 100 matchups).

Architecture is fixed by `--case`:

  Main (Crew Config #1: Semantic+LLM, Reactive)
    1A Symmetric: Crew & Imp both Semantic+LLM / Meeting+Skill / Reactive / Minimal
    1B Best-WR:   Crew Semantic+LLM, Imp Window — both Reactive
  Appendix (Crew Config #2: Window, Hierarchical)
    2A Symmetric: Crew & Imp both Window / Meeting+Skill / Hierarchical / Minimal
    2B Best-WR:   Crew Window+Hier, Imp Window+Reactive

Matchup index k is deterministic: k = i * N + j → (POOL[i], POOL[j]). The same
matchup index across different cases uses identical (imp_model, crew_model)
on identical slots → clean paired comparison.

Server splitting (4-way example, run on each server):
    # server 0 (gets matchups 0, 4, 8, ..., 48 — 13 cases)
    python main_1_aria_2vs6_multi_llm.py --case 1A --server-id 0 --num-servers 4
    # server 1 (gets matchups 1, 5, 9, ..., 45 — 12 cases)
    python main_1_aria_2vs6_multi_llm.py --case 1A --server-id 1 --num-servers 4
    # server 2 (gets matchups 2, 6, 10, ..., 46 — 12 cases)
    python main_1_aria_2vs6_multi_llm.py --case 1A --server-id 2 --num-servers 4
    # server 3 (gets matchups 3, 7, 11, ..., 47 — 12 cases)
    python main_1_aria_2vs6_multi_llm.py --case 1A --server-id 3 --num-servers 4

Other usage:
    # All 49 matchups for case 1A (single server)
    python main_1_aria_2vs6_multi_llm.py --case 1A

    # Resume / subset (within this server's slice):
    python main_1_aria_2vs6_multi_llm.py --case 1A --start-idx 12
    python main_1_aria_2vs6_multi_llm.py --case 1A --start-idx 0 --num-runs 5

    # List all 49 matchups without running anything:
    python main_1_aria_2vs6_multi_llm.py --list

    # Analyze logs without re-running:
    python main_1_aria_2vs6_multi_llm.py --analyze-only --out-dir <existing>
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

import yaml

SCRIPT_DIR = Path("/root/MineLand/scripts")
RUN_SCRIPT = SCRIPT_DIR / "amongus_2_imposter_6_crewmates_Aria.py"

PLAYER_NAMES = ["James", "Steve", "Jason", "Michael", "Olivia", "Noah", "Ethan", "Martin"]
IMPOSTER_NAMES = {"James", "Olivia"}

CHARACTERS = [
    {"name": "James",   "role": "imposter", "color": "Red"},
    {"name": "Steve",   "role": "crewmate", "color": "Blue"},
    {"name": "Jason",   "role": "crewmate", "color": "Green"},
    {"name": "Michael", "role": "crewmate", "color": "Yellow"},
    {"name": "Olivia",  "role": "imposter", "color": "Pink"},
    {"name": "Noah",    "role": "crewmate", "color": "White"},
    {"name": "Ethan",   "role": "crewmate", "color": "Light Blue"},
    {"name": "Martin",  "role": "crewmate", "color": "Black"},
]

# 12-model pool → 144 matchups per case.
# Excluded (2026-05-18/19):
#   - gemini-2.5-flash-lite: JSON parser fail in imposter/ghost prompts
#   - gemma3-4b/12b/27b: EmergencyModule false-positive + rate limits
# Replacements: gemini-3-flash-preview, gemma4-26b-a4b, gemma4-31b.
# (gemma4-31b uses provider_ignore=DeepInfra to skip vision-less turbo route.)
MODEL_POOL = [
    "gpt-4.1-mini",
    "gpt-5-mini",
    "gpt-5",
    "qwen3.5-27b",
    "qwen3.5-9b",
    "qwen3.6-27b",
    "kimi-k2.5",
    "gemini-2.5-flash",
    "gemini-3.1-flash-lite-preview",
    "gemini-3-flash-preview",
    "gemma4-26b-a4b",
    "gemma4-31b",
]
assert len(MODEL_POOL) == len(set(MODEL_POOL)), "MODEL_POOL must be unique"


# ──────────────────────────────────────────────────────────────
# Case definitions
# ──────────────────────────────────────────────────────────────

# Shared CLI flags that don't change across cases (state, module policy,
# verbose etc.). Per-case crew+imposter architecture is composed below.
COMMON_CLI = [
    "--state-mode", "privileged",
    "--module-policy", "vlm",
    "--vlm-kill-fallback", "-1",
    "--max-steps", "200",
    "--no-parallel", "--verbose",
]

# Crewmate architecture per case (matches image's Setting #A/#B).
#
#   Main (Crew Config #1: Semantic+LLM, Reactive)
#     1A Symmetric: Crew & Imp both Semantic+LLM, Meeting+Skill, Reactive, Minimal
#     1B Best-WR:   Crew Semantic+LLM, Imp Window
#   Appendix (Crew Config #2: Window, Hierarchical)
#     2A Symmetric: Crew & Imp both Window, Meeting+Skill, Hierarchical, Minimal
#     2B Best-WR:   Crew Window+Hier, Imp Window+Reactive
CASE_CREWMATE_CLI = {
    "1A": [
        "--memory-mode", "semantic",
        "--memory-belief-update", "llm",
        "--reflection-mode", "meeting",
        "--planning-mode", "reactive",
        "--planning-policy", "llm",
        "--prompt-style", "minimal",
        "--skill-memory",
        "--skill-memory-sources", "self,observed",
    ],
    "1B": [
        "--memory-mode", "semantic",
        "--memory-belief-update", "llm",
        "--reflection-mode", "meeting",
        "--planning-mode", "reactive",
        "--planning-policy", "llm",
        "--prompt-style", "minimal",
        "--skill-memory",
        "--skill-memory-sources", "self,observed",
    ],
    "2A": [
        "--memory-mode", "window",
        "--memory-belief-update", "rule",
        "--reflection-mode", "meeting",
        "--planning-mode", "hierarchical",
        "--planning-policy", "llm",
        "--prompt-style", "minimal",
        "--skill-memory",
        "--skill-memory-sources", "self,observed",
    ],
    "2B": [
        "--memory-mode", "window",
        "--memory-belief-update", "rule",
        "--reflection-mode", "meeting",
        "--planning-mode", "hierarchical",
        "--planning-policy", "llm",
        "--prompt-style", "minimal",
        "--skill-memory",
        "--skill-memory-sources", "self,observed",
    ],
}

CASE_IMPOSTER_CLI = {
    "1A": [
        "--imposter-memory-mode", "semantic",
        "--imposter-memory-belief-update", "llm",
        "--imposter-reflection-mode", "meeting",
        "--imposter-planning-mode", "reactive",
        "--imposter-planning-policy", "llm",
        "--imposter-prompt-style", "minimal",
        "--imposter-skill-memory",
        "--imposter-skill-memory-sources", "self,observed",
    ],
    "1B": [
        "--imposter-memory-mode", "window",
        "--imposter-memory-belief-update", "rule",
        "--imposter-reflection-mode", "meeting",
        "--imposter-planning-mode", "hierarchical",
        "--imposter-planning-policy", "llm",
        "--imposter-prompt-style", "minimal",
        "--imposter-skill-memory",
        "--imposter-skill-memory-sources", "self,observed",
    ],
    "2A": [
        "--imposter-memory-mode", "window",
        "--imposter-memory-belief-update", "rule",
        "--imposter-reflection-mode", "meeting",
        "--imposter-planning-mode", "hierarchical",
        "--imposter-planning-policy", "llm",
        "--imposter-prompt-style", "minimal",
        "--imposter-skill-memory",
        "--imposter-skill-memory-sources", "self,observed",
    ],
    "2B": [
        "--imposter-memory-mode", "window",
        "--imposter-memory-belief-update", "rule",
        "--imposter-reflection-mode", "meeting",
        "--imposter-planning-mode", "reactive",
        "--imposter-planning-policy", "llm",
        "--imposter-prompt-style", "minimal",
        "--imposter-skill-memory",
        "--imposter-skill-memory-sources", "self,observed",
    ],
}

CASE_LABELS = {
    "1A": "Symmetric (Semantic+LLM / Reactive both sides)",
    "1B": "Best-WR (Crew Semantic+LLM+Reactive, Imp Window+Hierarchical)",
    "2A": "Symmetric (Window / Hierarchical both sides)",
    "2B": "Best-WR (Crew Window+Hier, Imp Window+Reactive)",
}


def enumerate_matchups() -> list[tuple[str, str]]:
    """All N*N (imposter_model, crewmate_model) matchups in deterministic order.

    Matchup index k = i * N + j → (MODEL_POOL[i], MODEL_POOL[j]).
    """
    return [(imp, crew) for imp in MODEL_POOL for crew in MODEL_POOL]


def split_assignment(model_a: str, model_b: str) -> list[str]:
    """8-slot assignment where each of A,B takes 1 imposter + 3 crewmates.

    A: James (imp), Steve, Jason, Michael (3 crew)
    B: Olivia (imp), Noah, Ethan, Martin (3 crew)
    """
    a_slots = {0, 1, 2, 3}  # James(imp), Steve, Jason, Michael
    return [model_a if i in a_slots else model_b for i in range(len(CHARACTERS))]


def models_for_matchup_idx(matchup_idx: int) -> list[str]:
    """Build the 8-slot assignment for a given matchup index.

    Both imposter slots (James, Olivia) get `imposter_model`; all six crewmate
    slots get `crewmate_model`. Returned list aligns positionally with CHARACTERS.
    """
    matchups = enumerate_matchups()
    if matchup_idx < 0 or matchup_idx >= len(matchups):
        raise IndexError(f"matchup_idx {matchup_idx} out of range [0, {len(matchups)})")
    imp_model, crew_model = matchups[matchup_idx]
    out: list[str] = []
    for char in CHARACTERS:
        if char["role"] == "imposter":
            out.append(imp_model)
        else:
            out.append(crew_model)
    return out


def build_yaml_config(models: list[str], out_path: Path) -> None:
    agents = []
    for char, model in zip(CHARACTERS, models):
        agents.append({
            "name": char["name"],
            "role": char["role"],
            "color": char["color"],
            "model": model,
        })
    cfg = {"task_id": "amongus_2_imposter_6_crewmates", "agents": agents}
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False))


# ──────────────────────────────────────────────────────────────
# Game running
# ──────────────────────────────────────────────────────────────

def kill_leftover_servers(instance_id: int = 0) -> None:
    """Free only THIS instance's ports (server 25565+id, mineflayer 21301+id,
    rcon 25575+id).

    Port-scoped so several runners can share one container: the old
    name-matching pkill ('fabric-server', 'java -Xmx', 'node.*index') killed
    every game in the container, not just this lane's leftovers.
    """
    for port in (25565 + instance_id, 21301 + instance_id, 25575 + instance_id):
        subprocess.run(f"lsof -ti tcp:{port} | xargs -r kill -9",
                       shell=True, stderr=subprocess.DEVNULL)
    time.sleep(5)


def run_one_game(case: str, models: list[str], matchup_idx: int,
                 log_dir: Path, trace_prompts: bool = False,
                 instance_id: int = 0, max_steps: int | None = None) -> dict:
    log_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = log_dir / "agents.yaml"
    build_yaml_config(models, yaml_path)
    log_path = log_dir / "game.log"

    cmd = ([sys.executable, str(RUN_SCRIPT),
            "--config", str(yaml_path),
            "--agent-save-root", str(log_dir),
            "--instance-id", str(instance_id)]
           + COMMON_CLI
           + CASE_CREWMATE_CLI[case]
           + CASE_IMPOSTER_CLI[case])
    # Override comes after COMMON_CLI so argparse keeps the last --max-steps.
    if max_steps is not None:
        cmd += ["--max-steps", str(max_steps)]

    start = datetime.now()
    n_matchups = len(enumerate_matchups())
    imp_model = models[0]  # James is imposter slot 0
    crew_model = next(m for c, m in zip(CHARACTERS, models) if c["role"] == "crewmate")
    print(f"\n{'=' * 70}\n  CASE {case} — matchup {matchup_idx + 1}/{n_matchups} (idx={matchup_idx})"
          f"\n  imposter={imp_model}  crewmate={crew_model}"
          f"\n  start: {start.isoformat(timespec='seconds')}\n  log: {log_path}", flush=True)
    for char, model in zip(CHARACTERS, models):
        print(f"    {char['name']:8s} ({char['role']:8s}) → {model}", flush=True)
    print("=" * 70, flush=True)

    kill_leftover_servers(instance_id)

    env = os.environ.copy()
    if trace_prompts:
        # Enables tracing; actual output is routed per-agent into <log_dir>/<agent>/
        # (registered via each agent's save path), so point the fallback root at
        # log_dir itself rather than a separate (otherwise-empty) trace/ subdir.
        env["ARIA_TRACE_DIR"] = str(log_dir)
        print(f"  trace: {log_dir}/<agent>/ (per-call LLM input/output)", flush=True)

    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, env=env)

    end = datetime.now()
    duration = (end - start).total_seconds()
    print(f"\n>>> finished in {int(duration // 60)}m {int(duration % 60)}s "
          f"(exit={proc.returncode})", flush=True)

    parsed = parse_game_log(log_path)
    print(
        f"  RESULT: {parsed['win_type']} (winner={parsed['winner']})\n"
        f"  steps={parsed['steps']}  kills={parsed['kills']} {parsed['kill_victims']}\n"
        f"  ejections={parsed['ejections']}\n"
        f"  meetings={parsed['meetings']}  missions={parsed['missions_completed']}  "
        f"vote_acc={parsed['vote_accuracy']}\n"
        f"  tokens in/out={parsed['tokens_input']}/{parsed['tokens_output']}  "
        f"cost=${parsed['cost_usd']}",
        flush=True,
    )

    return {
        "case": case, "matchup_idx": matchup_idx,
        "imposter_model": imp_model,
        "crewmate_model": crew_model,
        "models": models,
        "start": start.isoformat(timespec="seconds"),
        "end": end.isoformat(timespec="seconds"),
        "duration_sec": round(duration, 1),
        "exit_code": proc.returncode,
        "log_path": str(log_path),
    }


# ──────────────────────────────────────────────────────────────
# Log analysis
# ──────────────────────────────────────────────────────────────

WIN_LINE = re.compile(r"RESULT:\s+([A-Z ]+)")
STEPS_LINE = re.compile(r"Steps\s*:\s*(\d+)")
TIME_LINE = re.compile(r"Time\s*:\s*(\d+)m\s+(\d+)s")
GHOST_LINE = re.compile(r"👻 (\w+) ghost updated to: 1")
MEETING_RECEIPT = re.compile(
    r"\[StateBuilder:(\w+)\] meeting_start received "
    r"\(meeting #(\d+), tick=(\d+), trigger=(\w+)\):\s*"
    r"(?:reporter=(\S+) victim=(\S+)|emergency button)"
)
VOTE_RECEIPT = re.compile(
    r"\[StateBuilder:(\w+)\] vote_result received "
    r"\(meeting #(\d+), tick=(\d+)\):\s*"
    r"(?:ejected=(\S+) role=(\S+)|SKIPPED)"
)
VOTE_REVEAL = re.compile(r'text="(\w+) was (?:a|an) (Crewmate|Imposter)\."')
VOTE_SKIPPED_TEXT = re.compile(r'text="No one was ejected \(SKIPPED\)\."')
MISSION_DONE = re.compile(r"<\w+> Mission \d+ Completed by (\w+)")
TOKEN_LINE = re.compile(r"TOTAL COST:\s*\$([\d.]+)")
TOKEN_TOTAL = re.compile(r"TOTAL\s+input=([\d,]+)\s+output=([\d,]+)")


def parse_game_log(log_path: Path) -> dict:
    out: dict = {
        "win_type": "UNKNOWN", "winner": "unknown", "steps": None, "duration_sec": None,
        "kills": 0, "kill_victims": [], "meetings": 0,
        "ejections": [], "skipped_meetings": 0,
        "imposters_ejected": 0, "crewmates_ejected": 0, "vote_accuracy": None,
        "missions_completed": 0,
        "self_reports": 0, "imposter_reports": 0,
        "tokens_input": None, "tokens_output": None, "cost_usd": None,
    }
    if not log_path.exists():
        return out

    text = log_path.read_text(errors="ignore")

    if m := WIN_LINE.search(text):
        win = m.group(1).strip()
        out["win_type"] = win
        if "MISSION WIN" in win:
            out["winner"] = "crewmate_mission"
        elif "TIMEOUT" in win:
            out["winner"] = "timeout_imposter"
        elif "IMPOSTER WIN" in win:
            out["winner"] = "imposter"
        elif "CREWMATE WIN" in win:
            out["winner"] = "crewmate_vote"

    if m := STEPS_LINE.search(text):
        out["steps"] = int(m.group(1))
    if m := TIME_LINE.search(text):
        out["duration_sec"] = int(m.group(1)) * 60 + int(m.group(2))

    ghost_seen: set[str] = set()
    ghost_order: list[str] = []
    for m in GHOST_LINE.finditer(text):
        v = m.group(1)
        if v not in ghost_seen:
            ghost_seen.add(v)
            ghost_order.append(v)

    meeting_ticks = set()
    self_reports = 0
    imposter_reports = 0
    for m in MEETING_RECEIPT.finditer(text):
        meeting_no = int(m.group(2))
        tick = int(m.group(3))
        trigger = m.group(4)
        reporter = m.group(5)
        victim = m.group(6)
        key = (meeting_no, tick)
        if key in meeting_ticks:
            continue
        meeting_ticks.add(key)
        if trigger == "report" and reporter:
            if reporter == victim:
                self_reports += 1
            if reporter in IMPOSTER_NAMES:
                imposter_reports += 1
    out["meetings"] = len(meeting_ticks)
    out["self_reports"] = self_reports
    out["imposter_reports"] = imposter_reports

    seen_votes: set[tuple] = set()
    imposters_ejected = 0
    crewmates_ejected = 0
    skipped = 0
    ejections: list[tuple[str, str]] = []
    for m in VOTE_RECEIPT.finditer(text):
        meeting_no = int(m.group(2))
        tick = int(m.group(3))
        ejected = m.group(4)
        role = m.group(5)
        key = (meeting_no, tick)
        if key in seen_votes:
            continue
        seen_votes.add(key)
        if ejected:
            ejections.append((ejected, role))
            if role == "imposter":
                imposters_ejected += 1
            elif role == "crewmate":
                crewmates_ejected += 1
        else:
            skipped += 1

    revealed: list[tuple[str, str]] = []
    seen_reveals: set[tuple[str, str]] = set()
    for m in VOTE_REVEAL.finditer(text):
        name = m.group(1)
        role = m.group(2).lower()
        key = (name, role)
        if key in seen_reveals:
            continue
        seen_reveals.add(key)
        revealed.append(key)
    if revealed:
        ejections = revealed
        imposters_ejected = sum(1 for _, role in ejections if role == "imposter")
        crewmates_ejected = sum(1 for _, role in ejections if role == "crewmate")
        skipped = min(skipped, max(len(meeting_ticks) - len(ejections), 0))
    elif skipped == 0 and VOTE_SKIPPED_TEXT.search(text):
        skipped = 1

    out["ejections"] = [{"name": n, "role": r} for n, r in ejections]
    out["imposters_ejected"] = imposters_ejected
    out["crewmates_ejected"] = crewmates_ejected
    out["skipped_meetings"] = skipped
    ejected_names = {n for n, _ in ejections}
    out["kill_victims"] = [v for v in ghost_order if v not in ejected_names]
    out["kills"] = len(out["kill_victims"])
    total_ejections = imposters_ejected + crewmates_ejected
    if total_ejections > 0:
        out["vote_accuracy"] = imposters_ejected / total_ejections

    out["missions_completed"] = len(MISSION_DONE.findall(text))

    if m := TOKEN_LINE.search(text):
        out["cost_usd"] = float(m.group(1))
    if m := TOKEN_TOTAL.search(text):
        out["tokens_input"] = int(m.group(1).replace(",", ""))
        out["tokens_output"] = int(m.group(2).replace(",", ""))

    return out


def analyze(out_dir: Path) -> None:
    rows: list[dict] = []
    for log_path in sorted(out_dir.rglob("game.log")):
        run_dir = log_path.parent
        # Find case<...> and matchup<NN>/pair/run<NN> tokens anywhere in the path
        # so the layout's trial/timestamp layers don't break parsing.
        parts = log_path.parts
        m_run = next((mm for p in parts
                      for mm in [re.fullmatch(r"(?:matchup|pair|run)(\d+)", p)] if mm), None)
        m_case = next((mm for p in parts
                       for mm in [re.fullmatch(r"case(.+)", p)] if mm), None)
        if not m_run or not m_case:
            continue
        matchup_idx = int(m_run.group(1))
        case = m_case.group(1)

        yaml_path = run_dir / "agents.yaml"
        models = []
        if yaml_path.exists():
            try:
                y = yaml.safe_load(yaml_path.read_text())
                models = [a["model"] for a in y.get("agents", [])]
            except Exception:
                models = []

        parsed = parse_game_log(log_path)
        imposter_model = models[0] if models else ""
        crewmate_model = next(
            (m_ for c_, m_ in zip(CHARACTERS, models) if c_["role"] == "crewmate"),
            "",
        )
        row = {
            "case": case, "matchup_idx": matchup_idx,
            "log_path": str(log_path),
            "imposter_model": imposter_model,
            "crewmate_model": crewmate_model,
            "models": ";".join(models),
            **parsed,
        }
        for slot_i, char in enumerate(CHARACTERS):
            row[f"model_{char['name']}"] = models[slot_i] if slot_i < len(models) else ""
        rows.append(row)

    if not rows:
        print(f"[analyze] no game.log files under {out_dir}")
        return

    runs_csv = out_dir / "all_runs.csv"
    with open(runs_csv, "w", newline="") as f:
        keys = list(rows[0].keys())
        keys = [k for k in keys if k not in ("kill_victims", "ejections")]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})

    runs_json = out_dir / "all_runs.json"
    runs_json.write_text(json.dumps(rows, indent=2, default=str))

    print(f"[analyze] {len(rows)} runs")
    print(f"  per-run csv:  {runs_csv}")
    print(f"  per-run json: {runs_json}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def print_all_matchups() -> None:
    matchups = enumerate_matchups()
    print(f"Pool ({len(MODEL_POOL)}): {MODEL_POOL}")
    print(f"All {len(MODEL_POOL)}*{len(MODEL_POOL)} = {len(matchups)} (imposter, crewmate) matchups:\n")
    for idx, (imp, crew) in enumerate(matchups):
        print(f"  matchup {idx:02d}: imposter={imp:18s} crewmate={crew}")


def next_trial_number(case_root: Path) -> int:
    """Smallest unused trial index for this case: max existing trial<N> + 1, else 1.

    Used when --trial is omitted so a 2nd run auto-bumps to trial2 instead of
    landing back in trial1. Never reuses an existing trial dir.
    """
    if not case_root.exists():
        return 1
    existing = [int(m.group(1))
                for p in case_root.glob("trial*") if p.is_dir()
                for m in [re.match(r"trial(\d+)$", p.name)] if m]
    return max(existing) + 1 if existing else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=str, choices=list(CASE_CREWMATE_CLI.keys()),
                    help=" | ".join(f"{k}={v}" for k, v in CASE_LABELS.items()))
    ap.add_argument("--server-id", type=int, default=0,
                    help="This server's index in [0, num_servers). With --num-servers, "
                         "round-robin filter selects matchups where idx %% num_servers == server_id.")
    ap.add_argument("--num-servers", type=int, default=1,
                    help="Total number of servers for round-robin split. Default 1 (no split).")
    ap.add_argument("--start-idx", type=int, default=0,
                    help="Within this server's filtered slice, start from this position. Default 0.")
    ap.add_argument("--num-runs", type=int, default=None,
                    help="How many matchups to run from --start-idx within this server's slice. "
                         "Default: all remaining.")
    ap.add_argument("--matchup-indices", type=str, default=None,
                    help="Comma-separated explicit matchup indices to run, e.g. '8,26,30,39,43,45'. "
                         "Overrides --server-id / --num-servers / --start-idx / --num-runs.")
    ap.add_argument("--split-models", type=str, default=None,
                    help="Mixed-role mode: 'A,B' assigns A=James-imp+Steve/Jason/Michael, "
                         "B=Olivia-imp+Noah/Ethan/Martin. Skips matchup-pool logic; runs "
                         "--num-runs games (default 1) with this fixed split.")
    ap.add_argument("--out-dir", default=None,
                    help="Output root. Default: repo-root RQ2/. Layout: "
                         "<root>/case<case>/trial<N>/matchup<NN>/<timestamp>/")
    ap.add_argument("--trial", type=int, default=None,
                    help="Trial (repetition) number — groups reruns under "
                         "case<case>/trial<N>/. Default: auto (next unused trial for this "
                         "case, so a 2nd run becomes trial2 and never overwrites trial1). "
                         "Concurrent lanes (--num-servers>1) must pass an explicit --trial so "
                         "they share one trial dir; run_multi_llm_one_docker.sh sets it.")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="Override per-game max steps (default from COMMON_CLI = 200). "
                         "Handy for quick smoke tests, e.g. --max-steps 30.")
    ap.add_argument("--trace-prompts", action="store_true",
                    help="Dump every LLM call's full input (system+human messages) and "
                         "output to per-call JSON files under <run>/trace/. Base64 images "
                         "are saved to <run>/trace/images/ and linked by relative path.")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--list", dest="list_matchups", action="store_true",
                    help="List all matchups and exit.")
    args = ap.parse_args()

    if args.list_matchups:
        print_all_matchups()
        return

    if args.analyze_only:
        if not args.out_dir:
            sys.exit("--analyze-only requires --out-dir")
        analyze(Path(args.out_dir))
        return

    if args.case is None:
        sys.exit("--case is required (1A | 1B | 2A | 2B)")

    if args.split_models:
        parts = [p.strip() for p in args.split_models.split(",") if p.strip()]
        if len(parts) != 2:
            sys.exit(f"--split-models requires exactly 2 model IDs, got {args.split_models!r}")
        model_a, model_b = parts
        models = split_assignment(model_a, model_b)
        num_runs = args.num_runs or 1

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        safe = lambda s: s.replace("/", "_")
        default_dirname = f"{timestamp}_case{args.case}_split_{safe(model_a)}_vs_{safe(model_b)}"
        out_dir = Path(args.out_dir) if args.out_dir else SCRIPT_DIR / "logs" / "multi_llm" / default_dirname
        case_dir = out_dir / f"case{args.case}"
        case_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "case": args.case,
            "mode": "split",
            "model_a": model_a,
            "model_b": model_b,
            "characters": CHARACTERS,
            "models_by_slot": models,
            "num_runs": num_runs,
        }
        (case_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        summary_csv = case_dir / "runs_summary.csv"
        fieldnames = ["case", "matchup_idx", "imposter_model", "crewmate_model", "models",
                      "start", "end", "duration_sec", "exit_code", "log_path"]
        if not summary_csv.exists():
            with open(summary_csv, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()

        print(f"\n{'#' * 70}")
        print(f"  Multi-LLM SPLIT MODE — case {args.case}")
        print(f"  A = {model_a}  →  James(imp), Steve, Jason, Michael")
        print(f"  B = {model_b}  →  Olivia(imp), Noah, Ethan, Martin")
        print(f"  Runs: {num_runs}")
        print(f"  Output: {case_dir}")
        print("#" * 70)

        for r in range(num_runs):
            log_dir = case_dir / f"run{r:02d}"
            row = run_one_game(args.case, models, r, log_dir, trace_prompts=args.trace_prompts,
                               instance_id=args.server_id, max_steps=args.max_steps)
            row_csv = {**row, "models": ";".join(row["models"])}
            with open(summary_csv, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(
                    {k_: row_csv.get(k_) for k_ in fieldnames}
                )

        print(f"\n{'#' * 70}\n  Analyzing...\n{'#' * 70}")
        analyze(out_dir)
        return

    if args.num_servers < 1:
        sys.exit("--num-servers must be >= 1")
    if not (0 <= args.server_id < args.num_servers):
        sys.exit(f"--server-id must be in [0, {args.num_servers})")

    matchups = enumerate_matchups()
    n_matchups = len(matchups)

    if args.matchup_indices:
        try:
            my_idx = sorted({int(x) for x in args.matchup_indices.split(",") if x.strip()})
        except ValueError:
            sys.exit(f"--matchup-indices must be comma-separated ints, got {args.matchup_indices!r}")
        for k in my_idx:
            if not (0 <= k < n_matchups):
                sys.exit(f"--matchup-indices: {k} out of range [0, {n_matchups})")
        server_slice = my_idx
    else:
        # Round-robin server filter: this server runs idx where idx %% N == server_id.
        server_slice = [k for k in range(n_matchups)
                        if k % args.num_servers == args.server_id]

        if not (0 <= args.start_idx < max(len(server_slice), 1)):
            sys.exit(f"--start-idx must be in [0, {len(server_slice)}) for this server's slice")
        sliced = server_slice[args.start_idx:]
        num_to_run = len(sliced) if args.num_runs is None else min(args.num_runs, len(sliced))
        my_idx = sliced[:num_to_run]

    out_dir = Path(args.out_dir) if args.out_dir else SCRIPT_DIR.parent / "RQ2"
    # Layout: <out>/case<case>/trial<N>/matchup<NN>/<timestamp>/ — the trial
    # groups repetitions and the per-game timestamp keeps reruns from colliding.
    case_root = out_dir / f"case{args.case}"
    if args.trial is None:
        trial_num = next_trial_number(case_root)
        if args.num_servers > 1:
            # Lanes that SHARE one out_dir must agree on the trial number; each
            # auto-picking independently could race to different trials. The
            # launcher (run_multi_llm_one_docker.sh) computes it once and passes an
            # explicit --trial. Separate-machine splits (run_main_split.sh) use a
            # distinct --out-dir per server, so per-process auto-pick is fine there.
            print(f"[warn] --num-servers>1 without --trial: auto-picked trial{trial_num}. "
                  f"If several lanes share THIS --out-dir, pass one explicit --trial "
                  f"(the launcher does) so they don't diverge.", file=sys.stderr)
    else:
        trial_num = args.trial
    trial_dir = case_root / f"trial{trial_num}"
    # Never overwrite existing logs on a FRESH full run. A fresh auto trial never
    # collides; an explicit --trial pointing at a trial that already holds games is
    # refused. Exceptions: concurrent lanes (num_servers>1) intentionally co-write one
    # trial dir; and an explicit --matchup-indices is a TARGETED re-run (only the named
    # matchups), so it must be allowed to add to a trial that already has other games.
    if args.num_servers == 1 and not args.matchup_indices and any(trial_dir.glob("matchup*/*/game.log")):
        sys.exit(f"{trial_dir} already contains games — refusing to overwrite. "
                 f"Omit --trial to auto-increment, or pass an unused --trial N.")
    trial_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "case": args.case,
        "model_pool": MODEL_POOL,
        "characters": CHARACTERS,
        "n_matchups_total": n_matchups,
        "server_id": args.server_id,
        "num_servers": args.num_servers,
        "server_slice": server_slice,
        "start_idx_in_slice": args.start_idx,
        "matchup_indices_to_run": my_idx,
        "matchup_assignments": [
            {
                "matchup_idx": k,
                "imposter_model": matchups[k][0],
                "crewmate_model": matchups[k][1],
                "models_by_slot": models_for_matchup_idx(k),
            }
            for k in my_idx
        ],
    }
    manifest["trial"] = trial_num
    # When several lanes share one trial_dir (multiple runners in one container),
    # keep manifest/summary per-lane so concurrent writers don't overwrite or
    # interleave each other. analyze() aggregates via rglob on game.log, so the
    # split bookkeeping files don't need merging. Single-lane runs are unchanged.
    suffix = "" if args.num_servers == 1 else f"_srv{args.server_id}"
    (trial_dir / f"manifest{suffix}.json").write_text(json.dumps(manifest, indent=2))

    summary_csv = trial_dir / f"runs_summary{suffix}.csv"
    fieldnames = ["case", "matchup_idx", "imposter_model", "crewmate_model", "models",
                  "start", "end", "duration_sec", "exit_code", "log_path"]
    if not summary_csv.exists():
        with open(summary_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    print(f"\n{'#' * 70}")
    print(f"  Multi-LLM 1v1 2v6 — case {args.case}  trial {trial_num}")
    print(f"  Server {args.server_id}/{args.num_servers}  |  this server slice: {server_slice}")
    print(f"  Running matchups: {my_idx}  ({len(my_idx)} of {n_matchups} total)")
    print(f"  Output: {trial_dir}")
    print("#" * 70)

    for k in my_idx:
        models = models_for_matchup_idx(k)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = trial_dir / f"matchup{k:02d}" / ts
        row = run_one_game(args.case, models, k, log_dir, trace_prompts=args.trace_prompts,
                           instance_id=args.server_id, max_steps=args.max_steps)
        row_csv = {
            **row,
            "models": ";".join(row["models"]),
        }
        with open(summary_csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(
                {k_: row_csv.get(k_) for k_ in fieldnames}
            )

    print(f"\n{'#' * 70}\n  Analyzing...\n{'#' * 70}")
    analyze(out_dir)


if __name__ == "__main__":
    main()
