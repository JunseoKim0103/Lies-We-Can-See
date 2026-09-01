"""Main runner for the Aria 2-imposter / 6-crewmate sweep (24 configs).

One file does everything:
  1. Defines the 24 architectural configs
  2. Splits across 3 servers (8 configs each) via --server-id
  3. Runs each config N times (default 3)
  4. Parses every game log for win type, length, kills, meetings, missions,
     vote accuracy, self-report rate, deception/detection metrics, token cost
  5. Writes per-run + per-config CSV/JSON + a markdown summary

Usage:
    # On server 0 (configs 0..7):
    python main_1_aria_2vs6.py --server-id 0
    # Defaults: --runs-per-config 3, --out-dir scripts/logs/sweep/<timestamp>

    # Analyze without re-running (e.g., partway through a sweep, or after
    # all 3 server slices finish — --analyze-only walks the entire out-dir
    # so it merges every serverN/ subdir into the final summary):
    python main_1_aria_2vs6.py --analyze-only --out-dir <existing dir>

Servers can run in parallel on separate machines pointing at the same
shared --out-dir, or sequentially on one machine. The slice you run is
analyzed automatically when it finishes; re-run --analyze-only on the
shared dir once every slice is done to produce the combined summary.
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
from statistics import mean, stdev

SCRIPT_DIR = Path("/root/MineLand/scripts")
RUN_SCRIPT = SCRIPT_DIR / "amongus_2_imposter_6_crewmates_Aria.py"

PLAYER_NAMES = ["James", "Steve", "Jason", "Michael", "Olivia", "Noah", "Ethan", "Martin"]
IMPOSTER_NAMES = {"James", "Olivia"}


# ──────────────────────────────────────────────────────────────
# 24 config matrix
# ──────────────────────────────────────────────────────────────

def build_24_configs() -> list[dict]:
    """3 (memory) × 2 (refl/skill) × 2 (planning) × 2 (prompt) = 24."""
    memories = [
        {"label": "off",      "memory_mode": "none",     "memory_belief_update": "rule"},
        {"label": "win",      "memory_mode": "window",   "memory_belief_update": "rule"},
        {"label": "sem-llm",  "memory_mode": "semantic", "memory_belief_update": "llm"},
    ]
    refl_skill = [
        {"label": "no-rs",   "reflection_mode": "none",
         "skill_memory": False, "skill_memory_sources": None},
        {"label": "meet-rs", "reflection_mode": "meeting",
         "skill_memory": True, "skill_memory_sources": "self,observed"},
    ]
    plannings = [
        {"label": "react", "planning_mode": "reactive",     "planning_policy": "llm"},
        {"label": "hier",  "planning_mode": "hierarchical", "planning_policy": "llm"},
    ]
    prompts = [
        {"label": "min", "prompt_style": "minimal"},
        {"label": "det", "prompt_style": "deterministic"},
    ]

    configs: list[dict] = []
    cid = 0
    for mem in memories:
        for rs in refl_skill:
            for pl in plannings:
                for pr in prompts:
                    label = "_".join([mem["label"], rs["label"], pl["label"], pr["label"]])
                    cfg = {
                        "id": cid, "label": label,
                        "state_mode": "privileged",
                        "module_policy": "vlm",
                        "vlm_kill_fallback": -1,
                        **{k: v for k, v in mem.items() if k != "label"},
                        **{k: v for k, v in rs.items() if k != "label"},
                        **{k: v for k, v in pl.items() if k != "label"},
                        **{k: v for k, v in pr.items() if k != "label"},
                    }
                    configs.append(cfg)
                    cid += 1
    assert len(configs) == 24, f"expected 24 configs, got {len(configs)}"
    return configs


# Crewmate architectural baseline (fixed across the entire sweep):
# privileged state, semantic+LLM memory, meeting reflection, skill memory
# with self,observed sources, reactive+LLM planning, minimal prompt.
# Sweep below only varies the imposter-side architecture.
CREWMATE_BASELINE_CLI = [
    "--state-mode", "privileged",
    "--memory-mode", "semantic",
    "--memory-belief-update", "llm",
    "--reflection-mode", "meeting",
    "--planning-mode", "reactive",
    "--planning-policy", "llm",
    "--prompt-style", "minimal",
    "--skill-memory",
    "--skill-memory-sources", "self,observed",
    "--module-policy", "vlm",
    "--vlm-kill-fallback", "-1",
    "--no-parallel", "--verbose",
]


def cfg_to_cli_args(cfg: dict) -> list[str]:
    """Crewmate baseline (fixed) + per-config imposter overrides."""
    args = list(CREWMATE_BASELINE_CLI)
    args += [
        "--imposter-memory-mode", cfg["memory_mode"],
        "--imposter-memory-belief-update", cfg["memory_belief_update"],
        "--imposter-reflection-mode", cfg["reflection_mode"],
        "--imposter-planning-mode", cfg["planning_mode"],
        "--imposter-planning-policy", cfg["planning_policy"],
        "--imposter-prompt-style", cfg["prompt_style"],
    ]
    if cfg["skill_memory"]:
        args.append("--imposter-skill-memory")
        if cfg["skill_memory_sources"]:
            args += ["--imposter-skill-memory-sources", cfg["skill_memory_sources"]]
    else:
        args.append("--no-imposter-skill-memory")
    return args


# ──────────────────────────────────────────────────────────────
# Game running
# ──────────────────────────────────────────────────────────────

def kill_leftover_servers() -> None:
    for cmd in (
        "pkill -9 -f 'python.*amongus' 2>/dev/null || true",
        "pkill -9 -f 'fabric-server' 2>/dev/null || true",
        "pkill -9 -f 'java -Xmx' 2>/dev/null || true",
        "pkill -9 -f 'node.*index' 2>/dev/null || true",
    ):
        subprocess.run(cmd, shell=True)
    time.sleep(5)


def run_one_game(cfg: dict, run_idx: int, log_dir: Path, extra_args: list[str] | None = None) -> dict:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "game.log"
    cmd = [sys.executable, str(RUN_SCRIPT)] + cfg_to_cli_args(cfg) + list(extra_args or [])

    start = datetime.now()
    print(f"\n{'=' * 70}\n  CONFIG {cfg['id']:02d} ({cfg['label']}) — run {run_idx + 1}"
          f"\n  start: {start.isoformat(timespec='seconds')}\n  log: {log_path}\n"
          + "=" * 70, flush=True)

    kill_leftover_servers()

    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)

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
        f"imp_ej={parsed['imposters_ejected']}  crew_ej={parsed['crewmates_ejected']}  "
        f"vote_acc={parsed['vote_accuracy']}\n"
        f"  tokens in/out={parsed['tokens_input']}/{parsed['tokens_output']}  "
        f"cost=${parsed['cost_usd']}",
        flush=True,
    )

    return {
        "config_id": cfg["id"], "config_label": cfg["label"], "run_idx": run_idx,
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
VOTE_TALLY_HEADER = re.compile(r"\[VoteTally\] meeting #(\d+) tick=(\d+)")
VOTE_TALLY_VOTE = re.compile(r"^\s+(\w+) → (\w+|skip)\s*$")
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
        "vote_correct_ratio": None, "imposter_survival_steps": None,
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

    # Death signals from scoreboard. This includes both kills and vote
    # ejections, so actual kill_victims is computed after ejections are parsed.
    ghost_seen: set[str] = set()
    ghost_order: list[str] = []
    for m in GHOST_LINE.finditer(text):
        v = m.group(1)
        if v not in ghost_seen:
            ghost_seen.add(v)
            ghost_order.append(v)

    # Meetings + reporter info (count distinct meeting numbers per agent
    # or per unique tick to avoid duplicate counting due to fanout)
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

    # Vote results — count ejections by unique meeting+tick
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

    # Some runs end or transition before StateBuilder receives vote_result,
    # but the Minecraft/system chat still fans out the final result to agents.
    # Use that as a fallback and de-duplicate fanout copies.
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

    # Missions
    out["missions_completed"] = len(MISSION_DONE.findall(text))

    # Token / cost
    if m := TOKEN_LINE.search(text):
        out["cost_usd"] = float(m.group(1))
    if m := TOKEN_TOTAL.search(text):
        out["tokens_input"] = int(m.group(1).replace(",", ""))
        out["tokens_output"] = int(m.group(2).replace(",", ""))

    return out


def aggregate_per_config(rows: list[dict]) -> list[dict]:
    """Group by config_id, compute aggregates."""
    by_cfg: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_cfg[r["config_id"]].append(r)

    summary: list[dict] = []
    for cid, group in sorted(by_cfg.items()):
        n = len(group)
        wins = Counter(g["winner"] for g in group)
        steps = [g["steps"] for g in group if g["steps"] is not None]
        kills = [g["kills"] for g in group]
        meetings = [g["meetings"] for g in group]
        missions = [g["missions_completed"] for g in group]
        durs = [g["duration_sec"] for g in group if g["duration_sec"] is not None]
        costs = [g["cost_usd"] for g in group if g["cost_usd"] is not None]
        accs = [g["vote_accuracy"] for g in group if g["vote_accuracy"] is not None]
        self_rep = [g["self_reports"] for g in group]
        imp_rep = [g["imposter_reports"] for g in group]
        imp_eject = sum(g["imposters_ejected"] for g in group)
        crew_eject = sum(g["crewmates_ejected"] for g in group)

        summary.append({
            "config_id": cid,
            "label": group[0]["config_label"],
            "n_games": n,
            "win_imposter":         wins.get("imposter", 0),
            "win_crewmate_vote":    wins.get("crewmate_vote", 0),
            "win_crewmate_mission": wins.get("crewmate_mission", 0),
            "win_timeout_imp":      wins.get("timeout_imposter", 0),
            "imp_win_rate": round(
                (wins.get("imposter", 0) + wins.get("timeout_imposter", 0)) / n, 3),
            "crew_win_rate": round(
                (wins.get("crewmate_vote", 0) + wins.get("crewmate_mission", 0)) / n, 3),
            "avg_steps":    round(mean(steps), 1) if steps else None,
            "avg_duration_s": round(mean(durs), 1) if durs else None,
            "avg_kills":    round(mean(kills), 2) if kills else None,
            "avg_meetings": round(mean(meetings), 2) if meetings else None,
            "avg_missions": round(mean(missions), 1) if missions else None,
            "imposters_ejected_total": imp_eject,
            "crewmates_ejected_total": crew_eject,
            "vote_accuracy": round(mean(accs), 3) if accs else None,
            "self_report_rate": round(mean(self_rep), 2) if self_rep else None,
            "imposter_report_rate": round(mean(imp_rep), 2) if imp_rep else None,
            "avg_cost_usd":   round(mean(costs), 4) if costs else None,
            "total_cost_usd": round(sum(costs), 4) if costs else None,
        })
    return summary


def write_markdown_summary(per_cfg: list[dict], path: Path) -> None:
    lines = ["# Sweep summary", ""]
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Total configs analyzed: {len(per_cfg)}")
    lines.append("")
    lines.append("| id | label | n | imp/crew win | imp_rate | crew_rate | "
                 "avg_steps | avg_kills | avg_meetings | avg_missions | "
                 "vote_acc | self_rep | imp_rep | $/game |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in per_cfg:
        wins = (f"{r['win_imposter']+r['win_timeout_imp']}/"
                f"{r['win_crewmate_vote']+r['win_crewmate_mission']}")
        lines.append(
            f"| {r['config_id']} | {r['label']} | {r['n_games']} | {wins} | "
            f"{r['imp_win_rate']} | {r['crew_win_rate']} | "
            f"{r['avg_steps']} | {r['avg_kills']} | {r['avg_meetings']} | "
            f"{r['avg_missions']} | {r['vote_accuracy']} | "
            f"{r['self_report_rate']} | {r['imposter_report_rate']} | "
            f"{r['avg_cost_usd']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def analyze(out_dir: Path) -> None:
    """Walk out_dir, parse every game.log, write CSVs + markdown."""
    rows: list[dict] = []
    # Find all game.log files under serverN/cfgXX_label/runYY/game.log
    for log_path in sorted(out_dir.rglob("game.log")):
        # Walk upward to determine config_id, config_label, run_idx
        run_dir = log_path.parent
        cfg_dir = run_dir.parent
        try:
            run_idx = int(run_dir.name.replace("run", ""))
        except ValueError:
            continue
        m = re.match(r"cfg(\d+)_(.+)", cfg_dir.name)
        if not m:
            continue
        cid = int(m.group(1))
        label = m.group(2)
        parsed = parse_game_log(log_path)
        rows.append({"config_id": cid, "config_label": label, "run_idx": run_idx,
                     "log_path": str(log_path), **parsed})

    if not rows:
        print(f"[analyze] no game.log files under {out_dir}")
        return

    runs_csv = out_dir / "all_runs.csv"
    with open(runs_csv, "w", newline="") as f:
        keys = list(rows[0].keys())
        # Drop list-typed fields from CSV to keep it readable
        keys = [k for k in keys if k not in ("kill_victims", "ejections")]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})

    runs_json = out_dir / "all_runs.json"
    runs_json.write_text(json.dumps(rows, indent=2, default=str))

    per_cfg = aggregate_per_config(rows)
    cfg_csv = out_dir / "per_config_summary.csv"
    with open(cfg_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_cfg[0].keys()))
        w.writeheader()
        w.writerows(per_cfg)

    write_markdown_summary(per_cfg, out_dir / "summary.md")

    print(f"[analyze] {len(rows)} runs across {len(per_cfg)} configs")
    print(f"  per-run:   {runs_csv}")
    print(f"  per-cfg:   {cfg_csv}")
    print(f"  markdown:  {out_dir / 'summary.md'}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-id", type=int, choices=[0, 1, 2],
                    help="0 → configs 0..7, 1 → 8..15, 2 → 16..23")
    ap.add_argument("--runs-per-config", type=int, default=3)
    ap.add_argument("--start-config-id", type=int, default=None,
                    help="Start from this global config id within the selected server slice")
    ap.add_argument("--start-run", type=int, default=1,
                    help="1-based run number to start from for --start-config-id only")
    ap.add_argument("--config-ids", type=str, default=None,
                    help="Comma-separated global config ids to run (overrides server slicing)")
    ap.add_argument("--out-dir", default=None,
                    help="Default: scripts/logs/sweep/<timestamp>/")
    ap.add_argument("--analyze-only", action="store_true",
                    help="Skip running games; just analyze logs in --out-dir")
    ap.add_argument("--model", type=str, default=None,
                    help="LLM model name passed through to inner script (e.g., qwen-3.6-27b)")
    ap.add_argument("--imposter-model", type=str, default=None,
                    help="LLM model for imposters (defaults to --model)")
    args = ap.parse_args()

    if args.analyze_only:
        if not args.out_dir:
            sys.exit("--analyze-only requires --out-dir")
        analyze(Path(args.out_dir))
        return

    if args.server_id is None and not args.config_ids:
        sys.exit("--server-id is required (0, 1, or 2) unless --config-ids given")

    configs = build_24_configs()
    if args.config_ids:
        want = {int(x) for x in args.config_ids.split(",") if x.strip()!=""}
        my_configs = [c for c in configs if c["id"] in want]
        slice_start = min(want); slice_end = max(want)
    else:
        slice_start = args.server_id * 8
        my_configs = configs[slice_start: slice_start + 8]
        slice_end = slice_start + 7

    if args.start_config_id is not None:
        if not (slice_start <= args.start_config_id <= slice_end):
            sys.exit(
                f"--start-config-id {args.start_config_id} is outside server{args.server_id} "
                f"slice ({slice_start}..{slice_end})"
            )
        my_configs = [cfg for cfg in my_configs if cfg["id"] >= args.start_config_id]

    if args.start_run < 1:
        sys.exit("--start-run is 1-based and must be >= 1")
    if args.start_run > args.runs_per_config:
        sys.exit("--start-run cannot be greater than --runs-per-config")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_tag = (args.model or "gpt-4.1-mini").replace("/", "-")
    if args.imposter_model and args.imposter_model != args.model:
        model_tag += f"_imp-{args.imposter_model.replace('/', '-')}"
    default_dirname = f"{timestamp}_{model_tag}"
    out_dir = Path(args.out_dir) if args.out_dir else SCRIPT_DIR / "logs" / "sweep" / default_dirname
    server_dir = out_dir / f"server{args.server_id}"
    server_dir.mkdir(parents=True, exist_ok=True)

    with open(server_dir / "configs.json", "w") as f:
        json.dump({"server_id": args.server_id,
                   "runs_per_config": args.runs_per_config,
                   "configs": my_configs}, f, indent=2)

    summary_csv = server_dir / "runs_summary.csv"
    fieldnames = ["config_id", "config_label", "run_idx", "start", "end",
                  "duration_sec", "exit_code", "log_path"]
    with open(summary_csv, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    print(f"\n{'#' * 70}")
    print(f"  Sweep slice: server{args.server_id} | configs "
          f"{slice_start}..{slice_start + 7} | {args.runs_per_config} runs each")
    if args.start_config_id is not None or args.start_run != 1:
        print(f"  Start point: config {args.start_config_id if args.start_config_id is not None else slice_start}, "
              f"run {args.start_run}")
    total_games = 0
    for cfg in my_configs:
        first_run = args.start_run - 1 if cfg["id"] == args.start_config_id else 0
        total_games += args.runs_per_config - first_run
    print(f"  Total games: {total_games}")
    print(f"  Output dir: {server_dir}")
    print("#" * 70)

    extra_args: list[str] = []
    if args.model:
        extra_args += ["--model", args.model]
    if args.imposter_model:
        extra_args += ["--imposter-model", args.imposter_model]

    for cfg in my_configs:
        first_run = args.start_run - 1 if cfg["id"] == args.start_config_id else 0
        for run_idx in range(first_run, args.runs_per_config):
            log_dir = server_dir / f"cfg{cfg['id']:02d}_{cfg['label']}" / f"run{run_idx:02d}"
            row = run_one_game(cfg, run_idx, log_dir, extra_args=extra_args)
            with open(summary_csv, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

    # Analyze own slice
    print(f"\n{'#' * 70}\n  Analyzing slice...\n{'#' * 70}")
    analyze(out_dir)


if __name__ == "__main__":
    main()
