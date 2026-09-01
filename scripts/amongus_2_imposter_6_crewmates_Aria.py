"""
Among Us 2 Imposter vs 5 Crewmates — Aria Agent (7 players total)

James (Red, Imposter) + Olivia (Pink, Imposter) vs
Steve (Blue), Jason (Green), Michael (Yellow), Noah (White), Ethan (Light Blue).

Usage:
    python amongus_2_imposter_6_crewmates_Aria.py --no-parallel
    python amongus_2_imposter_6_crewmates_Aria.py --preset full --no-parallel
    python amongus_2_imposter_6_crewmates_Aria.py --config configs/rq1_2v6.yaml --no-parallel
"""

import os
import sys
import signal
import argparse
import base64
import time
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np
from PIL import Image
from dotenv import load_dotenv
import cv2
import yaml

import mineland
from mineland.aria import Aria, AriaConfig
from mineland.aria.config import steve_equivalent, baseline, full
from mineland.aria.llm import MODEL_CONFIGS, get_model_config
from mineland.utils import base64_to_image


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

PRESETS = {
    "steve": steve_equivalent,
    "baseline": baseline,
    "full": full,
}

TASK_ID = "amongus_2_imposter_6_crewmates"
CHARACTERS = [
    {"name": "James",   "role": "imposter", "color": "Red",        "teammate": "Olivia"},
    {"name": "Steve",   "role": "crewmate", "color": "Blue"},
    {"name": "Jason",   "role": "crewmate", "color": "Green"},
    {"name": "Michael", "role": "crewmate", "color": "Yellow"},
    {"name": "Olivia",  "role": "imposter", "color": "Pink",       "teammate": "James"},
    {"name": "Noah",    "role": "crewmate", "color": "White"},
    {"name": "Ethan",   "role": "crewmate", "color": "Light Blue"},
    {"name": "Martin",  "role": "crewmate", "color": "Black"},
]
PLAYER_ID_MAP = {
    "James": 1, "Steve": 2, "Jason": 3, "Michael": 4,
    "Olivia": 5, "Noah": 6, "Ethan": 7, "Martin": 8,
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Among Us (2 imposter / 5 crewmate) with Aria agents")

    parser.add_argument("--config", "-c", type=str, default=None,
                        help="YAML config file (overrides other options)")
    parser.add_argument("--task", "-t", type=str, default=TASK_ID)
    parser.add_argument("--preset", type=str, default="steve",
                        choices=list(PRESETS.keys()),
                        help="Aria preset (default: steve)")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini",
                        help="LLM model name (used for crewmates, or all if --imposter-model not set)")
    parser.add_argument("--imposter-model", type=str, default=None,
                        help="LLM model name for imposters (defaults to --model)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="LLM temperature for all agents (default: 0.3)")
    parser.add_argument("--imposter-temperature", type=float, default=None,
                        help="LLM temperature for imposters only (overrides --temperature)")
    parser.add_argument("--prompt-style", type=str, default=None,
                        choices=["deterministic", "minimal"],
                        help="Override prompt style from preset")
    parser.add_argument("--memory-mode", type=str, default=None,
                        choices=["none", "window", "episodic", "semantic"])
    parser.add_argument("--memory-belief-update", type=str, default="llm",
                        choices=["rule", "llm"],
                        help="How SemanticMemory updates player beliefs (only used when memory-mode=semantic)")
    parser.add_argument("--planning-mode", type=str, default=None,
                        choices=["reactive", "shortterm", "hierarchical"])
    parser.add_argument("--reflection-mode", type=str, default=None,
                        choices=["none", "post_action", "periodic", "meeting"])
    parser.add_argument("--state-mode", type=str, default=None,
                        choices=["ego", "privileged"])
    parser.add_argument("--skill-memory", action="store_true", default=None,
                        help="Enable skill memory (requires reflection != none)")
    parser.add_argument("--no-skill-memory", dest="skill_memory", action="store_false")
    parser.add_argument("--skill-memory-sources", type=str, default=None,
                        help="Comma-separated sources for skill memory entries "
                             "(e.g. 'self' or 'self,observed'). Default = preset value.")
    parser.add_argument("--planning-policy", type=str, default=None,
                        choices=["rule", "llm"],
                        help="Planning sub-policy (rule=hardcoded priority, llm=model decides)")
    parser.add_argument("--module-policy", type=str, default=None,
                        choices=["rule", "vlm"],
                        help="Kill/report module policy (rule or vlm)")
    parser.add_argument("--vlm-kill-fallback", type=int, default=None,
                        help="After N consecutive VLM defer/stalk, force rule-based kill (0=disabled)")

    # Imposter-only architectural overrides (apply on top of the shared
    # values above when the agent's role is imposter).
    parser.add_argument("--imposter-prompt-style", type=str, default=None,
                        choices=["deterministic", "minimal"])
    parser.add_argument("--imposter-memory-mode", type=str, default=None,
                        choices=["none", "window", "episodic", "semantic"])
    parser.add_argument("--imposter-memory-belief-update", type=str, default=None,
                        choices=["rule", "llm"])
    parser.add_argument("--imposter-planning-mode", type=str, default=None,
                        choices=["reactive", "shortterm", "hierarchical"])
    parser.add_argument("--imposter-reflection-mode", type=str, default=None,
                        choices=["none", "post_action", "periodic", "meeting"])
    parser.add_argument("--imposter-skill-memory", action="store_true", default=None)
    parser.add_argument("--no-imposter-skill-memory", dest="imposter_skill_memory",
                        action="store_false")
    parser.add_argument("--imposter-skill-memory-sources", type=str, default=None)
    parser.add_argument("--imposter-planning-policy", type=str, default=None,
                        choices=["rule", "llm"])
    parser.add_argument("--parallel", action="store_true", default=True)
    parser.add_argument("--no-parallel", dest="parallel", action="store_false")
    parser.add_argument("--terminal-agent", type=str, default=None)
    parser.add_argument("--agent-save-root", type=str, default=None,
                        help="Base dir for per-agent storage (trace/images/memory). "
                             "If set, agents save to <root>/<name>; else ./RQ2/<session>/<name>.")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-meeting-speaks", type=int, default=3,
                        help="Per-agent speak turns in meeting (default 3 for 7 players)")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--instance-id", type=int, default=0,
                        help="Isolate concurrent games in one container: offsets ports "
                             "(server 25565+id, mineflayer 21301+id) and uses a private world dir.")

    return parser.parse_args()


args = parse_arguments()


def default_temperature_for_model(model: str) -> float:
    cfg = get_model_config(model)
    if cfg:
        if cfg.temperature_fixed is not None:
            return cfg.temperature_fixed
        return cfg.default_temperature
    if model.startswith("gpt-5"):
        return 1.0
    if model.startswith("qwen"):
        return 0.7
    return 0.3


def validate_model_or_exit(model: str) -> None:
    if model in MODEL_CONFIGS:
        return
    print(
        f"[ERROR] Unknown Aria model: {model}\n"
        f"Available models: {', '.join(MODEL_CONFIGS.keys())}"
    )
    sys.exit(1)


validate_model_or_exit(args.model)
if args.imposter_model:
    validate_model_or_exit(args.imposter_model)


# ──────────────────────────────────────────────────────────────
# Config resolution
# ──────────────────────────────────────────────────────────────

def load_personal_message(agent_name, task_id):
    """Load personal message from file."""
    path = os.path.join(os.path.dirname(__file__),
                        "personal_messages", task_id, f"{agent_name}.txt")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None


def build_aria_config(char: dict, model: str, preset_name: str, overrides: dict) -> AriaConfig:
    """Build AriaConfig for one character, applying overrides from YAML/CLI."""
    preset_fn = PRESETS[preset_name]
    kwargs = {
        "role": char["role"],
        "bot_name": char["name"],
        "llm_model": model,
        "vlm_model": model,
        "all_players": [{"name": c["name"], "color": c["color"]} for c in CHARACTERS],
        "player_id_map": PLAYER_ID_MAP,
        "personal_message": load_personal_message(char["name"], args.task),
        "teammate_imposter": char.get("teammate"),
    }
    for key, val in overrides.items():
        if val is not None:
            kwargs[key] = val

    if kwargs.get("planning_mode") == "hierarchical":
        kwargs.setdefault("planning_policy", "llm")

    return preset_fn(**kwargs)


if args.config:
    with open(args.config, "r") as f:
        yaml_cfg = yaml.safe_load(f)
    task_id = yaml_cfg.get("task_id", TASK_ID)
    agents_yaml = yaml_cfg["agents"]
else:
    task_id = args.task
    agents_yaml = None

cli_overrides = {
    "prompt_style": args.prompt_style,
    "memory_mode": args.memory_mode,
    "memory_belief_update": args.memory_belief_update,
    "planning_mode": args.planning_mode,
    "planning_policy": args.planning_policy,
    "reflection_mode": args.reflection_mode,
    "state_mode": args.state_mode,
    "module_policy": args.module_policy,
    "use_skill_memory": args.skill_memory,
    "vlm_kill_fallback": args.vlm_kill_fallback,
    "temperature": args.temperature,
}
if args.skill_memory_sources is not None:
    cli_overrides["skill_memory_sources"] = [
        s.strip() for s in args.skill_memory_sources.split(",") if s.strip()
    ]

imposter_cli_overrides = {
    "prompt_style": args.imposter_prompt_style,
    "memory_mode": args.imposter_memory_mode,
    "memory_belief_update": args.imposter_memory_belief_update,
    "planning_mode": args.imposter_planning_mode,
    "planning_policy": args.imposter_planning_policy,
    "reflection_mode": args.imposter_reflection_mode,
    "use_skill_memory": args.imposter_skill_memory,
}
if args.imposter_skill_memory_sources is not None:
    imposter_cli_overrides["skill_memory_sources"] = [
        s.strip() for s in args.imposter_skill_memory_sources.split(",") if s.strip()
    ]


def overrides_for_role(role: str) -> dict:
    """Base overrides + imposter-specific overrides when applicable."""
    merged = dict(cli_overrides)
    if role == "imposter":
        for k, v in imposter_cli_overrides.items():
            if v is not None:
                merged[k] = v
    return merged

# ──────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────

run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
if args.instance_id:
    # Keep concurrent instances from colliding on a same-second output path.
    run_time = f"{run_time}_inst{args.instance_id}"
game_session_time = run_time
# Unify run-level logs (main.log, per-agent .log, token_usage, configs) under
# --agent-save-root (RQ2/case/trial/matchup/<ts>) so copying RQ2 alone carries
# everything. Fall back to the standalone ./logs tree when no save-root is given.
log_base_dir = args.agent_save_root if args.agent_save_root else os.path.join("./logs", task_id, f"aria_{args.preset}", run_time)
os.makedirs(log_base_dir, exist_ok=True)

main_log_path = os.path.join(log_base_dir, "main.log")
main_log_file = open(main_log_path, "w", buffering=1)

thread_local = threading.local()


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()
    def flush(self):
        for st in self.streams:
            st.flush()


class ThreadAwareWriter:
    def __init__(self, default_writer, attr_name):
        self.default_writer = default_writer
        self.attr_name = attr_name
    def write(self, s):
        w = getattr(thread_local, self.attr_name, None) or self.default_writer
        return w.write(s)
    def flush(self):
        w = getattr(thread_local, self.attr_name, None) or self.default_writer
        w.flush()


default_stdout = Tee(sys.__stdout__, main_log_file)
default_stderr = Tee(sys.__stderr__, main_log_file)
sys.stdout = ThreadAwareWriter(default_stdout, "stdout_writer")
sys.stderr = ThreadAwareWriter(default_stderr, "stderr_writer")

print("Logging started.")
print("Main log file:", main_log_path)

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key


# ──────────────────────────────────────────────────────────────
# Image utils
# ──────────────────────────────────────────────────────────────

def to_hwc_rgb(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.size == 0 or a.ndim < 2:
        raise ValueError(f"Invalid image shape: {a.shape}")
    if a.ndim == 3 and a.shape[-1] in (3, 4):
        a = a[..., :3]
    elif a.ndim == 3 and a.shape[0] in (3, 4):
        a = np.transpose(a, (1, 2, 0))[..., :3]
    else:
        raise ValueError(f"Unsupported shape: {a.shape}")
    if np.issubdtype(a.dtype, np.floating):
        a = (a * 255.0).clip(0, 255).astype(np.uint8) if a.max() <= 1.0 + 1e-6 else a.clip(0, 255).astype(np.uint8)
    return a.astype(np.uint8)


# ──────────────────────────────────────────────────────────────
# Create environment
# ──────────────────────────────────────────────────────────────

agents_config_raw = [
    {"name": c["name"], "role": c["role"], "color": c["color"]}
    for c in CHARACTERS
]

def free_instance_ports(instance_id: int) -> None:
    """Kill only the processes holding THIS instance's ports.

    Replaces the old name-matching pkill, which would also kill every other
    game running in the same container.
    """
    import subprocess
    ports = [25565 + instance_id, 21301 + instance_id, 25575 + instance_id]
    for port in ports:
        subprocess.run(
            f"lsof -ti tcp:{port} | xargs -r kill -9",
            shell=True, stderr=subprocess.DEVNULL,
        )
    time.sleep(2)


free_instance_ports(args.instance_id)

mland = mineland.make(
    task_id=task_id,
    agents_count=len(CHARACTERS),
    agents_config=agents_config_raw,
    world_type="amongus_2im",
    headless=False,
    image_size=(360, 640),
    enable_auto_pause=False,
    instance_id=args.instance_id,
)

# ──────────────────────────────────────────────────────────────
# Initialize Aria agents
# ──────────────────────────────────────────────────────────────

agents = []
all_players = [{"name": c["name"], "color": c["color"]} for c in CHARACTERS]

print(f"\n{'='*60}")
print(f"  Initializing Aria Agents (preset={args.preset})")
print(f"{'='*60}")

for i, char in enumerate(CHARACTERS):
    role_overrides = overrides_for_role(char["role"])
    if agents_yaml and i < len(agents_yaml):
        agent_yaml = agents_yaml[i]
        default_model = args.imposter_model if (args.imposter_model and char["role"] == "imposter") else args.model
        model = agent_yaml.get("model", default_model)
        preset = agent_yaml.get("preset", args.preset)
        skip_keys = {"name", "role", "color", "model", "preset"}
        per_agent_overrides = {k: v for k, v in agent_yaml.items() if k not in skip_keys}
        for k, v in role_overrides.items():
            if v is not None:
                per_agent_overrides[k] = v
        if char["role"] == "imposter" and args.imposter_temperature is not None:
            per_agent_overrides["temperature"] = args.imposter_temperature
    else:
        model = (args.imposter_model or args.model) if char["role"] == "imposter" else args.model
        preset = args.preset
        per_agent_overrides = dict(role_overrides)

    if per_agent_overrides.get("temperature") is None:
        per_agent_overrides["temperature"] = default_temperature_for_model(model)

    if char["role"] == "imposter" and args.imposter_temperature is not None:
        per_agent_overrides["temperature"] = args.imposter_temperature

    if args.agent_save_root:
        agent_save_path = os.path.join(args.agent_save_root, char['name'])
    else:
        agent_save_path = f"./RQ2/{game_session_time}/{char['name']}"

    try:
        config = build_aria_config(char, model, preset, per_agent_overrides)
        config.save_path = agent_save_path
        agent = Aria(config)
        agents.append(agent)
        print(f"  {char['name']:10s} ({char['role']:10s}): {config.summary()}")
    except Exception as e:
        print(f"  [ERROR] Failed to create {char['name']}: {e}")
        sys.exit(1)

print(f"\n  Total: {len(agents)} Aria agents initialized\n")

# ──────────────────────────────────────────────────────────────
# Reset environment
# ──────────────────────────────────────────────────────────────

obs = mland.reset()
agents_count = len(obs)
agents_name = [obs[i]["name"] for i in range(agents_count)]
active_agents = min(agents_count, len(agents))

print("agents_count:", agents_count)
print("agents_name:", agents_name)

viewer_wait = agents_count * 1 + 3
print(f"Waiting {viewer_wait}s for viewers to initialize...")
time.sleep(viewer_wait)

agent_log_files = {}
main_log_lock = threading.Lock()
agent_log_locks = {}
for name in agents_name[:active_agents]:
    path = os.path.join(log_base_dir, f"{name}.log")
    agent_log_files[name] = open(path, "w", buffering=1)
    agent_log_locks[name] = threading.Lock()

# Frames follow --agent-save-root too: RQ2/.../<ts>/<agent>/images/frame_NNN.png
# (alongside that agent's memory), so RQ2 is a self-contained per-match/step dataset.
save_base_dir = args.agent_save_root if args.agent_save_root else os.path.join("./frames", task_id, f"aria_{args.preset}", run_time)
os.makedirs(save_base_dir, exist_ok=True)
agent_frame_dirs = {}
for name in agents_name[:active_agents]:
    d = os.path.join(save_base_dir, name, "images") if args.agent_save_root else os.path.join(save_base_dir, name)
    os.makedirs(d, exist_ok=True)
    agent_frame_dirs[name] = d

code_info = [None] * agents_count
task_info = None
done = False
event = None
agent_alive = {name: True for name in agents_name[:active_agents]}
meeting_speaker_idx = 0
MAX_MEETING_SPEAKS = args.max_meeting_speaks
meeting_speak_counts = {name: 0 for name in agents_name[:active_agents]}
prev_phase = 0
# Voting state: armed when all have spoken (timer set to 750), tally injected
# once every alive agent's vote_module._has_voted is True. Reset on next
# meeting_start.
vote_phase_armed = False
vote_result_injected = False
dead_players_at_last_meeting = set()

INITIAL_CREW_COUNT = sum(1 for c in CHARACTERS if c["role"] == "crewmate")
MISSIONS_PER_CREW = 3


def sync_global_mission_progress():
    """Aggregate crewmates' pending missions and inject into imposters' state.

    Before mission scoreboard packets arrive, crewmates' mission_status is
    empty — `len(pending_missions())` returns 0, which would falsely read as
    "all done". Fall back to the initial full count until every crewmate has
    received their assignment.
    """
    total_pending = 0
    populated = 0
    for ag in agents:
        sb = getattr(ag, "state_builder", None)
        if sb is not None and sb.role == "crewmate":
            total_pending += len(sb.pending_missions())
            if len(sb.mission_status) > 0:
                populated += 1
    if populated < INITIAL_CREW_COUNT:
        total_pending = INITIAL_CREW_COUNT * MISSIONS_PER_CREW
    for ag in agents:
        sb = getattr(ag, "state_builder", None)
        if sb is not None and sb.role == "imposter":
            sb.set_global_mission_progress(
                remaining=total_pending, initial_crew=INITIAL_CREW_COUNT,
            )


def extract_report_from_observations(observations, player_names):
    """Return (reporter, victim) from reliable server broadcast messages.

    This is intentionally kept in orchestration, not env_adapter, so meeting
    structured events still fire exactly once.
    """
    names = set(player_names)
    patterns = [
        re.compile(r"\b([A-Za-z0-9_]+)\s+reported\s+([A-Za-z0-9_]+)'s body!?"),
        re.compile(r"\b([A-Za-z0-9_]+)\s+reported\s+([A-Za-z0-9_]+)\s+body!?"),
    ]
    for ob in observations:
        events = getattr(ob, "event", None) or []
        for ev in events:
            msg = ev.get("message", "") if isinstance(ev, dict) else getattr(ev, "message", "")
            if not msg or "reported" not in msg:
                continue
            for pat in patterns:
                m = pat.search(msg)
                if not m:
                    continue
                reporter, victim = m.group(1), m.group(2)
                if reporter in names and victim in names:
                    return reporter, victim
    return None, None


def _event_field(event, key, default=None):
    if isinstance(event, dict):
        return event.get(key, default)
    return getattr(event, key, default)


def dedupe_events(events):
    """Preserve first-seen order while dropping duplicate Mineflayer events."""
    seen = set()
    deduped = []
    for event in events:
        key = (
            _event_field(event, "tick"),
            _event_field(event, "type"),
            _event_field(event, "message"),
            _event_field(event, "position"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(event)
    return deduped


def tally_and_inject_vote_result(agents_, agent_alive_, meeting_no, tick):
    """Tally intended vote targets and inject a provisional vote_result.

    The authoritative outcome is the server tellraw result parsed by
    env_adapter. This synthetic result is a fallback for runs where that
    tellraw is missing; StateBuilder will overwrite it when server_tellraw
    arrives.
    """
    from mineland.aria.env_adapter import GameEvent

    tally = {}
    skip_count = 0
    voter_lines = []
    for ag in agents_:
        name = ag.bot_name
        if not agent_alive_.get(name, True):
            continue
        vm = getattr(ag, "vote_module", None)
        if vm is None or not getattr(vm, "_has_voted", False):
            continue
        target = vm._last_vote_target
        voter_lines.append(f"  {name} → {target}")
        if not target or target == "skip":
            skip_count += 1
        else:
            tally[target] = tally.get(target, 0) + 1

    print(f"[VoteTally:provisional] meeting #{meeting_no} tick={tick}")
    for line in voter_lines:
        print(line)
    print(f"  Total: {tally} | skip={skip_count}")

    if not tally:
        ejected, role, skipped = None, None, True
    else:
        top_name = max(tally, key=lambda n: tally[n])
        top_count = tally[top_name]
        tied_with_skip = (skip_count >= top_count)
        tied_with_other = sum(1 for n, c in tally.items() if c == top_count) > 1
        if tied_with_skip or tied_with_other:
            ejected, role, skipped = None, None, True
        else:
            ejected = top_name
            skipped = False
            ejected_role = None
            for ag in agents_:
                if ag.bot_name == ejected:
                    ejected_role = ag.role
                    break
            role = ejected_role

    msg = (f"(orchestration provisional) vote_result: "
           f"ejected={ejected} role={role} skipped={skipped}")
    for ag in agents_:
        sb = getattr(ag, "state_builder", None)
        if sb is None:
            continue
        sb._process_event(GameEvent(
            type="vote_result", message=msg,
            data={
                "ejected": ejected,
                "ejected_role": role,
                "skipped": skipped,
                "source": "orchestration_tally",
                "authoritative": False,
            },
        ))
    return ejected, role, skipped


# ──────────────────────────────────────────────────────────────
# Agent execution helpers
# ──────────────────────────────────────────────────────────────

class ThreadSafeWriter:
    def __init__(self, agent_name, step_i):
        self.agent_name = agent_name
        self.step_i = step_i
        self.prefix = f"[step={step_i:03d}][{agent_name}] "
        self.show = (args.terminal_agent is None or agent_name == args.terminal_agent)

    def write(self, s):
        if not s or s == "\n":
            return
        lines = s.rstrip("\n").split("\n")
        for line in lines:
            full = f"{self.prefix}{line}\n"
            with agent_log_locks.get(self.agent_name, main_log_lock):
                f = agent_log_files.get(self.agent_name)
                if f:
                    f.write(full)
                    f.flush()
            with main_log_lock:
                main_log_file.write(full)
                main_log_file.flush()
            if self.show:
                sys.__stdout__.write(full)
                sys.__stdout__.flush()

    def flush(self):
        pass


def run_agent_reasoning(idx, step_i, cur_obs, cur_code_info, cur_done, cur_task_info):
    agent_name = agents_name[idx]
    writer = ThreadSafeWriter(agent_name, step_i)
    thread_local.stdout_writer = writer
    thread_local.stderr_writer = writer

    # Tag this agent's LLM calls with the env step index for prompt tracing.
    from mineland.aria.llm import set_current_step
    set_current_step(step_i)

    # Save the EXACT image fed to this agent's LLM this step, keyed by step_i, so
    # frame_{step}.png aligns 1:1 with game.log / state_timeline step. Only agents
    # that actually reason (input went to the LLM) get a frame for that step.
    try:
        _img = to_hwc_rgb(cur_obs[idx]["rgb"])
        Image.fromarray(_img).save(
            os.path.join(agent_frame_dirs[agent_name], f"frame_{step_i:03d}.png"))
    except Exception:
        pass

    try:
        action = agents[idx].run(
            cur_obs[idx], cur_code_info[idx], cur_done, cur_task_info,
            verbose=args.verbose,
        )
        return {"idx": idx, "action": action}
    except Exception as e:
        print(f"[ERROR] {agent_name} reasoning failed: {e}")
        import traceback
        traceback.print_exc()
        return {"idx": idx, "action": None}
    finally:
        thread_local.stdout_writer = None
        thread_local.stderr_writer = None


def _get_obs_val(o, key):
    val = getattr(o, key, None)
    if val is None and isinstance(o, dict):
        val = o.get(key)
    return val


def _is_meeting_talk(obs_list):
    for i in range(active_agents):
        if agent_alive.get(agents_name[i], True):
            return (_get_obs_val(obs_list[i], 'phase') == 1 and
                    _get_obs_val(obs_list[i], 'talk') == 1)
    return False


def _apply_meeting_dead_context(new_dead_since_last_meeting):
    """Ensure each StateBuilder sees every death before the current meeting.

    The server report names only the body that was found. The orchestration
    layer separately tracks every player who became ghost since the previous
    meeting. Because the report event may already be consumed before that list
    is attached, patch StateBuilder directly as well.
    """
    if not new_dead_since_last_meeting:
        return
    new_dead = sorted(set(new_dead_since_last_meeting))
    for agent in agents:
        sb = getattr(agent, "state_builder", None)
        if sb is None:
            continue
        existing = list(getattr(sb, "current_meeting_new_dead", []) or [])
        merged = sorted(set(existing) | set(new_dead))
        sb.current_meeting_new_dead = merged
        for dead_name in merged:
            if hasattr(sb, "_dead_players"):
                sb._dead_players.add(dead_name)
        print(f"[MeetingContext:{agent.bot_name}] deaths_since_previous_meeting={merged}")


# ──────────────────────────────────────────────────────────────
# Main game loop
# ──────────────────────────────────────────────────────────────

game_start_time = time.time()
_disconnect_detected = False

try:
    for step_i in range(args.max_steps):
        if step_i > 0 and step_i % 10 == 0:
            print(f"step: {step_i}, task_info: {task_info}")

        # Disconnect detection: any obs slot None → a bot dropped (mineflayer
        # flipped mineland_is_active=false on end/kicked/error). End the run
        # and mark as INVALID so it can be re-run.
        if step_i > 0:
            dropped = [agents_name[i] for i in range(active_agents) if obs[i] is None]
            if dropped:
                print(f"\n{'='*60}\n  DISCONNECT detected: {', '.join(dropped)} — ending game (INVALID)\n{'='*60}\n")
                _disconnect_detected = True
                break

        if step_i == 0:
            actions = mineland.Action.no_op(agents_count)
            obs, code_info, event, done, task_info = mland.step(action=actions)
        else:
            for idx in range(active_agents):
                name = agents_name[idx]
                ghost_val = _get_obs_val(obs[idx], 'ghost')
                if ghost_val == 1 and agent_alive.get(name, True):
                    print(f"\n{'='*60}\n  {name} has DIED (ghost=1)\n{'='*60}\n")
                    agent_alive[name] = False
                    for other in agents[:active_agents]:
                        sb = getattr(other, "state_builder", None)
                        if sb is not None and hasattr(sb, "_dead_players"):
                            sb._dead_players.add(name)

            all_spoke_done = all(
                meeting_speak_counts.get(agents_name[i], 0) >= MAX_MEETING_SPEAKS
                for i in range(active_agents)
                if agent_alive.get(agents_name[i], True)
            )

            if not args.parallel and _is_meeting_talk(obs) and not all_spoke_done:
                attempts = 0
                while attempts < active_agents:
                    name_candidate = agents_name[meeting_speaker_idx % active_agents]
                    alive = agent_alive.get(name_candidate, True)
                    has_turns = meeting_speak_counts.get(name_candidate, 0) < MAX_MEETING_SPEAKS
                    if alive and has_turns:
                        break
                    meeting_speaker_idx += 1
                    attempts += 1

                idx = meeting_speaker_idx % active_agents
                agent_name = agents_name[idx]
                meeting_speak_counts[agent_name] += 1
                print(f"[Meeting] step {step_i}, speaker={agent_name} (count={meeting_speak_counts[agent_name]}/{MAX_MEETING_SPEAKS})")

                # Drain pending events into every non-speaker's state_builder
                # before this step (mineflayer clears its event buffer on the
                # next mland.step()). Without this, broadcast events like
                # `(SERVER) X reported Y's body!` would only be processed by
                # whichever agent happens to be speaking that step — others
                # would never fire meeting_start.
                for _i in range(active_agents):
                    if _i == meeting_speaker_idx % active_agents:
                        continue
                    _ag = agents[_i]
                    _sb = getattr(_ag, "state_builder", None)
                    if _sb is None:
                        continue
                    try:
                        _gs = _ag.env_adapter.parse_obs(obs[_i])
                        _sb.update(_gs)
                    except Exception as _e:
                        if args.verbose:
                            print(f"[Meeting] state_builder drain failed for {agents_name[_i]}: {_e}")

                result = run_agent_reasoning(idx, step_i, obs, code_info, done, task_info)
                action = result["action"] if result else None

                actions = mineland.Action.no_op(agents_count)
                if action is not None:
                    actions[idx] = action
                obs, code_info, event, done, task_info = mland.step(actions)

                all_spoke_now = all(
                    meeting_speak_counts.get(agents_name[i], 0) >= MAX_MEETING_SPEAKS
                    for i in range(active_agents)
                    if agent_alive.get(agents_name[i], True)
                )
                if all_spoke_now:
                    for _ in range(2):
                        obs, code_info, event, done, task_info = mland.step(mineland.Action.no_op(agents_count))
                    print(f"[Meeting] All agents reached {MAX_MEETING_SPEAKS} speaks — VOTE transition")
                    mland.server_manager.execute("scoreboard players set meeting meeting_timer 750")
                    for agent in agents:
                        if hasattr(agent, 'state_builder'):
                            agent.state_builder._force_all_spoke = True
                    vote_phase_armed = True
                    vote_result_injected = False

                meeting_speaker_idx += 1

            else:
                print(f"[Sync] step {step_i}, reasoning for {active_agents} agents")

                actions_map = {}
                with ThreadPoolExecutor(max_workers=active_agents) as executor:
                    futures = {}
                    for idx in range(active_agents):
                        name = agents_name[idx]
                        alive = agent_alive.get(name, True)
                        is_crewmate = CHARACTERS[idx]["role"] == "crewmate"
                        if alive or is_crewmate:
                            future = executor.submit(
                                run_agent_reasoning, idx, step_i, obs, code_info, done, task_info
                            )
                            futures[future] = idx

                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            if result:
                                actions_map[result['idx']] = result['action']
                        except Exception as e:
                            print(f"[ERROR] Reasoning: {e}")
                            import traceback
                            traceback.print_exc()

                batch_actions = [
                    actions_map.get(idx) or mineland.Action.no_op(1)[0]
                    for idx in range(agents_count)
                ]
                new_obs, new_ci, new_ev, new_done, new_ti = mland.step(batch_actions)
                accumulated_events = [[] for _ in range(active_agents)]
                for idx in range(active_agents):
                    accumulated_events[idx].extend(getattr(new_obs[idx], "event", []) or [])

                resume_tick = 0
                while True:
                    still_running = any(
                        (new_ci[idx].get("is_running", False)
                         if isinstance(new_ci[idx], dict)
                         else getattr(new_ci[idx], "is_running", False))
                        for idx in range(active_agents)
                        if agent_alive.get(agents_name[idx], True) or CHARACTERS[idx]["role"] == "crewmate"
                    )
                    if not still_running:
                        break
                    resume_actions = [mineland.Action(type=mineland.Action.RESUME, code="")
                                      for _ in range(agents_count)]
                    new_obs, new_ci, new_ev, new_done, new_ti = mland.step(resume_actions)
                    for idx in range(active_agents):
                        accumulated_events[idx].extend(getattr(new_obs[idx], "event", []) or [])
                    resume_tick += 1
                    if new_done:
                        break

                print(f"[Sync] action complete after {resume_tick} resume(s)")

                for idx in range(active_agents):
                    new_obs[idx].event = dedupe_events(accumulated_events[idx])
                    obs[idx] = new_obs[idx]
                    code_info[idx] = new_ci[idx]
                event = new_ev
                done = new_done
                task_info = new_ti

            phase_detected = None
            for idx in range(active_agents):
                p = getattr(obs[idx], 'phase', None)
                if p is not None:
                    phase_detected = p
                    break

            # Inject vote_result as soon as every alive agent has cast a vote
            # — bypasses unreliable broadcast parsing of meeting_show_results /
            # meeting_reveal_result tellraw output.
            if vote_phase_armed and not vote_result_injected:
                all_voted = all(
                    getattr(getattr(ag, "vote_module", None), "_has_voted", False)
                    for ag in agents
                    if agent_alive.get(ag.bot_name, True)
                )
                if all_voted:
                    sb_any = next((getattr(ag, "state_builder", None) for ag in agents
                                   if getattr(ag, "state_builder", None) is not None), None)
                    meeting_no = getattr(sb_any, "_meeting_count", 0) if sb_any else 0
                    tick_now = getattr(sb_any, "tick", 0) if sb_any else 0
                    tally_and_inject_vote_result(agents, agent_alive, meeting_no, tick_now)
                    dead_players_at_last_meeting = {
                        n for n, alive in agent_alive.items() if not alive
                    }
                    vote_result_injected = True

            if phase_detected == 1 and prev_phase == 0:
                print(f"\n{'='*60}\n  MEETING PHASE DETECTED\n{'='*60}\n")

                _dead_now = {n for n, alive in agent_alive.items() if not alive}
                _new_dead_since_last_meeting = sorted(_dead_now - dead_players_at_last_meeting)
                dead_players_at_last_meeting = set(_dead_now)

                # Drain pending broadcast events into every agent's state_builder
                # BEFORE the no_op step below — that step calls mineflayer's
                # clearEvents() and would wipe the `(SERVER) X reported Y's body!`
                # message that fires meeting_start. Without this, only agents
                # who happened to run agent.run() on the right step would
                # process the broadcast.
                for _i in range(active_agents):
                    _ag = agents[_i]
                    _sb = getattr(_ag, "state_builder", None)
                    if _sb is None:
                        continue
                    try:
                        _gs = _ag.env_adapter.parse_obs(obs[_i])
                        for _event in _gs.events:
                            if _event.type == "meeting_start":
                                _event.data["new_dead_since_last_meeting"] = list(_new_dead_since_last_meeting)
                        _sb.update(_gs)
                    except Exception as _e:
                        if args.verbose:
                            print(f"[Meeting] phase-transition drain failed for {agents_name[_i]}: {_e}")

                meeting_speaker_idx = 0
                meeting_speak_counts = {name: 0 for name in agents_name[:active_agents]}
                vote_phase_armed = False
                vote_result_injected = False
                for agent in agents:
                    if hasattr(agent, 'state_builder'):
                        agent.state_builder._force_all_spoke = False
                        agent.state_builder.speech_counts = {}
                    if hasattr(agent, 'meeting_module'):
                        agent.meeting_module._speech_count = 0
                    if hasattr(agent, 'vote_module'):
                        agent.vote_module._has_voted = False
                        agent.vote_module._last_vote_target = None

                int_actions = mineland.Action.no_op(agents_count)
                int_obs, int_ci, int_ev, int_done, int_ti = mland.step(int_actions)
                for i in range(active_agents):
                    obs[i] = int_obs[i]
                    code_info[i] = int_ci[i]
                event = int_ev
                done = int_done
                task_info = int_ti

                # meeting_start event (with reporter/victim/trigger) is now
                # fired by env_adapter from the (SERVER) broadcast — confirmed
                # 100% reliable across runs. Orchestration does not duplicate
                # the inject (would cause meeting_start_index to change twice
                # → VoteModule would reset _has_voted twice → double-voting).
                # Just track new dead since last meeting for our own bookkeeping.
                if _new_dead_since_last_meeting:
                    print(f"  [Meeting] new dead since last: {_new_dead_since_last_meeting}")
                    _apply_meeting_dead_context(_new_dead_since_last_meeting)

                dead_agents = [n for n, alive in agent_alive.items() if not alive]
                if dead_agents:
                    print(f"  [Meeting] Dead players: {dead_agents}")
                    for dead_name in dead_agents:
                        for agent in agents:
                            if hasattr(agent, 'state_builder'):
                                agent.state_builder._dead_players.add(dead_name)

            prev_phase = phase_detected if phase_detected is not None else prev_phase

            sync_global_mission_progress()

        # Frames are now saved inside run_agent_reasoning, keyed by the step's
        # actual LLM-input image (post-action save removed to fix step misalignment).
        for idx in range(active_agents):
            game_end_val = None
            ao = obs[idx]
            if hasattr(ao, 'scoreboard'):
                sb = ao.scoreboard
                game_end_val = getattr(sb, 'game_end', None)
                if game_end_val is None and isinstance(sb, dict):
                    game_end_val = sb.get('game_end')
            elif isinstance(ao, dict):
                game_end_val = ao.get('scoreboard', {}).get('game_end')

            if game_end_val == 1:
                print(f"\n  GAME END detected (agent={agents_name[idx]})")
                done = True
                break

        if done:
            print(f"done at step {step_i}")
            break

finally:
    elapsed = time.time() - game_start_time
    elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"

    try:
        winner = None
        try:
            winner, _ = mland.sim.server_manager.get_game_result()
        except Exception:
            pass

        # Log file fallback — only match authoritative "<Server>" messages,
        # not player chat that may contain "imposter win" in conversation.
        if not winner:
            try:
                with open(main_log_path, 'r', errors='ignore') as f:
                    for line in f:
                        low = line.lower()
                        if '<server>' not in low:
                            continue
                        # Require the game-server entity as speaker
                        speaker_idx = low.find('<server>')
                        msg = low[speaker_idx + len('<server>'):]
                        if 'crew mission win' in msg:
                            winner = 'crewmate_mission'
                            break
                        elif 'crew win' in msg or 'crewmate win' in msg or 'crewmates win' in msg:
                            winner = 'crewmate'
                            break
                        elif 'imposter win' in msg or 'imposters win' in msg:
                            winner = 'imposter'
                            break
            except Exception:
                pass

        result_map = {
            'imposter': 'IMPOSTER WIN',
            'crewmate': 'CREWMATE WIN',
            'crewmate_mission': 'CREWMATE MISSION WIN',
        }
        if _disconnect_detected:
            result_str = 'INVALID (DISCONNECT)'
        else:
            result_str = result_map.get(winner, 'TIMEOUT IMPOSTER WIN')

        print(f"\n{'='*60}")
        print(f"  RESULT: {result_str}")
        print(f"  Steps : {step_i + 1}")
        print(f"  Time  : {elapsed_str}")
        print(f"{'='*60}\n")
    except Exception as e:
        print(f"[warn] {e}")

    try:
        summary_path = os.path.join(log_base_dir, "aria_configs.txt")
        with open(summary_path, "w") as f:
            for agent in agents:
                f.write(f"{agent.config.summary()}\n")
        print(f"Aria configs saved to: {summary_path}")
    except Exception:
        pass

    try:
        from mineland.aria.llm import get_token_summary
        cost_summary = get_token_summary(show_cost=True)
        print(f"\n{cost_summary}")
        cost_path = os.path.join(log_base_dir, "token_usage.txt")
        with open(cost_path, "w") as f:
            f.write(cost_summary)
        print(f"Token usage saved to: {cost_path}")
    except Exception as e:
        print(f"[warn] token summary: {e}")

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    for f in agent_log_files.values():
        try:
            f.close()
        except Exception:
            pass
    try:
        main_log_file.close()
    except Exception:
        pass

    try:
        mland.close()
        print(f"Finished. Logs: {log_base_dir}")
    except Exception:
        pass

    os._exit(0)
