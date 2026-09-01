"""
Among Us 1 Imposter vs 3 Crewmates — Aria Agent

Runs 4 Aria agents in an Among Us game. Supports:
- YAML config for per-agent AriaConfig presets
- CLI flags for quick experiments (--preset, --prompt-style, etc.)
- Same sync execution model as the Steve script

Usage:
    # Default: steve-equivalent config for all agents
    python amongus_1_imposter_3_crewmates_Aria.py --no-parallel

    # Specific preset
    python amongus_1_imposter_3_crewmates_Aria.py --preset full --no-parallel

    # YAML config
    python amongus_1_imposter_3_crewmates_Aria.py --config configs/1v3.yaml --no-parallel

    # Quick ablation: change prompt style only
    python amongus_1_imposter_3_crewmates_Aria.py --preset steve --prompt-style minimal --no-parallel
"""

import os
import sys
import signal
import argparse
import base64
import time
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
from mineland.utils import base64_to_image


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

PRESETS = {
    "steve": steve_equivalent,
    "baseline": baseline,
    "full": full,
}

TASK_ID = "amongus_1_imposter_3_crewmates"
CHARACTERS = [
    {"name": "James",   "role": "imposter", "color": "Red"},
    {"name": "Steve",   "role": "crewmate", "color": "Blue"},
    {"name": "Jason",   "role": "crewmate", "color": "Green"},
    {"name": "Michael", "role": "crewmate", "color": "Yellow"},
]
PLAYER_ID_MAP = {"James": 1, "Steve": 2, "Jason": 3, "Michael": 4}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Among Us with Aria agents")

    parser.add_argument("--config", "-c", type=str, default=None,
                        help="YAML config file (overrides other options)")
    parser.add_argument("--task", "-t", type=str, default=TASK_ID)
    parser.add_argument("--preset", type=str, default="steve",
                        choices=list(PRESETS.keys()),
                        help="Aria preset (default: steve)")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini",
                        help="LLM model name")
    parser.add_argument("--prompt-style", type=str, default=None,
                        choices=["deterministic", "minimal"],
                        help="Override prompt style from preset")
    parser.add_argument("--memory-mode", type=str, default=None,
                        choices=["none", "window", "episodic", "semantic"])
    parser.add_argument("--planning-mode", type=str, default=None,
                        choices=["reactive", "shortterm", "hierarchical"])
    parser.add_argument("--reflection-mode", type=str, default=None,
                        choices=["none", "post_action", "periodic", "meeting"])
    parser.add_argument("--state-mode", type=str, default=None,
                        choices=["ego", "privileged"])
    parser.add_argument("--module-policy", type=str, default=None,
                        choices=["rule", "vlm"],
                        help="Kill/report module policy (rule or vlm)")
    parser.add_argument("--parallel", action="store_true", default=True)
    parser.add_argument("--no-parallel", dest="parallel", action="store_false")
    parser.add_argument("--terminal-agent", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--verbose", action="store_true", default=False)

    return parser.parse_args()


args = parse_arguments()


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
    """Build AriaConfig for one character, applying overrides from YAML/CLI.

    All AriaConfig fields can be overridden. Non-None values in `overrides`
    take precedence over the preset defaults.
    """
    preset_fn = PRESETS[preset_name]
    kwargs = {
        "role": char["role"],
        "bot_name": char["name"],
        "llm_model": model,
        "vlm_model": model,
        "all_players": [{"name": c["name"], "color": c["color"]} for c in CHARACTERS],
        "player_id_map": PLAYER_ID_MAP,
        "personal_message": load_personal_message(char["name"], args.task),
    }
    # Apply ALL non-None overrides (supports every AriaConfig field)
    for key, val in overrides.items():
        if val is not None:
            kwargs[key] = val

    # hierarchical requires llm policy
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
    "planning_mode": args.planning_mode,
    "reflection_mode": args.reflection_mode,
    "state_mode": args.state_mode,
    "module_policy": args.module_policy,
}

# ──────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────

run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
game_session_time = run_time
log_base_dir = os.path.join("./logs", task_id, f"aria_{args.preset}", run_time)
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

load_dotenv()
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

mland = mineland.make(
    task_id=task_id,
    agents_count=len(CHARACTERS),
    agents_config=agents_config_raw,
    world_type="amongus",
    headless=False,
    image_size=(360, 640),
    enable_auto_pause=False,
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
    if agents_yaml and i < len(agents_yaml):
        # YAML per-agent config — pass ALL fields as overrides
        agent_yaml = agents_yaml[i]
        model = agent_yaml.get("model", args.model)
        preset = agent_yaml.get("preset", args.preset)
        # Everything except meta fields becomes an AriaConfig override
        skip_keys = {"name", "role", "color", "model", "preset"}
        per_agent_overrides = {k: v for k, v in agent_yaml.items() if k not in skip_keys}
        # CLI overrides take precedence over YAML
        for k, v in cli_overrides.items():
            if v is not None:
                per_agent_overrides[k] = v
    else:
        model = args.model
        preset = args.preset
        per_agent_overrides = cli_overrides

    agent_save_path = f"./storage/{game_session_time}/{char['name']}"

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

# Agent log files
agent_log_files = {}
main_log_lock = threading.Lock()
agent_log_locks = {}
for name in agents_name[:active_agents]:
    path = os.path.join(log_base_dir, f"{name}.log")
    agent_log_files[name] = open(path, "w", buffering=1)
    agent_log_locks[name] = threading.Lock()

# Frame dirs
save_base_dir = os.path.join("./frames", task_id, f"aria_{args.preset}", run_time)
os.makedirs(save_base_dir, exist_ok=True)
agent_frame_dirs = {}
for name in agents_name[:active_agents]:
    d = os.path.join(save_base_dir, name)
    os.makedirs(d, exist_ok=True)
    agent_frame_dirs[name] = d

# State tracking
code_info = [None] * agents_count
task_info = None
done = False
event = None
agent_alive = {name: True for name in agents_name[:active_agents]}
meeting_speaker_idx = 0
MAX_MEETING_SPEAKS = 5
meeting_speak_counts = {name: 0 for name in agents_name[:active_agents]}
prev_phase = 0

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
    """Run one agent's reasoning (returns action, no env step)."""
    agent_name = agents_name[idx]
    writer = ThreadSafeWriter(agent_name, step_i)
    thread_local.stdout_writer = writer
    thread_local.stderr_writer = writer

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


# ──────────────────────────────────────────────────────────────
# Main game loop
# ──────────────────────────────────────────────────────────────

game_start_time = time.time()

try:
    for step_i in range(args.max_steps):
        if step_i > 0 and step_i % 10 == 0:
            print(f"step: {step_i}, task_info: {task_info}")

        if step_i == 0:
            actions = mineland.Action.no_op(agents_count)
            obs, code_info, event, done, task_info = mland.step(action=actions)
        else:
            # Ghost check
            for idx in range(active_agents):
                name = agents_name[idx]
                ghost_val = _get_obs_val(obs[idx], 'ghost')
                if ghost_val == 1 and agent_alive.get(name, True):
                    print(f"\n{'='*60}\n  {name} has DIED (ghost=1)\n{'='*60}\n")
                    agent_alive[name] = False

            # Check meeting talk
            all_spoke_done = all(
                meeting_speak_counts.get(agents_name[i], 0) >= MAX_MEETING_SPEAKS
                for i in range(active_agents)
                if agent_alive.get(agents_name[i], True)
            )

            if not args.parallel and _is_meeting_talk(obs) and not all_spoke_done:
                # ── SYNC meeting: one agent speaks per step ──
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

                result = run_agent_reasoning(idx, step_i, obs, code_info, done, task_info)
                action = result["action"] if result else None

                actions = mineland.Action.no_op(agents_count)
                if action is not None:
                    actions[idx] = action
                obs, code_info, event, done, task_info = mland.step(actions)

                # Check all spoke
                all_spoke_now = all(
                    meeting_speak_counts.get(agents_name[i], 0) >= MAX_MEETING_SPEAKS
                    for i in range(active_agents)
                    if agent_alive.get(agents_name[i], True)
                )
                if all_spoke_now:
                    print(f"[Meeting] All agents reached {MAX_MEETING_SPEAKS} speaks — VOTE transition")
                    for agent in agents:
                        if hasattr(agent, 'state_builder'):
                            agent.state_builder._force_all_spoke = True

                meeting_speaker_idx += 1

            else:
                # ── SYNC mode: all reasoning, then batch action ──
                print(f"[Sync] step {step_i}, reasoning for {active_agents} agents")

                # Phase 1: parallel reasoning
                actions_map = {}
                with ThreadPoolExecutor(max_workers=active_agents) as executor:
                    futures = {}
                    for idx in range(active_agents):
                        if agent_alive.get(agents_name[idx], True):
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

                # Phase 2: batch step
                batch_actions = [
                    actions_map.get(idx) or mineland.Action.no_op(1)[0]
                    for idx in range(agents_count)
                ]
                new_obs, new_ci, new_ev, new_done, new_ti = mland.step(batch_actions)

                # Phase 3: wait for code completion
                resume_tick = 0
                while True:
                    still_running = any(
                        (new_ci[idx].get("is_running", False)
                         if isinstance(new_ci[idx], dict)
                         else getattr(new_ci[idx], "is_running", False))
                        for idx in range(active_agents)
                        if agent_alive.get(agents_name[idx], True)
                    )
                    if not still_running:
                        break
                    resume_actions = [mineland.Action(type=mineland.Action.RESUME, code="")
                                      for _ in range(agents_count)]
                    new_obs, new_ci, new_ev, new_done, new_ti = mland.step(resume_actions)
                    resume_tick += 1
                    if new_done:
                        break

                print(f"[Sync] action complete after {resume_tick} resume(s)")

                for idx in range(active_agents):
                    obs[idx] = new_obs[idx]
                    code_info[idx] = new_ci[idx]
                event = new_ev
                done = new_done
                task_info = new_ti

            # ── Phase transition handling ──
            phase_detected = None
            for idx in range(active_agents):
                p = getattr(obs[idx], 'phase', None)
                if p is not None:
                    phase_detected = p
                    break

            if phase_detected == 1 and prev_phase == 0:
                print(f"\n{'='*60}\n  MEETING PHASE DETECTED\n{'='*60}\n")
                meeting_speaker_idx = 0
                meeting_speak_counts = {name: 0 for name in agents_name[:active_agents]}
                for agent in agents:
                    if hasattr(agent, 'state_builder'):
                        agent.state_builder._force_all_spoke = False

                # Consume INTERRUPT
                int_actions = mineland.Action.no_op(agents_count)
                int_obs, int_ci, int_ev, int_done, int_ti = mland.step(int_actions)
                for i in range(active_agents):
                    obs[i] = int_obs[i]
                    code_info[i] = int_ci[i]
                event = int_ev
                done = int_done
                task_info = int_ti

                # Propagate deaths to all agents' state builders
                dead_agents = [n for n, alive in agent_alive.items() if not alive]
                if dead_agents:
                    print(f"  [Meeting] Dead players: {dead_agents}")
                    for dead_name in dead_agents:
                        for agent in agents:
                            if hasattr(agent, 'state_builder'):
                                agent.state_builder._dead_players.add(dead_name)

            prev_phase = phase_detected if phase_detected is not None else prev_phase

            sync_global_mission_progress()

        # Save frames
        for idx in range(active_agents):
            name = agents_name[idx]
            try:
                img = to_hwc_rgb(obs[idx]["rgb"])
                path = os.path.join(agent_frame_dirs[name], f"frame_{step_i:03d}.png")
                Image.fromarray(img).save(path)
            except Exception:
                pass

        # Game end check
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
    # ── Cleanup ──
    elapsed = time.time() - game_start_time
    elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"

    # Winner detection
    try:
        winner = getattr(task_info, 'winner', None)
        if not winner:
            try:
                result, msg = mland.sim.server_manager.get_game_result()
                if result:
                    winner = result
            except Exception:
                pass
        if not winner:
            try:
                with open(main_log_path, 'r', errors='ignore') as f:
                    for line in f:
                        low = line.lower()
                        if '<server>' not in low and '[server]' not in low:
                            continue
                        if 'imposter win' in low or 'imposters win' in low:
                            winner = 'imposter'
                            break
                        elif 'crew win' in low or 'crewmate win' in low or 'crewmates win' in low:
                            winner = 'crewmate'
                            break
            except Exception:
                pass

        print(f"\n{'='*60}")
        if winner:
            print(f"  RESULT: {winner.upper()} WIN")
        else:
            print(f"  RESULT: unknown")
        print(f"  Steps : {step_i + 1}")
        print(f"  Time  : {elapsed_str}")
        print(f"{'='*60}\n")
    except Exception as e:
        print(f"[warn] {e}")

    # Save Aria config summary
    try:
        summary_path = os.path.join(log_base_dir, "aria_configs.txt")
        with open(summary_path, "w") as f:
            for agent in agents:
                f.write(f"{agent.config.summary()}\n")
        print(f"Aria configs saved to: {summary_path}")
    except Exception:
        pass

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
