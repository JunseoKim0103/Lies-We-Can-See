# 🧠 `mineland/`: sandbox and agent harness

## MINEAMONGUS (`sim/`, `tasks/`)

A Fabric Minecraft server (1.19) is launched fresh for every match. Eight agents (2 imposters, 6 crewmates) share one contiguous world: labeled rooms joined by hallways, 20 task stations, and a central chamber where meetings are held.

All Among Us mechanics are enforced **server-side** by datapacks (`sim/server/`), not by the agent loop: role assignment, kill cooldown, body spawning, meeting triggers, vote tallying, win conditions. Agents cannot reach around the rules.

| | |
|---|---|
| `sim/sim.py`, `sim/bridge.py` | Gym-style step loop; one HTTP round-trip per step to the bot bridge |
| `sim/server_manager.py` | Per-match server instance; the `world_type` template is copied into an isolated world dir |
| `sim/mineflayer/` | Node.js bridge: one mineflayer bot per agent, plus the headless renderer |
| `tasks/amongus_task.py` | Task definition, mission pool, win conditions |
| `patches/` | Headless-RGB patches for the vendored renderer; `apply_patches.sh` copies them into `node_modules` after `npm ci` |

The world used by the 2 vs 6 experiments is `amongus_2im`; other templates in `sim/server/` are alternate layouts. `tasks/` holds only the Among Us task registry: the upstream MineLand task families (survival, harvest, combat and so on) have been removed.

> `sim/server/` and `sim/mineflayer/node_modules/` are large binaries and are **not tracked in git**. They ship inside the Docker image.

## ARIA (`aria/`)

ARIA is built as a controlled instrument rather than a maximally capable agent. Each cognitive component is an independent ablation axis, switchable from the command line.

| Axis | Values | Flag |
|---|---|---|
| State representation | `privileged` · `egocentric` | `--state-mode` |
| Memory | `window` · `semantic` (+ LLM belief update) · `none` | `--memory-mode`, `--memory-belief-update` |
| Planning | `reactive` · `hierarchical` | `--planning-mode` |
| Reflection & skill memory | off · `meeting` reflection + skill memory | `--reflection-mode`, `--skill-memory` |
| Prompt style | `minimal` · `deterministic` | `--prompt-style` |

A per-player suspicion belief runs across both state modes. State representation is held at `privileged` for every cognitive-component sweep: egocentric imposters cannot find targets, so no kills, bodies, meetings, or verbal deception occur at all.

| | |
|---|---|
| `aria_agent.py` | Per-agent decision cycle: observe → memory → plan → act → reflect |
| `config.py` | Axis definitions and the presets (`baseline`, `full`, `steve_equivalent`) |
| `llm.py` | Backbone routing and `MODEL_CONFIGS`: provider, temperature, vision support |
| `env_adapter.py` | Observation construction (RGB frames + structured state) and action dispatch |
| `modules/` | Decision modules: kill, report, surveillance, emergency, meeting, vote, move, mission |
| `planner/` | Reactive and hierarchical planners |
| `memory/` | Window and semantic memory back-ends |
| `reflection/` | Meeting-end reflector |
| `state/` | Privileged and egocentric state builders |
| `prompt_template/` | Prompt sets for the `minimal` and `deterministic` styles |
| `action/` | Action codegen |

### Adding a backbone

Add an entry to `MODEL_CONFIGS` in `aria/llm.py`:

```python
"my-model": ModelConfig("vendor/my-model", "openrouter", True, 0.7),
#            └ API model id             └ provider    └ vision  └ temperature
```

Then pass `--model my-model` (or `--imposter-model my-model`). Self-hosted endpoints are reached with `api_base_env="MY_MODEL_API_BASE"`, read from `scripts/.env`.
