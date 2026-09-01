# 🎬 `scripts/`: match runners and experiment drivers

Four files: two match runners, and one driver per research question.

| | |
|---|---|
| `amongus_2_imposter_6_crewmates_Aria.py` | Plays **one** 2 vs 6 match: launches the server, spawns 8 agents, runs to a win condition, writes `game.log`. This is the configuration used throughout the paper. |
| `amongus_1_imposter_3_crewmates_Aria.py` | The same, at 1 vs 3. Cheaper for smoke-testing changes. |
| `main_1_aria_2vs6.py` | **RQ1** driver: harness ablation, 24 imposter configurations under a fixed backbone |
| `main_1_aria_2vs6_multi_llm.py` | **RQ2** driver: cross-VLM round-robin, every backbone against every other |

Both drivers shell out to the 2 vs 6 match runner, once per game.

---

## RQ1: harness ablation

```bash
run_2vs6.sh --config-ids 8-23 --runs-per-config 3 --model gpt-4.1-mini \
            --out-dir scripts/logs/sweep/rq1_gpt
```

`run_2vs6.sh` (on `PATH` in the image) wraps the driver; calling it directly is equivalent:

```bash
python scripts/main_1_aria_2vs6.py --config-ids 8,9,10 --runs-per-config 3 \
       --model gpt-4.1-mini --out-dir scripts/logs/sweep/rq1_gpt
```

| Flag | Meaning |
|---|---|
| `--config-ids` | Configurations to run. The wrapper accepts ranges (`8-23`); the driver takes a comma list. |
| `--runs-per-config` | Repetitions per configuration (default 3) |
| `--model` | Backbone for **all** agents (self-play) |
| `--imposter-model` | Overrides the backbone for the imposters only |
| `--server-id` | Server slice for parallel runs; also selects an isolated world dir and port |
| `--start-run`, `--start-config-id`, `--end-config-id` | Resume a partially completed sweep |
| `--analyze-only` | Re-aggregate an existing `--out-dir` without playing |

### The 24-configuration grid

Crossing the four cognitive axes gives **3 (memory) × 2 (reflection & skill) × 2 (planning) × 2 (prompt) = 24** imposter configurations, indexed in that nesting order:

| Ids | Memory | |
|---|---|---|
| 0–7 | `off` | degenerate control, excluded from the paper |
| 8–15 | `win` | window memory |
| 16–23 | `sem-llm` | semantic memory with LLM belief update |

Within each block of eight: reflection & skill (`no-rs` → `meet-rs`), then planning (`react` → `hier`), then prompt (`min` → `det`). So id 8 resolves to `win_no-rs_react_min`, which is also the directory it writes to.

The fifth axis, **state representation**, is set in `configs/*.yaml` rather than by id. All cognitive-component sweeps use `privileged`; `egocentric` is reported only as the collapse condition.

Crewmate configurations are fixed within a sweep. The paper contrasts two: `(semantic, reactive)` and `(window, hierarchical)`, both with `meeting-on` reflection and `minimal` prompts.

Analysis is a cluster paired bootstrap over matched configuration pairs, with pre-registered one-sided directions (window > semantic, reflection-on > off, hierarchical > reactive; prompt has no a priori direction).

---

## RQ2: cross-VLM round-robin

```bash
python scripts/main_1_aria_2vs6_multi_llm.py --case 1A --server-id 0 --num-servers 4
```

Enumerates every (imposter backbone, crewmate backbone) pair from the 12-model pool: both imposter slots get the same imposter backbone, all six crewmate slots the same crewmate backbone. With 12 models that is 144 matchups per case; the paper runs 4 cases × 2 repetitions = **1,152 matches**.

| Case | Crewmate configuration | Imposter configuration |
|---|---|---|
| `1A` | semantic + LLM, reactive | symmetric, same as crewmate |
| `1B` | semantic + LLM, reactive | window, reactive (the RQ1-best imposter) |
| `2A` | window, hierarchical | symmetric |
| `2B` | window, hierarchical | window, reactive |

Matchup index `k` is deterministic (`k = i * N + j` over the pool), so the same index in different cases uses identical backbones in identical slots, giving a clean paired comparison. Splitting by `k % num_servers == server_id` makes work allocation reproducible across machines.

Per-model win rates are compared with Pearson correlation (Fisher *z* confidence intervals).

---

## Output layout

```
<out-dir>/
├── summary.md                       # per-config table: win rate, kills, meetings, vote accuracy, cost
├── all_runs.csv                     # one row per match
├── per_config_summary.csv           # aggregated per configuration
└── cfgNN_<mem>_<refl>_<plan>_<prompt>/
    └── runNN/
        └── game.log                 # full transcript: utterances, actions, votes, events
```

`game.log` is what the [judge pipeline](../judge/README.md) consumes.

> Run **one** driver per container. Two concurrent drivers collide on the Minecraft world and port 25565 and die silently with exit `-9`. For parallelism, give each a distinct `--server-id`, or use separate containers.

## Supporting files

| | |
|---|---|
| `configs/` | Per-experiment YAML: per-agent backbones, state mode, temperature overrides |
| `personal_messages/` | Role-specific prompt injections, keyed by task id and agent name |
| `.env.example` | Template for `OPENAI_API_KEY` / `OPENROUTER_API_KEY`. Copy it to `.env`, which is read by `python-dotenv` |
