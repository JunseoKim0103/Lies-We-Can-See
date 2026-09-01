# Match configurations

Three examples, one per way of running a match by hand. Sweeps do not need these
files: the RQ1 driver sets the axes from the command line, and the RQ2 driver
writes an `agents.yaml` per game.

| | |
|---|---|
| `rq1_2v6.yaml` | 2 imposters vs 6 crewmates, one backbone for every agent (self-play) |
| `rq2_2v6_cross_model.yaml` | The same roster with the imposters on a different backbone from the crewmates |
| `1v3.yaml` | 1 imposter vs 3 crewmates, cheap and quick, for smoke-testing a change |

```bash
python scripts/amongus_2_imposter_6_crewmates_Aria.py --config scripts/configs/rq1_2v6.yaml
python scripts/amongus_1_imposter_3_crewmates_Aria.py --config scripts/configs/1v3.yaml
```

## Format

```yaml
task_id: amongus_2_imposter_6_crewmates   # must match a directory in ../personal_messages/

agents:
  - name: James            # must match a .txt in ../personal_messages/<task_id>/
    role: imposter         # imposter | crewmate
    color: Red             # in-game body colour
    model: gpt-4.1-mini    # a key in mineland/aria/llm.py MODEL_CONFIGS
    preset: full           # optional: baseline | full | steve_equivalent
    temperature: 0.3       # optional
```

Any key beyond `name`, `role`, `color`, `model` and `preset` is passed to the
agent as a per-agent override, so ARIA axes can be set here as well:
`memory_mode`, `planning_mode`, `reflection_mode`, `prompt_style`, `state_mode`.

Agent counts and names must match the `task_id`: eight agents for
`amongus_2_imposter_6_crewmates`, four for `amongus_1_imposter_3_crewmates`.
