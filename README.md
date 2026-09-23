# Lies We Can See

Code for the paper **Lies We Can See: Joint Verbal and Non-Verbal Deception by VLM Agents in Embodied Social Interactions**.

![teaser](assets/teaser.png)

**MineAmongUs** is a 3D multimodal Among Us sandbox in Minecraft where 2 imposter agents deceive 6 crewmates through both what they say and what they do: stalking a target, checking for witnesses, fleeing an unreported body, then accusing the crewmate who found it. **ARIA** is the configurable VLM-agent harness that runs them, and an **LLM-as-a-Judge** scores every deceptive act.

## Overview

| Component | What it is |
|-----------|------------|
| **MineAmongUs** | The sandbox. 8 agents (2 imposters vs 6 crewmates) share one map. A match alternates a task phase (move, kill, fake missions) and a meeting phase (chat, accuse, vote) until one side wins. |
| **ARIA** | The agents. Five ablation axes (memory, planning, reflection & skill, prompt style, state representation), each with two settings, so the same backbone can be run under many harness configurations. |
| **LLM-as-a-Judge** | The measurement. Reads a `game.log` and labels deception with a 23-atom codebook across 6 families (human-LLM Cohen's κ = 0.709). |

The 6 families are Camouflage (NV-1), Pursuit & Kill (NV-2), Report & Emergency (NV-3), Falsification (V-1), Equivocation (V-2), and Concealment (V-3). See [`judge/README.md`](judge/README.md) for the full taxonomy.

The Minecraft server, Among Us world, bot bridge, headless renderer, and Python environment are packaged as a Docker image. The image is withheld during the anonymous review period and will be released with the paper. No Minecraft account is needed.

## Running a match

Inside the Docker image (released after review):

```bash
cp scripts/.env.example scripts/.env   # add your OPENAI_API_KEY
run_2vs6.sh                            # one match, ~15 min, ~$0.25
```

Results land in `scripts/logs/sweep/<timestamp>/`: `summary.md` for the table, `game.log` for the full transcript.

## Reproducing the paper

The single match above uses one harness configuration. The paper's two studies sweep many:

- **RQ1 (harness ablation)**: one backbone, 24 imposter configurations. See `scripts/main_1_aria_2vs6.py`.
- **RQ2 (cross-VLM round-robin)**: every backbone against every other. See `scripts/main_1_aria_2vs6_multi_llm.py`.

Score the resulting logs with the judge, then read [`scripts/README.md`](scripts/README.md) for the full sweep-and-score pipeline.

## Repository layout

```
mineamongus/
├── mineland/                     # sandbox + agent harness  ......... mineland/README.md
│   ├── sim/                      #   Python bridge, server & bot managers
│   │   ├── server/               #     Fabric 1.19 world + datapacks
│   │   └── mineflayer/           #     per-agent Node.js bots + HTTP step bridge
│   ├── aria/                     #   ARIA harness
│   │   ├── modules/              #     decision modules (kill, report, surveillance, meeting, vote, move, mission)
│   │   ├── planner/              #     reactive & hierarchical planners
│   │   ├── memory/               #     window & semantic memory back-ends
│   │   ├── reflection/           #     meeting-end reflector
│   │   ├── state/                #     privileged & egocentric state builders
│   │   ├── prompt_template/      #     minimal & deterministic prompt sets
│   │   └── action/               #     action codegen
│   ├── tasks/                    #   task definitions, including the Among Us task
│   └── patches/                  #   headless-RGB patches for the renderer
├── scripts/                      # 2 match runners + 1 driver per RQ  scripts/README.md
├── judge/                        # LLM-as-a-Judge, 23-atom codebook   judge/README.md
│   ├── codebook.py               #   the 23 atoms + the prompt text built from them
│   └── judge.py                  #   score a log, or a directory of logs
├── data/                         # how to reproduce a result          data/README.md
└── docker/                       # image, entrypoint, rebuilding      docker/README.md
```

## Intended use

This repository is for research on agent deception and alignment. The deceptive behaviors it elicits are the object of study, not a capability to deploy.

## Notes

1. Run **one** sweep per container. Two concurrent runners collide on the Minecraft world and port 25565.
2. Built on MineLand; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
3. Minecraft is a trademark of Mojang Studios; this is an independent research artifact.

MIT licensed.
