# 📦 `data/`

This repository ships code only. The game logs, judge outputs and human annotations are on the Hugging Face Hub.

> **Dataset:** to be added, released alongside the paper.

## Reproducing a result

```bash
# 1. play matches
run_2vs6.sh --config-ids 8-23 --runs-per-config 3 --model gpt-4.1-mini \
            --out-dir scripts/logs/sweep/mine

# 2. score the deception atoms
python judge/judge.py --sweep-dir scripts/logs/sweep/mine --two-pass --model gpt-5.4
```

Step 1 writes `summary.md` and `all_runs.csv` next to the per-match `game.log`s; step 2 writes one `findings.json` per match.

Win rates will not match the paper exactly, because the backbones are served APIs and are not deterministic even at fixed temperature. The paper's claims are cluster paired bootstraps over 192 matches, not single-match comparisons; see [`../scripts/README.md`](../scripts/README.md).
