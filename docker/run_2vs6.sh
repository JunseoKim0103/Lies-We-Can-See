#!/bin/bash
# Run the 2-imposter vs 6-crewmate experiment.
#
#   run_2vs6.sh [--config-ids 8] [--runs-per-config 1] [--model gpt-4.1-mini] [--out-dir DIR]
#
# With no arguments: one match of configuration 8 (a smoke test).
set -e
cd /root/MineLand

if [ ! -f scripts/.env ]; then
    echo "ERROR: scripts/.env is missing. Copy scripts/.env.example and fill in your API key." >&2
    exit 1
fi

CONFIG_IDS=8
RUNS=1
MODEL=gpt-4.1-mini
OUTDIR=""
EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --config-ids)       CONFIG_IDS="$2"; shift 2 ;;
        --runs-per-config)  RUNS="$2";       shift 2 ;;
        --model)            MODEL="$2";      shift 2 ;;
        --out-dir)          OUTDIR="$2";     shift 2 ;;
        *)                  EXTRA+=("$1");   shift ;;
    esac
done
[ -n "$OUTDIR" ] || OUTDIR="scripts/logs/sweep/$(date +%Y%m%d_%H%M%S)"

# The driver only accepts a comma list, so expand ranges like "8-15" here.
expand_ids() {
    local out=""
    local IFS=','
    for part in $1; do
        case "$part" in
            *-*)
                local lo=${part%%-*} hi=${part##*-}
                for ((i = lo; i <= hi; i++)); do out="$out,$i"; done
                ;;
            "") ;;
            *)  out="$out,$part" ;;
        esac
    done
    echo "${out#,}"
}
CONFIG_IDS=$(expand_ids "$CONFIG_IDS")

mkdir -p "$OUTDIR"
echo "[run] configs=$CONFIG_IDS runs=$RUNS model=$MODEL out=$OUTDIR"
python scripts/main_1_aria_2vs6.py \
    --config-ids "$CONFIG_IDS" \
    --runs-per-config "$RUNS" \
    --model "$MODEL" \
    --out-dir "$OUTDIR" "${EXTRA[@]}"

echo "[done] results: $OUTDIR/summary.md , $OUTDIR/all_runs.csv"
