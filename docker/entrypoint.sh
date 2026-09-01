#!/bin/bash
# Container entrypoint.
#   - starts Xvfb on :1 (the headless renderer crashes without a display)
#   - puts the conda environment and Node 18 at the front of PATH
set -e

export DISPLAY=${DISPLAY:-:1}
export PATH=/root/anaconda3/envs/mineland/bin:/root/.nvm/versions/node/v18.18.2/bin:$PATH

if ! pgrep -x Xvfb >/dev/null 2>&1; then
    Xvfb "$DISPLAY" -screen 0 1024x768x24 </dev/null >/var/log/xvfb.log 2>&1 &
    for _ in $(seq 1 20); do
        pgrep -x Xvfb >/dev/null 2>&1 && break
        sleep 0.5
    done
fi

if [ ! -f /root/MineLand/scripts/.env ]; then
    echo "[warn] /root/MineLand/scripts/.env is missing. Copy .env.example and fill in your API key." >&2
fi

exec "$@"
