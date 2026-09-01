# 🐳 `docker/` — the environment

## What's in the image

`ghcr.io/junseokim0103/mineamongus:paper` — `linux/amd64`, 7.8 GB unpacked (2.5 GB download).

| | |
|---|---|
| Fabric Minecraft server 1.19 + the Among Us world and datapacks | `mineland/sim/server/` |
| Node 18 (nvm) + mineflayer bot bridge and patched headless renderer | `mineland/sim/mineflayer/` |
| Xvfb, OpenJDK 17 | system |
| conda environment `mineland` (Python 3.11) | `/root/anaconda3/envs/mineland` |
| The code in this repository | `/root/MineLand` |

No Minecraft account is needed — the server runs with `online-mode=false`.

## Files here

| | |
|---|---|
| `entrypoint.sh` | Starts `Xvfb :1` and puts the conda environment and Node 18 on `PATH`. Runs automatically. |
| `run_2vs6.sh` | Convenience wrapper for the sweep driver; on `PATH` inside the container. |

## Running your own code

The repository holds the sources; the image holds the sources *plus* the binaries. Rebuild a thin layer with [`../Dockerfile`](../Dockerfile):

```bash
docker build -t mineamongus:dev .
docker run -it --shm-size=4g mineamongus:dev
```

`COPY` merges directories, so `mineland/sim/server/` and `mineland/sim/mineflayer/node_modules/` — which are not tracked in git — survive from the base image.

For a faster loop, bind-mount instead of rebuilding:

```bash
docker run -it --shm-size=4g \
  -v "$(pwd)/scripts:/root/MineLand/scripts" \
  -v "$(pwd)/mineland/aria:/root/MineLand/mineland/aria" \
  ghcr.io/junseokim0103/mineamongus:paper
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `getUniformLocation ... null` | `DISPLAY` is unset. Run `export DISPLAY=:1` — the entrypoint sets it, but `docker exec` sessions do not inherit it. |
| Renderer fails with a shared-memory error | The container was started without `--shm-size=4g`. |
| `MESA: ZINK: failed to choose pdev` | Harmless warning; the match still runs. |
| Driver exits with code `-9`, no traceback | Usually a second driver in the same container (world and port 25565 conflict), or `node` missing from `PATH`. |
| `FileNotFoundError: 'node'` | `export PATH=/root/.nvm/versions/node/v18.18.2/bin:$PATH` |
| API key not picked up | Keys load via `python-dotenv` from `scripts/.env`. Check the file exists and is not still named `.env.example`. |

## Note on the image layout

The image was produced by flattening a container, so it is a **single layer**. Pulling it is one 2.5 GB blob with no incremental resume — which is exactly why code changes should go through the thin `Dockerfile` above rather than a rebuild of the base.
