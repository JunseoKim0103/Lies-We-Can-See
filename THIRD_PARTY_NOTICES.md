# Third-party notices

This work is MIT licensed (see [LICENSE](LICENSE)). The components below are not
ours and carry their own terms.

## In this repository

| Component | License | Notes |
|---|---|---|
| [MineLand](https://github.com/cocacola-lab/MineLand) | MIT, © 2024 CoLa Lab | This repository is a derivative work. `mineland/sim/` and the `steve/` and `alex/` agents originate upstream. |
| [prismarine-viewer](https://github.com/PrismarineJS/prismarine-viewer) | MIT | Vendored as `prismarine-viewer-colalab`; the headless-RGB modifications in `mineland/patches/` are applied on top. |
| [mineflayer](https://github.com/PrismarineJS/mineflayer) | MIT | The bot bridge in `mineland/sim/mineflayer/`. |

## Required at runtime, not redistributed by this repository

The sandbox runs on a Fabric-modded Minecraft server. These are **not** part of
this repository and must be obtained by the user:

| Component | License | Notes |
|---|---|---|
| Minecraft: Java Edition server 1.19 | [Minecraft EULA](https://aka.ms/MinecraftEULA), © Mojang Studios | Proprietary. Not redistributable. Downloaded from Mojang at first run; you must accept the EULA yourself. |
| [Fabric Loader / Fabric API](https://fabricmc.net/) | Apache-2.0 | |
| [ReplayMod](https://www.replaymod.com/) | GPL-3.0 | Used for frame capture. |
| Multiworld, Multiplayer-Server-Pause, better-respawn, runtick, completeconfig, ModMenu | see each project | Fabric mods used for world switching, deterministic ticking and respawn control. |

Minecraft is a trademark of Mojang Studios. This project is not affiliated with,
endorsed by, or sponsored by Mojang Studios or Microsoft.

## Models and APIs

Backbones are accessed as hosted APIs (OpenAI, OpenRouter). Their terms of
service apply to your use; no model weights are distributed here.
