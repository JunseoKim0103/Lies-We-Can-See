# Overhead camera patches

These files patch `prismarine-viewer-colalab` inside `node_modules` so that a
third-person / overhead camera also works in headless mode.

## Why they are needed

By default prismarine-viewer does not draw the bot's own mesh when it renders
headlessly, because `worldView.js` drops `bot.entity` from the entity list.

The patch adds botMesh creation to `freecamera-base64.js`, which gives us:

- an overhead view in which **every player is visible, including the bot itself**
- a **third-person** camera viewpoint
- server-side operation with no browser involved

## Patched files

| File | Location | Change |
|------|----------|--------|
| `freecamera-base64.js` | `lib/` | adds the `showSelfMesh` option, `Entity` mesh creation/update logic and an `enableSelfMesh()` API, and moves `worldView.updatePosition` onto the camera position |
| `headless-base64.js` | `lib/` | returns a `{ viewer, worldView, views }` handle (a bare `return` becomes `return { ... }`) |
| `Entity.js` | `viewer/lib/entity/` | armor_stand head colour mapping: fixes the equipment slot index (4 to 5, for the 1.19 protocol) |
| `entities.js` | `viewer/lib/` | rebuilds the mesh when the `equipmentChanged` flag arrives, so armor_stand corpse colours update live |
| `worldView.js` | `viewer/lib/` | adds an `entityEquip` event listener so equipment changes reach the viewer |

## Applying them

```bash
# run after npm ci
bash patches/apply_patches.sh
```

Or by hand:

```bash
cp patches/freecamera-base64.js mineland/sim/mineflayer/node_modules/prismarine-viewer-colalab/lib/
cp patches/headless-base64.js mineland/sim/mineflayer/node_modules/prismarine-viewer-colalab/lib/
cp patches/Entity.js mineland/sim/mineflayer/node_modules/prismarine-viewer-colalab/viewer/lib/entity/
cp patches/entities.js mineland/sim/mineflayer/node_modules/prismarine-viewer-colalab/viewer/lib/
cp patches/worldView.js mineland/sim/mineflayer/node_modules/prismarine-viewer-colalab/viewer/lib/
```

## Usage (Python)

```python
# create an overhead camera (the bot's own mesh included)
mland.bridge.addOverheadCamera("overhead", bot_name="James", width=512, height=512)

# place the camera (looking straight down: pitch = -pi/2)
import math
mland.bridge.updateCameraLocation("overhead", [x, y + 30, z], yaw=0, pitch=-math.pi/2)

# capture a frame
b64 = mland.bridge.getCameraView("overhead")

# turn the self mesh on for an existing camera
mland.bridge.enableSelfMeshOnCamera("overhead")
```

## The existing camera API is unchanged

The usual `addCamera`, `getCameraView`, `updateCameraLocation` workflow still
behaves exactly as before. `addOverheadCamera` is a convenience wrapper that
simply defaults `showSelfMesh` to `true`.
