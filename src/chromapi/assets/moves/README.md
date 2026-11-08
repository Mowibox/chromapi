---
license: apache-2.0
task_categories:
  - robotics
tags:
  - chromapi
pretty_name: Chromaposes
configs:
  - config_name: default
    data_files: metadata.jsonl
---

# Chromaposes - Chromapi Moves Library

Recorded moves of the quadruped robot Chromapi quadruped, 
There is two categories of poses:

- body moves (mood gestures): the chassis moves over the four planted feet and comes back;
- leg moves: the weight shifts onto three feet and a front leg moves.

Layout: one `<name>.json` per move, indexed in `metadata.jsonl`.

```json
{"description": "When to play it.",
 "time": [0.0, 0.02, ...],
 "set_target_data": [{"body": [x, y, z, roll, pitch, yaw]}, ...]}

{"description": "When to play it.",
 "default_leg": "tl",
 "time": [0.0, 0.02, ...],
 "set_target_data": [{"leg": [coxa, femur, tibia]}, ...]}
```

- `time` [s], sampled at 50 Hz; the player interpolates linearly in between.
- `body`: chassis pose relative to the start pose, zero at the first and last sample.
  Feet frame: +X forward, +Y left, +Z up [m]; roll > 0 right, pitch > 0 nose down, yaw > 0 left [rad].
- `leg`: offsets [rad] of the front leg's joints from where it is when the leg move starts (after the weight
  shift), zero at the first and last sample. The same values play on `tl` or `tr` (mirrored joints).
- Optional `mood`: the ring animation and sound played with the move (a Chromapi mood name; default `happy`).

## Usage

Read by the Chromapi SDK 1.0.0 and later (`pip install 'chromapi[hub]'`):

```python
from chromapi.locomotion.recorded_move import RecordedMoves
moves = RecordedMoves("Mowibox/chromaposes")  # downloaded once, then read from the cache, offline too
```
