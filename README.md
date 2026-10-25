# Chromapi

Chromapi is an open-source 12-DoF quadruped robot that I design and build on my own.

![Chromapi](https://custom-icon-badges.demolab.com/badge/Chromapi-software-a957f9?logo=chromapi)
![Discord](https://img.shields.io/discord/1383790297594728518?color=7289DA&logo=discord&logoColor=white&label=Discord)
![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python)
[![Software: Apache 2.0](https://img.shields.io/badge/Software-Apache%202.0-lightgrey.svg)](https://www.apache.org/licenses/LICENSE-2.0)
![CI: Linters](https://github.com/Mowibox/chromapi/actions/workflows/lint.yml/badge.svg)
![CI: Unit Tests](https://github.com/Mowibox/chromapi/actions/workflows/pytest.yml/badge.svg)
![Issues](https://img.shields.io/github/issues/Mowibox/chromapi)
![Stars](https://img.shields.io/github/stars/Mowibox/chromapi?style=social)

<p align="center">
  <img alt="chromapi" src="./assets/chromapi.gif"/>
</p>

## Getting started

<p align="center">
  <img alt="chromapi" src="./assets/chromapi_logo.png" width="300">
</p>

To build the robot, start with the [Bill of Materials](https://mowibox.github.io/chroma/en/chromapi/bom/) and the [specifications](https://mowibox.github.io/chroma/en/chromapi/specifications/) on the Chroma website.

### Installation

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/) (or pip).

```bash
git clone https://github.com/Mowibox/chromapi
cd chromapi
uv sync --all-extras    # or: pip install -e ".[all]"
```

The `audio` extra enables sounds on the robot and the `sim` extra installs MuJoCo for simulation.

### In simulation

```python
from chromapi.chromapi import Chromapi

with Chromapi(backend="mujoco", launch_viewer=True) as robot:
    robot.wake_up()
    robot.walk_distance(0.3)  # [m]
    robot.rest()
```

### On the real robot

On the Raspberry Pi, once the [motherboard firmware](https://github.com/Mowibox/chromapi_motherboard) is flashed:

```bash
chromapi-setup-motors     # interactive: set each servo's ID and configuration, one servo at a time
chromapi-scan-motors      # check that all 12 servos answer on the bus
chromapi-diagnostics      # check the bridge, power monitor, IMU, audio and servos
```

The same code then runs on the robot with `Chromapi(backend="hardware")`, which talks to the motherboard over UART communication.

## Project overview

| Resource | Description | Location | Status |
| :--- | :--- | :--- | :--- |
| **📋 Bill of Materials** | Complete hardware, electronics, and 3D printed parts list to build Chromapi from scratch. | [Chroma Website](https://mowibox.github.io/chroma/en/chromapi/bom/) | **Available** |
| **💻 Software API** | Python control framework, hardware interfaces, and kinematic solvers. | [`/src/chromapi`](./src/chromapi/) | *Work in Progress* |
| **📐 CAD Model** | 3D Mechanical design and robot assembly workspace. | [Onshape Document](https://cad.onshape.com/documents/3b2f6609101a115f427bb3a2/w/dda5699ddfe216cfbe74c22b/e/50bb405a7750d04ab1c412db?renderMode=0&uiState=6a7266a5caff22fc791d5f8b) | **Available** |
| **📖 Documentation** | Specifications, design notes and communication protocol. | [Chroma Website](https://mowibox.github.io/chroma/en/chromapi/) | **Available** |
| **🛠️ Hardware Utilities** | Full system diagnostics and motor configuration tools. | [`/src/chromapi/utils`](./src/chromapi/utils) | **Available** |
| **🤖 Simulation Models** | URDF and MJCF models for MuJoCo simulation. | [`/src/chromapi/model`](./src/chromapi/model) | **Available** |
| **⚡ Electronics** | Custom PCB files, schematics, and low-level firmware. | [`chromapi_motherboard` repository](https://github.com/Mowibox/chromapi_motherboard) | **Available** |

## Authors

| |
| :---: |
| <img src="https://github.com/Mowibox.png" width="100"> |
| [**Ousmane THIONGANE**](https://mowibox.github.io) |

## Sponsors & Acknowledgements

This project has been made possible thanks to the sponsors listed below. Their contributions in manufacturing, assembly, and prototyping have greatly accelerated the development of Chromapi.

<div align="center">

| | |
| :---: | :---: |
| <a href="https://www.pcbway.com"><img src="./assets/pcbway.jpg" width="150" alt="PCBWay"></a> | <a href="https://www.ensea.fr"><img src="./assets/ensea.jpg" width="150" alt="ENSEA"></a> |
| [**PCBWay**](https://www.pcbway.com) | [**ENSEA**](https://www.ensea.fr) |
| PCB Manufacturing & Assembly | 3D Printing & Prototyping |

</div>

---

> Huge thanks to everyone who gave me feedback on the Chromapi's design, shared ideas, or simply took the time to brainstorm with me. This project wouldn't be where it is today without your help!

## Contributions

Contributions are always welcome!

* **Report Issues:** Found an error or have a feature request? Create a new issue [here.](https://github.com/Mowibox/chromapi/issues/new/choose)
* **Fix Bugs & Add Features:** Find out where you can lend a hand by checking out [existing issues.](https://github.com/Mowibox/chromapi/issues)

Before opening a pull request, don't forget to run the linters and unit tests: `uv run ruff check . && uv run mypy && uv run pytest`.

_By submitting a contribution, you agree that it is published under the license of the part it belongs to._

## License

This repository is multi-licensed:

| Part | Path | License |
| :--- | :--- | :--- |
| Software (Python SDK, tools, tests, scripts) | [`/src`](./src/), [`/tests`](./tests/), [`/scripts`](./scripts/) | [Apache 2.0](./LICENSE) |
| Robot description (URDF, MJCF, configuration files) | [`/src/chromapi/model`](./src/chromapi/model/) | [Apache 2.0](./LICENSE) |
| 3D meshes (`.stl`, `.part`) | [`/src/chromapi/model/*/assets`](./src/chromapi/model/) | [CC BY-NC-SA 4.0](./src/chromapi/model/LICENSE) |
| Documentation and media (README, images, GIFs) | [`/assets`](./assets/), this README | [CC BY-NC-SA 4.0](./src/chromapi/model/LICENSE) |

**Third-party content keeps its own license:** meshes of off-the-shelf parts (Raspberry Pi 4, IMX219 camera, Noctua NF-A4x10 fan, WS2812B ring, 18650 battery, speaker, microswitch, power switch, connectors, screws) come from their respective manufacturers. The ENSEA and PCBWay logos in [`/assets`](./assets/) are trademarks of their respective owners.
