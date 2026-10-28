# SPDX-License-Identifier: Apache-2.0
"""Kinematics layer for Chromapi.

- :mod:`chromapi.kinematics.types` - shared array/dict type aliases.
- :mod:`chromapi.kinematics.topology` - leg/joint naming (no numbers).
- :mod:`chromapi.kinematics.poses` - named joint configurations (rest/approach/wake-up/zero).
- :mod:`chromapi.kinematics.kromatics` - :class:`~chromapi.kinematics.kromatics.Kromatics`, the
  whole-body kinematic solver (a single QP over all 12 joints, via Rhoban's ``placo``/
  Pinocchio) used by :class:`chromapi.chromapi.Chromapi` and :mod:`chromapi.locomotion.gait`.
"""
