"""Kinematics layer for Chromapi.

- :mod:`chromapi.kinematics.types` - shared array/dict type aliases.
- :mod:`chromapi.kinematics.topology` - leg/joint naming (no numbers).
- :mod:`chromapi.kinematics.poses` - named joint configurations (rest/approach/wake-up/zero).
- :mod:`chromapi.kinematics.kromatics` - :class:`~chromapi.kinematics.kromatics.Kromatics`, the
  whole-body kinematic solver (a single QP over all 12 joints, via Rhoban's ``placo``/
  Pinocchio) used by :class:`chromapi.chromapi.Chromapi` and :mod:`chromapi.locomotion.gait`.
- :mod:`chromapi.kinematics.analytical_ik` - :class:`~chromapi.kinematics.analytical_ik.AnalyticalIK`,
  the closed-form per-leg alternative/fallback to :class:`~chromapi.kinematics.kromatics.Kromatics`'s
  QP (see that module's docstring for when to reach for it instead).
"""
