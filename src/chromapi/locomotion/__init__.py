"""Locomotion layer for Chromapi - gait generation on top of :mod:`chromapi.kinematics`.

- :mod:`chromapi.locomotion.gait` - :class:`~chromapi.locomotion.gait.GaitParams`,
  :class:`~chromapi.locomotion.gait.GaitEngine` and
  :class:`~chromapi.locomotion.gait.WalkMove` - open-loop quasi-static walking/turning, built on
  a single unified body twist (see that module's docstring). No balance/stability control yet
  (planned as a follow-up on top of this - see ``chromapi_locomotion_theorie.md``'s Module 5).
"""
