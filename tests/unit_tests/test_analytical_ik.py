"""Closed-form leg IK (from chromapi.kinematics.analytical_ik) against the URDF model."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from chromapi.kinematics.analytical_ik import AnalyticalIK
from chromapi.kinematics.kromatics import Kromatics
from chromapi.locomotion.gait import WALK_POSTURE, GaitEngine, GaitParams

URDF_PATH = Path(__file__).resolve().parents[2] / "src" / "chromapi" / "model" / "urdf" / "robot.urdf"


@pytest.fixture(scope="module")
def kinematics() -> Kromatics:
    """Kromatics model loaded from the package URDF."""
    return Kromatics(URDF_PATH, dt=0.02)


@pytest.fixture(scope="module")
def analytical(kinematics: Kromatics) -> AnalyticalIK:
    """AnalyticalIK built from that model (its report must pass)."""
    ik, report = AnalyticalIK.from_kromatics(kinematics)
    assert report.ok(), report.summary()
    return ik


def test_model_matches_urdf_forward_kinematics(kinematics: Kromatics) -> None:
    """The closed-form FK agrees with the URDF at numerical precision."""
    _, report = AnalyticalIK.from_kromatics(kinematics, n_samples=200, seed=7)
    assert max(report.max_error_m.values()) < 1e-6, report.summary()


def test_inverse_recovers_joint_angles_within_limits(analytical: AnalyticalIK) -> None:
    """FK -> IK round trip recovers the joint angles anywhere within the limits (no margin)."""
    rng = np.random.default_rng(0)
    for chain in analytical.chains.values():
        chain = dataclasses.replace(chain, workspace_margin=1.0)
        for _ in range(200):
            q = rng.uniform(chain.lower, chain.upper)
            solution = chain.inverse(chain.forward(*q), reference=tuple(q))
            assert solution is not None
            np.testing.assert_allclose(solution, q, atol=1e-6)


def test_walk_posture_matches_qp(kinematics: Kromatics, analytical: AnalyticalIK) -> None:
    """The walking stance is solved exactly (checked through the URDF FK)."""
    x_reach, y_reach, height = WALK_POSTURE
    targets = kinematics.stance_targets(height, x_reach, y_reach)
    q, converged = analytical.inverse_kinematics(targets)
    assert converged
    feet = kinematics.forward_kinematics(q)
    for leg, target in targets.items():
        np.testing.assert_allclose(feet[leg], target, atol=1e-6)


def test_near_full_extension_is_refused(analytical: AnalyticalIK) -> None:
    """A straight leg (femur and tibia aligned) is beyond WORKSPACE_MARGIN and refused."""
    for chain in analytical.chains.values():
        knee = np.linspace(chain.lower[2], chain.upper[2], 721)
        feet = [chain.forward(0.0, 0.0, q3) for q3 in knee]
        straight = max(feet, key=lambda foot: float(np.linalg.norm(foot - chain.points[1])))
        assert chain.inverse(straight) is None
        assert dataclasses.replace(chain, workspace_margin=1.0).inverse(straight) is not None


def test_unreachable_target_returns_none(kinematics: Kromatics, analytical: AnalyticalIK) -> None:
    """An out-of-reach target is reported as None, not a wrong solution."""
    far = kinematics.stance_targets(0.12, 0.11, 0.09)["tl"] + np.array([1.0, 0.0, 0.0])
    assert analytical.inverse_kinematics_leg("tl", far) is None


def test_gait_engine_keeps_analytical_solver(kinematics: Kromatics) -> None:
    """GaitEngine no longer disables the analytical solver on its first use."""
    engine = GaitEngine(kinematics, GaitParams(solver="analytical"))
    engine.set_velocity(0.0, -0.05, 0.0)
    for _ in range(50):
        engine.step(0.02)
    assert engine._analytical is not None
    assert not engine._analytical_disabled
