import numpy as np
import logging
import pytest

from xgw.qp_incremental_projector import IncrementalQPProjector

logger = logging.getLogger(__name__)


@pytest.fixture
def eps_abs():
    return 1e-5


@pytest.fixture
def eps_rel():
    return 1e-5


@pytest.fixture
def max_iter():
    return 10_000


@pytest.fixture
def nearly_parallel_halfspaces():
    eps = 1e-8
    A0 = np.array([
        [1, 1],
        [1, 1 + eps]
    ])
    b0 = np.array([1, 1])
    v = np.array([10, 10])
    return A0, b0, v


@pytest.fixture
def box():
    A0 = np.array([[-1, 0],
                [0, -1]])
    b0 = np.array([0.0, 0.0])
    v = np.array([2, 3])
    return A0, b0, v


def test_incremental_qp_projector(eps_abs, eps_rel, max_iter, box):

    A0, b0, v = box

    proj = IncrementalQPProjector(dim=A0.shape[1], A=A0, b=b0, eps_abs=eps_abs, eps_rel=eps_rel, max_iter=max_iter)

    # first solve toward v
    x1, obj1 = proj.solve(v)
    assert np.allclose(x1, v)
    logger.info(f"Initial projection: {x1}, objective value: {obj1}")

    # Add a new constraint x + y <= 3
    a_new = np.array([1.0, 1.0])
    b_new = x1.sum() - 1.0

    x2, obj2 = proj.solve_with_new_constraint(v, a_new, b_new)
    assert np.allclose(x2, np.array([1.5, 2.5]))
    logger.info(f"After adding half-plane: {x2}, objective value: {obj2}")