import numpy as np
import logging

from xgw.qp_incremental_projector import IncrementalQPProjector

import logging
logger = logging.getLogger(__name__)

def test_incremental_qp_projector():
    # dimension
    n = 2

    # initial constraints: x >= 0, y >= 0
    A0 = np.array([[-1, 0],
                [0, -1]])
    b0 = np.array([0.0, 0.0])

    proj = IncrementalQPProjector(dim=n, A=A0, b=b0)

    # first solve toward v
    v = np.array([2.0, 3.0])
    x1, _ = proj.solve(v)
    assert np.allclose(x1, v)
    logger.info("Initial projection: %s", x1)

    # Add a new constraint x + y <= 3
    a_new = np.array([1.0, 1.0])
    b_new = x1.sum() - 1.0

    x2, _ = proj.solve_with_new_constraint(v, a_new, b_new)
    assert np.allclose(x2, np.array([1.5, 2.5]))
    logger.info("After adding half-plane: {}".format(x2))