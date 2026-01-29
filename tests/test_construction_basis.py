import numpy as np
import logging
from xgw.hyperplane_approx import construct_basis_eij

logger = logging.getLogger(__name__)


def test_construct_basis():
    elem = np.meshgrid(np.array([3,5,8]), np.array([2,5,6]))
    n1_arr = np.ravel(elem[0])
    n2_arr = np.ravel(elem[1])
    for n1, n2 in zip(n1_arr, n2_arr):
        lin = np.linspace(0, 2, n1)
        xx, yy = np.meshgrid(lin, lin)
        space_x = np.vstack([xx.ravel(), yy.ravel()]).T
        lin = np.linspace(-1, 1, n2)
        xx, yy = np.meshgrid(lin, lin)
        space_y = np.vstack([xx.ravel(), yy.ravel()]).T
        
        N, dx = space_x.shape
        M, dy = space_y.shape
        f_base = np.zeros((N, M, dx*dy)) # constructing the function base f_{i,j}
        for i in range(dx):
            for j in range(dy):
                f_base[:,:, i+j*dx] = np.outer(space_x[:,i], space_y[:,j])
        
        e_base, R = construct_basis_eij(space_x, space_y)
        
        assert np.all(np.isclose(e_base@R, f_base))
