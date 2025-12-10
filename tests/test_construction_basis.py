import numpy as np
import logging
from xgw.Hyperplane_approx import construct_basis_eij

logger = logging.getLogger(__name__)

def test_construct_basis():
    elem = np.meshgrid(np.array([3,5,8]), np.array([2,5,6]))
    for i in range(len(elem[0])):
        lin = np.linspace(0, 2, elem[0][i])
        xx, yy = np.meshgrid(lin, lin)
        space_x = np.vstack([xx.ravel(), yy.ravel()]).T
        lin = np.linspace(-1, 1, elem[1][i])
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

