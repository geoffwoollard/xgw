import numpy as np

from xgw.Hyperplane_approx import DoubleRepresentation, Hausdorff


def test_Hausdorff():
    inner_square_vertices = [np.array(v) for v in [
        (0,1), (0,-1), (1,0), (-1,0)
    ]]

    outer_square_vertices = [np.array(v) for v in [
        (1,1), (1,-1), (-1,1), (-1,-1)
    ]]

    previous_solutions_to_reuse = {}
    P_plus = DoubleRepresentation()
    P_plus.add_V(outer_square_vertices)  
    P_minus = DoubleRepresentation()
    P_minus.add_V(inner_square_vertices)

    x_0, v_0, objective, previous_solutions_to_reuse = Hausdorff(P_plus, P_minus, previous_solutions_to_reuse)
    assert np.allclose(objective, np.linalg.norm(x_0 - v_0)**2)
    assert np.allclose(2*x_0, v_0) # since the closest point in the inner square to a vertex of the outer square is at half the distance
