import numpy as np
import matplotlib.pyplot as plt
from xgw.Hyperplane_approx import DoubleRepresentation, Hausdorff, new_direction,update_box



def test_Hausdorff(niter):
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
    
    for iter in range(niter):
        P_plus, P_minus, objective, previous_solutions_to_reuse = iteration_loop(P_plus, P_minus, {})
        plt.figure()
        list_min = np.array(P_minus.V)
        plt.scatter(list_min[:,0], list_min[:,1], c='k')
        list_pl = np.array(P_plus.V)
        plt.scatter(list_pl[:,0], list_pl[:,1], c='r')
        plt.plot(np.sin(np.linspace(0, 2*np.pi,1000)),np.cos(np.linspace(0, 2*np.pi,1000)), c='b')
        # plt.clf()
        
    A,b = P_plus.H
    check = True
    for i, elem in enumerate(P_minus.V):
        print(i)
        if not np.all(A@elem<=b+1e-15):
            print(np.max(A@elem-b))
            check =  False
    assert check
    plt.show()
        
    


def iteration_loop(P_plus, P_minus, previous_solutions_to_reuse):
    x_0, v_0, objective, previous_solutions_to_reuse = Hausdorff(P_plus, P_minus, previous_solutions_to_reuse)
    g = new_direction(x_0, v_0, P_minus)
    g_hat, g_star = 1, g/np.linalg.norm(g)
    P_plus, P_minus = update_box(P_plus, P_minus, [[g, g_hat]], [g_star])
    return P_plus, P_minus, objective, previous_solutions_to_reuse

if __name__ == '__main__':
    test_Hausdorff(50)