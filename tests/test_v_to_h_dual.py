import numpy as np
import pytest
from xgw.v_to_h import init_unit_cube
from xgw.v_to_h_dual import setup_dual_polytope, add_vertex_via_dual, center_polytope, center_polytope_facets, normalize_facets, primal_to_dual
from xgw.hyperplane_approx import build_B_from_H_and_V, DoubleDescription

def setup_dual_polytope_h_to_v_popcount(vertices, facets):

    vertices, center = center_polytope(vertices)
    facets = center_polytope_facets(facets, center)
    facets = normalize_facets(facets)
    dual_vertices = primal_to_dual(facets)

    dual_A = vertices
    dual_b = np.ones(len(vertices))
    
    dual_B_initialization = build_B_from_H_and_V(dual_A, dual_b, dual_vertices)
    dd_dual_polytope = DoubleDescription(implementation='h_to_v_popcount', V_initialization=dual_vertices, B_initialization=dual_B_initialization)
    dd_dual_polytope.V = dual_vertices.tolist()
    dd_dual_polytope.H = [dual_A, dual_b]

    dim = vertices.shape[1]
    return dd_dual_polytope, center, dim

@pytest.fixture
def dims():
    return [2]

def test_cube_single_vertex_expand(dims):
    for dim in dims:
        vertices, facets = init_unit_cube(dim)
        print(f"old vertices (len {len(vertices)}):\n", vertices)
        print("old facets:")
        for f in facets: print(f)

        

        x_new = np.array([0.5, 1.5]) 
        # dd_dual_polytope, center, dim = setup_dual_polytope(vertices, facets, implementation='cdd')
        dd_dual_polytope, center, dim = setup_dual_polytope_h_to_v_popcount(vertices, facets)
        print(f"dd_dual_polytope.V: {dd_dual_polytope.V}")
        print(f"dd_dual_polytope.H: {dd_dual_polytope.H}")
        print(f"\nAdding vertex {x_new} via dual...")
                    
        vertices, facets = add_vertex_via_dual(x_new, center, dd_dual_polytope, dim)
        print("new vertices:\n", vertices)
        print("new facets:")
        for f in facets: print(f)



# def test_cube_double_scale_expand(dims):
#     for dim in dims:
#         vertices, facets = init_unit_cube(dim)
#         print(f"old vertices (len {len(vertices)}):\n", vertices)
#         print("old facets:")
#         for f in facets: print(f)

#         bigger_cube_vertices = 3*vertices - 1

#         for x_new in bigger_cube_vertices.tolist():
#             # dd_dual_polytope, center, dim = setup_dual_polytope(vertices, facets, implementation='cdd')
#             dd_dual_polytope, center, dim = setup_dual_polytope_h_to_v_popcount(vertices, facets)
#             print(f"\nAdding vertex {x_new} via dual...")
                        
#             vertices, facets = add_vertex_via_dual(x_new, center, dd_dual_polytope, dim)
#             print("new vertices:\n", vertices)
#             print("new facets:")
#             for f in facets: print(f)

#         # canonicalize vertices for testing
#         vertices = np.round(vertices, decimals=5)
#         vertices = np.unique(vertices, axis=0)
#         assert len(vertices) == 2**dim
#         assert np.all(np.isin(bigger_cube_vertices, vertices))
