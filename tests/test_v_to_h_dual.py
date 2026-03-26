import numpy as np
import pytest
from xgw.v_to_h import init_unit_cube
from xgw.v_to_h_dual import setup_dual_polytope, add_vertex_via_dual
    

@pytest.fixture
def dims():
    return [2,3,4,5]

def test_cube_double_scale_expand(dims):
    for dim in dims:
        vertices, facets = init_unit_cube(dim)
        print(f"old vertices (len {len(vertices)}):\n", vertices)
        print("old facets:")
        for f in facets: print(f)

        bigger_cube_vertices = 3*vertices - 1

        for x_new in bigger_cube_vertices.tolist():
            dd_dual_polytope, center, dim = setup_dual_polytope(vertices, facets)
            vertices, facets = add_vertex_via_dual(x_new, center, dd_dual_polytope, dim)
            # print("new vertices:\n", vertices)
            # print("new facets:")
            # for f in facets: print(f)

        # canonicalize vertices for testing
        vertices = np.round(vertices, decimals=5)
        vertices = np.unique(vertices, axis=0)
        assert len(vertices) == 2**dim
        assert np.all(np.isin(bigger_cube_vertices, vertices))
