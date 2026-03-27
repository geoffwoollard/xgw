import numpy as np
import pytest
# import logging
from time import time

from xgw.utils import canonicalize_vertices
from xgw.v_to_h import init_unit_cube
from xgw.v_to_h_dual import setup_dual_polytope, add_vertex_via_dual


# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)


@pytest.fixture
def dims():
    return [2,3]

@pytest.fixture
def implementations():
    return ['h_to_v_popcount','cdd']

def test_cube_single_vertex_expand(dims, implementations):
    for dim in dims:
        for implementation in implementations:
            # logger.info(f"Testing dimension {dim} with implementation {implementation}...")
            unit_cube_vertices, facets = init_unit_cube(dim)
            A = np.array([f["a"] for f in facets])
            b = np.array([f["b"] for f in facets])
            # print(f"old vertices (len {len(unit_cube_vertices)}):\n", unit_cube_vertices)
            # print("old facets:")
            # for f in facets: print(f)

            x_new = 0.5 * np.ones(dim)  # add center point of cube as new vertex
            x_new[0] = 1.5  # move one vertex outside the cube to force expansion
            dd_dual_polytope, center, dim = setup_dual_polytope(unit_cube_vertices, A, b, implementation=implementation)
            # print(f"dd_dual_polytope.V: {dd_dual_polytope.V}")
            # print(f"dd_dual_polytope.H: {dd_dual_polytope.H}")
            # print(f"\nAdding vertex {x_new} via dual...")
                        
            vertices, A, b = add_vertex_via_dual(x_new, center, dd_dual_polytope, dim)
            # print("new vertices:\n", vertices)
            # print("new facets:")
            # for f in facets: print(f)

            ground_truth_vertices = unit_cube_vertices.tolist() + [x_new.tolist()]
            ground_truth_vertices = np.array(ground_truth_vertices)
            ground_truth_vertices = canonicalize_vertices(ground_truth_vertices)
            assert len(ground_truth_vertices) == len(unit_cube_vertices) + 1
            vertices = canonicalize_vertices(vertices)
            assert len(vertices) == len(ground_truth_vertices), f"Expected {len(ground_truth_vertices)} vertices but got {len(vertices)} for dimension {dim} with implementation {implementation}"
            assert np.all(np.isin(ground_truth_vertices, vertices)), f"Not all expected vertices are in the result for dimension {dim} with implementation {implementation}"

def test_cube_double_scale_expand(dims, implementations):
    for dim in dims:
        for implementation in implementations:
            vertices, facets = init_unit_cube(dim)
            A = np.array([f["a"] for f in facets])
            b = np.array([f["b"] for f in facets])
            # print(f"old vertices (len {len(vertices)}):\n", vertices)
            # print("old facets:")
            # for f in facets: print(f)

            bigger_cube_vertices = 3*vertices - 1

            s = time()
            for i, x_new in enumerate(bigger_cube_vertices.tolist()):
                print(f"Testing dimension {dim} with implementation {implementation} on polytope with {len(vertices)} vertices (iteration {i+1}/{len(bigger_cube_vertices)})...")
                # dd_dual_polytope, center, dim = setup_dual_polytope(vertices, facets, implementation='cdd')
                dd_dual_polytope, center, dim = setup_dual_polytope(vertices, A, b, implementation=implementation)
                print("setup_dual_polytope done")
                print(f"Adding vertex {x_new} via dual...")            
                vertices, A, b = add_vertex_via_dual(x_new, center, dd_dual_polytope, dim)
                print("add_vertex_via_dual done")
                # print("new vertices:\n", vertices)
                # print("new facets:")
                # for f in facets: print(f)
            e = time()
            print(f"Total time for dimension {dim} with implementation {implementation}: {e-s:.2f} seconds")

            # canonicalize vertices for testing
            vertices = canonicalize_vertices(vertices)
            bigger_cube_vertices = canonicalize_vertices(bigger_cube_vertices)
            # assert len(vertices) == 2**dim
            assert np.all(np.isin(bigger_cube_vertices, vertices)), f"Not all expected vertices are in the result for dimension {dim} with implementation {implementation}"


# if __name__ == "__main__":

#     for dim in [4]:
#         test_cube_double_scale_expand([dim], ['h_to_v_popcount','cdd'])