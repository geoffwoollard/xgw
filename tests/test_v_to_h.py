import numpy as np
import pytest

from xgw.v_to_h import compute_hyperplane, rebuild_adjacency, orient_outward, add_vertex, init_unit_cube

def init_simplex(d):
    vertices = np.eye(d)
    vertices = np.vstack([vertices, np.zeros(d)])

    facets = []
    n = len(vertices)

    for i in range(n):
        verts = [j for j in range(n) if j != i]
        pts = vertices[verts]
        a, b = compute_hyperplane(pts)

        interior = np.mean(vertices, axis=0)
        a, b = orient_outward(a, b, vertices, interior)

        facets.append({
            "verts": verts,
            "a": a,
            "b": b,
            "neighbors": []
        })

    rebuild_adjacency(facets, vertices, d)
    return vertices, facets

    
@pytest.fixture
def dims():
    return [2, 3, 4]

@pytest.fixture
def dim():
    return 3

def test_cube_single_vertex_expand(dims):
    '''Add one vertex on top. should only affect one facet.'''

    for dim in dims:
        vertices, facets = init_unit_cube(dim)
        # print("old vertices:\n", vertices)
        # print("old facets:")
        # for f in facets: print(f)
        assert len(vertices) == 2**dim
        new_vertex = 0.5*np.ones(dim)
        new_coord = 1.5
        new_vertex[-1] = new_coord
        vertices, facets = add_vertex(vertices, facets, new_vertex)
        assert len(vertices) == 2**dim + 1
        assert len(facets) == 2*dim - 1 + 2*(dim-1)
    # print("new vertices:\n", vertices)
    # print("new facets:")
    # for f in facets: print(f)

def test_cube_double_scale_expand(dims):
    '''NB: inefficient in 6d'''

    for dim in dims:
        vertices, facets = init_unit_cube(dim)
        # print("old vertices:\n", vertices)
        # print("old facets:")
        # for f in facets: print(f)
        outer_box_vertices = 3.0 * vertices
        outer_box_vertices -= outer_box_vertices.mean(axis=0)  # center at com of old box
        outer_box_vertices += vertices.mean(axis=0)  # center at com of old box
        # print("outer box vertices:\n", outer_box_vertices)
        for v in outer_box_vertices:  # skip origin
            # print("adding vertex:", v)
            vertices, facets = add_vertex(vertices, facets, v)

        assert len(vertices) == 2**dim
        # canonicalize vertices for testing
        vertices = np.round(vertices, decimals=5)
        vertices = np.unique(vertices, axis=0)
        assert len(vertices) == 2**dim
        assert np.all(np.isin(outer_box_vertices, vertices))
        # print("new vertices:", vertices)
        # print("new facets:")
        # for f in facets: print(f)


def simplex_expand(dim):
    # test
    vertices, facets = init_simplex(3)
    print("initial vertices:\n", vertices)

    x_new = np.array([2.0, 2.0, 2.0])
    vertices, facets = add_vertex(vertices, facets, x_new)
    print("num vertices:", len(vertices))
    print("num facets:", len(facets))
    
    print("new vertices:\n", vertices)
    print("new facets:")
    for f in facets: print(f)

if __name__ == "__main__":
    test_cube_single_vertex_expand([3])
