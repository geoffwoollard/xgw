import numpy as np
import itertools
import pytest

from xgw.v_to_h import compute_hyperplane, rebuild_adjacency, orient_outward, add_vertex

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


def init_unit_cube(d):
    """
    Initialize unit cube [0,1]^d

    Returns:
        vertices: (2^d, d)
        facets: list of facet dicts
    """

    # -----------------------------
    # Step 1: vertices
    # -----------------------------
    vertices = np.array(list(itertools.product([0.0, 1.0], repeat=d)))

    n_vertices = len(vertices)

    # -----------------------------
    # Step 2: facets
    # -----------------------------
    facets = []

    for i in range(d):
        # facet x_i = 0
        verts_0 = [j for j in range(n_vertices) if vertices[j, i] == 0.0]
        a0 = np.zeros(d)
        a0[i] = -1.0   # outward normal
        b0 = 0.0

        facets.append({
            "verts": verts_0,
            "a": a0,
            "b": b0,
            "neighbors": []
        })

        # facet x_i = 1
        verts_1 = [j for j in range(n_vertices) if vertices[j, i] == 1.0]
        a1 = np.zeros(d)
        a1[i] = 1.0
        b1 = 1.0

        facets.append({
            "verts": verts_1,
            "a": a1,
            "b": b1,
            "neighbors": []
        })

    # -----------------------------
    # Step 3: adjacency
    # -----------------------------
    def rebuild_cube_adjacency(facets, d):
        """
        facets are ordered as:
            [x0=0, x0=1, x1=0, x1=1, ..., x_{d-1}=0, x_{d-1}=1]
        """

        for f in facets:
            f["neighbors"] = []

        for i in range(len(facets)):
            axis_i = i // 2

            for j in range(len(facets)):
                if i == j:
                    continue

                axis_j = j // 2

                # neighbors if different axes
                if axis_i != axis_j:
                    facets[i]["neighbors"].append(j)
    rebuild_cube_adjacency(facets, d)
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
    test_cube_double_scale_expand([6])
