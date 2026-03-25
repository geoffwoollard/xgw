import numpy as np
import itertools

from xgw.v_to_h_popcount import bit, facet_ridges, add_vertex

def init_unit_cube(d):
    vertices = np.array(list(itertools.product([0.0, 1.0], repeat=d)))

    facets = []

    for i in range(d):
        mask0 = 0
        mask1 = 0

        for j, v in enumerate(vertices):
            if v[i] == 0:
                mask0 |= bit(j)
            else:
                mask1 |= bit(j)

        a0 = np.zeros(d); a0[i] = -1
        a1 = np.zeros(d); a1[i] = 1

        facets.append({"mask": mask0, "a": a0, "b": 0.0, "neighbors": set()})
        facets.append({"mask": mask1, "a": a1, "b": 1.0, "neighbors": set()})

    # build adjacency once
    ridge_map = {}
    for i, f in enumerate(facets):
        for r in facet_ridges(f["mask"]):
            ridge_map.setdefault(r, []).append(i)
            
    print(f"ridge_map for dim={d}:")
    for ridge, facet_indices in ridge_map.items():
        print(f"  ridge {bin(ridge)}: facets {facet_indices}")

    for fs in ridge_map.values():
        if len(fs) == 2:
            i, j = fs
            facets[i]["neighbors"].add(j)
            facets[j]["neighbors"].add(i)

    return vertices, facets

def test_cube_single_vertex_expand(dims):
    '''Add one vertex on top. should only affect one facet.'''

    for dim in dims:
        vertices, facets = init_unit_cube(dim)
        print("old vertices:\n", vertices)
        print("old facets:")
        for f in facets: print(f)
        assert len(vertices) == 2**dim
        new_vertex = 0.5*np.ones(dim)
        new_coord = 1.5
        new_vertex[-1] = new_coord
        vertices, facets = add_vertex(vertices, facets, new_vertex)
        assert len(vertices) == 2**dim + 1
        assert len(facets) == 2*dim - 1 + 2*(dim-1)

if __name__ == "__main__":
    vertices, facets = init_unit_cube(3)
    print("old vertices:\n", vertices)
    print("old facets:")
    for f in facets: print(f)

    # x_new = np.array([1.1, 1.1, 1.1])
    # vertices, facets = add_vertex(vertices, facets, x_new)

    # print(len(vertices))  # stays 8 (correct)
    # test_cube_single_vertex_expand([3])

