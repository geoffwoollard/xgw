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

    for fs in ridge_map.values():
        if len(fs) == 2:
            i, j = fs
            facets[i]["neighbors"].add(j)
            facets[j]["neighbors"].add(i)

    return vertices, facets

if __name__ == "__main__":
    vertices, facets = init_unit_cube(3)

    x_new = np.array([1.1, 1.1, 1.1])
    vertices, facets = add_vertex(vertices, facets, x_new)

    print(len(vertices))  # stays 8 (correct)