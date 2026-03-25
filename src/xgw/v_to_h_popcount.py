'''each facet has
{
    "mask": int,          # bitset of vertices
    "a": np.array(d),     # hyperplane
    "b": float,
    "neighbors": set()    # indices
}
'''

import numpy as np


def bit(i):
    return 1 << i

def iter_bits(mask):
    i = 0
    while mask:
        if mask & 1:
            yield i
        mask >>= 1
        i += 1

def popcount(x):
    return x.bit_count()

def compute_hyperplane(points):
    p0 = points[0]
    A = points[1:] - p0
    _, _, vh = np.linalg.svd(A)
    a = vh[-1]
    a = a / np.linalg.norm(a)
    b = a @ p0
    return a, b

def compute_hyperplane(points):
    p0 = points[0]
    A = points[1:] - p0
    _, _, vh = np.linalg.svd(A)
    a = vh[-1]
    a = a / np.linalg.norm(a)
    b = a @ p0
    return a, b

def is_visible(facet, x, tol=1e-10):
    return facet["a"] @ x > facet["b"] + tol

def facet_ridges(mask):
    verts = list(iter_bits(mask))
    for v in verts:
        yield mask & ~bit(v)

def add_vertex(vertices, facets, x_new, tol=1e-10):
    d = vertices.shape[1]

    # Step 0: append vertex
    v_new = len(vertices)
    vertices = np.vstack([vertices, x_new])

    # -----------------------------
    # Step 1: find visible facets
    # -----------------------------
    visible = set()
    stack = []

    for i, f in enumerate(facets):
        if is_visible(f, x_new, tol):
            stack.append(i)
            break

    if not stack:
        return vertices, facets

    while stack:
        i = stack.pop()
        if i in visible:
            continue
        if is_visible(facets[i], x_new, tol):
            visible.add(i)
            stack.extend(facets[i]["neighbors"])

    # -----------------------------
    # Step 2: collect horizon ridges
    # -----------------------------
    ridge_count = {}   # ridge_mask -> [facet indices]

    for i, f in enumerate(facets):
        if i not in visible:
            continue
        for r in facet_ridges(f["mask"]):
            ridge_count.setdefault(r, []).append(i)

    horizon = []
    for r, fs in ridge_count.items():
        if len(fs) == 1:
            # ridge appears only once → boundary
            horizon.append(r)

    # -----------------------------
    # Step 3: remove visible facets
    # -----------------------------
    old_to_new = {}
    new_facets = []

    for i, f in enumerate(facets):
        if i not in visible:
            old_to_new[i] = len(new_facets)
            new_facets.append(f)

    facets = new_facets

    # -----------------------------
    # Step 4: create new facets
    # -----------------------------
    created = []

    interior = vertices.mean(axis=0)

    for r in horizon:
        new_mask = r | bit(v_new)

        verts = list(iter_bits(new_mask))
        pts = vertices[verts[:d]]  # pick d points

        a, b = compute_hyperplane(pts)

        if a @ interior > b:
            a, b = -a, -b

        created.append({
            "mask": new_mask,
            "a": a,
            "b": b,
            "neighbors": set()
        })

    # -----------------------------
    # Step 5: build ridge map (fast adjacency)
    # -----------------------------
    ridge_map = {}

    def add_ridges(fi, mask):
        for r in facet_ridges(mask):
            ridge_map.setdefault(r, []).append(fi)

    # existing facets
    for i, f in enumerate(facets):
        add_ridges(i, f["mask"])

    # new facets
    offset = len(facets)
    for i, f in enumerate(created):
        add_ridges(offset + i, f["mask"])

    facets.extend(created)

    # -----------------------------
    # Step 6: rebuild neighbors from ridge_map
    # -----------------------------
    for f in facets:
        f["neighbors"] = set()

    for fs in ridge_map.values():
        if len(fs) == 2:
            i, j = fs
            facets[i]["neighbors"].add(j)
            facets[j]["neighbors"].add(i)

    # -----------------------------
    # Step 7: prune vertices
    # -----------------------------
    used_mask = 0
    for f in facets:
        used_mask |= f["mask"]

    used = list(iter_bits(used_mask))
    old_to_new_v = {v: i for i, v in enumerate(used)}

    vertices = vertices[used]

    for f in facets:
        new_mask = 0
        for v in iter_bits(f["mask"]):
            new_mask |= bit(old_to_new_v[v])
        f["mask"] = new_mask

    return vertices, facets