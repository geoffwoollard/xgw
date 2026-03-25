import numpy as np

# -----------------------------
# Geometry helpers
# -----------------------------

def compute_hyperplane(points):
    """
    points: (d, d) array (d affinely independent points in R^d)
    returns (a, b) s.t. a^T x = b
    """
    p0 = points[0]
    A = points[1:] - p0  # (d-1, d)

    _, _, vh = np.linalg.svd(A)
    a = vh[-1]
    a = a / np.linalg.norm(a)

    b = a @ p0
    return a, b


def is_visible(facet, x, tol=1e-10):
    return facet["a"] @ x > facet["b"] + tol


def orient_outward(a, b, vertices, interior_point):
    """
    Ensure normal points outward (interior point satisfies a^T x <= b)
    """
    if a @ interior_point > b:
        return -a, -b
    return a, b


def compute_interior_point(vertices):
    """Simple centroid (assumes bounded polytope)"""
    return np.mean(vertices, axis=0)


# -----------------------------
# Adjacency rebuild
# -----------------------------

# def rebuild_adjacency(facets):
#     """
#     Two facets are neighbors if they share exactly d-1 vertices
#     """
#     n = len(facets)
#     for f in facets:
#         f["neighbors"] = []

#     for i in range(n):
#         for j in range(i + 1, n):
#             vi = set(facets[i]["verts"])
#             vj = set(facets[j]["verts"])

#             if len(vi.intersection(vj)) == len(vi) - 1:
#                 facets[i]["neighbors"].append(j)
#                 facets[j]["neighbors"].append(i)
def affine_dim(points, tol=1e-10):
    """
    points: (k, d)
    returns affine dimension
    """
    if len(points) <= 1:
        return 0
    A = points - points[0]
    return np.linalg.matrix_rank(A, tol=tol)

def rebuild_adjacency(facets, vertices, d, tol=1e-10):
    n = len(facets)

    for f in facets:
        f["neighbors"] = []

    for i in range(n):
        vi = facets[i]["verts"]
        pts_i = vertices[vi]

        for j in range(i + 1, n):
            vj = facets[j]["verts"]

            common = list(set(vi) & set(vj))
            if len(common) < d - 1:
                continue

            pts_common = vertices[common]
            dim = affine_dim(pts_common, tol)

            if dim == d - 2:
                facets[i]["neighbors"].append(j)
                facets[j]["neighbors"].append(i)


def prune_vertices(vertices, facets):
    """
    Remove vertices not used by any facet.
    Reindex everything.
    """

    # Step 1: collect used vertices
    used = set()
    for f in facets:
        used.update(f["verts"])

    used = sorted(used)

    # Step 2: build reindex map
    old_to_new = {old: i for i, old in enumerate(used)}

    # Step 3: filter vertices
    vertices_new = vertices[used]

    # Step 4: update facets
    for f in facets:
        f["verts"] = [old_to_new[v] for v in f["verts"]]

    return vertices_new, facets


# -----------------------------
# Main algorithm
# -----------------------------

def add_vertex(vertices, facets, x_new, tol=1e-10):
    """
    vertices: (N, d) array
    facets: list of dicts:
        {
            "verts": list of d vertex indices,
            "a": normal,
            "b": offset,
            "neighbors": list of indices
        }
    """

    d = vertices.shape[1]

    # Step 0: add vertex
    v_new_idx = len(vertices)
    vertices = np.vstack([vertices, x_new])

    # Step 1: find visible facets via BFS
    visible = set()
    stack = []

    # find seed
    for i, f in enumerate(facets):
        if is_visible(f, x_new, tol):
            stack.append(i)
            break

    if not stack:
        # point inside → no change
        return vertices, facets

    while stack:
        i = stack.pop()
        if i in visible:
            continue

        if is_visible(facets[i], x_new, tol):
            visible.add(i)
            stack.extend(facets[i]["neighbors"])

    # Step 2: compute horizon ridges
    # key = ridge (tuple of sorted vertex indices)
    horizon = {}

    for i in visible:
        f = facets[i]
        for nb in f["neighbors"]:
            if nb not in visible:
                ridge = tuple(sorted(set(f["verts"]) & set(facets[nb]["verts"])))
                horizon[ridge] = nb  # store one adjacent non-visible facet

    # Step 3: remove visible facets
    new_facets_list = []
    old_to_new_idx = {}

    for i, f in enumerate(facets):
        if i not in visible:
            old_to_new_idx[i] = len(new_facets_list)
            new_facets_list.append(f)

    facets = new_facets_list

    # Step 4: create new facets
    interior_point = compute_interior_point(vertices)

    created_facets = []

    for ridge, nb_old in horizon.items():
        verts_new = list(ridge) + [v_new_idx]
        pts = vertices[verts_new]

        a, b = compute_hyperplane(pts)
        a, b = orient_outward(a, b, vertices, interior_point)

        created_facets.append({
            "verts": verts_new,
            "a": a,
            "b": b,
            "neighbors": []
        })

    # Step 5: combine facets
    facets.extend(created_facets)

    # Step 6: rebuild adjacency (robust)
    rebuild_adjacency(facets, vertices, d, tol)

    # Step 7: prune dead vertices
    vertices, facets = prune_vertices(vertices, facets)

    return vertices, facets