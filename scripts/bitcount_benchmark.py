import time
import numpy as np
from scipy import sparse

# Optional GPU support
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def make_masks(n, nbits=64, seed=0, use_python_int=False):
    rng = np.random.default_rng(seed)

    if use_python_int:
        # arbitrary precision Python ints
        return [
            int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
            for _ in range(n)
        ]
    else:
        # fixed-width uint64
        return rng.integers(
            0,
            np.iinfo(np.uint64).max,
            size=n,
            dtype=np.uint64,
        )



def adjacency_python(masks, r):
    n = len(masks)
    rows, cols = [], []

    t0 = time.perf_counter()

    for j in range(n):
        m_j = masks[j]
        for i in range(j):
            m_i = masks[i]
            shared = (int(m_i) & int(m_j)).bit_count()
            if shared >= r - 1:
                rows.append(i)
                cols.append(j)

    dt = time.perf_counter() - t0
    
    D_upper = sparse.coo_matrix(
        (np.ones(len(rows), dtype=bool), (rows, cols)),
        shape=(n, n), dtype=bool
    )
    D = D_upper + D_upper.T
    return D.tocsr(), dt


def adjacency_numpy(masks, r):
    n = len(masks)
    rows, cols = [], []

    t0 = time.perf_counter()

    for j in range(n):
        shared = np.bitwise_and(masks[:j], masks[j])
        bits = np.bitwise_count(shared)
        match_idx = np.where(bits >= (r - 1))[0]
        
        rows.extend(match_idx)
        cols.extend([j] * len(match_idx))

    dt = time.perf_counter() - t0
    
    D_upper = sparse.coo_matrix(
        (np.ones(len(rows), dtype=bool), (rows, cols)),
        shape=(n, n), dtype=bool
    )
    D = D_upper + D_upper.T
    return D.tocsr(), dt


# def adjacency_cupy(masks, r):
#     if not HAS_CUPY:
#         raise RuntimeError("cupy not installed")

#     masks_gpu = cp.asarray(masks)
#     n = len(masks)
#     rows, cols = [], []

#     _ = cp.bitwise_and(masks_gpu[:1], masks_gpu[0])
#     cp.cuda.Stream.null.synchronize()

#     t0 = time.perf_counter()

#     for j in range(n):
#         shared = cp.bitwise_and(masks_gpu[:j], masks_gpu[j])
#         bits = cp.bitwise_count(shared)
#         match_idx = cp.where(bits >= (r - 1))[0]
        
#         rows.extend(cp.asnumpy(match_idx))
#         cols.extend([j] * len(match_idx))

#     cp.cuda.Stream.null.synchronize()
#     dt = time.perf_counter() - t0
    
#     D_upper = sparse.coo_matrix(
#         (np.ones(len(rows), dtype=bool), (rows, cols)),
#         shape=(n, n), dtype=bool
#     )
#     D = D_upper + D_upper.T
#     return D.tocsr(), dt


def adjacency_cupy(masks, r):
    if not HAS_CUPY:
        raise RuntimeError("cupy not installed")

    masks_gpu = cp.asarray(masks)
    n = len(masks)

    t0 = time.perf_counter()

    # Vectorize: compare all pairs at once using broadcasting
    # masks_gpu[:, None] shape (n, 1), masks_gpu[None, :] shape (1, n)
    shared = cp.bitwise_and(masks_gpu[:, None], masks_gpu[None, :])  # (n, n)
    bits = cp.bitwise_count(shared)  # (n, n)
    
    # Find upper triangle matches
    i_idx, j_idx = cp.where((bits >= (r - 1)) & (cp.arange(n)[:, None] < cp.arange(n)[None, :]))

    cp.cuda.Stream.null.synchronize()
    dt = time.perf_counter() - t0

    rows = cp.asnumpy(i_idx)
    cols = cp.asnumpy(j_idx)

    D_upper = sparse.coo_matrix(
        (np.ones(len(rows), dtype=bool), (rows, cols)),
        shape=(n, n), dtype=bool
    )
    D = D_upper + D_upper.T
    return D.tocsr(), dt


if __name__ == "__main__":
    n = 5000
    r = 8

    print(f"\nGenerating {n} masks...")
    masks_uint64 = make_masks(n, use_python_int=False)
    masks_object = make_masks(n, use_python_int=True)

    print("\n--- Python int.bit_count() ---")
    D, t = adjacency_python(masks_object, r)
    print(f"nnz={D.nnz}, shape={D.shape}")
    print(f"time={t:.3f} s")

    print("\n--- NumPy uint64 vectorized ---")
    D, t = adjacency_numpy(masks_uint64, r)
    print(f"nnz={D.nnz}, shape={D.shape}")
    print(f"time={t:.3f} s")

    if HAS_CUPY:
        print("\n--- CuPy GPU ---")
        D, t = adjacency_cupy(masks_uint64, r)
        print(f"nnz={D.nnz}, shape={D.shape}")
        print(f"time={t:.3f} s")
    else:
        print("\nCuPy not installed")