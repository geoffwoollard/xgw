import time
import numpy as np

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
    count = 0

    t0 = time.perf_counter()

    for j in range(n):
        m_j = masks[j]

        for i in range(j):
            m_i = masks[i]

            # Python int path
            shared = (int(m_i) & int(m_j)).bit_count()

            if shared >= r - 1:
                count += 1

    dt = time.perf_counter() - t0
    return count, dt


def adjacency_numpy(masks, r):
    """
    CPU vectorized version using numpy uint64.
    Requires numpy >= 1.22 for np.bit_count.
    """
    n = len(masks)
    count = 0

    t0 = time.perf_counter()

    for j in range(n):
        shared = np.bitwise_and(masks[:j], masks[j])
        bits = np.bitwise_count(shared)

        count += np.count_nonzero(bits >= (r - 1))

    dt = time.perf_counter() - t0
    return count, dt


def adjacency_cupy(masks, r):
    """
    GPU version using CuPy.
    """
    if not HAS_CUPY:
        raise RuntimeError("cupy not installed")

    masks_gpu = cp.asarray(masks)

    n = len(masks)
    count = 0

    # warmup
    _ = cp.bitwise_and(masks_gpu[:1], masks_gpu[0])
    cp.cuda.Stream.null.synchronize()

    t0 = time.perf_counter()

    for j in range(n):
        shared = cp.bitwise_and(masks_gpu[:j], masks_gpu[j])

        # unpack bits bytewise
        bits = cp.unpackbits(
            shared.view(cp.uint8),
            axis=1
        ).sum(axis=1)

        count += int(cp.count_nonzero(bits >= (r - 1)).get())

    cp.cuda.Stream.null.synchronize()
    dt = time.perf_counter() - t0

    return count, dt


if __name__ == "__main__":

    n = 5000
    r = 8

    print(f"\nGenerating {n} masks...")

    masks_uint64 = make_masks(n, use_python_int=False)
    masks_object = make_masks(n, use_python_int=True)

    print("\n--- Python int.bit_count() ---")
    c, t = adjacency_python(masks_object, r)
    print(f"matches={c}")
    print(f"time={t:.3f} s")

    print("\n--- NumPy uint64 vectorized ---")
    c, t = adjacency_numpy(masks_uint64, r)
    print(f"matches={c}")
    print(f"time={t:.3f} s")

    if HAS_CUPY:
        print("\n--- CuPy GPU ---")
        c, t = adjacency_cupy(masks_uint64, r)
        print(f"matches={c}")
        print(f"time={t:.3f} s")
    else:
        print("\nCuPy not installed")