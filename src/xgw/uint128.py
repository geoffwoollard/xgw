def main():
    import numpy as np
    import time
    import math
    
    n = 1000
    max_bit = 100
    rng = np.random.default_rng(seed=42)

    # Generate as uint64, then convert to Python ints for larger bit widths
    if max_bit <= 64:
        ints = rng.integers(low=0, high=2**max_bit, size=(n, n), dtype=np.uint64)
    else:
        # Generate pairs of uint64 and combine into larger integers
        lo = rng.integers(low=0, high=2**64, size=(n, n), dtype=np.uint64)
        hi = rng.integers(low=0, high=2**(max_bit-64), size=(n, n), dtype=np.uint64)
        ints = np.array([[int(hi[i,j]) << 64 | int(lo[i,j]) for j in range(n)] for i in range(n)], dtype=object)
        print(ints)
        print(math.log2(max(int(x) for x in ints.flatten())))

    if max_bit <= 64:
        print(f"Testing bitwise count for uint64 with max_bit={max_bit} (fits in 64 bits).")
        ints_uint64 = ints.astype(np.uint64)
        t0 = time.perf_counter()
        bits = np.bitwise_count(ints_uint64)
        dt = time.perf_counter() - t0
        print(f"Bitwise count for dtype {ints_uint64.dtype} of {n}x{n} integers took {dt:.4f} seconds.")

    ints_object = np.array(ints, dtype=object)
    t0 = time.perf_counter()
    bits_object = np.bitwise_count(ints_object)
    dt = time.perf_counter() - t0
    print(f"Bitwise count for dtype {ints_object.dtype} of {n}x{n} integers took {dt:.4f} seconds.")



    mask64 = (1 << 64) - 1
    lo = np.array([x & mask64 for x in ints], dtype=np.uint64)
    hi = np.array([x >> 64 for x in ints], dtype=np.uint64)
    t0 = time.perf_counter()
    bits_hi = np.bitwise_count(hi)
    bits_lo = np.bitwise_count(lo)
    bits_total = bits_hi + bits_lo
    dt = time.perf_counter() - t0
    print(f"Bitwise count for dtype {ints.dtype} of {n}x{n} integers took {dt:.4f} seconds.")

    assert np.array_equal(bits_total, bits_object), "Bitwise count mismatch between object and split uint64 method."

if __name__ == '__main__':
    main()