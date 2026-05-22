def main():
    import numpy as np
    import time

    n = 1000
    ints_uint64 = np.random.randint(low=0, high=2**63, size=(n,n)).astype(np.uint64)

    t0 = time.perf_counter()
    bits = np.bitwise_count(ints_uint64)
    dt = time.perf_counter() - t0
    print(f"Bitwise count for dtype {ints_uint64.dtype} of {n}x{n} integers took {dt:.4f} seconds.")

    ints_object = np.array(ints_uint64, dtype=object)
    t0 = time.perf_counter()
    bits = np.bitwise_count(ints_object)
    dt = time.perf_counter() - t0
    print(f"Bitwise count for dtype {ints_object.dtype} of {n}x{n} integers took {dt:.4f} seconds.")


if __name__ == '__main__':
    main()