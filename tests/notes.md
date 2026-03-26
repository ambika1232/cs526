# Test ideas

Use these kernels for milestone demos:

1. `A[tid]` -> fully coalesced baseline
2. `A[tid * 2]` -> mild stride
3. `A[tid * 4]` -> clearly non-coalesced
4. `A[tid + 8]` -> coalesced with offset
5. `A[N - tid - 1]` -> reverse indexing / difficult case

For the midterm, it is enough to show that the pass can classify 1, 2, 3, and 4 correctly and report `UNKNOWN` or conservative output for 5.
