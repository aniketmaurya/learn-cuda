import torch
import triton
import triton.language as tl

# =============================================================================
# Tiled MatMul kernel
# C[M, N] = A[M, K] @ B[K, N]
# Uses 3 block sizes: BLOCK_M x BLOCK_N output tile, reduced over BLOCK_K.
# =============================================================================
@triton.jit
def matmul_tiled_kernel(
    A_ptr, B_ptr, C_ptr,              # pointers to A, B, C in device memory
    M, N, K,                          # problem sizes
    stride_am, stride_ak,             # A strides (row-major: stride_am=K, stride_ak=1)
    stride_bk, stride_bn,             # B strides (row-major: stride_bk=N, stride_bn=1)
    stride_cm, stride_cn,             # C strides (row-major: stride_cm=N, stride_cn=1)
    BLOCK_M: tl.constexpr,            # tile height in M dimension
    BLOCK_N: tl.constexpr,            # tile width  in N dimension
    BLOCK_K: tl.constexpr,            # reduction chunk size along K
):
    # ---- Which output tile do I compute? ------------------------------------
    pid_m = tl.program_id(0)                          # tile row index p
    pid_n = tl.program_id(1)                          # tile col index r
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M) # rows  of my tile  (pT + u)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N) # cols  of my tile  (rT + v)

    # ---- Accumulator for C_tile (BLOCK_M x BLOCK_N, fp32 for accuracy) -------
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ---- Loop over the K dimension in BLOCK_K chunks (q) ---------------------
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)          # k indices for this chunk (qT + t)

        # Pointers to the sub-tiles we need this iteration:
        # A_tile shape = (BLOCK_M, BLOCK_K) -> A[pT+u, qT+t]
        # B_tile shape = (BLOCK_K, BLOCK_N) -> B[qT+t, rT+v]
        A_tile_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        B_tile_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        # Masks for edges (when blocks go out of bounds)
        A_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        B_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load tiles (out-of-bounds -> 0)
        A_tile = tl.load(A_tile_ptrs, mask=A_mask, other=0.0)
        B_tile = tl.load(B_tile_ptrs, mask=B_mask, other=0.0)

        # Fused multiply-accumulate over this BLOCK_K slice:
        # acc[u, v] += sum_t A_tile[u, t] * B_tile[t, v]
        acc += tl.dot(A_tile, B_tile)

    # ---- Write my output tile to C ------------------------------------------
    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    C_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc, mask=C_mask)


# =============================================================================
# Tiny driver to run & verify
# =============================================================================
def matmul_tiled(A: torch.Tensor, B: torch.Tensor, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32):
    assert A.is_cuda and B.is_cuda, "Move tensors to CUDA"
    assert A.dtype == torch.float32 and B.dtype == torch.float32
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match"
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    # 2D launch grid: one program per output tile (p, r)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_tiled_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return C


if __name__ == "__main__":
    # Example (same matrices as your walkthrough)
    A = torch.tensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9,10,11,12],
        [13,14,15,16]
    ], dtype=torch.float32, device="cuda")

    B = torch.tensor([
        [1,0,0,1],
        [0,1,1,0],
        [1,1,0,0],
        [0,0,1,1]
    ], dtype=torch.float32, device="cuda")

    C = matmul_tiled(A, B, BLOCK_M=2, BLOCK_N=2, BLOCK_K=2)  # small tiles to mirror the manual example
    print("Triton result:\n", C.cpu().numpy())

    # Reference
    C_ref = A @ B
    print("PyTorch result:\n", C_ref.cpu().numpy())
    print("Match:", torch.allclose(C, C_ref))
