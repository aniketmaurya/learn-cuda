import torch
import triton
import triton.language as tl

def python_matmul(A: list[list], B: list[list]):
    assert len(A[0]) == len(B)
    m, n, p = len(A), len(B[0]), len(A[0])
    print("m, n, k", m, n, p)

    C = [[0.0 for _ in range(n)] for _ in range(m)]
    print(C)
    for i in range(m):
        for j in range(n):
            acc = 0.0
            for ck in range(p):
                acc += (A[i][ck] * B[ck][j])
            C[i][j] = acc
    return C

@triton.jit
def _kernel(A_ptr, 
            B_ptr,
            C_ptr, 
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            block_size_m: tl.constexpr, block_size_n: tl.constexpr, block_size_k: tl.constexpr):
    row_pid = tl.program_id(0)
    col_pid = tl.program_id(1)

    row_start = row_pid * block_size_m
    col_start = col_pid * block_size_n
    row_end = row_start + block_size_m
    col_end = col_start + block_size_n

    offset_m = tl.arange(0, block_size_m) + row_start
    offset_n = tl.arange(0, block_size_n) + col_start

    acc = tl.zeros((block_size_m, block_size_n), dtype=tl.float32)

    for k0 in range(0, K, block_size_k):
        "C[i][j] = a[i][k] * b[k][j]"
        offset_k = tl.arange(0, block_size_k) + k0
        A_tile_ptr = A_ptr + offset_m[:, None] * stride_am + offset_k[None:, ] * stride_ak
        B_tile_ptr = B_ptr + offset_k[:, None] * stride_bk + offset_n[None, :] * stride_bn

        A_mask = (offset_m[:, None]<M) & (offset_k[None, :] < K)
        B_mask = (offset_k[:, None] < K) & (offset_n[None, :] < N)

        A_tile = tl.load(A_tile_ptr, mask=A_mask, other=0.0)
        B_tile = tl.load(B_tile_ptr, mask=B_mask, other=0.0)

        acc += tl.dot(A_tile, B_tile)

    C_ptrs = C_ptr + offset_m[:, None] * stride_cm + offset_n[None, :] * stride_cn
    C_mask = (offset_m[:, None] < M) & (offset_n[None, :] < N)
    tl.store(C_ptrs, acc, mask=C_mask)


def blocked_matmul(A: torch.Tensor, B: torch.Tensor):
    m, k1 = A.shape
    k2, n = B.shape
    assert k1==k2
    k = k1
    C = torch.empty((m, n), device=A.device, dtype=torch.float32)

    BLOCK_M, BLOCK_N = 32, 32
    BLOCK_K = 32
    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(n, BLOCK_N))

    _kernel[grid](
        A, B, C, m, n, k,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    return C


if __name__ == "__main__":
    x = [[1,2,3,4],
         [5,6,7,8]]
    y = [[9, 10,],
         [11, 12],
         [13, 14],
         [15, 16]]
    z = python_matmul(x, y)
    print(z)

    print(torch.Tensor(x) @ torch.Tensor(y))
    
    x1 = torch.Tensor(x).cuda()
    y1 = torch.Tensor(y).cuda()
    z = blocked_matmul(x1, y1)
    print(z)
