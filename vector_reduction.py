import torch
import triton
import triton.language as tl


@triton.jit
def reduce_kernel(x_ptr, n_elements, partial_outputs_ptr, BLOCK_SIZE: tl.constexpr):
    # One program (≈ CUDA block) handles one tile of BLOCK_SIZE elements
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Lane indices inside the tile (contiguous for coalesced loads)
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Masked load: out-of-bounds lanes contribute 0.0
    vals = tl.load(x_ptr + offs, mask=mask, other=0.0)
    # Accumulate in fp32 for stability (safe even if x is fp16/bf16)
    vals = vals.to(tl.float32)

    # Per-program reduction → one scalar
    partial = tl.sum(vals, axis=0)

    # Store this program's partial sum at index pid
    tl.store(partial_outputs_ptr + pid, partial)


def reduce_sum_triton(x: torch.Tensor, block_size: int = 512, num_warps: int = 4) -> torch.Tensor:
    """
    Two-pass reduction:
      Pass-1 (GPU): per-tile partial sums
      Pass-2 (GPU): sum of partials
    Returns a 0-dim tensor on GPU (use .item() if you want a Python float).
    """
    assert x.is_cuda, "Move x to CUDA before calling."
    n = x.numel()
    # One partial per program
    num_programs = triton.cdiv(n, block_size)
    partials = torch.empty(num_programs, device=x.device, dtype=torch.float32)

    # ✅ Correct 1D grid: (num_programs,)
    reduce_kernel[(num_programs,)](
        x, n, partials,
        BLOCK_SIZE=block_size,
        num_warps=num_warps
    )

    # Pass-2 on GPU
    return partials.sum()


if __name__ == "__main__":
    torch.cuda.init()
    print("GPU:", torch.cuda.get_device_name(0))

    # Test input
    x = torch.randn(100_000, device="cuda", dtype=torch.float32).contiguous()

    # Triton reduction
    triton_sum = reduce_sum_triton(x, block_size=512, num_warps=4)
    # PyTorch baseline
    torch_sum = x.sum()

    # Compare
    diff = (triton_sum - torch_sum).abs().item()
    print("Triton:", triton_sum.item())
    print("PyTorch:", torch_sum.item())
    print("abs diff:", diff)
