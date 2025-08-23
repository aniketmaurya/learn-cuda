import torch

import triton
import triton.language as tl

@triton.jit
def _add(
    inp_a,
    inp_b,
    out_ptr,
    n_elem,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
    a = inp_a + offsets
    b = inp_b + offsets
    out_ptr = out_ptr + offsets
    mask=offsets<n_elem

    a = tl.load(a, mask=mask, other=0.0)
    b = tl.load(b, mask=mask, other=0.0)
    tl.store(out_ptr, a + b, mask=mask)

def add(a, b):
    c = torch.empty_like(a)
    n = c.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    _add[(grid)](a, b, c, n, BLOCK_SIZE=4096)
    return c

def main():
    size = 98432

    x = torch.randn(size).cuda()
    y = torch.randn(size).cuda()

    output_torch = x + y
    output_triton = add(x, y)
    print(output_torch)
    print(output_triton)
    print(f'The maximum difference between torch and triton is '
        f'{torch.max(torch.abs(output_torch - output_triton))}')

main()
