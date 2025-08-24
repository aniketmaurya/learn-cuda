import torch
import triton
import triton.language as tl


@triton.jit
def _softmax(x_ptr, y_ptr, row_stride,col_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row_start_ptr = x_ptr + pid * row_stride
    offsets = tl.arange(0, BLOCK_SIZE)
    outputs = (y_ptr + pid * row_stride) + offsets * col_stride
    mask = offsets < n_cols
    
    row_item_ptr = row_start_ptr + offsets * col_stride
    row = tl.load(row_item_ptr, mask=mask, other=-float("inf"))
    row_stable = row  - tl.max(row, axis=0)
    numerator = tl.exp(row_stable)
    denominator = tl.sum(numerator, axis=0)

    y = numerator / denominator

    tl.store(outputs, y, mask=mask)

def softmax(x):
    m, n = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n)
    grid = (m,)

    y = torch.empty_like(x)

    _softmax[grid](x, y, x.stride(0),x.stride(1), m, n, BLOCK_SIZE=BLOCK_SIZE)
    return y

if __name__=="__main__":
    torch.manual_seed(0)
    x = torch.randn(1823, 781).cuda()
    y_triton = softmax(x)
    y_torch = torch.softmax(x, axis=1)
    assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)
