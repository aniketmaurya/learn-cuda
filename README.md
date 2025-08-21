# CUDA Programming Notes

## General concepts

* A kernel is the unit of CUDA code that programmers typically write and compose, akin to a procedure or function in languages targeting CPUs.
* A thread block is a level of the CUDA programming model's thread hierarchy below a grid but above a warp . It is the CUDA programming model's abstract equivalent of the concrete cooperative thread arrays  in PTX /SASS .]
* At the highest level, multiple thread blocks  are organized into a thread block grid  that spans the entire GPU. Thread blocks  are strictly limited in their coordiation and communication.
*  threads  execute on individual cores , thread blocks  are scheduled onto SMs , and grids  utilize all available SMs  on the device.
* Shared memory is the level of the memory hierarchy  corresponding to the thread block  level of the thread group hierarchy in the CUDA programming model . It is generally expected to be much smaller but much faster (in throughput and latency) than the global memory .


## Vector Reduction 

* On the host, they are PyTorch tensors. When you call the kernel, Triton passes device pointers to their first elements to the JIT’ed kernel. Inside the kernel you indeed work with pointers.

* num_programs is the size of the grid along axis 0 (so your grid is (num_programs,)). A grid can be 1D/2D/3D, but here it’s 1D.

* In Triton, BLOCK_SIZE is not a thread count. It’s a compile-time tile size = number of elements each program processes (drives the shape of tl.arange, masks, vector loads/stores).

* in Triton, num_warps sets how many hardware threads run a program (block), while BLOCK_SIZE sets how many data elements that program processes.

### How they combine

* Threads per program (CTA) = num_warps × 32
e.g., `num_warps=4` → 128 threads.


* `Elements per program = BLOCK_SIZE`
e.g., BLOCK_SIZE=512 → 512 elements handled by that same program.

So with `num_warps=4` and `BLOCK_SIZE=512`, you have 128 threads processing 512 elements.
On average: 4 elements per thread (`512 / 128 = 4`).

