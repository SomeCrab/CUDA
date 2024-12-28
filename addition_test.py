import numpy as np
import cupy as cp
from numba import cuda

# Device info
device = cuda.get_current_device()
max_threads_per_block = device.MAX_THREADS_PER_BLOCK
warp_size = device.WARP_SIZE
print(f'Amount of Streaming Multiprocessors: {device.MULTIPROCESSOR_COUNT}\nWarp size: {warp_size}\nMax threads per block: {max_threads_per_block}')

# Define the CUDA kernel
@cuda.jit
def add_number_kernel(data):
    idx = cuda.grid(1) # Get the thread's unique index
    if idx < data.size: # Ensure the index is within bounds
        data[idx] += 99999

# Create array
data = np.arange(1, 10001, dtype=np.int32)

# Allocate and transfer a numpy ndarray or structured scalar to the device
d_data = cuda.to_device(data)

# Set block size
threads_per_block = min(max_threads_per_block, warp_size * 2)
blocks_per_grid = (data.size + (threads_per_block - 1)) // threads_per_block

# Execution
add_number_kernel[blocks_per_grid, threads_per_block](d_data)
result = d_data.copy_to_host()
print(result)

# # Same operation using cupy

# data = cp.arange(1, 10001, dtype=cp.int32)

# data += 99999

# # Copy result back to CPU
# result = cp.asnumpy(data)

# print(result[:10])