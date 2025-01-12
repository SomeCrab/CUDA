import os
import shutil
from PIL import Image
import numpy as np
from numba import cuda
import time

start_exe = time.time()

@cuda.jit
def upscale_batch_kernel(batch_in, batch_out, scale):
    batch_idx, y, x = cuda.grid(3) # id img, y row, x column
    if batch_idx < batch_in.shape[0] and y < batch_in.shape[1] and x < batch_in.shape[2]:
        for c in range(batch_in.shape[3]):
            batch_out[batch_idx, y*scale, x*scale, c] = batch_in[batch_idx, y, x, c]
            batch_out[batch_idx, y*scale, x*scale+1, c] = batch_in[batch_idx, y, x, c]
            batch_out[batch_idx, y*scale+1, x*scale, c] = batch_in[batch_idx, y, x, c]
            batch_out[batch_idx, y*scale+1, x*scale+1, c] = batch_in[batch_idx, y, x, c]

def process_batch(images, scale=2):
    # Convert all images to a single numpy array
    batch_in = np.array([np.array(Image.open(img)) for img in images]).astype(np.float32)
    #print(batch_in[0][0][0][3])
    #print(batch_in.shape[1])
    #print('batch_in')
    
    # Allocate GPU memory for input and output batches
    batch_in_gpu = cuda.to_device(batch_in)
    batch_out_shape = (batch_in.shape[0], batch_in.shape[1] * scale, batch_in.shape[2] * scale, batch_in.shape[3])
    batch_out_gpu = cuda.device_array(shape=batch_out_shape, dtype=np.float32)

    # Configure the threads and blocks for the GPU
    threads_per_block = (16, 16)
    blocks_per_grid_x = (batch_in.shape[2] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (batch_in.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid_z = len(images)
    blocks_per_grid = (blocks_per_grid_z, blocks_per_grid_y, blocks_per_grid_x)

    start_exe = time.time() # timer for single batch
    # Launch the CUDA kernel
    upscale_batch_kernel[blocks_per_grid, threads_per_block](batch_in_gpu, batch_out_gpu, scale)
    
    # Copy results back to CPU
    batch_out = batch_out_gpu.copy_to_host().astype(np.uint8)
    print(f'Time taken for single batch: {int(time.time() - start_exe)}')

    # Save each image
    for i, img in enumerate(batch_out):
        Image.fromarray(img).save(os.path.join('output', f'upscaled_{os.path.basename(images[i])}'))
        shutil.move(images[i], os.path.join('done', os.path.basename(images[i])))

def main():
    input_folder = "input" # images to process
    os.makedirs('output', exist_ok=True)
    os.makedirs('done', exist_ok=True)

    images = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp'))]
    
    for i in range(0, len(images), 64):
        batch = images[i:i+64]
        process_batch(batch)

if __name__ == "__main__":
    main()
    print(f'Total time: {int(time.time() - start_exe)}')