#include <stdio.h>
#include <cuda.h>

__global__ void uniqueId(int * arr){
    int blockId = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * (gridDim.x) + blockIdx.x; // from 1 grid POV: 3rd dim -> 2nd dim -> 1st dim
    int localThreadId = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * (blockDim.x) + threadIdx.x; // from 1 block POV: 3rd dim -> 2nd dim -> 1st dim

    int unique = blockId * (blockDim.x * blockDim.y * blockDim.z) + localThreadId; // from thread POV: gridPOV * across various blocks in the grid (skip the previous blocks) + block POV 
    
    printf( "Block(%d,%d,%d) Thread(%d,%d,%d) -> %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, unique);
}
int main(){
    dim3 grid(2, 2, 2);
    dim3 block(2, 2, 2);

    uniqueId<<<grid, block>>>(nullptr);

    cudaDeviceSynchronize();
    return 0;
}
