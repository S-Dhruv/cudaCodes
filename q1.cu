#include <stdio.h>
#include <cuda.h>

__global__ void memoryCoalescing(int input, int * arr){
    if(input < 0 || input > 31) return;

    int id = threadIdx.x;

    printf("\n %d accessing %d",id,arr[id * (32 - input)]);
}
int main(){

    int h_arr[1024];
    int * d_arr;
    
    for(int i = 0; i < 1024;i++){
        h_arr[i]=i;
    }

    int input;
    printf("Enter the coalescing degree: ");
    scanf(" %d", &input);

    cudaMalloc(&d_arr, 1024 * sizeof(int));
    cudaMemcpy(d_arr,h_arr, 1024 * sizeof(int), cudaMemcpyHostToDevice);
    
    memoryCoalescing<<<1,32>>>(input,d_arr);
    cudaDeviceSynchronize();

    return 0;
}
