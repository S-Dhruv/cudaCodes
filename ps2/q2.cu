#include<stdio.h>
#include<cuda.h>

#define N 1024
#define SIZE (N * N)
#define BYTES (SIZE * sizeof(int))

// Easy code, written only to understand how different approaches change the time taken to compute
// Output
// IJK: 141495.593750 ms
// IKJ: 96145.875000 ms
// JIK: 142146.781250 ms
// JKI: 143850.140625 ms
// KIJ: 96039.726562 ms -> fastest 
// KJI: 143988.875000 ms
__global__ void perm1(int * result, int * input1, int * input2, int rows, int cols){
    
    for(int i = 0; i < rows;i++){
        for(int j = 0; j < cols; j++){
            for (int k = 0 ; k < rows;k++){
                result[i * rows + j] += input1[i * rows + k] * input2[k * cols + j];
            }
        }
    }

}
__global__ void perm2(int * result, int * input1, int * input2, int rows, int cols){
    
    for(int i = 0; i < rows;i++){
        for (int k = 0 ; k < rows;k++){
            for(int j = 0; j < cols; j++){
                result[i * rows + j] += input1[i * rows + k] * input2[k * cols + j];
            }
        }
    }

}
__global__ void perm3(int * result, int * input1, int * input2, int rows, int cols){
    
    for(int j = 0; j < cols; j++){
        for(int i = 0; i < rows;i++){
            for (int k = 0 ; k < rows;k++){
                result[i * rows + j] += input1[i * rows + k] * input2[k * cols + j];
            }
        }
    }
}

__global__ void perm4(int * result, int * input1, int * input2, int rows, int cols){
    
    for(int j = 0; j < cols; j++){
        for (int k = 0 ; k < rows;k++){
            for(int i = 0; i < rows;i++){
                result[i * rows + j] += input1[i * rows + k] * input2[k * cols + j];
            }
        }
    }

}
__global__ void perm5(int * result, int * input1, int * input2, int rows, int cols){
    
    for (int k = 0 ; k < rows;k++){
        for(int i = 0; i < rows;i++){
            for(int j = 0; j < cols; j++){ 
                result[i * rows + j] += input1[i * rows + k] * input2[k * cols + j];
            }
        }
    }

}

__global__ void perm6(int * result, int * input1, int * input2, int rows, int cols){
    
    for (int k = 0 ; k < rows;k++){
        for(int j = 0; j < cols; j++){
            for(int i = 0; i < rows;i++){
                result[i * rows + j] += input1[i * rows + k] * input2[k * cols + j];
            }
        }
    }

}
int main() {
    int *h_input1 = (int *)malloc(BYTES);
    int *h_input2 = (int *)malloc(BYTES);
    int *h_result = (int *)malloc(BYTES);

    int *d_input1;
    int *d_input2;
    int *d_result;

    // Initialize input matrices
    for (int i = 0; i < SIZE; i++) {
        h_input1[i] = 1;
        h_input2[i] = 1;
    }

    // Allocate GPU memory
    cudaMalloc(&d_input1, BYTES);
    cudaMalloc(&d_input2, BYTES);
    cudaMalloc(&d_result, BYTES);

    // Copy inputs to GPU
    cudaMemcpy(d_input1, h_input1, BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, h_input2, BYTES, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float milliseconds;
    
    cudaMemset(d_result, 0, BYTES);

    cudaEventRecord(start);
    perm1<<<1, 1>>>(d_result, d_input1, d_input2, N, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("IJK: %f ms\n", milliseconds);

    cudaMemset(d_result, 0, BYTES);

    cudaEventRecord(start);
    perm2<<<1, 1>>>(d_result, d_input1, d_input2, N, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("IKJ: %f ms\n", milliseconds);

    cudaMemset(d_result, 0, BYTES);

    cudaEventRecord(start);
    perm3<<<1, 1>>>(d_result, d_input1, d_input2, N, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("JIK: %f ms\n", milliseconds);

    cudaMemset(d_result, 0, BYTES);

    cudaEventRecord(start);
    perm4<<<1, 1>>>(d_result, d_input1, d_input2, N, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("JKI: %f ms\n", milliseconds);

    cudaMemset(d_result, 0, BYTES);

    cudaEventRecord(start);
    perm5<<<1, 1>>>(d_result, d_input1, d_input2, N, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("KIJ: %f ms\n", milliseconds);


    cudaEventRecord(start);
    perm6<<<1, 1>>>(d_result, d_input1, d_input2, N, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("KJI: %f ms\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
