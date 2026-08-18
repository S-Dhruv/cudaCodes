#include <stdio.h>
#include <cuda.h>
#define N 8

struct Point {
    int *x;
    int *y;
};

__global__ void compute(int * x, int * y, int * globalCount, int * globalSum){

    int tid = (blockIdx.x * blockDim.x + threadIdx.x); 
    int id = tid * 4;

    int avgSum = 0;
    
    if(id >= N) return ;

    int end = id + 4;
    if (end > N) end = N;

    // for 4 elements of a 2d array == 8 elements of a 1d array
    for(int i = id; i < end; i++){
        avgSum += x[i];
    }
    avgSum /= 4;
    printf("Average Sum: %d\n",avgSum);

    //Checking if the y value is greater than avgSum, if it is, changing the value
    bool flag = false;
    for(int i = id; i < end ;i++){
        if(y[i] > avgSum){
            flag = true;
        }
    }

    //If none of the y values are greater than avgSum, add to globalSum
    if(!flag){
        int localSum = 0;
        for(int i = id; i < end ; i++){
            localSum += y[i];
        }
        atomicAdd(globalSum,localSum);
    }
    else{
        atomicAdd(globalCount,4);
        for(int i = id ; i < end ; i++){
            y[i] = avgSum;
        }
    }
}


int main(){
   int h_x[N] = {
        10, 20, 30, 40,
        5, 10, 15, 20
    };

    int h_y[N] = {
        2, 4, 6, 50,
        2, 4, 6, 8
    };

    int globalCount = 0;
    int globalSum = 0;

    int *d_x;
    int *d_y;
    int *d_count;
    int *d_sum;

    // Allocate GPU memory
    cudaMalloc((void**)&d_x, N * sizeof(int));
    cudaMalloc((void**)&d_y, N * sizeof(int));
    cudaMalloc((void**)&d_count, sizeof(int));
    cudaMalloc((void**)&d_sum, sizeof(int));

    // Copy x and y arrays to GPU
    cudaMemcpy(d_x, h_x, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize GPU counter and sum
    cudaMemcpy(d_count, &globalCount, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum, &globalSum, sizeof(int), cudaMemcpyHostToDevice);

    printf("Before kernel:\n");

    for (int i = 0; i < N; i++) {
        printf("Point %d: x = %d, y = %d\n", i, h_x[i], h_y[i]);
    }

    // 2 threads, each handles 4 elements
    compute<<<1, 2>>>(d_x, d_y, d_count, d_sum);

    cudaDeviceSynchronize();

    // Copy results back
    cudaMemcpy(h_y, d_y, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemcpy(&globalCount, d_count, sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaMemcpy(&globalSum, d_sum, sizeof(int),
               cudaMemcpyDeviceToHost);

    printf("\nAfter kernel:\n");

    for (int i = 0; i < N; i++) {
        printf("Point %d: x = %d, y = %d\n", i, h_x[i], h_y[i]);
    }

    printf("\nGlobal Count: %d\n", globalCount);
    printf("Global Sum: %d\n", globalSum);

    // Free GPU memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_count);
    cudaFree(d_sum);

    return 0;
}
