#include<stdio.h>
#include<cuda.h>
#define ROWS 3
#define COLS 4


__global__ void per_row_column_kernel(int * matrixA, int * matrixB, int * matrixOutput, int rows, int cols){
    //to implement A + B (T)
    //1d grid && 1d block
    // 1 row 1 thread 
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    
    for(int j = 0; j < cols; j++){
        matrixOutput[x * cols + j] = 0;
        for(int k = 0; k < rows; k++){

          int left = matrixA[x * cols + k ] + matrixB[k * rows + x];
          int right = matrixB[j * rows + k] - matrixA[k * cols + j];
          
          matrixOutput[x * cols + j] += left * right;
        }
    }
}
int main()
{
    // A is 3 x 4
    int cpu_A[ROWS][COLS] = {
        {6,18,-9,9},
        {-5,13,16,-8},
        {-6,3,-7,-10}
    };

    // B is 4 x 3
    int cpu_B[COLS][ROWS] = {
        {-3,8,10},
        {-3,-9,13},
        {13,-2,10},
        {14,19,120}
    };

    int cpu_Output[ROWS][COLS];
    

    int * gpu_A;
    int * gpu_B;
    int * gpu_Output;

    cudaMalloc(&gpu_A, sizeof(cpu_A)); 
    cudaMalloc(&gpu_B, sizeof(cpu_B));
    cudaMalloc(&gpu_Output, sizeof(cpu_Output));
    
    cudaMemset(gpu_Output, 0, sizeof(cpu_Output));
    
    cudaMemcpy(gpu_A,cpu_A,sizeof(cpu_A),cudaMemcpyHostToDevice); 
    cudaMemcpy(gpu_B,cpu_B,sizeof(cpu_B),cudaMemcpyHostToDevice);

    per_row_column_kernel<<<1,ROWS>>>(gpu_A,gpu_B,gpu_Output,ROWS,COLS);

    cudaMemcpy(cpu_Output, gpu_Output,sizeof(cpu_Output), cudaMemcpyDeviceToHost);

    printf("Output:\n");

    for(int i=0;i<ROWS;i++)
    {
        for(int j=0;j<COLS;j++)
            printf("%u ", cpu_Output[i][j]);

        printf("\n");
    }
}
