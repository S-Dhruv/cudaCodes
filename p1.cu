#include<stdio.h>
#include<cuda.h>
#define ROWS 3
#define COLS 4


__global__ void per_row_column_kernel(int * matrixA, int * matrixB, int * matrixOutput, int rows, int cols){
    //to implement A + B (T)
    //1d grid && 1d block
    // 1 row 1 thread 
    int x = blockIdx.x * blockDim.x + threadIdx.x;
   
    if(x >= rows) return;

    for(int j = 0; j < cols; j++){
        matrixOutput[x * cols + j] = 0;
        for(int k = 0; k < cols; k++){

          int left = matrixA[x * cols + k] + matrixB[k * rows + x];
          int right = matrixB[j * rows + k] - matrixA[k * cols + j];
          
          matrixOutput[x * cols + j] += left * right;
        }
    }
}

__global__ void per_column_row_kernel(int * matrixA, int * matrixB, int * matrixOutput, int rows, int cols){ 
      int threadBlock = blockDim.x * blockDim.y;
      int localThread = threadIdx.y * blockDim.x + threadIdx.x;

      int id = blockIdx.x * threadBlock + localThread; // blockId * 128 (block Dim) + y-axis * skip over + x-axis
      
      if(id >= cols) return;

      for(int i = 0; i < rows;i++){
          matrixOutput[i * cols + id] = 0;
          for(int k = 0; k < cols ; k++){
            
            int left = matrixA[i * cols + k] + matrixB[k * rows + i];
            int right = matrixB[id * rows + k] - matrixA[k * cols + id];
                   
            matrixOutput[i * cols + id] += left * right;
          }
      }
}

__global__ void per_element_kernel(int * matrixA, int * matrixB, int * matrixOutput, int rows, int cols){
        int globalCol = blockIdx.x * blockDim.x + threadIdx.x;
        int globalRow = blockIdx.y * blockDim.y + threadIdx.y;
        
        if(globalRow >= rows || globalCol >= cols) return;

        int sum = 0;

        for(int i = 0 ; i < cols ; i++){

          int left = matrixA[globalRow * cols + i] + matrixB[i * rows + globalRow];
          int right = matrixB[globalCol * rows + i] - matrixA[i * cols + globalCol];
          
          sum+= left * right;
        }
        matrixOutput[globalRow * cols + globalCol] = sum;
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
        {14,16,19}      // use the assignment values
    };

    int cpu_Output[ROWS][COLS];

    int *gpu_A;
    int *gpu_B;
    int *gpu_Output;

    cudaMalloc(&gpu_A, sizeof(cpu_A));
    cudaMalloc(&gpu_B, sizeof(cpu_B));
    cudaMalloc(&gpu_Output, sizeof(cpu_Output));

    cudaMemcpy(gpu_A, cpu_A, sizeof(cpu_A), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_B, cpu_B, sizeof(cpu_B), cudaMemcpyHostToDevice);

    //------------------------------------------
    // Per Row - Per Column Kernel
    //------------------------------------------

    cudaMemset(gpu_Output, 0, sizeof(cpu_Output));

    per_row_column_kernel<<<1, ROWS>>>(
        gpu_A,
        gpu_B,
        gpu_Output,
        ROWS,
        COLS);

    cudaDeviceSynchronize();

    cudaMemcpy(cpu_Output,
               gpu_Output,
               sizeof(cpu_Output),
               cudaMemcpyDeviceToHost);

    printf("\nPer Row-Column Kernel:\n");

    for(int i = 0; i < ROWS; i++)
    {
        for(int j = 0; j < COLS; j++)
            printf("%d ", cpu_Output[i][j]);
        printf("\n");
    }

    //------------------------------------------
    // Per Column - Per Row Kernel
    //------------------------------------------

    cudaMemset(gpu_Output, 0, sizeof(cpu_Output));

    dim3 block2(2,2);

    int totalThreads = block2.x * block2.y;

    dim3 grid2((COLS + totalThreads - 1) / totalThreads);

    per_column_row_kernel<<<grid2, block2>>>(
        gpu_A,
        gpu_B,
        gpu_Output,
        ROWS,
        COLS);

    cudaDeviceSynchronize();

    cudaMemcpy(cpu_Output,
               gpu_Output,
               sizeof(cpu_Output),
               cudaMemcpyDeviceToHost);

    printf("\nPer Column-Row Kernel:\n");

    for(int i = 0; i < ROWS; i++)
    {
        for(int j = 0; j < COLS; j++)
            printf("%d ", cpu_Output[i][j]);
        printf("\n");
    }

    //------------------------------------------
    // Per Element Kernel
    //------------------------------------------

    cudaMemset(gpu_Output, 0, sizeof(cpu_Output));

    dim3 block3(16,16);

    dim3 grid3(
        (COLS + block3.x - 1) / block3.x,
        (ROWS + block3.y - 1) / block3.y);

    per_element_kernel<<<grid3, block3>>>(
        gpu_A,
        gpu_B,
        gpu_Output,
        ROWS,
        COLS);

    cudaDeviceSynchronize();

    cudaMemcpy(cpu_Output,
               gpu_Output,
               sizeof(cpu_Output),
               cudaMemcpyDeviceToHost);

    printf("\nPer Element Kernel:\n");

    for(int i = 0; i < ROWS; i++)
    {
        for(int j = 0; j < COLS; j++)
            printf("%d ", cpu_Output[i][j]);
        printf("\n");
    }

    cudaFree(gpu_A);
    cudaFree(gpu_B);
    cudaFree(gpu_Output);

    return 0;
}
