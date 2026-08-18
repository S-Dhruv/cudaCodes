#include <stdio.h>
#include <cuda.h>

__global__ void dkernel(int * matrix, int rows, int cols, int * result){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(id >= rows * cols) return ;

    int rowNumber = id/cols;
    int partOfUpper = cols - rowNumber - 1;

    if(rowNumber * cols + (cols - partOfUpper) <= id && id < ((rowNumber+1)*cols)){
        result[id] = matrix[id] == 0 ? 1 : -1 ;
    }
}

int main(){
    
    int rows = 3, cols = 3;
   int h_arr[rows][cols] = {
        {1, 0, 0},
        {2, 3, 0},
        {4, 5, 6}
    };

    int *d_arr, *d_res;
    int h_res[rows * cols];

    // fill the h_res with all 0s
    for (int i = 0; i < rows * cols; i++) {
        h_res[i] = 0;
    }

    cudaMalloc(&d_arr, rows * cols * sizeof(int));
    cudaMalloc(&d_res, rows * cols * sizeof(int));

    cudaMemcpy(
        d_arr,
        h_arr,
        rows * cols * sizeof(int),
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_res,
        h_res,
        rows * cols * sizeof(int),
        cudaMemcpyHostToDevice
    );

    dkernel<<<1, rows * cols>>>(d_arr, rows, cols, d_res);

    cudaMemcpy(
        h_res,
        d_res,
        rows * cols * sizeof(int),
        cudaMemcpyDeviceToHost
    );

    // loop over result if there is -1 then print that the matrix is not lower
    int isLower = 1;

    for (int i = 0; i < rows * cols; i++) {
        if (h_res[i] == -1) {
            isLower = 0;
            break;
        }
    }

    if (isLower)
        printf("Matrix is lower triangular\n");
    else
        printf("Matrix is not lower triangular\n");

    cudaFree(d_arr);
    cudaFree(d_res);

    return 0;}
