#include <stdio.h>
#include <cuda.h>
#include <math.h>
__global__ void addingVectors(int * arr1, int * arr2, int * arr3, int sizeOfArray1, int sizeOfArray2){
    int id = threadIdx.x;
    if (id >= sizeOfArray1 && id >= sizeOfArray2){
        return ;
    }
    else if(id >= sizeOfArray2){
        arr3[id] = arr1[id];
        return;
    }
    else if(id >= sizeOfArray1){
        arr3[id] = arr2[id];
        return ;
    }

    arr3[id] = arr1[id] + arr2[id];
}

int main(){
    
    int s1;
    printf("Enter size of array1");
    scanf(" %d", &s1);

    int s2;
    printf("Enter size of array2");
    scanf(" %d", &s2);
   
    int maxSize = (s1 > s2) ? s1 : s2;
    
    int arr1[s1], arr2[s2], arr3[maxSize];
    
    printf("Enter the elements of array 1:");
    for(int i = 0; i < s1; i++){
        scanf(" %d", arr1[i]);
    }
    
    printf("Enter the elements of array 2:");
    for(int i = 0; i < s2; i++){
        scanf(" %d", arr2[i]);
    }

    int *gArr1, *gArr2, * gArr3;

    cudaMalloc(&gArr1,s1 * sizeof(int));
    cudaMalloc(&gArr2,s2 * sizeof(int));
    cudaMalloc(&gArr3, maxSize * sizeof(int));

    cudaMemcpy(gArr1,arr1,s1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gArr2,arr2, s2 * sizeof(int), cudaMemcpyHostToDevice);

    int blocks = ceil((float) maxSize / 256);

    addingVectors<<<blocks,256>>>(gArr1,gArr2,gArr3,s1,s2);
    cudaDeviceSynchronize();
    cudaMemcpy(arr3,gArr3,maxSize * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i < maxSize ; i++){ 
        printf(" %d", arr3[i]);
        printf("");
    }

    return 0;
}
