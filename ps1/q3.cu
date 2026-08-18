#include<stdio.h>
#include<cuda.h>
#include<string.h>

#define MAX_STRINGS 100
#define MAX_LEN 100

__global__ void printTheString(char * str, int numStrings){
    int blockId = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * (gridDim.x) + blockIdx.x;
    int threadId = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * (blockDim.x) + threadIdx.x;
    int id = blockId * (blockDim.x * blockDim.y * blockDim.z) + threadId;

    if(id >= numStrings){
        return ;
    }
    char * curr = str + (id * MAX_LEN);
    printf(" ID: %d string: %s",id,curr);
}

int main(){
    int numStrings;

    printf("Enter the number of strings: ");
    scanf("%d", &numStrings);

    char h_strings[MAX_STRINGS][MAX_LEN];

    printf("Enter the strings:\n");

    for(int i = 0; i < numStrings; i++){
        scanf("%99s", h_strings[i]);
    }

    char *d_strings;

    cudaMalloc(&d_strings,numStrings * MAX_LEN * sizeof(char));

    cudaMemcpy(d_strings,h_strings,numStrings * MAX_LEN * sizeof(char),cudaMemcpyHostToDevice);

    int threads = (numStrings < 256) ? numStrings : 256;
    int blocks = ceil((float)numStrings/threads);

    printTheString<<<blocks, threads>>>(d_strings,numStrings);
    //printTheString<<<1, numStrings>>>(d_strings, numStrings);


    cudaDeviceSynchronize();
    return 0;
}

