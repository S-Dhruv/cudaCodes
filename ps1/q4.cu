#include<stdio.h>
#include<cuda.h>

__global__ void divergence(int d){
    int currThread = threadIdx.x % 32;

    if(currThread < d){
        int x =0;
        x += 1;
    }
    else{
        int y = 0;
        y -= 1;
    }
}
int main()
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threads = 32;
    int blocks = 1024;

    printf("d\tTime (ms)\n");
    printf("---------------------\n");

    for(int d = 0; d <= 32; d++)
    {
        cudaEventRecord(start);

        // Launch many times to get measurable timing
        for(int i = 0; i < 1000; i++)
        {
            divergence<<<blocks, threads>>>(d);
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start, stop);

        printf("%2d\t%.5f\n", d, milliseconds);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaDeviceReset();

    return 0;
}
