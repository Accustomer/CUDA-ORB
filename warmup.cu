#include "warmup.h"
#include "cuda_utils.h"
#include <device_launch_parameters.h>


__global__ void gpuAdd(float* c, float* a, float* b, const int size)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < size)
    {
        c[tid] = a[tid] * b[tid] + 6.f;
        tid += gridDim.x * blockDim.x;
    }
}



void warmup()
{
    std::cout << "===== Warming Up =====" << std::endl;
    const int n = 1 << 16;
    const size_t nbytes = n * sizeof(float);

    float* h_data1 = (float*)malloc(nbytes);
    float* h_data2 = (float*)malloc(nbytes);
    for (int i = 0; i < n; i++)
    {
        h_data1[i] = (rand() % 100000) * 0.12345f;
        h_data2[i] = (rand() % 100000) * 0.054845f;
    }    

    float* d_data1 = NULL;
    float* d_data2 = NULL;
    float* d_data3 = NULL;
    CHECK(cudaMalloc((void**)&d_data1, nbytes));
    CHECK(cudaMalloc((void**)&d_data2, nbytes));
    CHECK(cudaMalloc((void**)&d_data3, nbytes));
    CHECK(cudaMemcpy(d_data1, h_data1, nbytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_data2, h_data2, nbytes, cudaMemcpyHostToDevice));
    
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    gpuAdd << <grid, block >> > (d_data3, d_data1, d_data2, n);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaFree(d_data1));
    CHECK(cudaFree(d_data2));
    CHECK(cudaFree(d_data3));
    free(h_data1);
    free(h_data2);
}