#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Check for CUDA errors
void checkCudaError(cudaError_t error, const char *functionName) {
    if (error != cudaSuccess) {
        fprintf(stderr, "Error in %s: %s\n", functionName, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

// ReLU Activation Function Kernel
__global__ void reluActivation(float *input, float *output, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        output[index] = fmaxf(0.0f, input[index]);
    }
}

// ... (Other activation functions remain the same)

// Function to initialize the vector with random values
void initVector(float *vec, int N) {
    for (int i = 0; i < N; i++) {
        vec[i] = rand() % 10 - 5; // random float between -5 and 5
    }
}

int main() {
    const int N = 1024;
    size_t bytes = N * sizeof(float);

    float *h_input = (float*)malloc(bytes);
    float *h_output = (float*)malloc(bytes);
    initVector(h_input, N);

    float *d_input, *d_output;
    printf("Allocating device memory...\n");
    checkCudaError(cudaMalloc(&d_input, bytes), "cudaMalloc d_input");
    checkCudaError(cudaMalloc(&d_output, bytes), "cudaMalloc d_output");

    printf("Copying data from host to device...\n");
    checkCudaError(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice), "cudaMemcpy host to device");

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    printf("Launching ReLU activation kernel...\n");
    reluActivation<<<gridSize, blockSize>>>(d_input, d_output, N);
    checkCudaError(cudaGetLastError(), "ReLU Kernel Execution");

    // ... (Similarly for other activation functions)

    printf("Copying result back to host...\n");
    checkCudaError(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy device to host");

    // ... (Print output or process it as needed)

    printf("Freeing device memory...\n");
    cudaFree(d_input);
    cudaFree(d_output);

    free(h_input);
    free(h_output);

    printf("Program completed successfully.\n");
    return 0;
}
