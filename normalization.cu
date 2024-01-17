#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define VECTOR_SIZE 1024
#define DROPOUT_RATE 0.2  // 20% dropout

__global__ void normalize(float *input, float *output, int n, float min, float max) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        output[index] = (input[index] - min) / (max - min);
    }
}

__global__ void dropout(float *input, float *output, int n, float rate) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        float randVal = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        output[index] = randVal < rate ? 0 : input[index];
    }
}

// Helper function to initialize vector with random values
void initVector(float *vec, int N) {
    for (int i = 0; i < N; ++i) {
        vec[i] = rand() % 100; // Random float between 0 and 99
    }
}

int main() {
    float *h_input = (float*)malloc(VECTOR_SIZE * sizeof(float));
    float *h_output = (float*)malloc(VECTOR_SIZE * sizeof(float));
    initVector(h_input, VECTOR_SIZE);

    float *d_input, *d_output;
    cudaMalloc(&d_input, VECTOR_SIZE * sizeof(float));
    cudaMalloc(&d_output, VECTOR_SIZE * sizeof(float));

    cudaMemcpy(d_input, h_input, VECTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Assuming min and max are known. In practice, they should be computed from the data.
    float min = 0.0f, max = 99.0f;

    normalize<<<(VECTOR_SIZE + 255) / 256, 256>>>(d_input, d_output, VECTOR_SIZE, min, max);

    // Implementing dropout - only for demonstration, not suitable for training as it does not maintain state.
    dropout<<<(VECTOR_SIZE + 255) / 256, 256>>>(d_input, d_output, VECTOR_SIZE, DROPOUT_RATE);

    cudaMemcpy(h_output, d_output, VECTOR_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Free resources
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
