#include <stdio.h>
#include <stdlib.h>

// CUDA Kernel for Vector Addition
__global__ void vectorAdd(int *a, int *b, int *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

// Function to initialize the vector with random values
void initVector(int *vec, int N) {
    for (int i = 0; i < N; i++) {
        vec[i] = rand() % 100; // random int between 0 and 99
    }
}

int main() {
    const int N = 1024; // Size of the vectors
    size_t bytes = N * sizeof(int);

    // Allocate memory on the host
    int *h_a = (int*)malloc(bytes);
    int *h_b = (int*)malloc(bytes);
    int *h_c = (int*)malloc(bytes);

    // Initialize vectors with random values
    initVector(h_a, N);
    initVector(h_b, N);

    // Allocate memory on the device
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Number of threads in each thread block
    int blockSize = 256;

    // Number of thread blocks in grid
    int gridSize = (int)ceil((float)N / blockSize);

    // Execute the kernel
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

    // Copy the result back to the host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Verification of the result (optional)
    bool success = true;
    for(int i = 0; i < N; i++) {
        if(h_a[i] + h_b[i] != h_c[i]) {
            printf("Error: Element %d did not match! (%d + %d != %d)\n", i, h_a[i], h_b[i], h_c[i]);
            success = false;
            break;
        }
    }
    if(success) {
        printf("Vector addition successful!\n");
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
