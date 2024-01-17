#include <stdio.h>
#include <stdlib.h>

// CUDA Kernel for Vector Multiplication
__global__ void vectorMultiply(int *a, int *b, int *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] * b[idx];
    }
}

// Function to initialize the vector with random values
void initVector(int *vec, int N) {
    for (int i = 0; i < N; i++) {
        vec[i] = rand() % 10; // random int between 0 and 9
    }
}

// Function to print the vector
void printVector(int *vec, int N, const char* name) {
    printf("%s: ", name);
    for (int i = 0; i < N; i++) {
        printf("%d ", vec[i]);
    }
    printf("\n");
}

int main() {
    const int N = 10; // Reduced size for demonstration purposes
    size_t bytes = N * sizeof(int);

    // Allocate memory on the host
    int *h_a = (int*)malloc(bytes);
    int *h_b = (int*)malloc(bytes);
    int *h_c = (int*)malloc(bytes);

    // Initialize vectors with random values
    initVector(h_a, N);
    initVector(h_b, N);

    // Print initial vectors
    printVector(h_a, N, "Vector A");
    printVector(h_b, N, "Vector B");

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
    vectorMultiply<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

    // Copy the result back to the host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Print the result vector
    printVector(h_c, N, "Vector C (Result)");

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
