#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define IMAGE_WIDTH  5
#define IMAGE_HEIGHT 5
#define FILTER_WIDTH  3
#define FILTER_HEIGHT 3

// CUDA Kernel for 2D Convolution
__global__ void convolution2D(int *image, int *filter, int *result, int imageWidth, int imageHeight) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < imageHeight && col < imageWidth) {
        int sum = 0;
        int startRow = row - FILTER_HEIGHT / 2;
        int startCol = col - FILTER_WIDTH / 2;

        for (int i = 0; i < FILTER_HEIGHT; i++) {
            for (int j = 0; j < FILTER_WIDTH; j++) {
                int curRow = startRow + i;
                int curCol = startCol + j;

                if (curRow >= 0 && curRow < imageHeight && curCol >= 0 && curCol < imageWidth) {
                    sum += filter[i * FILTER_WIDTH + j] * image[curRow * imageWidth + curCol];
                }
            }
        }
        result[row * imageWidth + col] = sum;
    }
}

// Function to initialize the matrix with random values
void initMatrix(int *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = rand() % 10; // random int between 0 and 9
    }
}

int main() {
    const int imageSize = IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(int);
    const int filterSize = FILTER_WIDTH * FILTER_HEIGHT * sizeof(int);

    int h_image[IMAGE_WIDTH * IMAGE_HEIGHT];
    int h_filter[FILTER_WIDTH * FILTER_HEIGHT];
    int h_result[IMAGE_WIDTH * IMAGE_HEIGHT] = {0};

    // Initialize image and filter with random values
    initMatrix(h_image, IMAGE_WIDTH, IMAGE_HEIGHT);
    initMatrix(h_filter, FILTER_WIDTH, FILTER_HEIGHT);

    // Allocate memory on the device
    int *d_image, *d_filter, *d_result;
    cudaMalloc(&d_image, imageSize);
    cudaMalloc(&d_filter, filterSize);
    cudaMalloc(&d_result, imageSize);

    // Copy data from host to device
    cudaMemcpy(d_image, h_image, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, filterSize, cudaMemcpyHostToDevice);

    // Define block size and grid size
    dim3 blockSize(16, 16);
    dim3 gridSize((IMAGE_WIDTH + blockSize.x - 1) / blockSize.x, (IMAGE_HEIGHT + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    convolution2D<<<gridSize, blockSize>>>(d_image, d_filter, d_result, IMAGE_WIDTH, IMAGE_HEIGHT);

    // Copy the result back to the host
    cudaMemcpy(h_result, d_result, imageSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_image);
    cudaFree(d_filter);
    cudaFree(d_result);

    // Print the result (optional)
    printf("Result:\n");
    for (int i = 0; i < IMAGE_HEIGHT; i++) {
        for (int j = 0; j < IMAGE_WIDTH; j++) {
            printf("%d ", h_result[i * IMAGE_WIDTH + j]);
        }
        printf("\n");
    }

    return 0;
}
