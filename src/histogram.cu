#include "histogram.h"

#define TOTAL_CHARS 256

__global__ void histo_kernel(char *buffer, long size, unsigned int *histo)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(i < size) {
		atomicAdd( &(histo[(buffer[i])]), 1);
		i += stride;
	}
	__syncthreads();
}

void calculateFrequencies(char *char_array, int input_str_length, unordered_map<char, int> &freq) {
    //cout << "input:" << text << endl;
    cudaError_t err = cudaSuccess;

	// Allocate the host input matrix h_A
    unsigned int *h_histo = (unsigned int *)malloc(TOTAL_CHARS * sizeof(int));

    // Verify that allocations succeeded
    if (h_histo == NULL)
    {
        fprintf(stderr, "Failed to allocate host histograms!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input matrices
    for (int i = 0; i < TOTAL_CHARS; ++i)
    {
      h_histo[i] = 0;
    }
    unsigned int *d_histo =NULL;
    err = cudaMalloc((void**)&d_histo, input_str_length * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device matrix B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Allocate the device output matrix
    char *d_char_array = NULL;
    err = cudaMalloc((void**)&d_char_array, input_str_length* sizeof(char));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device matrix C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_histo, h_histo, input_str_length * sizeof(int),cudaMemcpyHostToDevice);// FILL HERE
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	err = cudaMemcpy(d_char_array, char_array, input_str_length * sizeof(char),cudaMemcpyHostToDevice);// FILL HERE
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix B from host to device (error code %s)!\n", 	  cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	int blocksPerGrid = 7;// FILL HERE
    int threadsPerBlock = 128;// FILL HERE
	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

	cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

	cudaEventRecord(start);
	histo_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_char_array, input_str_length, d_histo);
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);

	float elapsed = 0;
	cudaEventElapsedTime(&elapsed, start, stop);
	printf("The elapsed time for histogram kernal exexution is %.2f ms\n", elapsed);

    cudaEventDestroy (start);
    cudaEventDestroy (stop);
    // <--

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch matrixMul kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	cudaThreadSynchronize();

    // Copy the device result matrix in device memory to the host result matrix
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_histo, d_histo, input_str_length * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaThreadSynchronize();

    //Testing
    for(int i = 0; i < TOTAL_CHARS; i++){
		if (h_histo[i] != 0) {
			//printf("\nindex: %c   Frequency: %d", i, h_histo[i]);
			freq.insert( std::pair<char,int>(i,h_histo[i]));
		}
    }

    cudaFree(d_histo);
    // Free host memory
	free(h_histo);
}
