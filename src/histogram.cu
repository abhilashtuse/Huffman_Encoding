#include "histogram.h"

#define TEN_KB 10250
#define TOTAL_CHARS 256

__constant__ char d_input_string_const[TEN_KB];

__global__ void histo_kernel(long size, unsigned int *histo)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(i < size) {
		atomicAdd( &(histo[(d_input_string_const[i])]), 1);
		i += stride;
	}
	__syncthreads();
}


void calculateFrequencies(char *char_array, int input_str_length, unordered_map<char, int> &freq) {
    //cout << "input:" << text << endl;
    cudaError_t err = cudaSuccess;

	// Allocate the host input matrix h_A
    int histo_size = TOTAL_CHARS * sizeof(int);
    unsigned int *h_histo = (unsigned int *)malloc(histo_size);

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
    err = cudaMalloc((void**)&d_histo, histo_size * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device matrix B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	err = cudaMemcpyToSymbol(d_input_string_const, char_array, input_str_length * sizeof(char));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy input string from host to device (error code %s)!\n", 	  cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	int blocksPerGrid = 7;// FILL HERE
    int threadsPerBlock = 128;// FILL HERE
	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

	cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

	cudaEventRecord(start);
	histo_kernel<<<blocksPerGrid, threadsPerBlock>>>(input_str_length, d_histo);
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
    err = cudaMemcpy(h_histo, d_histo, histo_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaThreadSynchronize();

    //Testing
    for(int i = 0; i < TOTAL_CHARS; i++){
		if (h_histo[i] != 0) {
			printf("\nindex: %c   Frequency: %d", i, h_histo[i]);
			freq.insert( std::pair<char,int>(i,h_histo[i]));
		}
    }

    cudaFree(d_histo);
    // Free host memory
	free(h_histo);
}
