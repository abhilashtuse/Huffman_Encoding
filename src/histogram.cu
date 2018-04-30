#include "histogram.h"

#define TOTAL_CHARS 256

__global__ void histo_kernel(char *buffer, long size, unsigned int *histo)
{

	//printf("\ninside histogram kernel function");
	__shared__ unsigned int histo_private[7];
	if (threadIdx.x < 7)
	{
	histo_private[threadIdx.x] = 0;
	//printf("\nMake private copies of histogram");
	}
	__syncthreads();

	int i = threadIdx.x + blockIdx.x * blockDim.x;

	__syncthreads();
	//printf("\nThread id:%d", i);
	// stride is total number of threads

	int stride = blockDim.x * gridDim.x;
	//printf("\nStride :%d", stride);
	while(i < size) {
		//printf("\n Filing histogram");
		//65 because we start with 'A'
		atomicAdd( &(histo[(buffer[i])]), 1);

		i += stride;
	}
	__syncthreads();

	if (threadIdx.x < 7) {
		atomicAdd(&(histo[threadIdx.x]), histo_private[threadIdx.x] );
	}
}

void calculateFrequencies(string &text, unordered_map<char, int> &freq) {
    //cout << "input:" << text << endl;
    cudaError_t err = cudaSuccess;
    int size = TOTAL_CHARS * sizeof(int);
    char char_array[text.length()];
    strcpy(char_array, text.c_str());

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
    err = cudaMalloc((void**)&d_histo, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device matrix B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Allocate the device output matrix
    char *d_char_array = NULL;
    err = cudaMalloc((void**)&d_char_array, text.length()* sizeof(char));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device matrix C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_histo, h_histo, size,cudaMemcpyHostToDevice);// FILL HERE
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	err = cudaMemcpy(d_char_array, char_array, text.length() * sizeof(char),cudaMemcpyHostToDevice);// FILL HERE
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
	histo_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_char_array, text.length(), d_histo);
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
    err = cudaMemcpy(h_histo, d_histo, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaThreadSynchronize();

    //Testing
    for(int i = 0; i < TOTAL_CHARS; i++){
      	//printf("\nindex: %d   Frequency: %d", i, h_histo[i]);
		if (h_histo[i] != 0)
			freq.insert( std::pair<char,int>(i,h_histo[i]));
    }

    cudaFree(d_histo);
    // Free host memory
	free(h_histo);
}
