#include <iostream>
#include <fstream>

#define TEN_KB 10250
#define TOTAL_CHARS 256
#define MAX_CODE_WIDTH 12 //pow(2, height(root)); //Worst case height of tree when all chars have same frequency

__constant__ char d_tree_arr_const[TEN_KB]; // Device side tree in array form
__constant__ char d_input_string_const[TEN_KB];
__device__ char d_map_table[TOTAL_CHARS * MAX_CODE_WIDTH];
__global__ void encode_kernel(char *d_map_table_test, int partition_size, int partition_index)
{
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  int j = 0;
  char str[MAX_CODE_WIDTH];
  char temp = d_tree_arr_const[tid + partition_size * partition_index];
  //memset(str,'\0', MAX_CODE_WIDTH);
  //printf("\ntid: %d node:%c", threadIdx.x, d_tree_arr_const[threadIdx.x]);
  //if(temp == 'e'){
    if(temp != '$' && temp != '*') {
      for(int i = tid ;i >0 ;) {
        if(i % 2 == 0) {
          //printf("1");
          i = (i - 2)/ 2;
          str[j] = '1';
        }
        else {
          //printf("0");
          i = (i - 1)/ 2;
          str[j] = '0';
        }
        j++;
        str[j] = '\0';
      }

      int k = 0;
      for (k = 0; k < MAX_CODE_WIDTH && str[k] != '\0'; k++) {
        d_map_table_test[temp*8 + k] = str[k];
      }
      d_map_table_test[temp*8 + k] = '\0';
    }
  __syncthreads();
}

__device__ char d_encoded_string[TEN_KB * MAX_CODE_WIDTH];
#if 1
__global__ void generateEncodedStringKernel()
{

    int tid = threadIdx.x + (blockIdx.x * blockDim.x);

    //#pragma unroll
    for (int i = 0; i < MAX_CODE_WIDTH; i++) {
      d_encoded_string[tid * MAX_CODE_WIDTH + i] =
      d_map_table[d_input_string_const[tid] * MAX_CODE_WIDTH + i];
    }
    #if 0
    if(tid == 0){
      printf("\nEncoded String:%c",d_input_string_const[1]);

      for (int i = 0; i < MAX_CODE_WIDTH; i++) {

        printf("%c",d_map_table[d_input_string_const[1] * MAX_CODE_WIDTH + i]);
      }
      printf("\n");
      for (int j = 0; j < 10; j++) {
        for (int i = 0; i < MAX_CODE_WIDTH; i++) {
          printf("%c", d_encoded_string[j * MAX_CODE_WIDTH + i]);
        }
      }
    }
    #endif
    __syncthreads();
}


void generateEncodedString(int input_str_array_length)
{
  int blocksPerGrid = (input_str_array_length / 1024) + 1;
  int threadsPerBlock = 1024;// FILL HERE
	printf("CUDA generateEncodedStringKernel kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
	cudaEventRecord(start);

  generateEncodedStringKernel<<<blocksPerGrid, threadsPerBlock>>>();

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float elapsed = 0;
	cudaEventElapsedTime(&elapsed, start, stop);
	printf("The elapsed time for generateEncodedString kernal exexution is %.2f ms\n", elapsed);
  cudaEventDestroy (start);
  cudaEventDestroy (stop);
  cudaThreadSynchronize();

}
#endif

void gpu_encode(char *h_tree_arr, int h_tree_arr_length, char *input_str_array, int input_str_array_length) {
    cudaError_t err = cudaSuccess;
    //Create Streams
    int numStreams = h_tree_arr_length/2048 + 1;
    cudaStream_t stream[numStreams];
    for (int i=0; i<numStreams; i++){
       cudaStreamCreate(&stream[i]);
    }

    err = cudaMemcpyToSymbol(d_tree_arr_const, h_tree_arr, h_tree_arr_length * sizeof(char));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy tree array from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpyToSymbol(d_input_string_const, input_str_array, input_str_array_length );
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy input string from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Testing start
    char *h_map_table[numStreams];
    char* d_map_table_test[numStreams];
    int partition_size = (TOTAL_CHARS * MAX_CODE_WIDTH * sizeof(char))/numStreams;
    for (int i=0; i < numStreams; i++){
        h_map_table[i] = NULL;
        h_map_table[i] = (char *)malloc(partition_size);
        memset(h_map_table[i], '\0', partition_size);


        d_map_table_test[i] = NULL;
        err = cudaMalloc ((void**)&d_map_table_test[i],partition_size);
        if (err != cudaSuccess)
            {
              fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
              exit(EXIT_FAILURE);
             }

    }


    //testing end


    cudaEvent_t start, stop;
    float elapsed = 0;
    int blocksPerGrid = 2;
    int threadsPerBlock = 1024;// FILL HERE
    printf("CUDA encode kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 9);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  	cudaEventRecord(start);

    //encode_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_map_table_test);


    for (int i=0; i< numStreams ; i++){
      encode_kernel<<<blocksPerGrid, threadsPerBlock, 20480,stream[i]>>>(d_map_table_test[i], partition_size, i);
      err = cudaMemcpyAsync (h_map_table[i], d_map_table_test[i],(partition_size), cudaMemcpyDeviceToHost, stream[i] );
    }

  	cudaEventRecord(stop);
  	cudaEventSynchronize(stop);
  	cudaEventElapsedTime(&elapsed, start, stop);
  	printf("The elapsed time for Encode kernal excution is %.2f ms\n", elapsed);
    cudaEventDestroy (start);
    cudaEventDestroy (stop);
    cudaThreadSynchronize();


    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_map_table, d_map_table_test, TOTAL_CHARS * MAX_CODE_WIDTH * sizeof(char), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
    fprintf(stderr, "Failed to copy d_map_table from device to host (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
    }

    cudaThreadSynchronize();
    printf("final encode table\n");
    for(int j = 0; j < MAX_CODE_WIDTH; j++) {
      printf("%c", h_map_table[0]['e'*MAX_CODE_WIDTH+j]);
    }


    /*for (int i = 0; i < TOTAL_CHARS; i++) {
      printf("%c:", i);
      for(int j = 0; j < MAX_CODE_WIDTH; j++) {
        printf("%c", h_map_table[i*MAX_CODE_WIDTH+j]);
      }
      printf("\n");
    }*/



    generateEncodedString(input_str_array_length);
}
