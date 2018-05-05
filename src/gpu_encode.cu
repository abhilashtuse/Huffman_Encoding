#include <iostream>
#include <fstream>

#define TEN_KB 10250
#define TOTAL_CHARS 256
#define MAX_CODE_WIDTH 8 //pow(2, height(root)); //Worst case height of tree when all chars have same frequency

__constant__ char d_tree_arr_const[TEN_KB]; // Device side tree in array form
__constant__ char d_input_string_const[TEN_KB];
__device__ char d_map_table[TOTAL_CHARS * MAX_CODE_WIDTH];

__global__ void generateEncodedStringKernel()
{
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    __shared__ char d_encode_table_shared[(TOTAL_CHARS * MAX_CODE_WIDTH) / 4];
    for (int i = 0; i < MAX_CODE_WIDTH; i++) {
        d_encode_table_shared[threadIdx.x * MAX_CODE_WIDTH + i] = d_map_table[d_input_string_const[tid] * MAX_CODE_WIDTH + i];
    }
    __syncthreads();
    if (tid == 0) {
        printf("In concat kernel:");
        for (int i = 0; i < MAX_CODE_WIDTH; i++) {
            printf("%c", d_encode_table_shared[threadIdx.x * MAX_CODE_WIDTH + i]);
        }
        printf("\n");
    }

    /*if (threadIdx.x == 0) {
        for (int j = 0; j < 32; j++) {
            for (int i = 0; i < MAX_CODE_WIDTH; i++) {
                encoded_string[j * MAX_CODE_WIDTH + i] = d_encode_table_shared[d_input_string_const[j] * MAX_CODE_WIDTH + i];
            }
        }
    }
    __syncthreads();*/
}

void generateEncodedString()
{
    int blocksPerGrid = 8; //(text.length() / 1024) + 1;
    int threadsPerBlock = 32; //1024;// FILL HERE
	printf("CUDA encode kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	cudaEventRecord(start);
    //printf("Before generateEncodedStringKernel");

    generateEncodedStringKernel<<<blocksPerGrid, threadsPerBlock>>>();

    //printf("After generateEncodedStringKernel");
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float elapsed = 0;
	cudaEventElapsedTime(&elapsed, start, stop);
	printf("The elapsed time for encode kernal exexution is %.2f ms\n", elapsed);
    cudaEventDestroy (start);
    cudaEventDestroy (stop);
    cudaThreadSynchronize();
}

__global__ void encode_kernel()
{
  int threadId = threadIdx.x;
  int j =0;
  __shared__ char str[MAX_CODE_WIDTH];
  if(d_tree_arr_const[threadId] != '$' && d_tree_arr_const[threadId] != '*') {
      for(int i = threadId ;i >0 ;) {
          if(i % 2 == 0){
            i = (i - 2)/ 2;
              str[j] = '1';

          }
          else{
            i = (i - 1)/ 2;
              str[j] = '0';

          }
          j++;
          str[j] = '\0';
      }
      int k = 0;

      //printf("\nCharacter :%c Encoding :", d_tree_arr_const[threadId]);
      #pragma unroll
      for (k = 0; k < MAX_CODE_WIDTH && str[k] != '\0' && (j-k-1) >= 0; k++) {
        d_map_table[d_tree_arr_const[threadId]*8 + k] = str[j-k -1];
        //printf("%c", d_map_table[d_tree_arr_const[threadId]*8 + k]);
      }
      d_map_table[d_tree_arr_const[threadId]*8 + k] = '\0';
    }
    __syncthreads();
}

void gpu_encode(char *h_tree_arr, int h_tree_arr_length, char *input_str_array, int input_str_array_length) {
    cudaError_t err = cudaSuccess;
    err = cudaMemcpyToSymbol(d_tree_arr_const, h_tree_arr, h_tree_arr_length * sizeof(char));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy tree array from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpyToSymbol(d_input_string_const, input_str_array, input_str_array_length * sizeof(char));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy input string from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaEvent_t start, stop;
    float elapsed = 0;
    int blocksPerGrid = 1;//(text.length() / 1024) + 1;
    int threadsPerBlock = 32;//1024;// FILL HERE
    printf("CUDA encode kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 9);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  	cudaEventRecord(start);

    encode_kernel<<<blocksPerGrid, threadsPerBlock>>>();

  	cudaEventRecord(stop);
  	cudaEventSynchronize(stop);
  	cudaEventElapsedTime(&elapsed, start, stop);
  	printf("The elapsed time for Encode kernal excution is %.2f ms\n", elapsed);
    cudaEventDestroy (start);
    cudaEventDestroy (stop);
    cudaThreadSynchronize();

#if 0
    printf("Final Encoded Table\n");
    //printf("\nCHAR:%c ENCODEDING:%s",h_tree_arr[4], h_d_map_table[97*MAX_CODE_WIDTH]);
    for (int i = 0; i < MAX_NODES ; i++) {
      if(h_tree_arr[i] != '$' && h_tree_arr[i] != '*' && h_tree_arr[i] != '\0') {
        printf("\nchar:%c:", h_tree_arr[i]);
        for (int j = 0; j < 8; j++) {
          char tmp = h_d_map_table[h_tree_arr[i]*MAX_CODE_WIDTH + j];
          if (tmp != '\0')
            printf("%c", tmp);
        }
      }
    }
    printf("\n\n");
#endif

    generateEncodedString();
    //cout << "\nOriginal string was :\n" << text << '\n';
    // print encoded string
    /*string str = "";
    for (char ch: text) {
        for (int i = 0; h_d_map_table[(ch * MAX_CODE_WIDTH) + i] != '\0'; i++)
            str += h_d_map_table[ch * MAX_CODE_WIDTH + i];
    }
    cout << "\nEncoded string is :\n" << str << '\n';*/
}
