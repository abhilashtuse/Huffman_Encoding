#include <iostream>
#include <fstream>
#include <string>
#include "binary_tree.h"

using namespace std;

extern __constant__ char d_tree_arr_const[];

__device__ void append(char *s, char c, int position)
{
	s[position] = c;
	s[position +1] = '\0';
}

__device__ int string_count = 0;

__global__
void decode_kernel(int k, int count)
{
  /*if(threadIdx.x == 0) {
    cudaStream_t s1, s2;
    //unsigned int flag = cudaStreamDefault;
    cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
    int l = 2*k+1;
    int r = 2*k+2;
    if (d_tree_arr[l] == '$' && d_tree_arr[r] == '$')
    {
        //printf("\nDecoded CHAR:%c",d_tree_arr[k]);
        string_count = count;
    }
    if (encoded_str[count] == '0' && d_tree_arr[l] != '$' ){
      decode_kernel<<<1,1,0, s1>>>(l, count + 1);
      cudaDeviceSynchronize();
    }

    if (encoded_str[count] == '1' && d_tree_arr[r] != '$') {
      //printf("got right 1 count:%d\n", count);
      //count++;

      decode_kernel<<<1,1,0, s1>>>( r, count + 1);
      cudaDeviceSynchronize();
      //cudaStreamDestroy(s2);
    }
  }
  __syncthreads();*/
}

__global__
void decode_parent_kernel(int string_size) {

  if(threadIdx.x == 0){
    while (string_count < string_size) {
      decode_kernel<<<1,1,0>>>(0, string_count);
      cudaDeviceSynchronize();
    }
  }
  __syncthreads();
}

void gpu_decode(int input_str_length) {
	cudaError_t err = cudaSuccess;
    char h_final_str[input_str_length];
    char *d_final_str = NULL;
    err = cudaMalloc((void**)&d_final_str, input_str_length* sizeof(char));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device input string (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
		#if 0
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

  	cudaEventRecord(start);
    int blocksPerGrid = 8; //(input_str_length / 1024) + 1;
    int threadsPerBlock = 32; //1024;// FILL HERE
    decode_parent_kernel<<<blocksPerGrid, threadsPerBlock>>>(input_str_length);
  	cudaEventRecord(stop);

  	cudaEventSynchronize(stop);

    float elapsed = 0;
  	cudaEventElapsedTime(&elapsed, start, stop);
  	//printf("The elapsed time for Decode kernal excution is %.2f ms\n", elapsed);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaThreadSynchronize();
		#endif
    err = cudaMemcpy(h_final_str, d_final_str, input_str_length, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy final string from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaThreadSynchronize();
    //cudaFree(d_str);
    //cudaFree(d_encode_map);
    //cudaFree(d_input_string);
    cudaFree(d_final_str);
}

void cpu_decode(Node* root, int &index, string str)
{
    if (root == nullptr) {
        return;
    }

    // found a leaf node
    if (!root->left && !root->right)
    {
      //  cout << root->ch;
        return;
    }

    index++;

    if (str[index] =='0')
        cpu_decode(root->left, index, str);
    else
        cpu_decode(root->right, index, str);
}
