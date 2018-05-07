#include <iostream>
#include <fstream>
#include <string>
#include "binary_tree.h"

using namespace std;

#define TEN_KB 10250
#define MAX_CODE_WIDTH 12 //pow(2, height(root)); //Worst case height of tree when all chars have same frequency

//extern __constant__ char d_tree_arr_const[TEN_KB];
extern __device__ char d_encoded_string[TEN_KB * MAX_CODE_WIDTH];

__device__ void append(char *s, char c, int position)
{
	s[position] = c;
	s[position +1] = '\0';
}

__device__ int string_count = 0;

__global__
void decode_kernel(int k, int count, char *d_tree_arr)
{
  if(threadIdx.x == 0) {
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
    if (d_encoded_string[count] == '0' && d_tree_arr[l] != '$' ){
      decode_kernel<<<1,1,0, s1>>>(l, count + 1, d_tree_arr);
      cudaDeviceSynchronize();
    }

    if (d_encoded_string[count] == '1' && d_tree_arr[r] != '$') {
      //printf("got right 1 count:%d\n", count);
      //count++;

      decode_kernel<<<1,1,0, s1>>>( r, count + 1, d_tree_arr);
      cudaDeviceSynchronize();
      //cudaStreamDestroy(s2);
    }
  }
  __syncthreads();
}

__global__
void decode_parent_kernel(char *d_tree_arr, int string_size) {

  if(threadIdx.x == 0){
    while (string_count < string_size) {
      decode_kernel<<<1,1,0>>>(0, string_count, d_tree_arr);
      cudaDeviceSynchronize();
    }
  }
  __syncthreads();
}

void gpu_decode(char *h_tree_arr, int h_tree_arr_length, int input_str_length) {
	cudaError_t err = cudaSuccess;
	char *d_tree_arr = NULL;
	err = cudaMalloc((void**)&d_tree_arr, h_tree_arr_length* sizeof(char));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device input string (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy(d_tree_arr, h_tree_arr, h_tree_arr_length * sizeof(char), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy tree array from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

    char h_final_str[input_str_length];
    char *d_final_str = NULL;
    err = cudaMalloc((void**)&d_final_str, input_str_length* sizeof(char));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device input string (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

  	cudaEventRecord(start);
    int blocksPerGrid = 8; //(input_str_length / 1024) + 1;
    int threadsPerBlock = 32; //1024;// FILL HERE
    decode_parent_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_tree_arr, input_str_length);
  	cudaEventRecord(stop);

  	cudaEventSynchronize(stop);

    float elapsed = 0;
  	cudaEventElapsedTime(&elapsed, start, stop);
  	printf("The elapsed time for Decode kernal excution is %.2f ms\n", elapsed);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaThreadSynchronize();

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
        cout << root->ch;
        return;
    }

    index++;

    if (str[index] =='0')
        cpu_decode(root->left, index, str);
    else
        cpu_decode(root->right, index, str);
}
