#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include "binary_tree.h"

using namespace std;

#define TEN_KB 10250
#define TOTAL_CHARS 256
#define MAX_CODE_WIDTH 12 //pow(2, height(root)); //Worst case height of tree when all chars have same frequency

__constant__ char d_tree_arr_const[TEN_KB]; // Device side tree in array form
extern __constant__ char d_input_string_const[TEN_KB];
__constant__ char d_map_table[TOTAL_CHARS * MAX_CODE_WIDTH];
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
__global__ void generateEncodedStringKernel()
{
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);

    //#pragma unroll
    for (int i = 0; i < MAX_CODE_WIDTH; i++) {
      d_encoded_string[tid * MAX_CODE_WIDTH + i] =
      d_map_table[d_input_string_const[tid] * MAX_CODE_WIDTH + i];
    }
    __syncthreads();
#if 0
    if (tid == 0) {
        printf("Final encoded string:\n");
        for (int j = 0; j < 50; j++) {
            //printf("\nchar: %c code:", d_input_string_const[j]);
            for (int i = 0; i < MAX_CODE_WIDTH; i++) {
                if(d_encoded_string[j * MAX_CODE_WIDTH + i] == '0' || d_encoded_string[j * MAX_CODE_WIDTH + i] == '1' )
                printf("%c", d_encoded_string[j * MAX_CODE_WIDTH + i]);
            }
        }
    }
#endif
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

// traverse the Huffman Tree and store Huffman Codes
// in a map.
void cpu_encode(Node* root, string str, unordered_map<char, string> &huffmanCode)
{
    if (root == nullptr)
        return;

    // found a leaf node
    if (!root->left && !root->right) {
        huffmanCode[root->ch] = str;
    }

    cpu_encode(root->left, str + "0", huffmanCode);
    cpu_encode(root->right, str + "1", huffmanCode);
}

void gpu_encode(char *h_tree_arr, int h_tree_arr_length, char *input_str_array, int input_str_array_length, Node *root) {
    cudaError_t err = cudaSuccess;
    //Create Streams
    int numStreams = h_tree_arr_length/2048 + 1;
    cudaStream_t stream[numStreams];
    for (int i=0; i<numStreams; i++){
       cudaStreamCreate(&stream[i]);
    }

    /*err = cudaMemcpyToSymbol(d_tree_arr_const, h_tree_arr, h_tree_arr_length * sizeof(char));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy tree array from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }*/
    err = cudaMemcpyToSymbol(d_input_string_const, input_str_array, input_str_array_length );
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy input string from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    char *h_map_table = NULL;
    int table_size = TOTAL_CHARS*MAX_CODE_WIDTH*sizeof(char);
    h_map_table = (char*) malloc(table_size);
    unordered_map<char, string> huffmanCode;
    cpu_encode(root, "", huffmanCode);//gpu_encode(h_tree_arr, h_tree_arr_length, input_str_array, text.length());

    /*cout << "\nHuffman Codes are :\n" << '\n';
    for (auto pair: huffmanCode) {
        cout << pair.first << " " << pair.second << '\n';
        for (int j = 0; j < MAX_CODE_WIDTH; j++) {
            if (j < pair.second.length())
                h_map_table[pair.first * MAX_CODE_WIDTH + j] = pair.second[j];
        }
        cout << endl;
    }

    err = cudaMemcpyToSymbol(d_map_table, h_map_table, table_size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy encode table from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }*/
    //generateEncodedString(input_str_array_length);
}
