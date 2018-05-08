#ifndef GPU_DECODE_H
#define GPU_DECODE_H
#include <string>

using namespace std;

void gpu_decode(int input_str_length);
__device__ void append(char *s, char c, int position);
__global__ void decode_kernel(char *d_encoded_str, int k, int count);
__global__ void decode_parent_kernel(char *d_encoded_str, int string_size);
void cpu_decode(Node* root, int &index, string str);

#endif  /* GPU_DECODE_H */
