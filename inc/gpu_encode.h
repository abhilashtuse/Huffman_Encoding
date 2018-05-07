#ifndef GPU_ENCODE_H
#define GPU_ENCODE_H

__global__ void generateEncodedStringKernel();
void generateEncodedString(int input_str_array_length, Node *root);
__global__ void encode_kernel();
void gpu_encode(char *input_str_array, int input_str_array_length, Node *root);
void cpu_encode(Node* root, string str, unordered_map<char, string> &huffmanCode);

#endif  /* GPU_ENCODE_H */
