#ifndef GPU_ENCODE_H
#define GPU_ENCODE_H

__global__ void generateEncodedStringKernel();
void generateEncodedString();
__global__ void encode_kernel();
void gpu_encode(char *h_tree_arr, int h_tree_arr_length, char *input_str_array, int input_str_array_length, Node *root);
void cpu_encode(Node* root, string str, unordered_map<char, string> &huffmanCode);

#endif  /* GPU_ENCODE_H */
