#ifndef HUFFMAN_H
#define HUFFMAN_H

#include <iostream>
#include <string>
#include <queue>
#include <unordered_map>
#include "binary_tree.h"

using namespace std;

__global__ void encode_kernel(char *d_encode_table, char *d_input_string, char *d_encoded_string);
void buildHuffmanTree(string text);
void encode(Node* root, string str, unordered_map<char, string> &huffmanCode);
void decode(Node* root, int &index, string str);
#endif /* HUFFMAN_H */
