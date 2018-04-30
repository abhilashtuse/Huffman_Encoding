#ifndef HUFFMAN_H
#define HUFFMAN_H

#include <iostream>
#include <string>
#include <queue>
#include <unordered_map>

using namespace std;

// A Tree node
struct Node
{
    char ch;
    int freq;
    Node *left, *right;
};

void buildHuffmanTree(string text);
Node* getNode(char ch, int freq, Node* left, Node* right);
void encode(Node* root, string str, unordered_map<char, string> &huffmanCode);
void decode(Node* root, int &index, string str);
#endif /* HUFFMAN_H */
