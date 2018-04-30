#ifndef BINARY_TREE_H
#define BINARY_TREE_H

// A Tree node
struct Node
{
    char ch;
    int freq;
    Node *left, *right;
};

Node* getNode(char ch, int freq, Node* left, Node* right);
int height(Node* node);
void printLevelOrder(Node *root, char *arr, int height);

#endif /* BINARY_TREE_H */
