#include "huffman.h"
#include "histogram.h"
#include "gpu_encode.h"
#include "gpu_decode.h"

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>


#define MEM_WIDTH 256*8

#define MAX_NODES 511 //(pow(2, 9) - 1)

// Comparison object to be used to order the heap
struct comp
{
    bool operator()(Node* l, Node* r)
    {
        // highest priority item has lowest frequency
        return l->freq > r->freq;
    }
};

// Builds Huffman Tree and decode given input text
void buildHuffmanTree(string text)
{
    int nodes = 0;
    char input_str_array[text.length()];
    strcpy(input_str_array, text.c_str());

    // count frequency of appearance of each character and store it in a map
    unordered_map<char, int> freq;
    calculateFrequencies(input_str_array, text.length(), freq);

    // Create a priority queue to store live nodes of
    // Huffman tree;
    priority_queue<Node*, vector<Node*>, comp> pq;

    // Create a leaf node for each character and add it
    // to the priority queue.
    for (auto pair: freq) {
        pq.push(getNode(pair.first, pair.second, nullptr, nullptr));
        nodes++;
    }

    // do till there is more than one node in the queue
    while (pq.size() != 1)
    {
        // Remove the two nodes of highest priority
        // (lowest frequency) from the queue
        Node *left = pq.top(); pq.pop();
        Node *right = pq.top();    pq.pop();

        // Create a new internal node with these two nodes
        // as children and with frequency equal to the sum
        // of the two nodes' frequencies. Add the new node
        // to the priority queue.
        int sum = left->freq + right->freq;
        pq.push(getNode('\0', sum, left, right));
        nodes++;
    }

    printf("Total nodes:%d\n", nodes);
    // root stores pointer to root of Huffman Tree
    Node* root = pq.top();

    int treeHeight = height(root);
    int h_tree_arr_length = pow(2,treeHeight+1);
    char *h_tree_arr = (char *)malloc(sizeof(char) * h_tree_arr_length);
    memset(h_tree_arr, '\0', sizeof(char) * h_tree_arr_length);

    convertTreeToArray(root, h_tree_arr, treeHeight);
    printf("Tree converted to array :\n");
    /*for (int i = 0; i < 15; i++){
      //  h_tree_arr[i] = 65 + i;
        printf("%c->", h_tree_arr[i]);
    }*/

    cout << endl;

    gpu_encode(h_tree_arr, h_tree_arr_length, input_str_array, text.length());

    //gpu_decode(text.length());

    free(h_tree_arr);
}
