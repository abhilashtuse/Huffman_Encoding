#include "huffman.h"
#include "histogram.h"

// Huffman coding algorithm
int main()
{
	string text = "Huffman coding is a data compression algorithm.";
	//string text = "Huffman";
	buildHuffmanTree(text);
    return 0;
}
