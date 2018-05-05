#include <fstream>
#include <sstream>
#include "huffman.h"
#include "histogram.h"

// Huffman coding algorithm
int main()
{
	//string text = "Huffman coding is a data compression algorithm.";
	std::string filename = "SampleTextFile_10kb.txt";
 	std::ifstream file(filename.c_str());
	std::stringstream buffer;
    buffer << file.rdbuf();
    std::string text = buffer.str();
	//string text = "Huffman";
	buildHuffmanTree(text);
    return 0;
}
