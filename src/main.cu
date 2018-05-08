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
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	buildHuffmanTree(text);

	cudaEventRecord(stop);

	cudaEventSynchronize(stop);

	float elapsed = 0;
	cudaEventElapsedTime(&elapsed, start, stop);
	printf("The elapsed time for Algorithm excution is %.2f ms\n", elapsed);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	//cudaThreadSynchronize();
    return 0;
}
