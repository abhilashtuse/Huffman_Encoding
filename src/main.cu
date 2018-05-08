#include "huffman.h"
#include "histogram.h"

// Huffman coding algorithm
int main()
{
	string text = "Huffman coding is a data compression algorithm.";
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	buildHuffmanTree(text);

	cudaEventRecord(stop);

	cudaEventSynchronize(stop);

	float elapsed = 0;
	cudaEventElapsedTime(&elapsed, start, stop);
	cout << "The elapsed time for Algorithm excution is : " << elapsed;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
    return 0;
}
