#include "huffman.h"
#include "histogram.h"

// Huffman coding algorithm
int main()
{
	/*cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);*/

	string text = "Huffman coding is a data compression algorithm.";
	buildHuffmanTree(text);
	//unordered_map<char, int> freq;
	//calculateFrequencies(text, freq);
	//for (auto pair: freq) {
	//	cout << pair.first << " " << pair.second << endl;
	//}
	/*cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float elapsed = 0;
	cudaEventElapsedTime(&elapsed, start, stop);
	printf("The elapsed time for main exexution is %.2f ms\n", elapsed);

	cudaEventDestroy (start);
	cudaEventDestroy (stop);*/

    return 0;
}
