#include <fstream>
#include <sstream>
#include "huffman.h"
#include "histogram.h"

std::ofstream out_file("../output/out.txt");;

// Huffman coding algorithm
int main(int argc, char** argv)
{
	std::string inp_filename;
	if(argc < 2) {
        cerr << "Usage: " << argv[0] << " input_file_path" << endl;
		return 1;
	}
	else
    {
        inp_filename = argv[1];
    }

	std::ifstream inp_file(inp_filename.c_str());
	if (!inp_file.is_open()) {
		cerr << "Error: Invalid input file: " << inp_filename << endl;
		return 1;
	}

	if (!out_file.is_open()) {
		cerr << "Error: could not create output/out.txt file: " << endl;
		return 1;
	}

	std::stringstream buffer;
	buffer << inp_file.rdbuf();
	std::string text = buffer.str();

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	buildHuffmanTree(text);

	cudaEventRecord(stop);

	cudaEventSynchronize(stop);

	float elapsed = 0;
	cudaEventElapsedTime(&elapsed, start, stop);
	//cout << "The elapsed time for Algorithm excution is : " << elapsed;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	inp_file.close();
	out_file.close();
    return 0;
}
