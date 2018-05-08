# Parallelizing Huffman Coding Compression Algorithm with CUDA
Huffman coding is a lossless data compression algorithm. The idea is to assign variable-length codes to input characters. Lengths of the assigned codes are based on the frequencies of corresponding characters. The most frequent character gets the smallest code and the least frequent character gets the largest code. Code assigned to one character is not prefix of code assigned to any other character.

Used histogram for calculating frequency of individual characters in the input string.
Generating Huffman tree using priority queue on CPU side. Huffman tree is converted to binary heap array in order to use it on GPU side.


## Getting Started

Download and unpack the Huffman_Encoding package from the following link: https://github.com/abhilashtuse/Huffman_Encoding by pressing the green link: "Clone or download"

### Compile Project
Go to src directory and compile using make.
```
ubuntu@tegra-ubuntu:~/Huffman_Encoding$ cd src
ubuntu@tegra-ubuntu:~/Huffman_Encoding/src$ make
```

## Running the tests
To run the project use following command and pass input file path as an argument.
```
ubuntu@tegra-ubuntu:~/Huffman_Encoding/src$ ./main ../input/SampleTextFile_1kb.txt
```

After this, output file will be generated in the output directory. To check the difference between generated
output file and input file use following command.
```
ubuntu@tegra-ubuntu:~/Huffman_Encoding/src$ diff ../output/out.txt ../input/SampleTextFile_1kb.txt
```

You can check output for different input data size.
```
ubuntu@tegra-ubuntu:~/Huffman_Encoding/src$ ./main ../input/SampleTextFile_2kb.txt && diff ../output/out.txt ../input/SampleTextFile_2kb.txt
ubuntu@tegra-ubuntu:~/Huffman_Encoding/src$ ./main ../input/SampleTextFile_5kb.txt && diff ../output/out.txt ../input/SampleTextFile_5kb.txt
ubuntu@tegra-ubuntu:~/Huffman_Encoding/src$ ./main ../input/SampleTextFile_10kb.txt && diff ../output/out.txt ../input/SampleTextFile_10kb.txt
ubuntu@tegra-ubuntu:~/Huffman_Encoding/src$ ./main ../input/SampleTextFile_20kb.txt && diff ../output/out.txt ../input/SampleTextFile_20kb.txt
ubuntu@tegra-ubuntu:~/Huffman_Encoding/src$ ./main ../input/SampleTextFile_30kb.txt && diff ../output/out.txt ../input/SampleTextFile_30kb.txt
```

Note: To get the end-to-end execution time, you can comment line 116 in gpu_decode.cu
and uncomment line 49 in main.cu. This will give the algorithm execution time and output file won't be generated.
Use following command to check the end-to-end execution time.
```
ubuntu@tegra-ubuntu:~/Huffman_Encoding/src$ make
ubuntu@tegra-ubuntu:~/Huffman_Encoding/src$ ./main ../input/SampleTextFile_1kb.txt
```

## Authors

* **Abhilash Tuse** - [abhilashtuse](https://github.com/abhilashtuse)
* **Vishal Shrivastava** - [VishalShrivastava](https://github.com/VishalShrivastava)
