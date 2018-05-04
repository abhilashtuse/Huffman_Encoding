#include "huffman.h"
#include "histogram.h"
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#define TOTAL_CHARS 256
#define MEM_WIDTH 256*8
#define MAX_CODE_WIDTH 8 //pow(2, height(root)); //Worst case height of tree when all chars have same frequency
#define MAX_NODES 511//(pow(2, 9) - 1)


// Comparison object to be used to order the heap
struct comp
{
    bool operator()(Node* l, Node* r)
    {
        // highest priority item has lowest frequency
        return l->freq > r->freq;
    }
};

// traverse the Huffman Tree and store Huffman Codes
// in a map.
void encode(Node* root, string str, unordered_map<char, string> &huffmanCode)
{
    if (root == nullptr)
        return;

    // found a leaf node
    if (!root->left && !root->right) {
        huffmanCode[root->ch] = str;
    }

    encode(root->left, str + "0", huffmanCode);
    encode(root->right, str + "1", huffmanCode);
}

// traverse the Huffman Tree and decode the encoded string
void decode(Node* root, int &index, string str)
{
    if (root == nullptr) {
        return;
    }

    // found a leaf node
    if (!root->left && !root->right)
    {
        cout << root->ch;
        return;
    }

    index++;

    if (str[index] =='0')
        decode(root->left, index, str);
    else
        decode(root->right, index, str);
}

__constant__ char d_tree_arr [MAX_NODES];
__global__ void encode_kernel(int k, int count,char *str,  char *encode_map)
{
  if(threadIdx.x == 0){
    cudaStream_t s1, s2;
    unsigned int flag = cudaStreamDefault;
    cudaStreamCreateWithFlags(&s1, flag);
    cudaStreamCreateWithFlags(&s2, flag);
    //  printf("\nInside Kernel : %d and count : %d and string : %s",k, count, str);
    int l = 2*k+1;
    int r = 2*k+2;
    //__shared__ char s_str[MAX_CODE_WIDTH];
    if (d_tree_arr[l] == '$' && d_tree_arr[r] == '$')
    {
      //s_str = str[];
      int i = 0;
      #pragma unroll
      for ( ;i < MAX_CODE_WIDTH && str[i] != '\0'; i++) {
          encode_map[d_tree_arr[k]*MAX_CODE_WIDTH + i] = str[i];
      }
      encode_map[d_tree_arr[k]*MAX_CODE_WIDTH + i+1] = '\0';
		  // printf("\nCHAR:%c ENCODEDING:%s",d_tree_arr[k], str);
      //fill_Map <<<1,1,0, s1>>>
	  }
    //printf("\nCHAR:%c", d_tree_arr[k]);
    if (d_tree_arr[l] != '$'){
      //if(count != -1)
      str[count] = '0';
      str[count+1] = '\0';
      //append(str,'0', count);
      //printf("Appending: 0");
      //  printf("\nCHAR:%c LEFT ENCODEDING:%s",d_tree_arr[k], str);
      encode_kernel<<<1,1,0, s1>>>(l,count + 1, str, encode_map);
      //append(str,'\0',count);
      cudaDeviceSynchronize();
    }

    if (d_tree_arr[r] != '$') {
      //  printf("Appending: 1");
      //  if(count != -1)
      str[count] = '1';
      str[count+1] = '\0';
      //append(str,'1',count);
      //printf("\nCHAR:%c RIGHT ENCODEDING:%s",d_tree_arr[k], str);

      encode_kernel<<<1,1,0, s2>>>(r, count + 1, str, encode_map);
      cudaDeviceSynchronize();
    }
  }
  __syncthreads();
}

__global__ void my_concat_kernel(char *encode_table, char *input_string, char *encoded_string, size_t input_size)
{
    //printf("In encode_kernel input string:%c\n", input_string[threadIdx.x]);
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    //printf("tid:%d\n", tid);
    //int row = tid / MAX_CODE_WIDTH;
    //int col = tid % MAX_CODE_WIDTH;
    if (tid < input_size) {
        for (int i = 0; i < MAX_CODE_WIDTH; i++) {
            encoded_string[tid * MAX_CODE_WIDTH + i] = encode_table[input_string[tid] * MAX_CODE_WIDTH + i];
        }
    }
    __syncthreads();
}

void generateEncodedString(unordered_map<char, string> huffmanCode, string &text, string &final_encoded_string)
{
    cudaError_t err = cudaSuccess;
    int table_size = TOTAL_CHARS * MAX_CODE_WIDTH * sizeof(char);
    //cout << "table size: " << table_size << endl;
    char *h_encode_table = (char*) malloc(table_size);
    int encode_string_size = text.length() * MAX_CODE_WIDTH * sizeof(char);
    char *h_encoded_string = (char*) malloc(encode_string_size);
    if (h_encode_table == NULL || h_encoded_string == NULL) {
        fprintf(stderr, "Failed to allocate host table or string (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    memset(h_encode_table, 0, table_size);
    memset(h_encoded_string, 0, encode_string_size);

    //cout << "Huffman Codes are :\n" << '\n';
    for (auto pair: huffmanCode) {
        //cout << pair.first << " " << pair.second << '\n';
        for (int j = 0; j < MAX_CODE_WIDTH; j++) {
            if (j < pair.second.length())
                h_encode_table[pair.first * MAX_CODE_WIDTH + j] = pair.second[j];
        }
        //cout << endl;
    }

    // Allocate encode kernel variables
    char h_input_string[text.length()];
    strcpy(h_input_string, text.c_str());
    //cout << "input text:" << text << endl;
    //cout << "input string in host array:" << h_input_string << endl;
    char *d_input_string = NULL;
    err = cudaMalloc((void**)&d_input_string, text.length()* sizeof(char));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device input string (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_input_string, h_input_string, text.length() * sizeof(char),cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy input string from host to device (error code %s)!\n", 	  cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    char *d_encode_table = NULL;
    err = cudaMalloc((void**)&d_encode_table, table_size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device encode table (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_encode_table, h_encode_table, table_size,cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy encode table from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    char *d_encoded_string = NULL;
    err = cudaMalloc((void**)&d_encoded_string, encode_string_size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device encode string (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int blocksPerGrid = (text.length() / 1024) + 1;
    int threadsPerBlock = 1024;// FILL HERE
	printf("CUDA encode kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

	cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

	cudaEventRecord(start);
    //printf("Before my_concat_kernel");
    my_concat_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_encode_table, d_input_string, d_encoded_string, text.length());
    //printf("After my_concat_kernel");
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);

	float elapsed = 0;
	cudaEventElapsedTime(&elapsed, start, stop);
	printf("The elapsed time for encode kernal exexution is %.2f ms\n", elapsed);

    cudaEventDestroy (start);
    cudaEventDestroy (stop);
    cudaThreadSynchronize();

    err = cudaMemcpy(h_encoded_string, d_encoded_string, encode_string_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy encoded string from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaThreadSynchronize();

    printf("After kernel execution result:\n");
    for(int i = 0; i < text.length()*8; i++) {
        printf("%c", h_encoded_string[i]);
        if (h_encoded_string[i] != 0)
            final_encoded_string += h_encoded_string[i];
    }
    // Free Device variables
    cudaFree(d_input_string);
    cudaFree(d_encode_table);
    cudaFree(d_encoded_string);

    // Free host memory
    free(h_encode_table);
    free(h_encoded_string);

}

// Builds Huffman Tree and decode given input text
void buildHuffmanTree(string text)
{
    cudaError_t err = cudaSuccess;
    int nodes = 0;
    // count frequency of appearance of each character and store it in a map
    unordered_map<char, int> freq;
    calculateFrequencies(text, freq);
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

    //int inx = 0;
    //preorder(pq, a, &inx);
    printf("Total nodes:%d\n", nodes);
    // root stores pointer to root of Huffman Tree
    Node* root = pq.top();
    int treeHeight = height(root);
    char *h_arr = (char *)malloc(sizeof(char)*MAX_NODES);
    memset(h_arr, '\0', MAX_NODES);
    printLevelOrder(root, h_arr, treeHeight);
    printf("Tree converted to array :\n");
    for (int i = 0; i < 15; i++){
      //  h_arr[i] = 65 + i;
        printf("%c->", h_arr[i]);
    }

    char *d_str = NULL;
    err = cudaMalloc((void**)&d_str, MAX_CODE_WIDTH * sizeof(char));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device string(str) (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaMemset(d_str, '\0', MAX_CODE_WIDTH);

    err = cudaMemcpyToSymbol(d_tree_arr, h_arr, MAX_NODES*sizeof(char));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy tree array from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int table_size = TOTAL_CHARS * MAX_CODE_WIDTH * sizeof(char);
    char *h_encode_map = (char*) malloc(table_size);
    if (h_encode_map == NULL)
    {
        fprintf(stderr, "Failed to allocate host encode map (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    memset(h_encode_map, 0, table_size);

    char *d_encode_map = NULL;
    err = cudaMalloc((void**)&d_encode_map, table_size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device encode map (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    unordered_map<char, string> huffmanCode;
    //encode(root, "", huffmanCode);
    int blocksPerGrid = 1;//(text.length() / 1024) + 1;
    int threadsPerBlock = 1;//1024;// FILL HERE
    printf("CUDA encode kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 9);

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

  	cudaEventRecord(start);
    encode_kernel<<<blocksPerGrid, threadsPerBlock>>>(0, 0, d_str, d_encode_map);
  	cudaEventRecord(stop);

  	cudaEventSynchronize(stop);

  	float elapsed = 0;
  	cudaEventElapsedTime(&elapsed, start, stop);
  	printf("The elapsed time for Encode kernal excution is %.2f ms\n", elapsed);

    cudaEventDestroy (start);
    cudaEventDestroy (stop);
    cudaThreadSynchronize();

    err = cudaMemcpy(h_encode_map, d_encode_map, table_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy encode map from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

#if 1
    printf("Final Encoded Table\n");
    //printf("\nCHAR:%c ENCODEDING:%s",h_arr[4], h_encode_map[97*MAX_CODE_WIDTH]);
    for (int i = 0; i < MAX_NODES ; i++) {
      if(h_arr[i] != '$' && h_arr[i] != '*' && h_arr[i] != '\0') {
        printf("\nchar:%c:", h_arr[i]);
        for (int j = 0; j < 8; j++) {
          char tmp = h_encode_map[h_arr[i]*MAX_CODE_WIDTH + j];
          if (tmp != '\0')
            printf("%c", tmp);
        }
      }
    }
    printf("\n\n");
#endif

    cudaThreadSynchronize();
    //generateEncodedString(huffmanCode, text, str);
    cout << "\nOriginal string was :\n" << text << '\n';
    // print encoded string
    string str = "";
    for (char ch: text) {
        for (int i = 0; h_encode_map[(ch * MAX_CODE_WIDTH) + i] != '\0'; i++)
            str += h_encode_map[ch * MAX_CODE_WIDTH + i];
    }
    cout << "\nEncoded string is :\n" << str << '\n';

    // traverse the Huffman Tree again and this time
    // decode the encoded string
    int index = -1;
    cout << "\nDecoded string is: \n";
    while (index < (int)str.size() - 2) {
        decode(root, index, str);
    }
    cout << endl;
}
