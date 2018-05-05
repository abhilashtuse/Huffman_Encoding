#ifndef HISTOGRAM_H
#define HISTOGRAM_H
#include <string>
#include <queue>
#include <unordered_map>

using namespace std;

__global__ void histo_kernel(char *buffer, long size, unsigned int *histo);
void calculateFrequencies(char *char_array, int input_str_length, unordered_map<char, int> &freq);
#endif /* HISTOGRAM_H */
