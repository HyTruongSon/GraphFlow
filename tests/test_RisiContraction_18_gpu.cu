// Framework: GraphFlow
// Author: Machine Learning Group of UChicago
// Main Contributor: Hy Truong Son
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#include <iostream>
#include <cstring>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <ctime>
#include <sys/time.h>

#include "../GraphFlow_gpu/GraphFlow.h"

using namespace std;

int N;
int nChanels;
const int nContractions = RisiContraction_18_gpu::nContractions;

const double epsilon = 1e-8;

const double RANDOM_SEED = 123456789;

// Get the millisecond
void time_ms(long int &ms) {
	struct timeval tp;
	gettimeofday(&tp, NULL);
	ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;
}

// Difference in milliseconds
long int difftime_ms(long int &end, long int &start) {
	return end - start;
}

int get_random(int number) {
	return rand() % number;
}

bool compare_tensor(Tensor3D *A, Tensor3D *B) {
	if (A -> nRows != B -> nRows) {
		return false;
	}
	if (A -> nColumns != B -> nColumns) {
		return false;
	}
	if (A -> nDepth != B -> nDepth) {
		return false;
	}
	if (A -> size != B -> size) {
		return false;
	}
	for (int i = 0; i < A -> size; ++i) {
		if (abs(A -> value[i] != B -> value[i]) > epsilon) {
			return false;
		}
	}
	return true;
}

int main(int argc, char **argv) {
	if (argc < 3) {
		cerr << "Not enough parameters!" << endl;
		return 0;
	} 

	N = atoi(argv[1]);
	nChanels = atoi(argv[2]);

	cout << "-------------------------------------------------------" << endl;
	cout << "N = " << N << endl;
	cout << "nChanels = " << nChanels << endl;

	srand(RANDOM_SEED);

	long int start;
	long int end;

	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);

	cout << "Number of GPU: " << deviceCount << endl << endl;

	RisiContraction_18_gpu *contract = new RisiContraction_18_gpu(N, nChanels);
	RisiContraction_18 *ground_truth = new RisiContraction_18(N, nChanels);
	StackTensor3D *stack = new StackTensor3D(N, N, N, nChanels);
	
	// Generate tensors
	Tensor3D **tensors = new Tensor3D* [N];
	for (int i = 0; i < N; ++i) {
		tensors[i] = new Tensor3D(N, N, nChanels);
		for (int chanel = 0; chanel < nChanels; ++chanel) {
			for (int row = 0; row < N; ++row) {
				for (int column = row; column < N; ++column) {
					int index = tensors[i] -> index(row, column, chanel);
					int index_ = tensors[i] -> index(column, row, chanel);

					int random = get_random(10); 
					tensors[i] -> value[index] = random;
					tensors[i] -> value[index_] = random; 
				}
			}
		}
	}	

	// Generate the adjacency matrix
	Matrix *adj = new Matrix(N, N);
	for (int i = 0; i < N; ++i) {
		adj -> value[adj -> index(i, i)] = 1;
		for (int j = i + 1; j < N; ++j) {
			int random = get_random(2);
			adj -> value[adj -> index(i, j)] = random;
			adj -> value[adj -> index(j, i)] = random;
		}
	}

	// Similuation for RisiContraction_18_gpu
	stack -> clear();
	for (int i = 0; i < N; ++i) {
		stack -> add_tensor(tensors[i]);
	}

	contract -> setParameter(stack, adj);

	ground_truth -> clear();
	for (int i = 0; i < N; ++i) {
		ground_truth -> add_tensor(tensors[i]);
	}
	ground_truth -> set_adjacency(adj);

	// Check the sizes
	assert(contract -> size == ground_truth -> size);
	assert(contract -> nRows == ground_truth -> nRows);
	assert(contract -> nColumns == ground_truth -> nColumns);
	assert(contract -> nDepth == ground_truth -> nDepth);

	// Forward pass
	stack -> forward();

	time_ms(start);
	contract -> forward();
	time_ms(end);

	cout << "GPU forward time: " << difftime_ms(end, start) << endl;

	time_ms(start);
	ground_truth -> forward();
	time_ms(end);

	cout << "CPU forward time: " << difftime_ms(end, start) << endl;

	Tensor3D **results = new Tensor3D* [nContractions + 1];
	for (int Case = 1; Case <= nContractions; ++Case) {
		results[Case] = new Tensor3D(N, N, nChanels);
		for (int row = 0; row < N; ++row) {
			for (int column = 0; column < N; ++column) {
				for (int chanel = 0; chanel < nChanels; ++chanel) {
					int index = results[Case] -> index(row, column, chanel);
					int index_ = contract -> index(row, column, (Case - 1) * nChanels + chanel);
					results[Case] -> value[index] = contract -> value[index_];
				}
			}
		}
	}

	// Check the redundancy
	bool *free = new bool [nContractions + 1];
	for (int Case = 1; Case <= nContractions; ++Case) {
		free[Case] = true;
	}

	cout << "Check the uniqueness of tensor contractions:" << endl;
	int count = 0;
	for (int i = 1; i <= nContractions; ++i) {
		if (free[i]) {
			++count;
			cout << count << ": ";
			for (int j = i; j <= nContractions; ++j) {
				if (compare_tensor(results[i], results[j])) {
					free[j] = false;
					cout << j << " ";
				}
			}
			cout << endl;
		}
	}

	double error = 0.0;
	for (int i = 0; i < ground_truth -> size; ++i) {
		error += abs(ground_truth -> value[i] - contract -> value[i]);
	}
	cout << "Forward absolute error: " << error << endl << endl;

	// Randomized the gradient
	for (int i = 0; i < ground_truth -> size; ++i) {
		ground_truth -> gradient[i] = get_random(100); 
		contract -> gradient[i] = ground_truth -> gradient[i];
	}

	time_ms(start);
	contract -> backward();
	time_ms(end);

	cout << "GPU backward time: " << difftime_ms(end, start) << endl;

	time_ms(start);
	ground_truth -> backward();
	time_ms(end);

	cout << "CPU backward time: " << difftime_ms(end, start) << endl;

	error = 0.0;
	for (int a = 0; a < N; ++a) {
		for (int b = 0; b < N; ++b) {
			for (int c = 0; c < N; ++c) {
				for (int f = 0; f < nChanels; ++f) {
					double CPU = tensors[a] -> gradient[tensors[a] -> index(b, c, f)];
					double GPU = stack -> gradient[stack -> index(a, b, c, f)];

					error += abs(CPU - GPU);
				}
			}
		}
	}
	cout << "Backward absolute error: " << error << endl;

	return 0;
}
