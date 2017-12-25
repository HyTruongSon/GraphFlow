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

#include "../GraphFlow/GraphFlow.h"

using namespace std;

const int N = 10;
const int nChanels = 5;
const int nContractions = RisiContraction_50::nContractions;

const double epsilon = 1e-8;

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
	RisiContraction_50 *obj = new RisiContraction_50(N, nChanels);
	
	// Generate tensors
	Tensor3D **tensors = new Tensor3D* [N];
	for (int i = 0; i < N; ++i) {
		tensors[i] = new Tensor3D(N, N, nChanels);
		for (int chanel = 0; chanel < nChanels; ++chanel) {
			for (int row = 0; row < N; ++row) {
				for (int column = row; column < N; ++column) {
					int index = tensors[i] -> index(row, column, chanel);
					int index_ = tensors[i] -> index(column, row, chanel);

					int random = get_random(100); 
					tensors[i] -> value[index] = random;
					tensors[i] -> value[index_] = random; 
				}
			}
		}
	}	

	// Generate the adjacency matrix
	Matrix *adj = new Matrix(N, N);
	for (int i = 0; i < N; ++i) {
		adj -> value[adj -> index(i, i)] = 0;
		for (int j = i + 1; j < N; ++j) {
			int random = get_random(2);
			adj -> value[adj -> index(i, j)] = random;
			adj -> value[adj -> index(j, i)] = random;
		}
	}

	// Similuation for RisiContraction_50
	obj -> clear();
	for (int i = 0; i < N; ++i) {
		obj -> add_tensor(tensors[i]);
	}
	obj -> set_adjacency(adj);

	// Forward pass
	obj -> forward();

	Tensor3D **results = new Tensor3D* [nContractions + 1];
	for (int Case = 1; Case <= nContractions; ++Case) {
		results[Case] = new Tensor3D(N, N, nChanels);
		for (int row = 0; row < N; ++row) {
			for (int column = 0; column < N; ++column) {
				for (int chanel = 0; chanel < nChanels; ++chanel) {
					int index = results[Case] -> index(row, column, chanel);
					int index_ = obj -> index(row, column, (Case - 1) * nChanels + chanel);
					results[Case] -> value[index] = obj -> value[index_];
				}
			}
		}
	}

	// Check the redundancy
	bool *free = new bool [nContractions + 1];
	for (int Case = 1; Case <= nContractions; ++Case) {
		free[Case] = true;
	}

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

	return 0;
}
