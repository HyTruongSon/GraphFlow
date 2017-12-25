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
#include <assert.h>

#include "../GraphFlow/GraphFlow.h"

using namespace std;

const int N = 20;
const int nChanels = 10;
const double epsilon = 1e-8;

int main(int argc, char **argv) {
	Tensor3D *tensor3D = new Tensor3D(N, N, nChanels);
	Reshape2D *reshape2D = new Reshape2D(N * N, nChanels);
	Reshape3D *reshape3D = new Reshape3D(N, N, nChanels);

	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			for (int v = 0; v < nChanels; ++v) {
				int index = tensor3D -> index(i, j, v);
				tensor3D -> value[index] = rand() % 100;
			}
		}
	}

	reshape2D -> setParameter(tensor3D, N * N, nChanels);
	reshape3D -> setParameter(reshape2D, N, N, nChanels);

	reshape2D -> forward();
	reshape3D -> forward();

	assert(tensor3D -> nRows == reshape3D -> nRows);
	assert(tensor3D -> nColumns == reshape3D -> nColumns);
	assert(tensor3D -> nDepth == reshape3D -> nDepth);
	assert(tensor3D -> size == reshape3D -> size);

	for (int i = 0; i < tensor3D -> size; ++i) {
		assert(abs(tensor3D -> value[i] - reshape3D -> value[i]) < epsilon);
	}

	cout << "CORRECT" << endl;

	return 0;
}