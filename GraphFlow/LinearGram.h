// Framework: GraphFlow
// Class: LinearGram
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __LINEARGRAM_H_INCLUDED__
#define __LINEARGRAM_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <assert.h>

#include "Matrix.h"

using namespace std;

class LinearGram: public Matrix {
public:
	LinearGram(int nVectors) : Matrix(nVectors, nVectors) {
		this -> nVectors = nVectors;
		vectors.clear();
	}

	void add_vector(Vector *vect) {
		if (vectors.size() > 0) {
			assert(vect -> size == vectors[0] -> size);
		}
		vectors.push_back(vect);
	}

	void clear() {
		vectors.clear();
	}

	void forward() {
		assert(nVectors == vectors.size());

		for (int i = 0; i < size; ++i) {
			value[i] = 0.0;
		}

		for (int x = 0; x < nVectors; ++x) {
			for (int y = 0; y < nVectors; ++y) {
				int i = index(x, y);
				for (int j = 0; j < vectors[x] -> size; ++j) {
					value[i] += vectors[x] -> value[j] * vectors[y] -> value[j];
				}
			}
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}

	void backward() {
		assert(nVectors == vectors.size());

		for (int x = 0; x < nVectors; ++x) {
			for (int y = 0; y < nVectors; ++y) {
				int i = index(x, y);
				for (int j = 0; j < vectors[x] -> size; ++j) {
					vectors[x] -> gradient[j] += gradient[i] * vectors[y] -> value[j];
					vectors[y] -> gradient[j] += gradient[i] * vectors[x] -> value[j];
				}
			}
		}
	}

	int nVectors;
	vector < Vector* > vectors;

	~LinearGram() {
	}
};

#endif