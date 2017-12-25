// Framework: GraphFlow
// Class: ConcatVectors
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __CONCATVECTORS_H_INCLUDED__
#define __CONCATVECTORS_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <assert.h>

#include "Vector.h"

using namespace std;

class ConcatVectors: public Vector {
public:
	ConcatVectors(int size) : Vector(size) {
		vectors.clear();
	}

	void add_vector(Vector *vect) {
		vectors.push_back(vect);
	}

	void clear() {
		vectors.clear();
	}

	void forward() {
		assert(vectors.size() > 0);

		for (int i = 0; i < size; ++i) {
			value[i] = 0.0;
		}

		int count = 0;
		for (int v = 0; v < vectors.size(); ++v) {
			for (int i = 0; i < vectors[v] -> size; ++i) {
				value[count] = vectors[v] -> value[i];
				++count;
			}
		}
		assert(count == size);

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}

	void backward() {
		assert(vectors.size() > 0);

		int count = 0;
		for (int v = 0; v < vectors.size(); ++v) {
			for (int i = 0; i < vectors[v] -> size; ++i) {
				vectors[v] -> gradient[i] += gradient[count];
				++count;
			}
		}
		assert(count == size);
	}

	vector < Vector* > vectors;

	~ConcatVectors() {
	}
};

#endif