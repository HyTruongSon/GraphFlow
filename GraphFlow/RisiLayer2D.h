// Framework: GraphFlow
// Class: RisiLayer2D
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __RISILAYER2D_H_INCLUDED__
#define __RISILAYER2D_H_INCLUDED__

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

class RisiLayer2D: public Vector {
public:
	RisiLayer2D(int size) : Vector(size) {
		vectors.clear();
	}

	void add_vector(Vector *vect) {
		vectors.push_back(vect);
	}

	void clear() {
		vectors.clear();
	}

	void forward() {
		for (int i = 0; i < size; ++i) {
			value[i] = 0.0;
		}

		for (int i = 0; i < size; ++i) {
			for (int k = 0; k < size; ++k) {
				for (int u = 0; u < vectors.size(); ++u) {
					for (int v = u + 1; v < vectors.size(); ++v) {
						value[i] += vectors[u] -> value[i] * vectors[v] -> value[k];
						value[i] += vectors[u] -> value[k] * vectors[v] -> value[i];
					}
				}
			}
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}

	void backward() {
		for (int i = 0; i < size; ++i) {
			for (int k = 0; k < size; ++k) {
				for (int u = 0; u < vectors.size(); ++u) {
					for (int v = u + 1; v < vectors.size(); ++v) {
						vectors[u] -> gradient[i] += gradient[i] * vectors[v] -> value[k];
						vectors[v] -> gradient[k] += gradient[i] * vectors[u] -> value[i];

						vectors[u] -> gradient[k] += gradient[i] * vectors[v] -> value[i];
						vectors[v] -> gradient[i] += gradient[i] * vectors[u] -> value[k];
					}
				}
			}
		}
	}

	int input_size;
	vector < Vector* > vectors;

	~RisiLayer2D() {
	}
};

#endif