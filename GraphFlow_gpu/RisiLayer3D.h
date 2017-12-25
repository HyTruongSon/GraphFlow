// Framework: GraphFlow
// Class: RisiLayer3D
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __RISILAYER3D_H_INCLUDED__
#define __RISILAYER3D_H_INCLUDED__

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

class RisiLayer3D: public Vector {
public:
	RisiLayer3D(int input_size) : Vector(input_size * input_size * input_size) {
		this -> input_size = input_size;
		vectors.clear();
	}

	void add_vector(Vector *vect) {
		assert(input_size == vect -> size);
		vectors.push_back(vect);
	}

	void clear() {
		vectors.clear();
	}

	int index(int x, int y, int z) {
		return z * (input_size * input_size) + y * input_size + x;
	}

	void forward() {
		for (int i = 0; i < size; ++i) {
			value[i] = 0.0;
		}

		int pos;
		for (int x = 0; x < input_size; ++x) {
			for (int y = 0; y < input_size; ++y) {
				for (int z = 0; z < input_size; ++z) {
					pos = index(x, y, z);
					for (int i = 0; i < vectors.size(); ++i) {
						for (int j = 0; j < vectors.size(); ++j) {
							if (i != j) {
								for (int v = 0; v < vectors.size(); ++v) {
									if ((v != i) && (v != j)) {
										value[pos] += ( vectors[i] -> value[x] ) * ( vectors[j] -> value[y] ) * ( vectors[v] -> value[z] );
									}
								}
							}
						}
					}
				}
			}
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}

	void backward() {
		int pos;
		for (int x = 0; x < input_size; ++x) {
			for (int y = 0; y < input_size; ++y) {
				for (int z = 0; z < input_size; ++z) {
					pos = index(x, y, z);
					for (int i = 0; i < vectors.size(); ++i) {
						for (int j = 0; j < vectors.size(); ++j) {
							if (i != j) {
								for (int v = 0; v < vectors.size(); ++v) {
									if ((v != i) && (v != j)) {
										vectors[i] -> gradient[x] += gradient[pos] * ( vectors[j] -> value[y] ) * ( vectors[v] -> value[z] );

										vectors[j] -> gradient[y] += gradient[pos] * ( vectors[i] -> value[x] ) * ( vectors[v] -> value[z] );

										vectors[v] -> gradient[z] += gradient[pos] * ( vectors[i] -> value[x] ) * ( vectors[j] -> value[y] );
									}
								}
							}
						}
					}
				}
			}
		}
	}

	int input_size;
	vector < Vector* > vectors;

	~RisiLayer3D() {
	}
};

#endif