// Framework: GraphFlow
// Class: VertexRepresentation
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __VERTEXREPRESENTATION_H_INCLUDED__
#define __VERTEXREPRESENTATION_H_INCLUDED__

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

class VertexRepresentation: public Vector {
public:
	VertexRepresentation(int max_nVertices, Vector *feature, Vector *weight, int vertex) : Vector(max_nVertices) {
		assert(feature -> size == weight -> size);
		assert(vertex >= 0);
		assert(vertex < max_nVertices);

		this -> feature = feature;
		this -> weight = weight;
		this -> vertex = vertex;
	}

	void forward() {
		for (int i = 0; i < size; ++i) {
			value[i] = 0.0;
		}

		for (int i = 0; i < feature -> size; ++i) {
			value[vertex] += feature -> value[i] * weight -> value[i];
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}

	void backward() {
		for (int i = 0; i < feature -> size; ++i) {
			feature -> gradient[i] += gradient[vertex] * weight -> value[i];
			weight -> gradient[i] += gradient[vertex] * feature -> value[i];
		}
	}

	Vector* feature;
	Vector* weight;
	int vertex;

	~VertexRepresentation() {
	}
};

#endif
