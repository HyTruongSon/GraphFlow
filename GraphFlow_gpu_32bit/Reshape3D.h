// Framework: GraphFlow
// Class: Reshape3D
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __RESHAPE3D_H_INCLUDED__
#define __RESHAPE3D_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <assert.h>

#include "Vector.h"
#include "Tensor3D.h"

class Reshape3D: public Tensor3D {
public:
	Reshape3D(int max_nRows, int max_nColumns, int max_nDepth) : Tensor3D(max_nRows, max_nColumns, max_nDepth) {

	}

	Reshape3D(Vector *input, int nRows, int nColumns, int nDepth) : Tensor3D(nRows, nColumns, nDepth) {
		assert(input -> size == nRows * nColumns * nDepth);
		this -> input = input;
	}

	void setParameter(Vector *input, int nRows, int nColumns, int nDepth) {
		assert(input -> size == nRows * nColumns * nDepth);
		this -> input = input;

		this -> nRows = nRows;
		this -> nColumns = nColumns;
		this -> nDepth = nDepth;
		size = this -> nRows * this -> nColumns * this -> nDepth;
	}

	void forward() {
		int count = 0;
		for (int row = 0; row < nRows; ++row) {
			for (int column = 0; column < nColumns; ++column) {
				for (int depth = 0; depth < nDepth; ++depth) {
					int i = index(row, column, depth);
					value[i] = input -> value[count];
					++count;
				}
			}
		}
		assert(count == input -> size);

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}

	void backward() {
		int count = 0;
		for (int row = 0; row < nRows; ++row) {
			for (int column = 0; column < nColumns; ++column) {
				for (int depth = 0; depth < nDepth; ++depth) {
					int i = index(row, column, depth);
					input -> gradient[count] += gradient[i];
					++count;
				}
			}
		}
		assert(count == input -> size);
	}

	Vector *input;

	~Reshape3D() {
	}
};

#endif
