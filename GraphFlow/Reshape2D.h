// Framework: GraphFlow
// Class: Reshape2D
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __RESHAPE2D_H_INCLUDED__
#define __RESHAPE2D_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <assert.h>

#include "Vector.h"
#include "Matrix.h"

class Reshape2D: public Matrix {
public:
	Reshape2D(int max_nRows, int max_nColumns) : Matrix(max_nRows, max_nColumns) {

	}

	Reshape2D(Vector *input, int nRows, int nColumns) : Matrix(nRows, nColumns) {
		assert(input -> size == nRows * nColumns);
		this -> input = input;
	}

	void setParameter(Vector *input, int nRows, int nColumns) {
		assert(input -> size == nRows * nColumns);

		this -> input = input;
		this -> nRows = nRows;
		this -> nColumns = nColumns;
		size = this -> nRows * this -> nColumns;
	}

	void forward() {
		int count = 0;
		for (int row = 0; row < nRows; ++row) {
			for (int column = 0; column < nColumns; ++column) {
				int i = index(row, column);
				value[i] = input -> value[count];
				++count;
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
				int i = index(row, column);
				input -> gradient[count] += gradient[i];
				++count;
			}
		}
		assert(count == input -> size);
	}

	Vector *input;

	~Reshape2D() {
	}
};

#endif