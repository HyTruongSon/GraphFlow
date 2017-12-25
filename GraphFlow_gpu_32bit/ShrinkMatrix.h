// Framework: GraphFlow
// Class: ShrinkMatrix
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __SHRINKMATRIX_H_INCLUDED__
#define __SHRINKMATRIX_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <assert.h>

#include "Vector.h"
#include "Matrix.h"

class ShrinkMatrix: public Vector {
public:
	ShrinkMatrix(int max_size) : Vector(max_size) {

	}

	ShrinkMatrix(Matrix *input, int dim) : Vector(dim * input -> nRows + (1 - dim) * input -> nColumns) {
		this -> input = input;
		this -> dim = dim;
	}

	void setParameter(Matrix *input, int dim) {
		this -> input = input;
		this -> dim = dim;

		if (dim == 0) {
			size = this -> input -> nColumns;
		} else {
			size = this -> input -> nRows;
		}
	}

	void forward() {
		if (dim == 0) {
			for (int column = 0; column < input -> nColumns; ++column) {
				value[column] = 0.0;
				for (int row = 0; row < input -> nRows; ++row) {
					int i = input -> index(row, column);
					value[column] += input -> value[i];
				}
			}
		} else {
			for (int row = 0; row < input -> nRows; ++row) {
				value[row] = 0.0;
				for (int column = 0; column < input -> nColumns; ++column) {
					int i = input -> index(row, column);
					value[row] += input -> value[i];
				}
			}
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}

	void backward() {
		if (dim == 0) {
			for (int column = 0; column < input -> nColumns; ++column) {
				for (int row = 0; row < input -> nRows; ++row) {
					int i = input -> index(row, column);
					input -> gradient[i] += gradient[column];
				}
			}
		} else {
			for (int row = 0; row < input -> nRows; ++row) {
				for (int column = 0; column < input -> nColumns; ++column) {
					int i = input -> index(row, column);
					input -> gradient[i] += gradient[row];
				}
			}
		}
	}

	Matrix *input;
	int dim;

	~ShrinkMatrix() {
	}
};

#endif
