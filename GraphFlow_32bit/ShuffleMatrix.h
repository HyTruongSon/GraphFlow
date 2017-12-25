// Framework: GraphFlow
// Class: ShuffleMatrix
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __SHUFFLEMATRIX_H_INCLUDED__
#define __SHUFFLEMATRIX_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <assert.h>

#include "Matrix.h"
#include "Tensor3D.h"

class ShuffleMatrix: public Matrix {
public:
	ShuffleMatrix(Matrix *input, Vector *sequence) : Matrix(sequence -> size, input -> nColumns) {
		this -> input = input;
		this -> sequence = sequence;
	}

	void forward() {
		for (int i = 0; i < size; ++i) {
			value[i] = 0.0;
		}

		for (int row = 0; row < nRows; ++row) {
			int ind = sequence -> value[row];
			for (int column = 0; column < nColumns; ++column) {
				value[index(row, column)] = input -> value[input -> index(ind, column)];
			}
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}

	void backward() {
		for (int row = 0; row < nRows; ++row) {
			int ind = sequence -> value[row];
			for (int column = 0; column < nColumns; ++column) {
				input -> gradient[input -> index(ind, column)] += gradient[index(row, column)];
			}
		}
	}

	Matrix *input;
	Vector *sequence;

	~ShuffleMatrix() {
	}
};

#endif
