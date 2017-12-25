// Framework: GraphFlow
// Class: Transpose
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __TRANSPOSE_H_INCLUDED__
#define __TRANSPOSE_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <assert.h>

#include "Vector.h"

class Transpose: public Matrix {
public:
	Transpose(int max_nRows, int max_nColumns) : Matrix(max_nRows, max_nColumns) {

	}

	Transpose(Matrix *input) : Matrix(input -> nColumns, input -> nRows) {
		this -> input = input;
	}

	void setParameter(Matrix *input) {
		this -> input = input;
		
		nRows = this -> input -> nColumns;
		nColumns = this -> input -> nRows;
		size = nRows * nColumns;
	}

	void forward() {
		for (int i = 0; i < nRows; ++i) {
			for (int j = 0; j < nColumns; ++j) {
				value[i * nColumns + j] = input -> value[j * nRows + i];
			}
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}

	void backward() {
		for (int i = 0; i < nRows; ++i) {
			for (int j = 0; j < nColumns; ++j) {
				input -> gradient[j * nRows + i] += gradient[i * nColumns + j];
			}
		}
	}

	Matrix *input;

	~Transpose() {
	}
};

#endif
