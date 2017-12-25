// Framework: GraphFlow
// Class: ScalarMatMul
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __SCALARMATMUL_H_INCLUDED__
#define __SCALARMATMUL_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <assert.h>

#include "Vector.h"

class ScalarMatMul: public Matrix {
public:
	ScalarMatMul(Vector *scalar, Matrix *matrix) : Matrix(matrix -> nColumns, matrix -> nRows) {
		this -> scalar = scalar;
		this -> matrix = matrix;
	}

	void forward() {
		for (int i = 0; i < nRows; ++i) {
			for (int j = 0; j < nColumns; ++j) {
				int ind = index(i, j);
				value[ind] = scalar -> value[0] * matrix -> value[ind];
			}
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}

	void backward() {
		for (int i = 0; i < nRows; ++i) {
			for (int j = 0; j < nColumns; ++j) {
				int ind = index(i, j);
				scalar -> gradient[0] += gradient[ind] * matrix -> value[ind];
				matrix -> gradient[ind] += gradient[ind] * scalar -> value[0];
			}
		}
	}

	Vector *scalar;
	Matrix *matrix;

	~ScalarMatMul() {
	}
};

#endif