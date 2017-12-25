// Framework: GraphFlow
// Class: VectorAddMatrix
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __VECTORADDMATRIX_H_INCLUDED__
#define __VECTORADDMATRIX_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <assert.h>

#include "Vector.h"
#include "Matrix.h"

class VectorAddMatrix: public Matrix {
public:
	VectorAddMatrix(int max_nRows, int max_nColumns) : Matrix(max_nRows, max_nColumns) {

	}

	VectorAddMatrix(Vector *first, Matrix *second) : Matrix(second -> nRows, second -> nColumns) {
		this -> first = first;
		this -> second = second;

		assert(this -> first -> size == this -> second -> nColumns);
	}

	void setParameter(Vector *first, Matrix *second) {
		this -> first = first;
		this -> second = second;

		assert(this -> first -> size == this -> second -> nColumns);

		nRows = second -> nRows;
		nColumns = second -> nColumns;
		size = nRows * nColumns;
	}

	void forward() {
		for (int row = 0; row < nRows; ++row) {
			for (int column = 0; column < nColumns; ++column) {
				int i = index(row, column);
				value[i] = first -> value[column] + second -> value[i];
			}
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}

	void backward() {
		for (int row = 0; row < nRows; ++row) {
			for (int column = 0; column < nColumns; ++column) {
				int i = index(row, column);
				first -> gradient[column] += gradient[i];
				second -> gradient[i] += gradient[i];
			}
		}
	}

	Vector *first;
	Matrix *second;

	~VectorAddMatrix() {
	}
};

#endif