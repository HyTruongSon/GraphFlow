// Framework: GraphFlow
// Class: MatrixConcat
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __MATRIXCONCAT_H_INCLUDED__
#define __MATRIXCONCAT_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <assert.h>

#include "Matrix.h"

class MatrixConcat: public Matrix {
public:
	MatrixConcat(int max_nRows, int max_nColumns) : Matrix(max_nRows, max_nColumns) {

	}

	MatrixConcat(Matrix *first, Matrix *second) : Matrix(first -> nRows, first -> nColumns + second -> nColumns) {
		assert(first -> nRows == second -> nRows);

		this -> first = first;
		this -> second = second;
	}

	void setParameter(Matrix *first, Matrix *second) {
		assert(first -> nRows == second -> nRows);

		this -> first = first;
		this -> second = second;

		nRows = first -> nRows;
		nColumns = first -> nColumns + second -> nColumns;
		size = nRows * nColumns;
	}

	void forward() {
		for (int i = 0; i < nRows; ++i) {
			for (int j = 0; j < first -> nColumns; ++j) {
				int ind = index(i, j);
				int f = first -> index(i, j);
				value[ind] = first -> value[f];
			}
		}

		for (int i = 0; i < nRows; ++i) {
			for (int j = 0; j < second -> nColumns; ++j) {
				int ind = index(i, first -> nColumns + j);
				int s = second -> index(i, j);
				value[ind] = second -> value[s];
			}
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}

	void backward() {
		for (int i = 0; i < nRows; ++i) {
			for (int j = 0; j < first -> nColumns; ++j) {
				int ind = index(i, j);
				int f = first -> index(i, j);
				first -> gradient[f] += gradient[ind];
			}
		}

		for (int i = 0; i < nRows; ++i) {
			for (int j = 0; j < second -> nColumns; ++j) {
				int ind = index(i, first -> nColumns + j);
				int s = second -> index(i, j);
				second -> gradient[s] += gradient[ind];
			}
		}
	}

	Matrix *first;
	Matrix *second;

	~MatrixConcat() {
	}
};

#endif
