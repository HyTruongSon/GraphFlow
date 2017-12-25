// Framework: GraphFlow
// Class: MatMul
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __MATMUL_H_INCLUDED__
#define __MATMUL_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <assert.h>

#include "Matrix.h"

class MatMul: public Matrix {
public:
	MatMul(int max_first_nRows, int max_first_nColumns, int max_second_nRows, int max_second_nColumns) : Matrix(max_first_nRows, max_second_nColumns) {
	}

	MatMul(int max_nRows, int max_nColumns) : Matrix(max_nRows, max_nColumns) {

	}

	MatMul(Matrix *first, Matrix *second) : Matrix(first -> nRows, second -> nColumns) {
		assert(first -> nColumns == second -> nRows);

		this -> first = first;
		this -> second = second;
	}

	void setParameter(Matrix *first, Matrix *second) {
		assert(first -> nColumns == second -> nRows);
		
		this -> first = first;
		this -> second = second;

		nRows = first -> nRows;
		nColumns = second -> nColumns;
		size = nRows * nColumns;
	}

	void forward() {
		for (int i = 0; i < size; ++i) {
			value[i] = 0.0;
		}

		for (int i = 0; i < nRows; ++i) {
			for (int j = 0; j < nColumns; ++j) {
				int v = index(i, j);
				for (int k = 0; k < first -> nColumns; ++k) {
					int f = first -> index(i, k);
					int s = second -> index(k, j);
					value[v] += first -> value[f] * second -> value[s];
				}
			}
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}

	void backward() {
		for (int i = 0; i < nRows; ++i) {
			for (int j = 0; j < nColumns; ++j) {
				int v = index(i, j);
				for (int k = 0; k < first -> nColumns; ++k) {
					int f = first -> index(i, k);
					int s = second -> index(k, j);

					first -> gradient[f] += gradient[v] * second -> value[s];
					second -> gradient[s] += gradient[v] * first -> value[f];
				}
			}
		}
	}

	Matrix *first;
	Matrix *second;

	~MatMul() {
	}
};

#endif