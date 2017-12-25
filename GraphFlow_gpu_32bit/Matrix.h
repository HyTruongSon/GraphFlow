// Framework: GraphFlow
// Class: Matrix
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __MATRIX_H_INCLUDED__
#define __MATRIX_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <assert.h>

#include "Vector.h"

class Matrix: public Vector {
public:
	Matrix(int nRows, int nColumns) : Vector(nRows * nColumns) {
		this -> nRows = nRows;
		this -> nColumns = nColumns;
	}

	void setParameter(int nRows, int nColumns) {
		this -> nRows = nRows;
		this -> nColumns = nColumns;
		size = this -> nRows * this -> nColumns;
	}

	int index(int row, int column) {
		return row * nColumns + column;
	}

	float valueAt(int row, int column) {
		return value[index(row, column)];
	}

	Matrix* multiply(Matrix *another) {
		assert(nColumns == another -> nRows);
		Matrix *product = new Matrix(nRows, another -> nColumns);
		for (int i = 0; i < nRows; ++i) {
			for (int j = 0; j < another -> nColumns; ++j) {
				int p = product -> index(i, j);
				for (int v = 0; v < nColumns; ++v) {
					int t = index(i, v);
					int a = another -> index(v, j);
					product -> value[p] += value[t] * another -> value[a];
 				}
			}
		}
		return product;
	}

	void forward() {
		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}

	void backward() {
	}

	int nRows;
	int nColumns;

	~Matrix() {
	}
};

#endif
