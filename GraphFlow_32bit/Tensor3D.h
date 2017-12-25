// Framework: GraphFlow
// Class: Tensor3D
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __TENSOR3D_H_INCLUDED__
#define __TENSOR3D_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <assert.h>

#include "Vector.h"

class Tensor3D: public Vector {
public:
	Tensor3D(int nRows, int nColumns, int nDepth) : Vector(nRows * nColumns * nDepth) {
		this -> nRows = nRows;
		this -> nColumns = nColumns;
		this -> nDepth = nDepth;
	}

	void setParameter(int nRows, int nColumns, int nDepth) {
		this -> nRows = nRows;
		this -> nColumns = nColumns;
		this -> nDepth = nDepth;

		size = this -> nRows * this -> nColumns * this -> nDepth;
	}

	int index(int row, int column, int depth) {
		return (row * nColumns + column) * nDepth + depth;
	}

	float valueAt(int row, int column, int depth) {
		return value[index(row, column, depth)];
	}

	float gradientAt(int row, int column, int depth) {
		return gradient[index(row, column, depth)];
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
	int nDepth;

	~Tensor3D() {
	}
};

#endif
