// Framework: GraphFlow
// Class: Tensor4D
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __TENSOR4D_H_INCLUDED__
#define __TENSOR4D_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <assert.h>

#include "Vector.h"

class Tensor4D: public Vector {
public:
	Tensor4D(int nRows, int nColumns, int nChanels1, int nChanels2) : Vector(nRows * nColumns * nChanels1 * nChanels2) {
		this -> nRows = nRows;
		this -> nColumns = nColumns;
		this -> nChanels1 = nChanels1;
		this -> nChanels2 = nChanels2;
	}

	int index(int row, int column, int chanel1, int chanel2) {
		return ((row * nColumns + column) * nChanels1 + chanel1) * nChanels2 + chanel2;
	}

	float valueAt(int row, int column, int chanel1, int chanel2) {
		return value[index(row, column, chanel1, chanel2)];
	}

	float gradientAt(int row, int column, int chanel1, int chanel2) {
		return gradient[index(row, column, chanel1, chanel2)];
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
	int nChanels1;
	int nChanels2;

	~Tensor4D() {
	}
};

#endif
