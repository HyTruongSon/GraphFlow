// Framework: GraphFlow
// Class: MatVecMul
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __MATVECMUL_H_INCLUDED__
#define __MATVECMUL_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <assert.h>

#include "Vector.h"

class MatVecMul: public Vector {
public:
	MatVecMul(int max_size) : Vector(max_size) {

	}

	MatVecMul(Matrix *W, Vector *x) : Vector(W -> nRows) {
		this -> W = W;
		this -> x = x;

		assert(this -> W -> nColumns == this -> x -> size);
	}

	void setParameter(Matrix *W, Vector *x) {
		this -> W = W;
		this -> x = x;

		assert(this -> W -> nColumns == this -> x -> size);
	}

	void forward() {
		for (int i = 0; i < W -> nRows; ++i) {
			value[i] = 0.0;
			for (int j = 0; j < W -> nColumns; ++j) {
				value[i] += W -> value[i * (W -> nColumns) + j] * x -> value[j];
			}
		}

		for (int i = 0; i < W -> nRows; ++i) {
			gradient[i] = 0.0;
		}
	}

	void backward() {
		// Gradient for W
		for (int i = 0; i < W -> nRows; ++i) {
			for (int j = 0; j < W -> nColumns; ++j) {
				W -> gradient[i * (W -> nColumns) + j] += gradient[i] * (x -> value[j]);
			}
		}

		// Gradient for x
		for (int j = 0; j < W -> nColumns; ++j) {
			for (int i = 0; i < W -> nRows; ++i) {
				x -> gradient[j] += gradient[i] * (W -> value[i * (W -> nColumns) + j]);
			}
		}
	}

	Matrix *W;
	Vector *x;

	~MatVecMul() {
	}
};

#endif
