// Framework: GraphFlow
// Class: Softmax2D
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __SOFTMAX2D_H_INCLUDED__
#define __SOFTMAX2D_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <assert.h>

#include "Matrix.h"

using namespace std;

class Softmax2D: public Matrix {
public:
	Softmax2D(int max_nRows, int max_nColumns) : Matrix(max_nRows, max_nColumns) {

	}

	Softmax2D(Matrix *input) : Matrix(input -> nRows, input -> nColumns) {
		this -> input = input;
	}

	void setParameter(Matrix *input) {
		this -> input = input;

		nRows = this -> nRows;
		nColumns = this -> nColumns;
		size = this -> nRows * this -> nColumns;
	}

	void forward() { 
		float MAX = input -> value[0];
		for (int i = 1; i < size; ++i) {
			MAX = max(MAX, input -> value[i]);
		}

		float sum = 0.0;
		for (int i = 0; i < size; ++i) {
			value[i] = exp(input -> value[i] - MAX);
			sum += value[i];
		}

		for (int i = 0; i < size; ++i) {
			value[i] /= sum;
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}
		
	void backward() {
		for (int i = 0; i < size; ++i) {
			input -> gradient[i] += gradient[i] * value[i] * (1.0 - value[i]);
		}
	}

	Matrix *input;

	~Softmax2D() {
	}
};

#endif
