// Framework: GraphFlow
// Class: LeakyReLU2D
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __LEAKYRELU2D_H_INCLUDED__
#define __LEAKYRELU2D_H_INCLUDED__

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

class LeakyReLU2D: public Matrix {
public:
	LeakyReLU2D(int max_nRows, int max_nColumns) : Matrix(max_nRows, max_nColumns) {

	}

	LeakyReLU2D(Matrix *input) : Matrix(input -> nRows, input -> nColumns) {
		this -> input = input;
		alpha = 0.01;
	}

	LeakyReLU2D(Matrix *input, float alpha) : Matrix(input -> nRows, input -> nColumns) {
		this -> input = input;
		this -> alpha = alpha;
	}

	void setParameter(Matrix *input) {
		this -> input = input;
		alpha = 0.01;

		nRows = input -> nRows;
		nColumns = input -> nColumns;
		size = nRows * nColumns;
	}

	void setParameter(Matrix *input, float alpha) {
		this -> input = input;
		this -> alpha = alpha;

		nRows = input -> nRows;
		nColumns = input -> nColumns;
		size = nRows * nColumns;
	}

	void forward() { 
		for (int i = 0; i < size; ++i) {
			if (input -> value[i] > 0.0) {
				value[i] = input -> value[i];
			} else {
				value[i] = alpha * input -> value[i];
			}
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}
		
	void backward() {
		for (int i = 0; i < size; ++i) {
			if (input -> value[i] > 0.0) {
				input -> gradient[i] += gradient[i];
			} else {
				input -> gradient[i] += gradient[i] * alpha;
			}
		}
	}

	Matrix *input;
	float alpha;

	~LeakyReLU2D() {
	}
};

#endif
