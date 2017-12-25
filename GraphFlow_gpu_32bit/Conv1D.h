// Framework: GraphFlow
// Class: Conv1D
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __CONV1D_H_INCLUDED__
#define __CONV1D_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <assert.h>

#include "Matrix.h"
#include "Tensor3D.h"

class Conv1D: public Matrix {
public:
	Conv1D(Matrix *input, Tensor3D *filter, Vector *bias, int stride, int pad)
	: Matrix(ceil(float(input -> nRows + 2 * pad - filter -> nRows + 1) / float(stride)), filter -> nDepth) {
		nInputChanels = filter -> nColumns;
		nOutputChanels = filter -> nDepth;

		assert(input -> nColumns == nInputChanels);
		assert(filter -> nColumns == nInputChanels);
		assert(filter -> nDepth == nOutputChanels);
		assert(bias -> size == nOutputChanels);
		assert(nColumns == nOutputChanels);
		assert(filter -> nRows <= input -> nRows + 2 * pad);

		this -> input = input;
		this -> filter = filter;
		this -> bias = bias;
		this -> stride = stride;
		this -> pad = pad;
	}

	void forward() {
		for (int i = 0; i < size; ++i) {
			value[i] = 0.0;
		}

		int i = 0;
		int j = 0;
		while (i < input -> nRows + 2 * pad - filter -> nRows + 1) {
			for (int c2 = 0; c2 < nOutputChanels; ++c2) {
				for (int c1 = 0; c1 < nInputChanels; ++c1) {
					for (int v = 0; v < filter -> nRows; ++v) {
						if ((i + v >= pad) && (i + v < input -> nRows + pad)) {
							float input_value = input -> value[input -> index(i + v - pad, c1)];
							float filter_value = filter -> value[filter -> index(v, c1, c2)];
							value[index(j, c2)] += input_value * filter_value;
						}
					}
				}
				value[index(j, c2)] += bias -> value[c2];
			}
			++j;
			i += stride;
		}
		assert(j == nRows);

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}

	void backward() {
		int i = 0;
		int j = 0;
		while (i < input -> nRows + 2 * pad - filter -> nRows + 1) {
			for (int c2 = 0; c2 < nOutputChanels; ++c2) {
				float grad = gradient[index(j, c2)];
				for (int c1 = 0; c1 < nInputChanels; ++c1) {
					for (int v = 0; v < filter -> nRows; ++v) {
						if ((i + v >= pad) && (i + v < input -> nRows + pad)) {
							float input_value = input -> value[input -> index(i + v - pad, c1)];
							float filter_value = filter -> value[filter -> index(v, c1, c2)];
							
							input -> gradient[input -> index(i + v - pad, c1)] += grad * filter_value;
							filter -> gradient[filter -> index(v, c1, c2)] += grad * input_value;
						}
					}
				}
				bias -> gradient[c2] += grad;
			}
			++j;
			i += stride;
		}
		assert(j == nRows);
	}

	int ceil(float number) {
		if (abs(number - int(number)) < 1e-8) {
			return int(number);
		}
		return int(number) + 1;
	} 

	Matrix *input;
	Tensor3D *filter;
	Vector *bias;
	int stride;
	int pad;
	int nInputChanels;
	int nOutputChanels;

	~Conv1D() {
	}
};

#endif
