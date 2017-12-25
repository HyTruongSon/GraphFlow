// Framework: GraphFlow
// Class: Conv2D
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __CONV2D_H_INCLUDED__
#define __CONV2D_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <assert.h>

#include "Tensor3D.h"
#include "Tensor4D.h"

class Conv2D: public Tensor3D {
public:
	Conv2D(Tensor3D *input, Tensor4D *filter, Matrix *bias, int stride, int pad) 
	: Tensor3D(ceil(double(input -> nRows + 2 * pad - filter -> nRows + 1) / double(stride)), ceil(double(input -> nColumns + 2 * pad - filter -> nColumns + 1) / double(stride)), filter -> nChanels2) {
		assert(filter -> nRows == filter -> nColumns);
		assert(input -> nDepth == filter -> nChanels1);
		assert(filter -> nRows <= input -> nRows + 2 * pad);
		assert(filter -> nColumns <= input -> nColumns + 2 * pad);
		assert(filter -> nChanels1 == bias -> nRows);
		assert(filter -> nChanels2 == bias -> nColumns);

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

		int row = 0;
		int r = 0;
		while (row + filter -> nRows - 1 < input -> nRows + 2 * pad) {
			int column = 0;
			int c = 0;
			while (column + filter -> nColumns - 1 < input -> nColumns + 2 * pad) {
				for (int x = 0; x < filter -> nRows; ++x) {
					for (int y = 0; y < filter -> nColumns; ++y) {
						if ((row + x >= pad) && (row + x < input -> nRows + pad) && (column + y >= pad) && (column + y < input -> nColumns + pad)) {
							for (int chanel1 = 0; chanel1 < filter -> nChanels1; ++chanel1) {
								int i = input -> index(row + x - pad, column + y - pad, chanel1);
								for (int chanel2 = 0; chanel2 < filter -> nChanels2; ++chanel2) {
									int f = filter -> index(x, y, chanel1, chanel2);
									int v = index(r, c, chanel2);
									value[v] += input -> value[i] * filter -> value[f];
								}
							}
						}
					}
				}
				column += stride;
				++c;
			}
			assert(c == nColumns);
			row += stride;
			++r;
		}

		assert(r == nRows);

		for (int r = 0; r < nRows; ++r) {
			for (int c = 0; c < nColumns; ++c) {
				for (int chanel2 = 0; chanel2 < filter -> nChanels2; ++chanel2) {
					int v = index(r, c, chanel2);
					for (int chanel1 = 0; chanel1 < filter -> nChanels1; ++chanel1) {
						int b = bias -> index(chanel1, chanel2);
						value[v] += bias -> value[b];
					}
				}
			}
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}

	void backward() {
		int row = 0;
		int r = 0;
		while (row + filter -> nRows - 1 < input -> nRows + 2 * pad) {
			int column = 0;
			int c = 0;
			while (column + filter -> nColumns - 1 < input -> nColumns + 2 * pad) {
				for (int x = 0; x < filter -> nRows; ++x) {
					for (int y = 0; y < filter -> nColumns; ++y) {
						if ((row + x >= pad) && (row + x < input -> nRows + pad) && (column + y >= pad) && (column + y < input -> nColumns + pad)) {
							for (int chanel1 = 0; chanel1 < filter -> nChanels1; ++chanel1) {
								int i = input -> index(row + x - pad, column + y - pad, chanel1);
								for (int chanel2 = 0; chanel2 < filter -> nChanels2; ++chanel2) {
									int f = filter -> index(x, y, chanel1, chanel2);
									int v = index(r, c, chanel2);
									
									input -> gradient[i] += gradient[v] * filter -> value[f];
									filter -> gradient[f] += gradient[v] * input -> value[i];
								}
							}
						}
					}
				}
				column += stride;
				++c;
			}
			assert(c == nColumns);
			row += stride;
			++r;
		}

		assert(r == nRows);

		for (int r = 0; r < nRows; ++r) {
			for (int c = 0; c < nColumns; ++c) {
				for (int chanel2 = 0; chanel2 < filter -> nChanels2; ++chanel2) {
					int v = index(r, c, chanel2);
					for (int chanel1 = 0; chanel1 < filter -> nChanels1; ++chanel1) {
						int b = bias -> index(chanel1, chanel2);
						bias -> gradient[b] += gradient[v];
					}
				}
			}
		}
	}

	int ceil(double number) {
		if (abs(number - int(number)) < 1e-8) {
			return int(number);
		}
		return int(number) + 1;
	} 

	Tensor3D *input;
	Tensor4D *filter;
	Matrix *bias;
	int stride;
	int pad;

	~Conv2D() {
	}
};

#endif