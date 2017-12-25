// Framework: GraphFlow
// Class: AveragePool2D
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __AVERAGEPOOL2D_H_INCLUDED__
#define __AVERAGEPOOL2D_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <assert.h>

#include "Tensor3D.h"

class AveragePool2D: public Tensor3D {
public:
	AveragePool2D(Tensor3D *input, int window, int stride) 
	: Tensor3D(ceil(float(input -> nRows - window + 1) / float(stride)), ceil(float(input -> nColumns - window + 1) / float(stride)), input -> nDepth) {
		this -> input = input;
		this -> window = window;
		this -> stride = stride;
	}

	void forward() {
		for (int i = 0; i < size; ++i) {
			value[i] = 0.0;
		}

		int row = 0;
		int r = 0;

		while (row + window - 1 < input -> nRows) {
			int column = 0;
			int c = 0;

			while (column + window - 1 < input -> nColumns) {
				for (int d = 0; d < nDepth; ++d) {
					int v = index(r, c, d);
					
					for (int x = 0; x < window; ++x) {
						for (int y = 0; y < window; ++y) {
							int i = input -> index(row + x, column + y, d);
							value[v] += input -> value[i];
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

		for (int i = 0; i < size; ++i) {
			value[i] /= window * window;
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}

	void backward() {
		int row = 0;
		int r = 0;

		while (row + window - 1 < input -> nRows) {
			int column = 0;
			int c = 0;

			while (column + window - 1 < input -> nColumns) {
				for (int d = 0; d < nDepth; ++d) {
					int v = index(r, c, d);
					
					for (int x = 0; x < window; ++x) {
						for (int y = 0; y < window; ++y) {
							int i = input -> index(row + x, column + y, d);
							input -> gradient[i] += gradient[v] / (window * window);
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
	}

	int ceil(float number) {
		if (abs(number - int(number)) < 1e-8) {
			return int(number);
		}
		return int(number) + 1;
	} 

	Tensor3D *input;
	int window;
	int stride;

	~AveragePool2D() {
	}
};

#endif
