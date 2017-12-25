// Framework: GraphFlow
// Class: MaxPool2D
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __MAXPOOL2D_H_INCLUDED__
#define __MAXPOOL2D_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <assert.h>

#include "Tensor3D.h"

class MaxPool2D: public Tensor3D {
public:
	MaxPool2D(Tensor3D *input, int window, int stride) 
	: Tensor3D(ceil(double(input -> nRows - window + 1) / double(stride)), ceil(double(input -> nColumns - window + 1) / double(stride)), input -> nDepth) {
		this -> input = input;
		this -> window = window;
		this -> stride = stride;

		position = new pair<int, int> [size];
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
					int i = input -> index(row, column, d);

					value[v] = input -> value[i];
					position[v].first = row;
					position[v].second = column;
				
					for (int x = 0; x < window; ++x) {
						for (int y = 0; y < window; ++y) {
							i = input -> index(row + x, column + y, d);

							if (value[v] < input -> value[i]) {
								value[v] = input -> value[i];
								position[v].first = row + x;
								position[v].second = column + y;
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

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}

	void backward() {
		for (int r = 0; r < nRows; ++r) {
			for (int c = 0; c < nColumns; ++c) {
				for (int d = 0; d < nDepth; ++d) {
					int v = index(r, c, d);
					int x = position[v].first;
					int y = position[v].second;
					int i = input -> index(x, y, d);
					input -> gradient[i] += gradient[v];
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
	pair<int, int> *position;
	int window;
	int stride;

	~MaxPool2D() {
		delete position;
	}
};

#endif