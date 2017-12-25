// Framework: GraphFlow
// Class: Norm3D
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __NORM3D_H_INCLUDED__
#define __NORM3D_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <assert.h>

#include "Tensor3D.h"

class Norm3D: public Tensor3D {
public:
	Norm3D(Tensor3D *input) : Tensor3D(input -> nRows, input -> nColumns, input -> nDepth) {
		this -> input = input;
		range = new pair < float, float > [input -> nDepth];
	}

	void forward() {
		for (int depth = 0; depth < nDepth; ++depth) {
			float MIN = 1e9;
			float MAX = -1e9;

			for (int row = 0; row < nRows; ++row) {
				for (int column = 0; column < nColumns; ++column) {
					int i = index(row, column, depth);
					MIN = min(MIN, input -> value[i]);
					MAX = max(MAX, input -> value[i]);
				}
			}

			range[depth].first = MIN;

			if (MIN < MAX) {
				range[depth].second = MAX - MIN;
			} else {
				range[depth].second = 1.0;
			}

			for (int row = 0; row < nRows; ++row) {
				for (int column = 0; column < nColumns; ++column) {
					int i = index(row, column, depth);
					value[i] = (input -> value[i] - range[depth].first) / range[depth].second;	
				}
			}
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}

	void backward() {
		for (int depth = 0; depth < nDepth; ++depth) {
			for (int row = 0; row < nRows; ++row) {
				for (int column = 0; column < nColumns; ++column) {
					int i = index(row, column, depth);
					input -> gradient[i] += gradient[i] / range[depth].second;
				}
			}
		}
	}

	pair < float, float > *range;
	Tensor3D *input;

	~Norm3D() {
		delete[] range;
	}
};

#endif
