// Framework: GraphFlow
// Class: Reshape4D
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __RESHAPE4D_H_INCLUDED__
#define __RESHAPE4D_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <assert.h>

#include "Vector.h"
#include "Tensor4D.h"

class Reshape4D: public Tensor4D {
public:
	Reshape4D(Vector *input, int nRows, int nColumns, int nChanels1, int nChanels2) : Tensor4D(nRows, nColumns, nChanels1, nChanels2) {
		assert(input -> size == nRows * nColumns * nChanels1 * nChanels2);
		this -> input = input;
	}

	void forward() {
		int count = 0;
		for (int row = 0; row < nRows; ++row) {
			for (int column = 0; column < nColumns; ++column) {
				for (int chanel1 = 0; chanel1 < nChanels1; ++chanel1) {
					for (int chanel2 = 0; chanel2 < nChanels2; ++chanel2) {
						int i = index(row, column, chanel1, chanel2);
						value[i] = input -> value[count];
						++count;
					}
				}
			}
		}
		assert(count == input -> size);

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}

	void backward() {
		int count = 0;
		for (int row = 0; row < nRows; ++row) {
			for (int column = 0; column < nColumns; ++column) {
				for (int chanel1 = 0; chanel1 < nChanels1; ++chanel1) {
					for (int chanel2 = 0; chanel2 < nChanels2; ++chanel2) {
						int i = index(row, column, chanel1, chanel2);
						input -> gradient[count] += gradient[i];
						++count;
					}
				}
			}
		}
		assert(count == input -> size);
	}

	Vector *input;

	~Reshape4D() {
	}
};

#endif