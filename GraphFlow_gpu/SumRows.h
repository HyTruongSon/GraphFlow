// Framework: GraphFlow
// Class: SumRows
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __SUMROWS_H_INCLUDED__
#define __SUMROWS_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <assert.h>

#include "Vector.h"
#include "Matrix.h"

using namespace std;

class SumRows: public Vector {
public:
	SumRows(Matrix *input) : Vector(input -> nColumns) {
		this -> input = input;
	}

	void forward() {
		for (int i = 0; i < size; ++i) {
			value[i] = 0.0;
		}

		for (int row = 0; row < input -> nRows; ++row) {
			for (int column = 0; column < input -> nColumns; ++column) {
				int i = input -> index(row, column);
				value[column] += input -> value[i];
			}
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}

	void backward() {
		for (int row = 0; row < input -> nRows; ++row) {
			for (int column = 0; column < input -> nColumns; ++column) {
				int i = input -> index(row, column);
				input -> gradient[i] += gradient[column];
			}
		}
	}

	Matrix *input;

	~SumRows() {
	}
};

#endif