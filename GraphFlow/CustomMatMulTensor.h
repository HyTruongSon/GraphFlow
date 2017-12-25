// Framework: GraphFlow
// Class: CustomMatMulTensor
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __CUSTOMMATMULTENSOR_H_INCLUDED__
#define __CUSTOMMATMULTENSOR_H_INCLUDED__

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

class CustomMatMulTensor: public Tensor3D {
public:
	CustomMatMulTensor(int max_nRows, int max_nColumns, int max_nDepth) : Tensor3D(max_nRows, max_nColumns, max_nDepth) {

	}

	CustomMatMulTensor(Matrix *first, Tensor3D *second) : Tensor3D(second -> nRows, second -> nColumns, first -> nRows) {
		assert(first -> nColumns == second -> nDepth);

		this -> first = first;
		this -> second = second;
	}

	void setParameter(Matrix *first, Tensor3D *second) {
		assert(first -> nColumns == second -> nDepth);

		this -> first = first;
		this -> second = second;

		nRows = second -> nRows;
		nColumns = second -> nColumns;
		nDepth = first -> nRows;
		size = nRows * nColumns * nDepth;
	}

	void forward() {
		for (int i = 0; i < size; ++i) {
			value[i] = 0.0;
		}

		for (int i = 0; i < nRows; ++i) {
			for (int j = 0; j < nColumns; ++j) {
				for (int k = 0; k < nDepth; ++k) {
					int ind = index(i, j, k);
					for (int v = 0; v < second -> nDepth; ++v) {
						int f = first -> index(k, v);
						int s = second -> index(i, j, v);
						value[ind] += first -> value[f] * second -> value[s];
					}
				}
			}
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}

	void backward() {
		for (int i = 0; i < nRows; ++i) {
			for (int j = 0; j < nColumns; ++j) {
				for (int k = 0; k < nDepth; ++k) {
					int ind = index(i, j, k);
					for (int v = 0; v < second -> nDepth; ++v) {
						int f = first -> index(k, v);
						int s = second -> index(i, j, v);

						first -> gradient[f] += gradient[ind] * second -> value[s];
						second -> gradient[s] += gradient[ind] * first -> value[f];
					}
				}
			}
		}
	}

	Matrix *first;
	Tensor3D *second;

	~CustomMatMulTensor() {
	}
};

#endif