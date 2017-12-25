// Framework: GraphFlow
// Class: TensorMatMul
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __TENSORMATMUL_H_INCLUDED__
#define __TENSORMATMUL_H_INCLUDED__

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

class TensorMatMul: public Tensor3D {
public:
	TensorMatMul(int max_nRows, int max_nColumns, int max_nDepth) : Tensor3D(max_nRows, max_nColumns, max_nDepth) {

	}

	TensorMatMul(Tensor3D *first, Matrix *second) : Tensor3D(first -> nRows, second -> nColumns, first -> nDepth) {
		assert(first -> nColumns == second -> nRows);

		this -> first = first;
		this -> second = second;
	}

	void setParameter(Tensor3D *first, Matrix *second) {
		assert(first -> nColumns == second -> nRows);
		this -> first = first;
		this -> second = second;

		nRows = first -> nRows;
		nColumns = second -> nColumns;
		nDepth = first -> nDepth;
		size = nRows * nColumns * nDepth;
	}

	void forward() {
		for (int i = 0; i < size; ++i) {
			value[i] = 0.0;
		}

		for (int d = 0; d < nDepth; ++d) {
			for (int i = 0; i < nRows; ++i) {
				for (int j = 0; j < nColumns; ++j) {
					int v = index(i, j, d);
					for (int k = 0; k < first -> nColumns; ++k) {
						int f = first -> index(i, k, d);
						int s = second -> index(k, j);
						value[v] += first -> value[f] * second -> value[s];
					}
				}
			}
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}

	void backward() {
		for (int d = 0; d < nDepth; ++d) {
			for (int i = 0; i < nRows; ++i) {
				for (int j = 0; j < nColumns; ++j) {
					int v = index(i, j, d);
					for (int k = 0; k < first -> nColumns; ++k) {
						int f = first -> index(i, k, d);
						int s = second -> index(k, j);

						first -> gradient[f] += gradient[v] * second -> value[s];
						second -> gradient[s] += gradient[v] * first -> value[f];
					}
				}
			}
		}
	}

	Tensor3D *first;
	Matrix *second;

	~TensorMatMul() {
	}
};

#endif