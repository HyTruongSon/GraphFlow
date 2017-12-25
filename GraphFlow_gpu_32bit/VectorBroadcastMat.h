// Framework: GraphFlow
// Class: VectorBroadcastMat
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __VECTORBROADCASTMAT_H_INCLUDED__
#define __VECTORBROADCASTMAT_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <assert.h>

#include "Tensor3D.h"

class VectorBroadcastMat: public Tensor3D {
public:
	VectorBroadcastMat(int max_nRows, int max_nColumns, int max_nDepth) : Tensor3D(max_nRows, max_nColumns, max_nDepth) {

	}

	VectorBroadcastMat(Vector *first, Matrix *second) : Tensor3D(second -> nRows, second -> nColumns, first -> size) {
		this -> first = first;
		this -> second = second;
	}

	void setParameter(Vector *first, Matrix *second) {
		this -> first = first;
		this -> second = second;

		nRows = second -> nRows;
		nColumns = second -> nColumns;
		nDepth = first -> size;
		size = nRows * nColumns * nDepth;
	}

	void forward() {
		for (int i = 0; i < nRows; ++i) {
			for (int j = 0; j < nColumns; ++j) {
				for (int d = 0; d < nDepth; ++d) {
					int ind = index(i, j, d);
					value[ind] = second -> value[second -> index(i, j)] * first -> value[d];
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
				for (int d = 0; d < nDepth; ++d) {
					int ind = index(i, j, d);
					second -> gradient[second -> index(i, j)] += gradient[ind] * first -> value[d];
					first -> gradient[d] += gradient[ind] * second -> value[second -> index(i, j)];
				}
			}
		}
	}

	Vector *first;
	Matrix *second;

	~VectorBroadcastMat() {
	}
};

#endif
