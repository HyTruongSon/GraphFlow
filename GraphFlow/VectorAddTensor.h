// Framework: GraphFlow
// Class: VectorAddTensor
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __VECTORADDTENSOR_H_INCLUDED__
#define __VECTORADDTENSOR_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <assert.h>

#include "Tensor3D.h"

class VectorAddTensor: public Tensor3D {
public:
	VectorAddTensor(int max_nRows, int max_nColumns, int max_nDepth) : Tensor3D(max_nRows, max_nColumns, max_nDepth) {

	}

	VectorAddTensor(Vector *first, Tensor3D *second) : Tensor3D(second -> nRows, second -> nColumns, second -> nDepth) {
		assert(first -> size == second -> nDepth);

		this -> first = first;
		this -> second = second;
	}

	void setParameter(Vector *first, Tensor3D *second) {
		assert(first -> size == second -> nDepth);

		this -> first = first;
		this -> second = second;

		nRows = second -> nRows;
		nColumns = second -> nColumns;
		nDepth = second -> nDepth;
		size = nRows * nColumns * nDepth;
	}

	void forward() {
		for (int i = 0; i < nRows; ++i) {
			for (int j = 0; j < nColumns; ++j) {
				for (int d = 0; d < nDepth; ++d) {
					int ind = index(i, j, d);
					value[ind] = second -> value[ind] + first -> value[d];
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
					second -> gradient[ind] += gradient[ind];
					first -> gradient[d] += gradient[ind];
				}
			}
		}
	}

	Vector *first;
	Tensor3D *second;

	~VectorAddTensor() {
	}
};

#endif