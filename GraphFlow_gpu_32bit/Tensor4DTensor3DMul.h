// Framework: GraphFlow
// Class: Tensor4DTensor3DMul
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __TENSOR4DTENSOR3DMUL_H_INCLUDED__
#define __TENSOR4DTENSOR3DMUL_H_INCLUDED__

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

class Tensor4DTensor3DMul: public Tensor3D {
public:
	Tensor4DTensor3DMul(int max_nRows, int max_nColumns, int max_nDepth) : Tensor3D(max_nRows, max_nColumns, max_nDepth) {

	}

	Tensor4DTensor3DMul(Tensor4D *first, Tensor3D *second) : Tensor3D(first -> nRows, second -> nColumns, first -> nChanels2) {
		assert(first -> nColumns == second -> nRows);
		assert(first -> nChanels1 == second -> nDepth);

		this -> first = first;
		this -> second = second;
	}

	void setParameter(Tensor4D *first, Tensor3D *second) {
		assert(first -> nColumns == second -> nRows);
		assert(first -> nChanels1 == second -> nDepth);

		this -> first = first;
		this -> second = second;

		nRows = first -> nRows;
		nColumns = second -> nColumns;
		nDepth = first -> nChanels2;
		size = nRows * nColumns * nDepth;
	}

	void forward() {
		for (int i = 0; i < size; ++i) {
			value[i] = 0.0;
		}

		for (int d = 0; d < nDepth; ++d) {
			for (int c = 0; c < first -> nChanels1; ++c) {
				for (int i = 0; i < nRows; ++i) {
					for (int j = 0; j < nColumns; ++j) {
						int v = index(i, j, d);
						for (int k = 0; k < first -> nColumns; ++k) {
							int f = first -> index(i, k, c, d);
							int s = second -> index(k, j, c);
							value[v] += first -> value[f] * second -> value[s];
						}
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
			for (int c = 0; c < first -> nChanels1; ++c) {
				for (int i = 0; i < nRows; ++i) {
					for (int j = 0; j < nColumns; ++j) {
						int v = index(i, j, d);
						for (int k = 0; k < first -> nColumns; ++k) {
							int f = first -> index(i, k, c, d);
							int s = second -> index(k, j, c);

							first -> gradient[f] += gradient[v] * second -> value[s];
							second -> gradient[s] += gradient[v] * first -> value[f];
						}
					}
				}
			}
		}
	}

	Tensor4D *first;
	Tensor3D *second;

	~Tensor4DTensor3DMul() {
	}
};

#endif
