// Framework: GraphFlow
// Class: Tensor3DConcat
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __TENSOR3DCONCAT_H_INCLUDED__
#define __TENSOR3DCONCAT_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <assert.h>

#include "Tensor3D.h"

class Tensor3DConcat: public Tensor3D {
public:
	Tensor3DConcat(int max_nRows, int max_nColumns, int max_nDepth) : Tensor3D(max_nRows, max_nColumns, max_nDepth) {

	}

	Tensor3DConcat(Tensor3D *first, Tensor3D *second) : Tensor3D(first -> nRows, first -> nColumns, first -> nDepth + second -> nDepth) {
		assert(first -> nRows == second -> nRows);
		assert(first -> nColumns == second -> nColumns);

		this -> first = first;
		this -> second = second;
	}

	void setParameter(Tensor3D *first, Tensor3D *second) {
		assert(first -> nRows == second -> nRows);
		assert(first -> nColumns == second -> nColumns);

		this -> first = first;
		this -> second = second;

		nRows = first -> nRows;
		nColumns = first -> nColumns;
		nDepth = first -> nDepth + second -> nDepth;
		size = nRows * nColumns * nDepth;
	}

	void forward() {
		for (int i = 0; i < nRows; ++i) {
			for (int j = 0; j < nColumns; ++j) {
				for (int v = 0; v < first -> nDepth; ++v) {
					int ind = index(i, j, v);
					int f = first -> index(i, j, v);
					value[ind] = first -> value[f];
				}
			}
		}

		for (int i = 0; i < nRows; ++i) {
			for (int j = 0; j < nColumns; ++j) {
				for (int v = 0; v < second -> nDepth; ++v) {
					int ind = index(i, j, first -> nDepth + v);
					int s = second -> index(i, j, v);
					value[ind] = second -> value[s];
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
				for (int v = 0; v < first -> nDepth; ++v) {
					int ind = index(i, j, v);
					int f = first -> index(i, j, v);

					first -> gradient[f] += gradient[ind];
				}
			}
		}

		for (int i = 0; i < nRows; ++i) {
			for (int j = 0; j < nColumns; ++j) {
				for (int v = 0; v < second -> nDepth; ++v) {
					int ind = index(i, j, first -> nDepth + v);
					int s = second -> index(i, j, v);

					second -> gradient[s] += gradient[ind];
				}
			}
		}
	}

	Tensor3D *first;
	Tensor3D *second;

	~Tensor3DConcat() {
	}
};

#endif
