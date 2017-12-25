// Framework: GraphFlow
// Class: Tensor4DConcat
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __TENSOR4DCONCAT_H_INCLUDED__
#define __TENSOR4DCONCAT_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <assert.h>

#include "Tensor4D.h"

class Tensor4DConcat: public Tensor4D {
public:
	Tensor4DConcat(int max_nRows, int max_nColumns, int max_nChanels1, int max_nChanels2) : Tensor4D(max_nRows, max_nColumns, max_nChanels1, max_nChanels2) {

	}

	Tensor4DConcat(Tensor4D *first, Tensor4D *second) : Tensor4D(first -> nRows, first -> nColumns, first -> nChanels1, first -> nChanels2 + second -> nChanels2) {
		assert(first -> nRows == second -> nRows);
		assert(first -> nColumns == second -> nColumns);
		assert(first -> nChanels1 == second -> nChanels1);

		this -> first = first;
		this -> second = second;
	}

	void setParameter(Tensor4D *first, Tensor4D *second) {
		assert(first -> nRows == second -> nRows);
		assert(first -> nColumns == second -> nColumns);
		assert(first -> nChanels1 == second -> nChanels1);

		this -> first = first;
		this -> second = second;

		nRows = first -> nRows;
		nColumns = first -> nColumns;
		nChanels1 = first -> nChanels1;
		nChanels2 = first -> nChanels2 + second -> nChanels2;
		size = nRows * nColumns * nChanels1 * nChanels2;
	}

	void forward() {
		for (int i = 0; i < nRows; ++i) {
			for (int j = 0; j < nColumns; ++j) {
				for (int u = 0; u < nChanels1; ++u) {
					for (int v = 0; v < first -> nChanels2; ++v) {
						int ind = index(i, j, u, v);
						int f = first -> index(i, j, u, v);
						value[ind] = first -> value[f];
					}
				}
			}
		}

		for (int i = 0; i < nRows; ++i) {
			for (int j = 0; j < nColumns; ++j) {
				for (int u = 0; u < nChanels1; ++u) {
					for (int v = 0; v < second -> nChanels2; ++v) {
						int ind = index(i, j, u, first -> nChanels1 + v);
						int s = second -> index(i, j, u, v);
						value[ind] = second -> value[s];
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
				for (int u = 0; u < nChanels1; ++u) {
					for (int v = 0; v < first -> nChanels2; ++v) {
						int ind = index(i, j, u, v);
						int f = first -> index(i, j, u, v);

						first -> gradient[f] += gradient[ind];
					}
				}
			}
		}

		for (int i = 0; i < nRows; ++i) {
			for (int j = 0; j < nColumns; ++j) {
				for (int u = 0; u < nChanels1; ++u) {
					for (int v = 0; v < second -> nChanels2; ++v) {
						int ind = index(i, j, u, first -> nChanels1 + v);
						int s = second -> index(i, j, u, v);

						second -> gradient[s] += gradient[ind];
					}
				}
			}
		}
	}

	Tensor4D *first;
	Tensor4D *second;

	~Tensor4DConcat() {
	}
};

#endif