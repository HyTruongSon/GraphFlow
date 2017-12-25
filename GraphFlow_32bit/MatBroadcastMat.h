// Framework: GraphFlow
// Class: MatBroadcastMat
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __MATBROADCASTMAT_H_INCLUDED__
#define __MATBROADCASTMAT_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <assert.h>

#include "Matrix.h"
#include "Tensor4D.h"

class MatBroadcastMat: public Tensor4D {
public:
	MatBroadcastMat(int max_nRows, int max_nColumns, int max_nChanels1, int max_nChanels2) : Tensor4D(max_nRows, max_nColumns, max_nChanels1, max_nChanels2) {

	}

	MatBroadcastMat(Matrix *first, Matrix *second) : Tensor4D(first -> nRows, first -> nColumns, second -> nRows, second -> nColumns) {
		this -> first = first;
		this -> second = second;
	}

	void setParameter(Matrix *first, Matrix *second) {
		this -> first = first;
		this -> second = second;

		nRows = first -> nRows;
		nColumns = first -> nColumns;
		nChanels1 = second -> nRows;
		nChanels2 = second -> nColumns;
		size = nRows * nColumns * nChanels1 * nChanels2;
	}

	void forward() {
		for (int i = 0; i < first -> nRows; ++i) {
			for (int j = 0; j < first -> nColumns; ++j) {
				int f = first -> index(i, j);
				for (int u = 0; u < second -> nRows; ++u) {
					for (int v = 0; v < second -> nColumns; ++v) {
						int s = second -> index(u, v);
						int ind = index(i, j, u, v);
						value[ind] = first -> value[f] * second -> value[s];
					}
				}
			}
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}

	void backward() {
		for (int i = 0; i < first -> nRows; ++i) {
			for (int j = 0; j < first -> nColumns; ++j) {
				int f = first -> index(i, j);
				for (int u = 0; u < second -> nRows; ++u) {
					for (int v = 0; v < second -> nColumns; ++v) {
						int s = second -> index(u, v);
						int ind = index(i, j, u, v);

						first -> gradient[f] += gradient[ind] * second -> value[s];
						second -> gradient[s] += gradient[ind] * first -> value[f];
					}
				}
			}
		}
	}

	Matrix *first;
	Matrix *second;

	~MatBroadcastMat() {
	}
};

#endif
