// Framework: GraphFlow
// Class: StackTensor3D
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __STACKTENSOR3D_H_INCLUDED__
#define __STACKTENSOR3D_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <assert.h>
#include <thread>

#include "Tensor3D.h"
#include "Tensor4D.h"

using namespace std;

class StackTensor3D: public Tensor4D {
public:
	StackTensor3D(int nRows, int nColumns, int nChanels1, int nChanels2) : Tensor4D(nRows, nColumns, nChanels1, nChanels2) {
		tensors.clear();
	}

	void setParameter(int nRows, int nColumns, int nChanels1, int nChanels2) {
		this -> nRows = nRows;
		this -> nColumns = nColumns;
		this -> nChanels1 = nChanels1;
		this -> nChanels2 = nChanels2;

		size = nRows * nColumns * nChanels1 * nChanels2;

		tensors.clear();
	}

	void add_tensor(Tensor3D *tensor) {
		assert(tensor -> nRows == nColumns);
		assert(tensor -> nColumns == nChanels1);
		assert(tensor -> nDepth == nChanels2);

		tensors.push_back(tensor);
	}

	void clear() {
		tensors.clear();
	}

	void forward() {
		assert(tensors.size() == nRows);
		
		for (int row = 0; row < nRows; ++row) {
			for (int column = 0; column < nColumns; ++column) {
				for (int chanel1 = 0; chanel1 < nChanels1; ++chanel1) {
					for (int chanel2 = 0; chanel2 < nChanels2; ++chanel2) {
						int i = index(row, column, chanel1, chanel2);
						int j = tensors[row] -> index(column, chanel1, chanel2);

						value[i] = tensors[row] -> value[j];
					}
				}
			}
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}

	void backward() {
		assert(tensors.size() == nRows);
		
		for (int row = 0; row < nRows; ++row) {
			for (int column = 0; column < nColumns; ++column) {
				for (int chanel1 = 0; chanel1 < nChanels1; ++chanel1) {
					for (int chanel2 = 0; chanel2 < nChanels2; ++chanel2) {
						int i = index(row, column, chanel1, chanel2);
						int j = tensors[row] -> index(column, chanel1, chanel2);

						tensors[row] -> gradient[j] += gradient[i];
					}
				}
			}
		}
	}

	// List of tensors
	vector < Tensor3D* > tensors;

	~StackTensor3D() {
		tensors.clear();
	}
};

#endif