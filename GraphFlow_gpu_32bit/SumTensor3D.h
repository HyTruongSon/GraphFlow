// Framework: GraphFlow
// Class: SumTensor3D
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __SUMTENSOR3D_H_INCLUDED__
#define __SUMTENSOR3D_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <assert.h>

#include "Tensor3D.h"

using namespace std;

class SumTensor3D: public Tensor3D {
public:
	SumTensor3D(int nRows, int nColumns, int nDepth) : Tensor3D(nRows, nColumns, nDepth) {
		tensors.clear();
	}

	void setParameter(int nRows, int nColumns, int nDepth) {
		this -> nRows = nRows;
		this -> nColumns = nColumns;
		this -> nDepth = nDepth;
		size = this -> nRows * this -> nColumns * this -> nDepth;

		tensors.clear();
	}

	void add_tensor(Tensor3D *tensor) {
		assert(nRows == tensor -> nRows);
		assert(nColumns == tensor -> nColumns);
		assert(nDepth == tensor -> nDepth);
		tensors.push_back(tensor);
	}

	void clear() {
		tensors.clear();
	}

	void forward() {
		for (int i = 0; i < size; ++i) {
			value[i] = 0.0;
		}

		for (int v = 0; v < tensors.size(); ++v) {
			for (int i = 0; i < size; ++i) {
				value[i] += tensors[v] -> value[i];
			}
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}

	void backward() {
		for (int v = 0; v < tensors.size(); ++v) {
			for (int i = 0; i < size; ++i) {
				tensors[v] -> gradient[i] += gradient[i];
			}
		}
	}

	vector < Tensor3D* > tensors;

	~SumTensor3D() {
		tensors.clear();
	}
};

#endif
