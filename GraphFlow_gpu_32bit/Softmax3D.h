// Framework: GraphFlow
// Class: Softmax3D
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __SOFTMAX3D_H_INCLUDED__
#define __SOFTMAX3D_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <assert.h>

#include "Tensor3D.h"

using namespace std;

class Softmax3D: public Tensor3D {
public:
	Softmax3D(int max_nRows, int max_nColumns, int max_nDepth) : Tensor3D(max_nRows, max_nColumns, max_nDepth) {
	}

	void setParameter(Tensor3D *input) {
		this -> input = input;
		nRows = this -> input -> nRows;
		nColumns = this -> input -> nColumns;
		nDepth = this -> input -> nDepth;
		size = this -> input -> size;
	}

	void forward() { 
		float MAX = input -> value[0];
		for (int i = 1; i < size; ++i) {
			MAX = max(MAX, input -> value[i]);
		}

		float sum = 0.0;
		for (int i = 0; i < size; ++i) {
			value[i] = exp(input -> value[i] - MAX);
			sum += value[i];
		}

		for (int i = 0; i < size; ++i) {
			value[i] /= sum;
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}
		
	void backward() {
		for (int i = 0; i < size; ++i) {
			input -> gradient[i] += gradient[i] * value[i] * (1.0 - value[i]);
		}
	}

	Tensor3D *input;

	~Softmax3D() {
	}
};

#endif
