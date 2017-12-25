// Framework: GraphFlow
// Class: LeakyReLU3D
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __LEAKYRELU3D_H_INCLUDED__
#define __LEAKYRELU3D_H_INCLUDED__

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

class LeakyReLU3D: public Tensor3D {
public:
	LeakyReLU3D(int max_nRows, int max_nColumns, int max_nDepth) : Tensor3D(max_nRows, max_nColumns, max_nDepth) {

	}

	LeakyReLU3D(Tensor3D *input) : Tensor3D(input -> nRows, input -> nColumns, input -> nDepth) {
		this -> input = input;
		alpha = 0.01;
	}

	LeakyReLU3D(Tensor3D *input, double alpha) : Tensor3D(input -> nRows, input -> nColumns, input -> nDepth) {
		this -> input = input;
		this -> alpha = alpha;
	}

	void setParameter(Tensor3D *input) {
		this -> input = input;
		alpha = 0.01;

		nRows = input -> nRows;
		nColumns = input -> nColumns;
		nDepth = input -> nDepth;
		size = nRows * nColumns * nDepth;
	}

	void setParameter(Tensor3D *input, double alpha) {
		this -> input = input;
		this -> alpha = alpha;

		nRows = input -> nRows;
		nColumns = input -> nColumns;
		nDepth = input -> nDepth;
		size = nRows * nColumns * nDepth;
	}


	void forward() { 
		for (int i = 0; i < size; ++i) {
			if (input -> value[i] > 0.0) {
				value[i] = input -> value[i];
			} else {
				value[i] = alpha * input -> value[i];
			}
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}
		
	void backward() {
		for (int i = 0; i < size; ++i) {
			if (input -> value[i] > 0.0) {
				input -> gradient[i] += gradient[i];
			} else {
				input -> gradient[i] += gradient[i] * alpha;
			}
		}
	}

	Tensor3D *input;
	double alpha;

	~LeakyReLU3D() {
	}
};

#endif