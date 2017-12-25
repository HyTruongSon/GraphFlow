// Framework: GraphFlow
// Class: ShrinkTensor
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __SHRINKTENSOR_H_INCLUDED__
#define __SHRINKTENSOR_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <assert.h>

#include "Vector.h"
#include "Tensor3D.h"

class ShrinkTensor: public Vector {
public:
	ShrinkTensor(int max_size) : Vector(max_size) {

	}

	ShrinkTensor(Tensor3D *input) : Vector(input -> nDepth) {
		this -> input = input;
	}

	void setParameter(Tensor3D *input) {
		this -> input = input;
		
		size = input -> nDepth;
	}

	void forward() {
		for (int d = 0; d < input -> nDepth; ++d) {
			value[d] = 0.0;
			for (int i = 0; i < input -> nRows; ++i) {
				for (int j = 0; j < input -> nColumns; ++j) {
					int ind = input -> index(i, j, d);
					value[d] += input -> value[ind];
				}
			}
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}

	void backward() {
		for (int d = 0; d < input -> nDepth; ++d) {
			for (int i = 0; i < input -> nRows; ++i) {
				for (int j = 0; j < input -> nColumns; ++j) {
					int ind = input -> index(i, j, d);
					input -> gradient[ind] += gradient[d];
				}
			}
		}
	}

	Tensor3D *input;

	~ShrinkTensor() {
	}
};

#endif