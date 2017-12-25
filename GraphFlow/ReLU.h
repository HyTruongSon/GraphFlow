// Framework: GraphFlow
// Class: ReLU
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __RELU_H_INCLUDED__
#define __RELU_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <assert.h>

#include "Vector.h"

using namespace std;

class ReLU: public Vector {
public:
	ReLU(Vector *input) : Vector(input -> size) {
		this -> input = input;
	}

	void forward() { 
		for (int i = 0; i < size; ++i) {
			value[i] = max(0.0, input -> value[i]);
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}
		
	void backward() {
		for (int i = 0; i < size; ++i) {
			if (value[i] > 0.0) {
				input -> gradient[i] += gradient[i];
			}
		}
	}

	Vector *input;

	~ReLU() {
	}
};

#endif