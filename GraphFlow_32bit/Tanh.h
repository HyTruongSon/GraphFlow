// Framework: GraphFlow
// Class: Tanh
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __TANH_H_INCLUDED__
#define __TANH_H_INCLUDED__

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

class Tanh: public Vector {
public:
	Tanh(Vector *input) : Vector(input -> size) {
		this -> input = input;
	}

	void forward() { 
		float exponent;
		for (int i = 0; i < size; ++i) {
			exponent = exp(- 2.0 * input -> value[i]);
			value[i] = (1.0 - exponent) / (1.0 + exponent);
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}
		
	void backward() {
		for (int i = 0; i < size; ++i) {
			input -> gradient[i] += gradient[i] * (1.0 - value[i] * value[i]);
		}
	}

	Vector *input;

	~Tanh() {
	}
};

#endif
