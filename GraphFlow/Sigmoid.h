// Framework: GraphFlow
// Class: Sigmoid
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __SIGMOID_H_INCLUDED__
#define __SIGMOID_H_INCLUDED__

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

class Sigmoid: public Vector {
public:
	Sigmoid(Vector *input) : Vector(input -> size) {
		this -> input = input;
	}

	void forward() { 
		for (int i = 0; i < size; ++i) {
			value[i] = 1.0 / (1.0 + exp(-(input -> value[i])));
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

	Vector *input;

	~Sigmoid() {
	}
};

#endif