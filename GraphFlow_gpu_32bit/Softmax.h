// Framework: GraphFlow
// Class: Softmax
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __SOFTMAX_H_INCLUDED__
#define __SOFTMAX_H_INCLUDED__

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

class Softmax: public Vector {
public:
	Softmax(int max_size) : Vector(max_size) {

	}

	Softmax(Vector *input) : Vector(input -> size) {
		this -> input = input;
	}

	void setParameter(Vector *input) {
		this -> input = input;
		size = input -> size;
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

	Vector *input;

	~Softmax() {
	}
};

#endif
