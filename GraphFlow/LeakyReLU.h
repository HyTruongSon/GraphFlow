// Framework: GraphFlow
// Class: LeakyReLU
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __LEAKYRELU_H_INCLUDED__
#define __LEAKYRELU_H_INCLUDED__

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

class LeakyReLU: public Vector {
public:
	LeakyReLU(int max_size) : Vector(max_size) {

	}

	LeakyReLU(Vector *input) : Vector(input -> size) {
		this -> input = input;
		alpha = 0.01;
	}

	LeakyReLU(Vector *input, double alpha) : Vector(input -> size) {
		this -> input = input;
		this -> alpha = alpha;
	}

	void setParameter(Vector *input)  {
		this -> input = input;
		alpha = 0.01;

		size = input -> size;
	}

	void setParameter(Vector *input, double alpha) {
		this -> input = input;
		this -> alpha = alpha;

		size = input -> size;
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

	Vector *input;
	double alpha;

	~LeakyReLU() {
	}
};

#endif