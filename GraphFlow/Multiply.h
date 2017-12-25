// Framework: GraphFlow
// Class: Multiply
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __MULTIPLY_H_INCLUDED__
#define __MULTIPLY_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <assert.h>

#include "Vector.h"

class Multiply: public Vector {
public:
	Multiply(Vector *first, Vector *second) : Vector(first -> size) {
		this -> first = first;
		this -> second = second;

		assert(this -> first -> size == this -> second -> size);
	}

	void forward() {
		for (int i = 0; i < size; ++i) {
			value[i] = first -> value[i] * second -> value[i];
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}

	void backward() {
		for (int i = 0; i < size; ++i) {
			first -> gradient[i] += gradient[i] * second -> value[i];
			second -> gradient[i] += gradient[i] * first -> value[i];
		}
	}

	Vector *first;
	Vector *second;

	~Multiply() {
	}
};

#endif