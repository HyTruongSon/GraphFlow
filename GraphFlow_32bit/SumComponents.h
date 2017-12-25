// Framework: GraphFlow
// Class: SumComponents
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __SUMCOMPONENTS_H_INCLUDED__
#define __SUMCOMPONENTS_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <assert.h>

#include "Vector.h"

using namespace std;

class SumComponents: public Vector {
public:
	SumComponents(Vector *input) : Vector(1) {
		this -> input = input;
	}

	void forward() {
		value[0] = 0.0;
		for (int i = 0; i < input -> size; ++i) {
			value[0] += input -> value[i];
		}

		gradient[0] = 0.0;
	}

	void backward() {
		for (int i = 0; i < input -> size; ++i) {
			input -> gradient[i] += gradient[0];
		}
	}

	Vector* input;

	~SumComponents() {
	}
};

#endif
