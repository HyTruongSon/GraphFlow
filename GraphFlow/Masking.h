// Framework: GraphFlow
// Class: Masking
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __MASKING_H_INCLUDED__
#define __MASKING_H_INCLUDED__

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

class Masking: public Vector {
public:
	Masking(Vector *input, Vector *mask) : Vector(input -> size) {
		assert(input -> size == mask -> size);

		this -> input = input;
		this -> mask = mask;
	}

	void forward() { 
		for (int i = 0; i < size; ++i) {
			if (mask -> value[i] > 0.0) {
				value[i] = input -> value[i];
			} else {
				value[i] = 0.0;
			}
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}
		
	void backward() {
		for (int i = 0; i < size; ++i) {
			if (mask -> value[i] > 0.0) {
				input -> gradient[i] += gradient[i];
			}
		}
	}

	Vector *input;
	Vector *mask;

	~Masking() {
	}
};

#endif