// Framework: GraphFlow
// Class: DropOut
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __DROPOUT_H_INCLUDED__
#define __DROPOUT_H_INCLUDED__

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

class DropOut: public Vector {
public:
	DropOut(Vector *input, float probability) : Vector(input -> size) {
		this -> input = input;
		this -> probability = probability;
		trainMode = true;
		mask = new bool [size];
	}

	void setMode(bool mode) {
		trainMode = mode;
	}

	float rand_uniform() {
		int RANGE = 1e4;
		int number = abs(rand() % RANGE);
		return float(number) / float(RANGE);
	}

	void forward() { 
		if (trainMode) {
			for (int i = 0; i < size; ++i) {
				if (rand_uniform() <= probability) {
					mask[i] = true;
				} else {
					mask[i] = false;
				}
			}

			for (int i = 0; i < size; ++i) {
				if (mask[i]) {
					value[i] = input -> value[i];
				} else {
					value[i] = 0.0;
				}
			}
		} else {
			for (int i = 0; i < size; ++i) {
				value[i] = probability * input -> value[i];
			}
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}
		
	void backward() {
		for (int i = 0; i < size; ++i) {
			if (mask[i]) {
				input -> gradient[i] += gradient[i];
			}
		}
	}

	Vector *input;
	float probability;
	bool trainMode;
	bool *mask;

	~DropOut() {
		delete[] mask;
	}
};

#endif
