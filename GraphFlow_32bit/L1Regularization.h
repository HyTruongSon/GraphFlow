// Framework: GraphFlow
// Class: L1Regularization
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __L1REGULARIZATION_H_INCLUDED__
#define __L1REGULARIZATION_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <assert.h>

#include "Vector.h"

class L1Regularization: public Entity {
public:
	L1Regularization(float lambda) {
		this -> lambda = lambda;
		params.clear();

		size = 1;
		value = new float [size];
		gradient = new float [size];
	}

	void add(Vector *param) {
		params.push_back(param);
	}

	void clear() {
		params.clear();
	}

	void forward() {
		value[0] = 0.0;
		for (int i = 0; i < params.size(); ++i) {
			for (int j = 0; j < params[i] -> size; ++j) {
				value[0] += abs(params[i] -> value[j]);
			}
		}
		value[0] *= lambda;

		gradient[0] = 0.0;
	}

	float sign(float number) {
		if (abs(number) < 1e-8) {
			return 0.0;
		}
		if (number > 0.0) {
			return 1.0;
		}
		return -1.0;
	}

	void backward() {
		gradient[0] = 1.0;
		for (int i = 0; i < params.size(); ++i) {
			for (int j = 0; j < params[i] -> size; ++j) {
				params[i] -> gradient[j] += lambda * sign(params[i] -> value[j]);
			}
		}
	}

	float getLoss() {
		return value[0];
	}

	float lambda;
	vector < Vector* > params;

	int size;
	float *value;
	float *gradient;

	~L1Regularization() {
		delete value;
		delete gradient;
	}
};

#endif
