// Framework: GraphFlow
// Class: SumGradients
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __SUMGRADIENTS_H_INCLUDED__
#define __SUMGRADIENTS_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

#include "Vector.h"

using namespace std;

class SumGradients {
public:
	SumGradients() {
		parameters.clear();
		sum_gradients.clear();
	}

	void add(Vector *parameter) {
		Vector *sum_gradient = new Vector(parameter -> size);

		parameters.push_back(parameter);
		sum_gradients.push_back(sum_gradient);
	}

	void clear() {
		for (int i = 0; i < sum_gradients.size(); ++i) {
			delete sum_gradients[i];
		}
		
		parameters.clear();
		sum_gradients.clear();
	}

	void cache_gradients() {
		for (int i = 0; i < parameters.size(); ++i) {
			for (int j = 0; j < parameters[i] -> size; ++j) {
				sum_gradients[i] -> gradient[j] += parameters[i] -> gradient[j];
			}
		}
	}

	void reset_sum_gradients() {
		for (int i = 0; i < parameters.size(); ++i) {
			for (int j = 0; j < parameters[i] -> size; ++j) {
				sum_gradients[i] -> gradient[j] = 0.0;
			}
		}
	}

	void get_sum_gradients() {
		for (int i = 0; i < parameters.size(); ++i) {
			for (int j = 0; j < parameters[i] -> size; ++j) {
				parameters[i] -> gradient[j] = sum_gradients[i] -> gradient[j];
			}
		}
	}

	vector < Vector* > parameters;
	vector < Vector* > sum_gradients;

	~SumGradients() {
		for (int i = 0; i < sum_gradients.size(); ++i) {
			delete sum_gradients[i];
		}
	}
};

#endif
