// Framework: GraphFlow
// Class: Momentum
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __MOMENTUM_H_INCLUDED__
#define __MOMENTUM_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

#include "Vector.h"

using namespace std;

class Momentum {
public:
	Momentum() {
		gamma = 0.9;
		params.clear();
		moments.clear();
	}

	Momentum(float gamma) {
		this -> gamma = gamma;
		params.clear();
		moments.clear();
	}

	void add(Vector *param) {
		params.push_back(param);
		moments.push_back(new Vector (param -> size));

		for (int i = 0; i < param -> size; ++i) {
			moments[moments.size() - 1] -> value[i] = 0.0;
		}
	}

	void clear() {
		for (int i = 0; i < moments.size(); ++i) {
			delete moments[i];
		}
		
		params.clear();
		moments.clear();
	}

	void Learn(float learning_rate) {
		for (int i = 0; i < params.size(); ++i) {
			for (int j = 0; j < params[i] -> size; ++j) {
				moments[i] -> value[j] = gamma * moments[i] -> value[j] + learning_rate * params[i] -> gradient[j];
				params[i] -> value[j] -= moments[i] -> value[j];
			}
		}
	}

	void Learn(float learning_rate, int nBatch) {
		for (int i = 0; i < params.size(); ++i) {
			for (int j = 0; j < params[i] -> size; ++j) {
				moments[i] -> value[j] = gamma * moments[i] -> value[j] + learning_rate * params[i] -> gradient[j] / nBatch;
				params[i] -> value[j] -= moments[i] -> value[j];
			}
		}
	}

	float gamma;
	vector < Vector* > params;
	vector < Vector* > moments;

	~Momentum() {
		for (int i = 0; i < moments.size(); ++i) {
			delete moments[i];
		}
	}
};

#endif
