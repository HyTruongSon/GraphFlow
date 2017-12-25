// Framework: GraphFlow
// Class: SGD
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __SGD_H_INCLUDED__
#define __SGD_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

#include "Vector.h"

using namespace std;

class SGD {
public:
	SGD() {
		params.clear();
	}

	void add(Vector *param) {
		params.push_back(param);
	}

	void clear() {
		params.clear();
	}

	void Learn(double learning_rate) {
		for (int i = 0; i < params.size(); ++i) {
			for (int j = 0; j < params[i] -> size; ++j) {
				params[i] -> value[j] -= learning_rate * params[i] -> gradient[j];
			}
		}
	}

	void Learn(double learning_rate, int nBatch) {
		for (int i = 0; i < params.size(); ++i) {
			for (int j = 0; j < params[i] -> size; ++j) {
				params[i] -> value[j] -= learning_rate * params[i] -> gradient[j] / nBatch;
			}
		}
	}

	vector < Vector* > params;

	~SGD() {
	}
};

#endif