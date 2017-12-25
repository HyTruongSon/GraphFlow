// Framework: GraphFlow
// Class: CacheParameters
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __CACHEPARAMETERS_H_INCLUDED__
#define __CACHEPARAMETERS_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

#include "Vector.h"

using namespace std;

class CacheParameters {
public:
	CacheParameters() {
		parameters.clear();
		previous_parameters.clear();
	}

	void add(Vector *parameter) {
		Vector *previous_parameter = new Vector(parameter -> size);

		parameters.push_back(parameter);
		previous_parameters.push_back(previous_parameter);
	}

	void clear() {
		for (int i = 0; i < previous_parameters.size(); ++i) {
			delete previous_parameters[i];
		}

		parameters.clear();
		previous_parameters.clear();
	}

	void cache_parameters() {
		for (int i = 0; i < parameters.size(); ++i) {
			for (int j = 0; j < parameters[i] -> size; ++j) {
				previous_parameters[i] -> value[j] = parameters[i] -> value[j];
			}
		}
	}

	void restore_parameters() {
		for (int i = 0; i < parameters.size(); ++i) {
			for (int j = 0; j < parameters[i] -> size; ++j) {
				parameters[i] -> value[j] = previous_parameters[i] -> value[j];
			}
		}
	}

	vector < Vector* > parameters;
	vector < Vector* > previous_parameters;

	~CacheParameters() {
		for (int i = 0; i < previous_parameters.size(); ++i) {
			delete previous_parameters[i];
		}
	}
};

#endif
