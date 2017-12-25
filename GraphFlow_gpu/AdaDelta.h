// Framework: GraphFlow
// Class: AdaDelta
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __ADADELTA_H_INCLUDED__
#define __ADADELTA_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

#include "Vector.h"

using namespace std;

// Reference: https://arxiv.org/pdf/1212.5701.pdf

class AdaDelta {
public:
	AdaDelta() {
		p = 0.95;
		epsilon = 1e-6;

		params.clear();
		expected_g.clear();
		expected_deltax.clear();
	}

	AdaDelta(double p, double epsilon) {
		this -> p = p;
		this -> epsilon = epsilon;

		params.clear();
		expected_g.clear();
		expected_deltax.clear();
	}

	void add(Vector *param) {
		params.push_back(param);
		expected_g.push_back(new Vector (param -> size));
		expected_deltax.push_back(new Vector (param -> size));

		for (int i = 0; i < param -> size; ++i) {
			expected_g[expected_g.size() - 1] -> value[i] = 0.0;
			expected_deltax[expected_deltax.size() - 1] -> value[i] = 0.0;
		}
	}

	void clear() {
		for (int i = 0; i < expected_g.size(); ++i) {
			delete expected_g[i];
			delete expected_deltax[i];
		}

		params.clear();
		expected_g.clear();
		expected_deltax.clear();
	}

	void Learn(double alpha) {
		double g;
		double deltax;

		for (int i = 0; i < params.size(); ++i) {
			for (int j = 0; j < params[i] -> size; ++j) {
				// Compute Gradient
				g = params[i] -> gradient[j];

				// Accumulate Gradient
				expected_g[i] -> value[j] = p * expected_g[i] -> value[j] + (1.0 - p) * g * g;

				// Compute Update
				deltax = - sqrt(expected_deltax[i] -> value[j] + epsilon) / sqrt(expected_g[i] -> value[j] + epsilon) * g;

				// Accumulate Updates
				expected_deltax[i] -> value[j] = p * expected_deltax[i] -> value[j] + (1.0 - p) * deltax * deltax;

				// Apply Update
				params[i] -> value[j] += deltax;
			}
		}
	}

	void Learn(double alpha, int nBatch) {
		double g;
		double deltax;

		for (int i = 0; i < params.size(); ++i) {
			for (int j = 0; j < params[i] -> size; ++j) {
				// Compute Gradient
				g = params[i] -> gradient[j] / nBatch;

				// Accumulate Gradient
				expected_g[i] -> value[j] = p * expected_g[i] -> value[j] + (1.0 - p) * g * g;

				// Compute Update
				deltax = - sqrt(expected_deltax[i] -> value[j] + epsilon) / sqrt(expected_g[i] -> value[j] + epsilon) * g;

				// Accumulate Updates
				expected_deltax[i] -> value[j] = p * expected_deltax[i] -> value[j] + (1.0 - p) * deltax * deltax;

				// Apply Update
				params[i] -> value[j] += deltax;
			}
		}
	}

	double p;
	double epsilon;

	vector < Vector* > params;
	vector < Vector* > expected_g;
	vector < Vector* > expected_deltax;

	~AdaDelta() {
		for (int i = 0; i < expected_g.size(); ++i) {
			delete expected_g[i];
			delete expected_deltax[i];
		}
	}
};

#endif