// Framework: GraphFlow
// Class: AdaMax
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __ADAMAX_H_INCLUDED__
#define __ADAMAX_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

#include "Vector.h"

using namespace std;

// Reference: https://arxiv.org/pdf/1412.6980.pdf

class AdaMax {
public:
	AdaMax() {
		beta1 = 0.9;
		beta2 = 0.999;

		beta1_t = 1.0;

		params.clear();
		first_order.clear();
		infinity_norm.clear();
	}

	AdaMax(float beta1, float beta2) {
		this -> beta1 = beta1;
		this -> beta2 = beta2;

		beta1_t = 1.0;

		params.clear();
		first_order.clear();
		infinity_norm.clear();
	}

	void add(Vector *param) {
		params.push_back(param);
		first_order.push_back(new Vector (param -> size));
		infinity_norm.push_back(0.0);

		for (int i = 0; i < param -> size; ++i) {
			first_order[first_order.size() - 1] -> value[i] = 0.0;
		}
	}

	void clear() {
		beta1_t = 1.0;

		for (int i = 0; i < first_order.size(); ++i) {
			delete first_order[i];
		}

		params.clear();
		first_order.clear();
		infinity_norm.clear();
	}

	void Learn(float alpha) {
		float g;

		beta1_t *= beta1;
		for (int i = 0; i < params.size(); ++i) {
			float g_norm = 0.0;
			for (int j = 0; j < params[i] -> size; ++j) {
				// Get gradients w.r.t. stochastic objective at timestep t
				g = params[i] -> gradient[j];

				// Update biased first moment estimate
				first_order[i] -> value[j] = beta1 * first_order[i] -> value[j] + (1.0 - beta1) * g;

				// Compute the infinity norm
				g_norm = max(g_norm, abs(g));
			}

			// Update the exponentially weighted infinity norm
			infinity_norm[i] = max(beta2 * infinity_norm[i], g_norm);

			for (int j = 0; j < params[i] -> size; ++j) {
				// Update parameter
				params[i] -> value[j] -= alpha / (1.0 - beta1_t) * first_order[i] -> value[j] / infinity_norm[i];
			}
		}
	}

	void Learn(float alpha, int nBatch) {
		float g;

		beta1_t *= beta1;
		for (int i = 0; i < params.size(); ++i) {
			float g_norm = 0.0;
			for (int j = 0; j < params[i] -> size; ++j) {
				// Get gradients w.r.t. stochastic objective at timestep t
				g = params[i] -> gradient[j] / nBatch;

				// Update biased first moment estimate
				first_order[i] -> value[j] = beta1 * first_order[i] -> value[j] + (1.0 - beta1) * g;

				// Compute the infinity norm
				g_norm = max(g_norm, abs(g));
			}

			// Update the exponentially weighted infinity norm
			infinity_norm[i] = max(beta2 * infinity_norm[i], g_norm);

			for (int j = 0; j < params[i] -> size; ++j) {
				// Update parameter
				params[i] -> value[j] -= alpha / (1.0 - beta1_t) * first_order[i] -> value[j] / infinity_norm[i];
			}
		}
	}

	float beta1;
	float beta2;

	float beta1_t;

	vector < Vector* > params;
	vector < Vector* > first_order;
	vector < float > infinity_norm;

	~AdaMax() {
		for (int i = 0; i < first_order.size(); ++i) {
			delete first_order[i];
		}
	}
};

#endif
