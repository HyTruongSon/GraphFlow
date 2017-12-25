// Framework: GraphFlow
// Class: Adam
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __ADAM_H_INCLUDED__
#define __ADAM_H_INCLUDED__

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

class Adam {
public:
	Adam() {
		beta1 = 0.9;
		beta2 = 0.999;
		epsilon = 1e-8;

		beta1_t = 1.0;
		beta2_t = 1.0;

		params.clear();
		first_order.clear();
		second_order.clear();
	}

	Adam(float beta1, float beta2, float epsilon) {
		this -> beta1 = beta1;
		this -> beta2 = beta2;
		this -> epsilon = epsilon;

		beta1_t = 1.0;
		beta2_t = 1.0;

		params.clear();
		first_order.clear();
		second_order.clear();
	}

	void add(Vector *param) {
		params.push_back(param);
		first_order.push_back(new Vector (param -> size));
		second_order.push_back(new Vector (param -> size));

		for (int i = 0; i < param -> size; ++i) {
			first_order[first_order.size() - 1] -> value[i] = 0.0;
			second_order[second_order.size() - 1] -> value[i] = 0.0;
		}
	}

	void clear() {
		beta1_t = 1.0;
		beta2_t = 1.0;

		for (int i = 0; i < first_order.size(); ++i) {
			delete first_order[i];
			delete second_order[i];
		}

		params.clear();
		first_order.clear();
		second_order.clear();
	}

	void Learn(float alpha) {
		float g;
		float m_hat;
		float v_hat;

		beta1_t *= beta1;
		beta2_t *= beta2;

		for (int i = 0; i < params.size(); ++i) {
			for (int j = 0; j < params[i] -> size; ++j) {
				// Get gradients w.r.t. stochastic objective at timestep t
				g = params[i] -> gradient[j];

				// Update biased first moment estimate
				first_order[i] -> value[j] = beta1 * first_order[i] -> value[j] + (1.0 - beta1) * g;

				// Update biased second raw moment estimate
				second_order[i] -> value[j] = beta2 * second_order[i] -> value[j] + (1.0 - beta2) * g * g;

				// Compute bias-corrected first moment estimate
				m_hat = first_order[i] -> value[j] / (1.0 - beta1_t);

				// Compute bias-corrected second raw moment estimate
				v_hat = second_order[i] -> value[j] / (1.0 - beta2_t);

				// Update parameter
				params[i] -> value[j] -= alpha * m_hat / (sqrt(v_hat) + epsilon);
			}
		}
	}

	void Learn(float alpha, int nBatch) {
		float g;
		float m_hat;
		float v_hat;

		for (int i = 0; i < params.size(); ++i) {
			for (int j = 0; j < params[i] -> size; ++j) {
				// Get gradients w.r.t. stochastic objective at timestep t
				g = params[i] -> gradient[j] / nBatch;

				// Update biased first moment estimate
				first_order[i] -> value[j] = beta1 * first_order[i] -> value[j] + (1.0 - beta1) * g;

				// Update biased second raw moment estimate
				second_order[i] -> value[j] = beta2 * second_order[i] -> value[j] + (1.0 - beta2) * g * g;

				// Compute bias-corrected first moment estimate
				beta1_t *= beta1;
				m_hat = first_order[i] -> value[j] / (1.0 - beta1_t);

				// Compute bias-corrected second raw moment estimate
				beta2_t *= beta2;
				v_hat = second_order[i] -> value[j] / (1.0 - beta2_t);

				// Update parameter
				params[i] -> value[j] -= alpha * m_hat / (sqrt(v_hat) + epsilon);
			}
		}
	}

	float beta1;
	float beta2;
	float epsilon;

	float beta1_t;
	float beta2_t;

	vector < Vector* > params;
	vector < Vector* > first_order;
	vector < Vector* > second_order;

	~Adam() {
		for (int i = 0; i < first_order.size(); ++i) {
			delete first_order[i];
			delete second_order[i];
		}
	}
};

#endif
