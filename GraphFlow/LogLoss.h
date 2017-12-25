// Framework: GraphFlow
// Class: LogLoss
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __LOGLOSS_H_INCLUDED__
#define __LOGLOSS_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <assert.h>

#include "Vector.h"

const double LOG_ZERO = -256.0;

class LogLoss: public Entity {
public:
	LogLoss(Entity *score, Entity *target) {
		this -> score = (Vector *) score;
		this -> target = (Vector *) target;

		assert(this -> target -> size == 1);

		probability = new double [this -> score -> size];

		size = 1;
		value = new double [size];
		gradient = new double [size];
	}

	void forward() {
		int label = (int)(target -> value[0]);

		double MAX = score -> value[0];
		for (int i = 1; i < score -> size; ++i) {
			MAX = max(MAX, score -> value[i]);
		}

		double sum = 0.0;
		for (int i = 0; i < score -> size; ++i) {
			probability[i] = exp(score -> value[i] - MAX);
			sum += probability[i];
		}

		for (int i = 0; i < score -> size; ++i) {
			probability[i] /= sum;
		}

		if (probability[label] <= 0.0) {
			value[0] = LOG_ZERO;
		} else {
			value[0] = log(probability[label]);
		}

		gradient[0] = 0.0;
	}

	void backward() {
		gradient[0] = -1.0;
		
		int label = (int)(target -> value[0]);

		for (int i = 0; i < score -> size; ++i) {
			if (i == label) {
				score -> gradient[i] += probability[i] - 1.0;
			} else {
				score -> gradient[i] += probability[i];
			}
		}
	}

	double getLoss() {
		return value[0];
	}

	double *probability;
	Vector *score;
	Vector *target;

	int size;
	double *value;
	double *gradient;

	~LogLoss() {
		delete[] value;
		delete[] gradient;
	}
};

#endif