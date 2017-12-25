// Framework: GraphFlow
// Class: SquaredLoss
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __SQUAREDLOSS_H_INCLUDED__
#define __SQUAREDLOSS_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <assert.h>

#include "Vector.h"

class SquaredLoss: public Entity {
public:
	SquaredLoss() {
		size = 1;
		value = new float [size];
		gradient = new float [size];
	}

	SquaredLoss(Entity *predict, Entity *target) {
		this -> predict = (Vector *) predict;
		this -> target = (Vector *) target;

		assert(this -> predict -> size == this -> target -> size);

		size = 1;
		value = new float [size];
		gradient = new float [size];
	}

	void setParameter(Entity *predict, Entity *target) {
		this -> predict = (Vector *) predict;
		this -> target = (Vector *) target;

		assert(this -> predict -> size == this -> target -> size);
	}

	void forward() {
		value[0] = 0.0;
		for (int i = 0; i < predict -> size; ++i) {
			value[0] += (predict -> value[i] - target -> value[i]) * (predict -> value[i] - target -> value[i]);
		}
		value[0] *= 0.5;

		gradient[0] = 0.0;
	}

	void backward() {
		gradient[0] = 1.0;
		for (int i = 0; i < predict -> size; ++i) {
			predict -> gradient[i] += predict -> value[i] - target -> value[i];
			target -> gradient[i] += target -> value[i] - predict -> value[i];
		}
	}

	float getLoss() {
		return value[0];
	}

	Vector *predict;
	Vector *target;

	int size;
	float *value;
	float *gradient;

	~SquaredLoss() {
		delete value;
		delete gradient;
	}
};

#endif
