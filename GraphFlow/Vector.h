// Framework: GraphFlow
// Class: Vector
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __VECTOR_H_INCLUDED__
#define __VECTOR_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <assert.h>

#include "Entity.h"

class Vector: public Entity {
public:
	Vector(int size) {
		this -> size = size;
		value = new double [this -> size];
		gradient = new double [this -> size];
	}

	void forward() {
		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}

	void backward() {
	}

	int size;
	double *value;
	double *gradient;

	~Vector() {
		delete[] value;
		delete[] gradient;
	}
};

#endif