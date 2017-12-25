// Framework: GraphFlow
// Class: InnerProduct
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __INNERPRODUCT_H_INCLUDED__
#define __INNERPRODUCT_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <assert.h>

#include "Vector.h"

using namespace std;

class InnerProduct: public Vector {
public:
	InnerProduct() : Vector(1) {

	}

	InnerProduct(Entity *first, Entity *second) : Vector(1) {
		this -> first = (Vector *) first;
		this -> second = (Vector *) second;
		assert(this -> first -> size == this -> second -> size);
	}

	void setParameter(Entity *first, Entity *second) {
		this -> first = (Vector *) first;
		this -> second = (Vector *) second;
		assert(this -> first -> size == this -> second -> size);
	}

	void forward() {
		value[0] = 0.0;
		for (int i = 0; i < first -> size; ++i) {
			value[0] += first -> value[i] * second -> value[i]; 
		}

		gradient[0] = 0.0;
	}

	void backward() {
		for (int i = 0; i < first -> size; ++i) {
			first -> gradient[i] += gradient[0] * second -> value[i];
			second -> gradient[i] += gradient[0] * first -> value[i];
		}
	}

	Vector *first;
	Vector *second;

	~InnerProduct() {
	}
};

#endif
