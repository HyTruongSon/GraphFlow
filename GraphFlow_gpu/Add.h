// Framework: GraphFlow
// Class: Add
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __ADD_H_INCLUDED__
#define __ADD_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <assert.h>

#include "Vector.h"

class Add: public Vector {
public:
	Add(Vector *first, Vector *second) : Vector(first -> size) {
		this -> first = first;
		this -> second = second;

		assert(this -> first -> size == this -> second -> size);
	}

	void forward() {
		for (int i = 0; i < size; ++i) {
			value[i] = first -> value[i] + second -> value[i];
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}

	void backward() {
		for (int i = 0; i < size; ++i) {
			first -> gradient[i] += gradient[i];
			second -> gradient[i] += gradient[i];
		}
	}

	Vector *first;
	Vector *second;

	~Add() {
	}
};

#endif