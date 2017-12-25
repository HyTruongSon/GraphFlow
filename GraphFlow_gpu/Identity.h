// Framework: GraphFlow
// Class: Identity
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __IDENTITY_H_INCLUDED__
#define __IDENTITY_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <assert.h>

#include "Vector.h"

using namespace std;

class Identity: public Vector {
public:
	Identity(Vector *input) : Vector(input -> size) {
		this -> input = input;
	}

	void forward() { 
		for (int i = 0; i < size; ++i) {
			value[i] = input -> value[i];
		}
	}
		
	void backward() {
		for (int i = 0; i < size; ++i) {
			input -> gradient[i] += gradient[i];
		}
	}

	Vector *input;

	~Identity() {
		delete input;
	}
};

#endif