// Framework: GraphFlow
// Class: ConCat
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __CONCAT_H_INCLUDED__
#define __CONCAT_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>
#include <assert.h>

#include "Vector.h"

using namespace std;

class ConCat: public Vector {
public:
	ConCat(Vector *first, Vector *second) : Vector(first -> size + second -> size) {
		this -> first = first;
		this -> second = second;
	}

	void forward() { 
		int count = 0;
		for (int i = 0; i < first -> size; ++i) {
			value[count] = first -> value[i];
			++count;
		}

		for (int i = 0; i < second -> size; ++i) {
			value[count] = second -> value[i];
			++count;
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}
		
	void backward() {
		int count = 0;
		for (int i = 0; i < first -> size; ++i) {
			first -> gradient[i] += gradient[count];
			++count;
		}

		for (int i = 0; i < second -> size; ++i) {
			second -> gradient[i] += gradient[count];
			++count;
		}
	}

	Vector *first;
	Vector *second;

	~ConCat() {
	}
};

#endif