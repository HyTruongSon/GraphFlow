// Framework: GraphFlow
// Class: OuterProduct
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __OUTERPRODUCT_H_INCLUDED__
#define __OUTERPRODUCT_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <assert.h>

#include "Matrix.h"

using namespace std;

class OuterProduct: public Matrix {
public:
	OuterProduct(Vector *first, Vector *second) : Matrix(first -> size, second -> size) {
		this -> first = first;
		this -> second = second;
	}

	void forward() {
		for (int i = 0; i < size; ++i) {
			value[i] = 0.0;
		}

		for (int f = 0; f < first -> size; ++f) {
			for (int s = 0; s < second -> size; ++s) {
				int i = index(f, s);
				value[i] = first -> value[f] * second -> value[s];
			}
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}

	void backward() {
		for (int f = 0; f < first -> size; ++f) {
			for (int s = 0; s < second -> size; ++s) {
				int i = index(f, s);
				first -> gradient[f] += gradient[i] * second -> value[s];
				second -> gradient[s] += gradient[i] * first -> value[f];
			}
		}
	}

	Vector *first;
	Vector *second;

	~OuterProduct() {
	}
};

#endif