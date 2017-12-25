// Framework: GraphFlow
// Class: Sort
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __SORT_H_INCLUDED__
#define __SORT_H_INCLUDED__

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

class Sort: public Vector {
public:
	Sort(Vector *input) : Vector(input -> size) {
		this -> input = input;
		arr = new pair<double, int> [this -> input -> size];
	}

	void forward() { 
		for (int i = 0; i < size; ++i) {
			arr[i].first = input -> value[i];
			arr[i].second = i;
		}

		sort(arr, arr + size);

		for (int i = 0; i < size; ++i) {
			value[i] = arr[i].first;
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}
		
	void backward() {
		for (int i = 0; i < size; ++i) {
			int index = arr[i].second;
			input -> gradient[index] += gradient[i];
		}
	}

	pair<double, int> *arr;
	Vector *input;

	~Sort() {
		delete arr;
	}
};

#endif