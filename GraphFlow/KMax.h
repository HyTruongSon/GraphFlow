// Framework: GraphFlow
// Class: KMax
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __KMAX_H_INCLUDED__
#define __KMAX_H_INCLUDED__

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

class KMax: public Vector {
public:
	KMax(Vector *input, int K) : Vector(K) {
		this -> input = input;
		this -> K = K;
		arr = new pair<double, int> [this -> input -> size];
	}

	void forward() { 
		for (int i = 0; i < input -> size; ++i) {
			arr[i].first = input -> value[i];
			arr[i].second = i;
		}

		sort(arr, arr + input -> size);

		int count = 0;
		for (int i = input -> size - K; i < input -> size; ++i) {
			int index = arr[i].second;
			value[count] = input -> value[index];
			++count;
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}
		
	void backward() {
		int count = 0;
		for (int i = input -> size - K; i < input -> size; ++i) {
			int index = arr[i].second;
			input -> gradient[index] += gradient[count];
			++count;
		}
	}

	int K;
	pair<double, int> *arr;
	Vector *input;

	~KMax() {
		delete arr;
	}
};

#endif