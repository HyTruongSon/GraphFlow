// Framework: GraphFlow
// Class: RisiContraction_4
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __RISICONTRACTION_4_H_INCLUDED__
#define __RISICONTRACTION_4_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <assert.h>
#include <thread>

#include "Tensor3D.h"

using namespace std;

class RisiContraction_4: public Tensor3D {
public:
	RisiContraction_4(int max_nRows, int max_nColumns, int max_nDepth) : Tensor3D(max_nRows, max_nColumns, max_nDepth) {
		tensors.clear();
	}

	RisiContraction_4(int N, int nChanels) : Tensor3D(N, N, nContractions * nChanels) {
		this -> N = N;
		this -> nChanels = nChanels;

		tensors.clear();
	}

	void setParameter(int N, int nChanels) {
		this -> N = N;
		this -> nChanels = nChanels;

		this -> nRows = N;
		this -> nColumns = N;
		this -> nDepth = nChanels * nContractions;
		size = nRows * nColumns * nDepth;

		tensors.clear();
	}

	void add_tensor(Tensor3D *tensor) {
		assert(tensor -> nRows == N);
		assert(tensor -> nColumns == N);
		assert(tensor -> nDepth == nChanels);
		tensors.push_back(tensor);
	}

	void clear() {
		tensors.clear();
	}

	float value_at(int a, int b, int c, int f) {
		return tensors[a] -> value[tensors[a] -> index(b, c, f)];
	}

	void set_gradient_for(int a, int b, int c, int f, float grad) {
		tensors[a] -> gradient[tensors[a] -> index(b, c, f)] += grad;
	}

	void forward() {
		assert(tensors.size() == N);

		for (int i = 0; i < size; ++i) {
			value[i] = 0.0;
		}

		int ind;
		float delta;
		int a, b, c, f;

		for (f = 0; f < nChanels; ++f) {
			for (a = 0; a < N; ++a) {
				for (b = 0; b < N; ++b) {
					for (c = 0; c < N; ++c) {

						delta = value_at(a, b, c, f);

						// Case 1: Fix a, b. Contract c.
						ind = index(a, b, 0 * nChanels + f);
						value[ind] += delta;

						// Case 2: Fix b, c. Contract a.
						ind = index(b, c, 1 * nChanels + f);
						value[ind] += delta;
					}
				}
			}
		}

		for (f = 0; f < nChanels; ++f) {
			for (a = 0; a < N; ++a) {
				for (c = 0; c < N; ++c) {

					// Case 3: (a = b, c).
					// a == b
					ind = index(a, c, 2 * nChanels + f);
					value[ind] += value_at(a, a, c, f);
				}
			}
		}

		for (f = 0; f < nChanels; ++f) {
			for (a = 0; a < N; ++a) {
				for (b = 0; b < N; ++b) {

					// Case 4: (a, b = c).
					// b == c
					ind = index(a, b, 3 * nChanels + f);
					value[ind] += value_at(a, b, b, f);
				}
			}
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}

	void backward() {
		assert(tensors.size() == N);
		
		int ind;
		int a, b, c, f;

		for (f = 0; f < nChanels; ++f) {
			for (a = 0; a < N; ++a) {
				for (b = 0; b < N; ++b) {
					for (c = 0; c < N; ++c) {

						// Case 1: Fix a, b. Contract c.
						ind = index(a, b, 0 * nChanels + f);
						set_gradient_for(a, b, c, f, gradient[ind]);

						// Case 2: Fix b, c. Contract a.
						ind = index(b, c, 1 * nChanels + f);
						set_gradient_for(a, b, c, f, gradient[ind]);
					}
				}
			}
		}

		for (f = 0; f < nChanels; ++f) {
			for (a = 0; a < N; ++a) {
				for (c = 0; c < N; ++c) {

					// Case 3: (a = b, c).
					// a == b
					ind = index(a, c, 2 * nChanels + f);
					set_gradient_for(a, a, c, f, gradient[ind]);
				}
			}
		}

		for (f = 0; f < nChanels; ++f) {
			for (a = 0; a < N; ++a) {
				for (b = 0; b < N; ++b) {

					// Case 4: (a, b = c).
					// b == c
					ind = index(a, b, 3 * nChanels + f);
					set_gradient_for(a, b, b, f, gradient[ind]);
				}
			}
		}
	}


	// Number of contractions implemented in this class
	static const int nContractions = 4;

	// The size of the receptive field
	int N;

	// Number of chanels
	int nChanels;

	// Neighbors' representations
	vector < Tensor3D* > tensors;

	~RisiContraction_4() {
		tensors.clear();
	}
};

#endif
