// Framework: GraphFlow
// Class: RisiContraction_10
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __RISICONTRACTION_10_H_INCLUDED__
#define __RISICONTRACTION_10_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <assert.h>

#include "Tensor3D.h"

using namespace std;

class RisiContraction_10: public Tensor3D {
public:
	RisiContraction_10(int max_nRows, int max_nColumns, int max_nDepth) : Tensor3D(max_nRows, max_nColumns, max_nDepth) {
		tensors.clear();
	}

	RisiContraction_10(int N, int nChanels) : Tensor3D(N, N, nContractions * nChanels) {
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

	void set_adjacency(Matrix *adj) {
		assert(adj -> nRows == N);
		assert(adj -> nColumns == N);
		this -> adj = adj;
	}

	void clear() {
		tensors.clear();
	}

	double value_at(int a, int b, int c, int d, int e, int f) {
		return tensors[a] -> value[tensors[a] -> index(b, c, f)] * adj -> value[adj -> index(d, e)];
	}

	void set_gradient_for(int a, int b, int c, int d, int e, int f, double grad) {
		tensors[a] -> gradient[tensors[a] -> index(b, c, f)] += grad * adj -> value[adj -> index(d, e)];
	}

	void forward() {
		assert(tensors.size() == N);

		for (int i = 0; i < size; ++i) {
			value[i] = 0.0;
		}

		int Case;
		int ind;

		for (int f = 0; f < nChanels; ++f) {
			for (int a = 0; a < N; ++a) {
				for (int b = 0; b < N; ++b) {
					for (int c = 0; c < N; ++c) {
						for (int d = 0; d < N; ++d) {
							for (int e = 0; e < N; ++e) {

								// +-----------+
								// | 1 + 1 + 1 |
								// +-----------+

								// Case 1: Fix a, b. Contract c, d, e.
								Case = 1;
								ind = index(a, b, (Case - 1) * nChanels + f);
								value[ind] += value_at(a, b, c, d, e, f);

								// Case 2: Fix a, c. Contract b, d, e.
								Case = 2;
								ind = index(a, c, (Case - 1) * nChanels + f);
								value[ind] += value_at(a, b, c, d, e, f);

								// Case 3: Fix a, d. Contract b, c, e.
								Case = 3;
								ind = index(a, d, (Case - 1) * nChanels + f);
								value[ind] += value_at(a, b, c, d, e, f);

								// Case 4: Fix a, e. Contract b, c, d.
								Case = 4;
								ind = index(a, e, (Case - 1) * nChanels + f);
								value[ind] += value_at(a, b, c, d, e, f);

								// Case 5: Fix b, c. Contract a, d, e.
								Case = 5;
								ind = index(b, c, (Case - 1) * nChanels + f);
								value[ind] += value_at(a, b, c, d, e, f);

								// Case 6: Fix b, d. Contract a, c, e.
								Case = 6;
								ind = index(b, d, (Case - 1) * nChanels + f);
								value[ind] += value_at(a, b, c, d, e, f);

								// Case 7: Fix b, e. Contract a, c, d.
								Case = 7;
								ind = index(b, e, (Case - 1) * nChanels + f);
								value[ind] += value_at(a, b, c, d, e, f);

								// Case 8: Fix c, d. Contract a, b, e.
								Case = 8;
								ind = index(c, d, (Case - 1) * nChanels + f);
								value[ind] += value_at(a, b, c, d, e, f);

								// Case 9: Fix c, e. Contract a, b, d.
								Case = 9;
								ind = index(c, e, (Case - 1) * nChanels + f);
								value[ind] += value_at(a, b, c, d, e, f);

								// Case 10: Fix d, e. Contract a, b, c.
								Case = 10;
								ind = index(d, e, (Case - 1) * nChanels + f);
								value[ind] += value_at(a, b, c, d, e, f);
							}
						}
					}
				}
			}
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}

	void backward() {
		int Case;
		int ind;

		for (int f = 0; f < nChanels; ++f) {
			for (int a = 0; a < N; ++a) {
				for (int b = 0; b < N; ++b) {
					for (int c = 0; c < N; ++c) {
						for (int d = 0; d < N; ++d) {
							for (int e = 0; e < N; ++e) {

								// +-----------+
								// | 1 + 1 + 1 |
								// +-----------+

								// Case 1: Fix a, b. Contract c, d, e.
								Case = 1;
								ind = index(a, b, (Case - 1) * nChanels + f);
								set_gradient_for(a, b, c, d, e, f, gradient[ind]);

								// Case 2: Fix a, c. Contract b, d, e.
								Case = 2;
								ind = index(a, c, (Case - 1) * nChanels + f);
								set_gradient_for(a, b, c, d, e, f, gradient[ind]);

								// Case 3: Fix a, d. Contract b, c, e.
								Case = 3;
								ind = index(a, d, (Case - 1) * nChanels + f);
								set_gradient_for(a, b, c, d, e, f, gradient[ind]);

								// Case 4: Fix a, e. Contract b, c, d.
								Case = 4;
								ind = index(a, e, (Case - 1) * nChanels + f);
								set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								
								// Case 5: Fix b, c. Contract a, d, e.
								Case = 5;
								ind = index(b, c, (Case - 1) * nChanels + f);
								set_gradient_for(a, b, c, d, e, f, gradient[ind]);

								// Case 6: Fix b, d. Contract a, c, e.
								Case = 6;
								ind = index(b, d, (Case - 1) * nChanels + f);
								set_gradient_for(a, b, c, d, e, f, gradient[ind]);

								// Case 7: Fix b, e. Contract a, c, d.
								Case = 7;
								ind = index(b, e, (Case - 1) * nChanels + f);
								set_gradient_for(a, b, c, d, e, f, gradient[ind]);

								// Case 8: Fix c, d. Contract a, b, e.
								Case = 8;
								ind = index(c, d, (Case - 1) * nChanels + f);
								set_gradient_for(a, b, c, d, e, f, gradient[ind]);

								// Case 9: Fix c, e. Contract a, b, d.
								Case = 9;
								ind = index(c, e, (Case - 1) * nChanels + f);
								set_gradient_for(a, b, c, d, e, f, gradient[ind]);

								// Case 10: Fix d, e. Contract a, b, c.
								Case = 10;
								ind = index(d, e, (Case - 1) * nChanels + f);
								set_gradient_for(a, b, c, d, e, f, gradient[ind]);
							}
						}
					}
				}
			}
		}
	}

	// Number of contractions implemented in this class
	static const int nContractions = 10;

	// The size of the receptive field
	int N;

	// Number of chanels
	int nChanels;

	// Neighbors' representations
	vector < Tensor3D* > tensors;

	// The reduced adjacency matrix
	Matrix *adj;

	~RisiContraction_10() {
		tensors.clear();
	}
};

#endif