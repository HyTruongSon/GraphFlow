// Framework: GraphFlow
// Class: RisiContraction_50
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __RISICONTRACTION_50_H_INCLUDED__
#define __RISICONTRACTION_50_H_INCLUDED__

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

class RisiContraction_50: public Tensor3D {
public:
	RisiContraction_50(int max_nRows, int max_nColumns, int max_nDepth) : Tensor3D(max_nRows, max_nColumns, max_nDepth) {
		tensors.clear();
	}

	RisiContraction_50(int N, int nChanels) : Tensor3D(N, N, nContractions * nChanels) {
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

	float value_at(int a, int b, int c, int d, int e, int f) {
		return tensors[a] -> value[tensors[a] -> index(b, c, f)] * adj -> value[adj -> index(d, e)];
	}

	void set_gradient_for(int a, int b, int c, int d, int e, int f, float grad) {
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

								// +-------+
								// | 1 + 2 |
								// +-------+

								// Case 11: (a, b). Contract (c, d). Singleton (e).
								Case = 11;
								ind = index(a, b, (Case - 1) * nChanels + f);
								if (c == d) {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 12: (a, b). Contract (c, e). Singleton (d).
								Case = 12;
								ind = index(a, b, (Case - 1) * nChanels + f);
								if (c == e) {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 13: (a, b). Contract (d, e). Singleton (c).
								Case = 13;
								ind = index(a, b, (Case - 1) * nChanels + f);
								if (d == e) {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 14: (a, c). Contract (b, d). Singleton (e).
								Case = 14;
								ind = index(a, c, (Case - 1) * nChanels + f);
								if (b == d) {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 15: (a, c). Contract (b, e). Singleton (d).
								Case = 15;
								ind = index(a, c, (Case - 1) * nChanels + f);
								if (b == e) {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 16: (a, c). Contract (d, e). Singleton (b).
								Case = 16;
								ind = index(a, c, (Case - 1) * nChanels + f);
								if (d == e) {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 17: (a, d). Contract (b, c). Singleton (e).
								Case = 17;
								ind = index(a, d, (Case - 1) * nChanels + f);
								if (b == c) {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 18: (a, d). Contract (b, e). Singleton (c).
								Case = 18;
								ind = index(a, d, (Case - 1) * nChanels + f);
								if (b == e) {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 19: (a, d). Contract (c, e). Singleton (b).
								Case = 19;
								ind = index(a, d, (Case - 1) * nChanels + f);
								if (c == e) {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 20: (a, e). Contract (b, c). Singleton (d).
								Case = 20;
								ind = index(a, e, (Case - 1) * nChanels + f);
								if (b == c) {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 21: (a, e). Contract (b, d). Singleton (c).
								Case = 21;
								ind = index(a, e, (Case - 1) * nChanels + f);
								if (b == d) {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 22: (a, e). Contract (c, d). Singleton (b).
								Case = 22;
								ind = index(a, e, (Case - 1) * nChanels + f);
								if (c == d) {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 23: (b, c). Contract (a, d). Singleton (e).
								Case = 23;
								ind = index(b, c, (Case - 1) * nChanels + f);
								if (a == d) {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 24: (b, c). Contract (a, e). Singleton (d).
								Case = 24;
								ind = index(b, c, (Case - 1) * nChanels + f);
								if (a == e) {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 25: (b, c). Contract (d, e). Singleton (a).
								Case = 25;
								ind = index(b, c, (Case - 1) * nChanels + f);
								if (d == e) {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 26: (b, d). Contract (a, c). Singleton (e).
								Case = 26;
								ind = index(b, d, (Case - 1) * nChanels + f);
								if (a == c) {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 27: (b, d). Contract (a, e). Singleton (c).
								Case = 27;
								ind = index(b, d, (Case - 1) * nChanels + f);
								if (a == e) {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 28: (b, d). Contract (c, e). Singleton (a).
								Case = 28;
								ind = index(b, d, (Case - 1) * nChanels + f);
								if (c == e) {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 29: (b, e). Contract (a, c). Singleton (d).
								Case = 29;
								ind = index(b, e, (Case - 1) * nChanels + f);
								if (a == c) {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 30: (b, e). Contract (a, d). Singleton (c).
								Case = 30;
								ind = index(b, e, (Case - 1) * nChanels + f);
								if (a == d) {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 31: (b, e). Contract (c, d). Singleton (a).
								Case = 31;
								ind = index(b, e, (Case - 1) * nChanels + f);
								if (c == d) {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 32: (c, d). Contract (a, b). Singleton (e).
								Case = 32;
								ind = index(c, d, (Case - 1) * nChanels + f);
								if (a == b) {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 33: (c, d). Contract (a, e). Singleton (b).
								Case = 33;
								ind = index(c, d, (Case - 1) * nChanels + f);
								if (a == e) {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 34: (c, d). Contract (b, e). Singleton (a).
								Case = 34;
								ind = index(c, d, (Case - 1) * nChanels + f);
								if (b == e) {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 35: (c, e). Contract (a, b). Singleton (d).
								Case = 35;
								ind = index(c, e, (Case - 1) * nChanels + f);
								if (a == b) {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 36: (c, e). Contract (a, d). Singleton (b).
								Case = 36;
								ind = index(c, e, (Case - 1) * nChanels + f);
								if (a == d) {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 37: (c, e). Contract (b, d). Singleton (a).
								Case = 37;
								ind = index(c, e, (Case - 1) * nChanels + f);
								if (b == d) {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 38: (d, e). Contract (a, b). Singleton (c).
								Case = 38;
								ind = index(d, e, (Case - 1) * nChanels + f);
								if (a == b) {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 39: (d, e). Contract (a, c). Singleton (b).
								Case = 39;
								ind = index(d, e, (Case - 1) * nChanels + f);
								if (a == c) {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 40: (d, e). Contract (b, c). Singleton (a).
								Case = 40;
								ind = index(d, e, (Case - 1) * nChanels + f);
								if (b == c) {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// +---+
								// | 3 |
								// +---+

								// Case 41: (a, b). Contract (c, d, e).
								Case = 41;
								ind = index(a, b, (Case - 1) * nChanels + f);
								if ((c == d) && (d == e))  {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 42: (a, c). Contract (b, d, e).
								Case = 42;
								ind = index(a, c, (Case - 1) * nChanels + f);
								if ((b == d) && (d == e))  {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 43: (a, d). Contract (b, c, e).
								Case = 43;
								ind = index(a, d, (Case - 1) * nChanels + f);
								if ((b == c) && (c == e))  {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 44: (a, e). Contract (b, c, d).
								Case = 44;
								ind = index(a, e, (Case - 1) * nChanels + f);
								if ((b == c) && (c == d))  {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 45: (b, c). Contract (a, d, e).
								Case = 45;
								ind = index(b, c, (Case - 1) * nChanels + f);
								if ((a == d) && (d == e))  {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 46: (b, d). Contract (a, c, e).
								Case = 46;
								ind = index(b, d, (Case - 1) * nChanels + f);
								if ((a == c) && (c == e))  {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 47: (b, e). Contract (a, c, d).
								Case = 47;
								ind = index(b, e, (Case - 1) * nChanels + f);
								if ((a == c) && (c == d))  {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 48: (c, d). Contract (a, b, e).
								Case = 48;
								ind = index(c, d, (Case - 1) * nChanels + f);
								if ((a == b) && (b == e))  {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 49: (c, e). Contract (a, b, d).
								Case = 49;
								ind = index(c, e, (Case - 1) * nChanels + f);
								if ((a == b) && (b == d))  {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 50: (d, e). Contract (a, b, c).
								Case = 50;
								ind = index(d, e, (Case - 1) * nChanels + f);
								if ((a == b) && (b == c))  {
									value[ind] += value_at(a, b, c, d, e, f);
								}
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

								// +-------+
								// | 1 + 2 |
								// +-------+

								// Case 11: (a, b). Contract (c, d). Singleton (e).
								Case = 11;
								ind = index(a, b, (Case - 1) * nChanels + f);
								if (c == d) {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 12: (a, b). Contract (c, e). Singleton (d).
								Case = 12;
								ind = index(a, b, (Case - 1) * nChanels + f);
								if (c == e) {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 13: (a, b). Contract (d, e). Singleton (c).
								Case = 13;
								ind = index(a, b, (Case - 1) * nChanels + f);
								if (d == e) {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 14: (a, c). Contract (b, d). Singleton (e).
								Case = 14;
								ind = index(a, c, (Case - 1) * nChanels + f);
								if (b == d) {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 15: (a, c). Contract (b, e). Singleton (d).
								Case = 15;
								ind = index(a, c, (Case - 1) * nChanels + f);
								if (b == e) {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 16: (a, c). Contract (d, e). Singleton (b).
								Case = 16;
								ind = index(a, c, (Case - 1) * nChanels + f);
								if (d == e) {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 17: (a, d). Contract (b, c). Singleton (e).
								Case = 17;
								ind = index(a, d, (Case - 1) * nChanels + f);
								if (b == c) {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 18: (a, d). Contract (b, e). Singleton (c).
								Case = 18;
								ind = index(a, d, (Case - 1) * nChanels + f);
								if (b == e) {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 19: (a, d). Contract (c, e). Singleton (b).
								Case = 19;
								ind = index(a, d, (Case - 1) * nChanels + f);
								if (c == e) {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 20: (a, e). Contract (b, c). Singleton (d).
								Case = 20;
								ind = index(a, e, (Case - 1) * nChanels + f);
								if (b == c) {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 21: (a, e). Contract (b, d). Singleton (c).
								Case = 21;
								ind = index(a, e, (Case - 1) * nChanels + f);
								if (b == d) {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 22: (a, e). Contract (c, d). Singleton (b).
								Case = 22;
								ind = index(a, e, (Case - 1) * nChanels + f);
								if (c == d) {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 23: (b, c). Contract (a, d). Singleton (e).
								Case = 23;
								ind = index(b, c, (Case - 1) * nChanels + f);
								if (a == d) {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 24: (b, c). Contract (a, e). Singleton (d).
								Case = 24;
								ind = index(b, c, (Case - 1) * nChanels + f);
								if (a == e) {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 25: (b, c). Contract (d, e). Singleton (a).
								Case = 25;
								ind = index(b, c, (Case - 1) * nChanels + f);
								if (d == e) {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 26: (b, d). Contract (a, c). Singleton (e).
								Case = 26;
								ind = index(b, d, (Case - 1) * nChanels + f);
								if (a == c) {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 27: (b, d). Contract (a, e). Singleton (c).
								Case = 27;
								ind = index(b, d, (Case - 1) * nChanels + f);
								if (a == e) {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 28: (b, d). Contract (c, e). Singleton (a).
								Case = 28;
								ind = index(b, d, (Case - 1) * nChanels + f);
								if (c == e) {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 29: (b, e). Contract (a, c). Singleton (d).
								Case = 29;
								ind = index(b, e, (Case - 1) * nChanels + f);
								if (a == c) {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 30: (b, e). Contract (a, d). Singleton (c).
								Case = 30;
								ind = index(b, e, (Case - 1) * nChanels + f);
								if (a == d) {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 31: (b, e). Contract (c, d). Singleton (a).
								Case = 31;
								ind = index(b, e, (Case - 1) * nChanels + f);
								if (c == d) {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 32: (c, d). Contract (a, b). Singleton (e).
								Case = 32;
								ind = index(c, d, (Case - 1) * nChanels + f);
								if (a == b) {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 33: (c, d). Contract (a, e). Singleton (b).
								Case = 33;
								ind = index(c, d, (Case - 1) * nChanels + f);
								if (a == e) {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 34: (c, d). Contract (b, e). Singleton (a).
								Case = 34;
								ind = index(c, d, (Case - 1) * nChanels + f);
								if (b == e) {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 35: (c, e). Contract (a, b). Singleton (d).
								Case = 35;
								ind = index(c, e, (Case - 1) * nChanels + f);
								if (a == b) {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 36: (c, e). Contract (a, d). Singleton (b).
								Case = 36;
								ind = index(c, e, (Case - 1) * nChanels + f);
								if (a == d) {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 37: (c, e). Contract (b, d). Singleton (a).
								Case = 37;
								ind = index(c, e, (Case - 1) * nChanels + f);
								if (b == d) {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 38: (d, e). Contract (a, b). Singleton (c).
								Case = 38;
								ind = index(d, e, (Case - 1) * nChanels + f);
								if (a == b) {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 39: (d, e). Contract (a, c). Singleton (b).
								Case = 39;
								ind = index(d, e, (Case - 1) * nChanels + f);
								if (a == c) {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 40: (d, e). Contract (b, c). Singleton (a).
								Case = 40;
								ind = index(d, e, (Case - 1) * nChanels + f);
								if (b == c) {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// +---+
								// | 3 |
								// +---+

								// Case 41: (a, b). Contract (c, d, e).
								Case = 41;
								ind = index(a, b, (Case - 1) * nChanels + f);
								if ((c == d) && (d == e))  {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 42: (a, c). Contract (b, d, e).
								Case = 42;
								ind = index(a, c, (Case - 1) * nChanels + f);
								if ((b == d) && (d == e))  {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 43: (a, d). Contract (b, c, e).
								Case = 43;
								ind = index(a, d, (Case - 1) * nChanels + f);
								if ((b == c) && (c == e))  {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 44: (a, e). Contract (b, c, d).
								Case = 44;
								ind = index(a, e, (Case - 1) * nChanels + f);
								if ((b == c) && (c == d))  {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 45: (b, c). Contract (a, d, e).
								Case = 45;
								ind = index(b, c, (Case - 1) * nChanels + f);
								if ((a == d) && (d == e))  {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 46: (b, d). Contract (a, c, e).
								Case = 46;
								ind = index(b, d, (Case - 1) * nChanels + f);
								if ((a == c) && (c == e))  {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 47: (b, e). Contract (a, c, d).
								Case = 47;
								ind = index(b, e, (Case - 1) * nChanels + f);
								if ((a == c) && (c == d))  {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 48: (c, d). Contract (a, b, e).
								Case = 48;
								ind = index(c, d, (Case - 1) * nChanels + f);
								if ((a == b) && (b == e))  {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 49: (c, e). Contract (a, b, d).
								Case = 49;
								ind = index(c, e, (Case - 1) * nChanels + f);
								if ((a == b) && (b == d))  {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 50: (d, e). Contract (a, b, c).
								Case = 50;
								ind = index(d, e, (Case - 1) * nChanels + f);
								if ((a == b) && (b == c))  {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}
							}
						}
					}
				}
			}
		}
	}

	// Number of contractions implemented in this class
	static const int nContractions = 50;

	// The size of the receptive field
	int N;

	// Number of chanels
	int nChanels;

	// Neighbors' representations
	vector < Tensor3D* > tensors;

	// The reduced adjacency matrix
	Matrix *adj;

	~RisiContraction_50() {
		tensors.clear();
	}
};

#endif
