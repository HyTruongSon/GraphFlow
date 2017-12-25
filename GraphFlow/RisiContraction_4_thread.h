// Framework: GraphFlow
// Class: RisiContraction_4_thread
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __RISICONTRACTION_4_THREAD_H_INCLUDED__
#define __RISICONTRACTION_4_THREAD_H_INCLUDED__

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

class RisiContraction_4_thread: public Tensor3D {
public:
	RisiContraction_4_thread(int max_nRows, int max_nColumns, int max_nDepth) : Tensor3D(max_nRows, max_nColumns, max_nDepth) {
		assert(DEPRECATED == false);

		tensors.clear();
	}

	RisiContraction_4_thread(int N, int nChanels) : Tensor3D(N, N, nContractions * nChanels) {
		assert(DEPRECATED == false);

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

	double value_at(int a, int b, int c, int f) {
		return tensors[a] -> value[tensors[a] -> index(b, c, f)];
	}

	void set_gradient_for(int a, int b, int c, int f, double grad) {
		tensors[a] -> gradient[tensors[a] -> index(b, c, f)] += grad;
	}

	// Contraction 1
	static void forward_job_0(RisiContraction_4_thread *obj) {
		int nChanels = obj -> nChanels;
		int N = obj -> N;

		int Case;
		int ind;

		for (int f = 0; f < nChanels; ++f) {
			for (int a = 0; a < N; ++a) {
				for (int b = 0; b < N; ++b) {
					for (int c = 0; c < N; ++c) {

						// Case 1: Fix a, b. Contract c.
						Case = 1;
						ind = obj -> index(a, b, (Case - 1) * nChanels + f);
						obj -> value[ind] += obj -> value_at(a, b, c, f);
					}
				}
			}
		}
	}

	// Contraction 2
	static void forward_job_1(RisiContraction_4_thread *obj) {
		int nChanels = obj -> nChanels;
		int N = obj -> N;

		int Case;
		int ind;

		for (int f = 0; f < nChanels; ++f) {
			for (int a = 0; a < N; ++a) {
				for (int b = 0; b < N; ++b) {
					for (int c = 0; c < N; ++c) {

						// Case 2: Fix b, c. Contract a.
						Case = 2;
						ind = obj -> index(b, c, (Case - 1) * nChanels + f);
						obj -> value[ind] += obj -> value_at(a, b, c, f);
					}
				}
			}
		}
	}

	// Contraction 3 - 4
	static void forward_job_2(RisiContraction_4_thread *obj) {
		int nChanels = obj -> nChanels;
		int N = obj -> N;

		int Case;
		int ind;

		for (int f = 0; f < nChanels; ++f) {
			for (int a = 0; a < N; ++a) {
				for (int c = 0; c < N; ++c) {

					// Case 3: (a = b, c).
					// a == b
					Case = 3;
					ind = obj -> index(a, c, (Case - 1) * nChanels + f);
					obj -> value[ind] += obj -> value_at(a, a, c, f);
				}
			}
		}

		for (int f = 0; f < nChanels; ++f) {
			for (int a = 0; a < N; ++a) {
				for (int b = 0; b < N; ++b) {

					// Case 4: (a, b = c).
					// b == c
					Case = 4;
					ind = obj -> index(a, b, (Case - 1) * nChanels + f);
					obj -> value[ind] += obj -> value_at(a, b, b, f);
				}
			}
		}
	}

	// Contraction 1
	static void backward_job_0(RisiContraction_4_thread *obj) {
		int nChanels = obj -> nChanels;
		int N = obj -> N;

		int Case;
		int ind;

		for (int f = 0; f < nChanels; ++f) {
			for (int a = 0; a < N; ++a) {
				for (int b = 0; b < N; ++b) {
					for (int c = 0; c < N; ++c) {

						// Case 1: Fix a, b. Contract c.
						Case = 1;
						ind = obj -> index(a, b, (Case - 1) * nChanels + f);
						obj -> set_gradient_for(a, b, c, f, obj -> gradient[ind]);
					}
				}
			}
		}
	}

	// Contraction 2
	static void backward_job_1(RisiContraction_4_thread *obj) {
		int nChanels = obj -> nChanels;
		int N = obj -> N;

		int Case;
		int ind;

		for (int f = 0; f < nChanels; ++f) {
			for (int a = 0; a < N; ++a) {
				for (int b = 0; b < N; ++b) {
					for (int c = 0; c < N; ++c) {

						// Case 2: Fix b, c. Contract a.
						Case = 2;
						ind = obj -> index(b, c, (Case - 1) * nChanels + f);
						obj -> set_gradient_for(a, b, c, f, obj -> gradient[ind]);
					}
				}
			}
		}
	}

	// Contraction 3 - 4
	static void backward_job_2(RisiContraction_4_thread *obj) {
		int nChanels = obj -> nChanels;
		int N = obj -> N;

		int Case;
		int ind;

		for (int f = 0; f < nChanels; ++f) {
			for (int a = 0; a < N; ++a) {
				for (int c = 0; c < N; ++c) {

					// Case 3: (a = b, c).
					// a == b
					Case = 3;
					ind = obj -> index(a, c, (Case - 1) * nChanels + f);
					obj -> set_gradient_for(a, a, c, f, obj -> gradient[ind]);
				}
			}
		}

		for (int f = 0; f < nChanels; ++f) {
			for (int a = 0; a < N; ++a) {
				for (int b = 0; b < N; ++b) {

					// Case 4: (a, b = c).
					// b == c
					Case = 4;
					ind = obj -> index(a, b, (Case - 1) * nChanels + f);
					obj -> set_gradient_for(a, b, b, f, obj -> gradient[ind]);
				}
			}
		}
	}

	void forward() {
		assert(tensors.size() == N);
		assert(nThreads == 3);

		for (int i = 0; i < size; ++i) {
			value[i] = 0.0;
		}

		forward_thread[0] = std::thread(forward_job_0, this);
		forward_thread[1] = std::thread(forward_job_1, this);
		forward_thread[2] = std::thread(forward_job_2, this);

		for (int t = 0; t < nThreads; ++t) {
			forward_thread[t].join();
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}

	void backward() {
		assert(tensors.size() == N);
		assert(nThreads == 3);

		backward_thread[0] = std::thread(backward_job_0, this);
		backward_thread[1] = std::thread(backward_job_1, this);
		backward_thread[2] = std::thread(backward_job_2, this);

		for (int t = 0; t < nThreads; ++t) {
			backward_thread[t].join();
		}
	}

	// DEPRECATED
	static const bool DEPRECATED = true;

	// Number of contractions implemented in this class
	static const int nContractions = 4;

	// Number of threads
	static const int nThreads = 3;

	// Threads for Forward
	std::thread forward_thread[nThreads];

	// Threads for Backward
	std::thread backward_thread[nThreads];

	// The size of the receptive field
	int N;

	// Number of chanels
	int nChanels;

	// Neighbors' representations
	vector < Tensor3D* > tensors;

	~RisiContraction_4_thread() {
		tensors.clear();
	}
};

#endif