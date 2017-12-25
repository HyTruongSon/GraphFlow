// Framework: GraphFlow
// Class: StackTensor3D_thread
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __STACKTENSOR3D_THREAD_H_INCLUDED__
#define __STACKTENSOR3D_THREAD_H_INCLUDED__

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
#include "Tensor4D.h"

using namespace std;

class StackTensor3D_thread: public Tensor4D {
public:
	StackTensor3D_thread(int nRows, int nColumns, int nChanels1, int nChanels2) : Tensor4D(nRows, nColumns, nChanels1, nChanels2) {
		tensors.clear();

		forward_thread = new std::thread [nRows];
		backward_thread = new std::thread [nRows];
	}

	void setParameter(int nRows, int nColumns, int nChanels1, int nChanels2) {
		this -> nRows = nRows;
		this -> nColumns = nColumns;
		this -> nChanels1 = nChanels1;
		this -> nChanels2 = nChanels2;

		size = nRows * nColumns * nChanels1 * nChanels2;

		tensors.clear();
	}

	void add_tensor(Tensor3D *tensor) {
		assert(tensor -> nRows == nColumns);
		assert(tensor -> nColumns == nChanels1);
		assert(tensor -> nDepth == nChanels2);

		tensors.push_back(tensor);
	}

	void clear() {
		tensors.clear();
	}

	void forward_single_thread() {
		assert(tensors.size() == nRows);
		
		for (int row = 0; row < nRows; ++row) {
			for (int column = 0; column < nColumns; ++column) {
				for (int chanel1 = 0; chanel1 < nChanels1; ++chanel1) {
					for (int chanel2 = 0; chanel2 < nChanels2; ++chanel2) {
						int i = index(row, column, chanel1, chanel2);
						int j = tensors[row] -> index(column, chanel1, chanel2);

						value[i] = tensors[row] -> value[j];
					}
				}
			}
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}

	static void forward_job(StackTensor3D_thread *obj, int row) {
		int nColumns = obj -> nColumns;
		int nChanels1 = obj -> nChanels1;
		int nChanels2 = obj -> nChanels2;

		for (int column = 0; column < nColumns; ++column) {
			for (int chanel1 = 0; chanel1 < nChanels1; ++chanel1) {
				for (int chanel2 = 0; chanel2 < nChanels2; ++chanel2) {
					int i = obj -> index(row, column, chanel1, chanel2);
					int j = obj -> tensors[row] -> index(column, chanel1, chanel2);

					obj -> value[i] = obj -> tensors[row] -> value[j];
				}
			}
		}
	}

	void forward_multi_threads() {
		assert(tensors.size() == nRows);
		
		for (int row = 0; row < nRows; ++row) {
			forward_thread[row] = std::thread(forward_job, this, row);
		}

		for (int row = 0; row < nRows; ++row) {
			forward_thread[row].join();
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}

	void forward() {
		if (nRows <= THRESHOLD) {
			forward_single_thread();
		} else {
			forward_multi_threads();
		}
	}

	void backward_single_thread() {
		assert(tensors.size() == nRows);
		
		for (int row = 0; row < nRows; ++row) {
			for (int column = 0; column < nColumns; ++column) {
				for (int chanel1 = 0; chanel1 < nChanels1; ++chanel1) {
					for (int chanel2 = 0; chanel2 < nChanels2; ++chanel2) {
						int i = index(row, column, chanel1, chanel2);
						int j = tensors[row] -> index(column, chanel1, chanel2);

						tensors[row] -> gradient[j] += gradient[i];
					}
				}
			}
		}
	}

	static void backward_job(StackTensor3D_thread *obj, int row) {
		int nColumns = obj -> nColumns;
		int nChanels1 = obj -> nChanels1;
		int nChanels2 = obj -> nChanels2;

		for (int column = 0; column < nColumns; ++column) {
			for (int chanel1 = 0; chanel1 < nChanels1; ++chanel1) {
				for (int chanel2 = 0; chanel2 < nChanels2; ++chanel2) {
					int i = obj -> index(row, column, chanel1, chanel2);
					int j = obj -> tensors[row] -> index(column, chanel1, chanel2);

					obj -> tensors[row] -> gradient[j] += obj -> gradient[i];
				}
			}
		}
	}

	void backward_multi_threads() {
		assert(tensors.size() == nRows);
		
		for (int row = 0; row < nRows; ++row) {
			backward_thread[row] = std::thread(backward_job, this, row);
		}

		for (int row = 0; row < nRows; ++row) {
			backward_thread[row].join();
		}
	}

	void backward() {
		if (nRows <= THRESHOLD) {
			backward_single_thread();
		} else {
			backward_multi_threads();
		}
	}

	// Threshold to decide running on CPU or GPU
	static const int THRESHOLD = 4;

	// List of tensors
	vector < Tensor3D* > tensors;

	// Threads for Forward
	std::thread *forward_thread;

	// Threads for Backward
	std::thread *backward_thread;

	~StackTensor3D_thread() {
		tensors.clear();
	}
};

#endif