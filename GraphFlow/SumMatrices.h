// Framework: GraphFlow
// Class: SumMatrices
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __SUMMATRICES_H_INCLUDED__
#define __SUMMATRICES_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <assert.h>

#include "Matrix.h"

using namespace std;

class SumMatrices: public Matrix {
public:
	SumMatrices(int nRows, int nColumns) : Matrix(nRows, nColumns) {
		matrices.clear();
	}

	void setParameter(int nRows, int nColumns) {
		this -> nRows = nRows;
		this -> nColumns = nColumns;
		size = this -> nRows * this -> nColumns;

		matrices.clear();
	}

	void add_matrix(Matrix *matrix) {
		assert(nRows == matrix -> nRows);
		assert(nColumns == matrix -> nColumns);
		matrices.push_back(matrix);
	}

	void clear() {
		matrices.clear();
	}

	void forward() {
		for (int i = 0; i < size; ++i) {
			value[i] = 0.0;
		}

		for (int v = 0; v < matrices.size(); ++v) {
			for (int i = 0; i < size; ++i) {
				value[i] += matrices[v] -> value[i];
			}
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}

	void backward() {
		for (int v = 0; v < matrices.size(); ++v) {
			for (int i = 0; i < size; ++i) {
				matrices[v] -> gradient[i] += gradient[i];
			}
		}
	}

	vector < Matrix* > matrices;

	~SumMatrices() {
		matrices.clear();
	}
};

#endif