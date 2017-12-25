// Framework: GraphFlow
// Class: DenseGraph
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __DENSEGRAPH_H_INCLUDED__
#define __DENSEGRAPH_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

#include "Matrix.h"

using namespace std;

class DenseGraph {
public:
	DenseGraph(int nVertices, int nFeatures) {
		this -> nVertices = nVertices;
		this -> nFeatures = nFeatures;

		memory_allocation();
	}

	void memory_allocation() {
		adj = new int* [nVertices];
		feature = new float* [nVertices];
		coulomb = new float* [nVertices];
		distance = new float* [nVertices];

		for (int v = 0; v < nVertices; ++v) {
			adj[v] = new int [nVertices];
			feature[v] = new float [nFeatures];
			coulomb[v] = new float [nVertices];
			distance[v] = new float [nVertices];
		}

		for (int u = 0; u < nVertices; ++u) {
			for (int v = 0; v < nVertices; ++v) {
				adj[u][v] = 0;
			}
		}

		for (int v = 0; v < nVertices; ++v) {
			for (int f = 0; f < nFeatures; ++f) {
				feature[v][f] = 0.0;
			}
		}

		for (int u = 0; u < nVertices; ++u) {
			for (int v = 0; v < nVertices; ++v) {
				coulomb[u][v] = 0.0;
			}
		}

		for (int u = 0; u < nVertices; ++u) {
			for (int v = 0; v < nVertices; ++v) {
				distance[u][v] = 0.0;
			}
		}
	}

	void create_norm_adj() {
		Matrix *A = new Matrix(nVertices, nVertices);
		Matrix *D = new Matrix(nVertices, nVertices);

		for (int i = 0; i < nVertices; ++i) {
			for (int j = 0; j < nVertices; ++j) {
				int a = A -> index(i, j);
				A -> value[a] = adj[i][j];
				if (i == j) {
					A -> value[a] += 1.0;
				}
			}
		}

		for (int i = 0; i < nVertices; ++i) {
			for (int j = 0; j < nVertices; ++j) {
				int d = D -> index(i, j);
				D -> value[d] = 0.0;
			}
		}

		for (int i = 0; i < nVertices; ++i) {
			for (int j = 0; j < nVertices; ++j) {
				int a = A -> index(i, j);
				int d = D -> index(i, i);
				D -> value[d] += A -> value[a];
			}
		}

		for (int i = 0; i < nVertices; ++i) {
			int d = D -> index(i, i);
			if (D -> value[d] > 0.0) {
				D -> value[d] = 1.0 / sqrt(D -> value[d]);
			}
		}

		Matrix *P = D -> multiply(A);
		norm_adj = P -> multiply(D);

		delete A;
		delete D;
		delete P;
	}

	int nVertices;
	int nFeatures;
	int **adj;
	float **feature;
	float **coulomb;
	float **distance;
	Matrix *norm_adj;

	~DenseGraph() {
		for (int i = 0; i < nVertices; ++i) {
			delete adj[i];
			delete feature[i];
			delete coulomb[i];
			delete distance[i];
		}
		delete adj;
		delete feature;
		delete coulomb;
		delete distance;
		delete norm_adj;
	}
};

#endif

