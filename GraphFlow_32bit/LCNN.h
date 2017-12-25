// Framework: GraphFlow
// Class: Learning Convolutional Neural Networks for Graphs
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __LCNN_H_INCLUDED__
#define __LCNN_H_INCLUDED__

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <assert.h>

#include "DenseGraph.h"
#include "SumGradients.h"
#include "CacheParameters.h"
#include "Momentum.h"
#include "GraphFlow.h"

using namespace std;

const int INF = 1e9;

class LCNN {
public:
	LCNN(int nVertices, int nFeatures, int nNeighbors, int nDepth, int nChanels1, int nChanels2, int nDense, float momentum_param) {
		this -> nVertices = nVertices;
		this -> nFeatures = nFeatures;
		this -> nNeighbors = nNeighbors;
		this -> nDepth = nDepth;
		this -> nChanels1 = nChanels1;
		this -> nChanels2 = nChanels2;
		this -> nDense = nDense;
		this -> momentum_param = momentum_param;

		computation_graph();
		weights_initialization();
	}

	void computation_graph() {
		// +--------------------------+
		// | Component initialization |
		// +--------------------------+

		// Sequence
		sequence = new Vector(nVertices * nNeighbors);

		// Original feature
		feature = new Matrix(nVertices, nFeatures);

		// Histogram WL feature
		wl_feature = new Matrix(nVertices, nFeatures * (nDepth + 1));

		// First convolution layer
		firstInput = new ShuffleMatrix(wl_feature, sequence);
		firstFilter = new Tensor3D(nNeighbors, nFeatures * (nDepth + 1), nChanels1);
		firstBias = new Vector(nChanels1);
		firstConv = new Conv1D(firstInput, firstFilter, firstBias, nNeighbors, 0);
		firstReLU = new LeakyReLU(firstConv);
		firstReshape = new Reshape2D(firstReLU, firstConv -> nRows, firstConv -> nColumns);

		// Second convolution layer
		secondInput = new ShuffleMatrix(firstReshape, sequence);
		secondFilter = new Tensor3D(nNeighbors, nChanels1, nChanels2);
		secondBias = new Vector(nChanels2);
		secondConv = new Conv1D(secondInput, secondFilter, secondBias, nNeighbors, 0);
		secondReLU = new LeakyReLU(secondConv);
		secondReshape = new Reshape2D(secondReLU, secondConv -> nRows, secondConv -> nColumns);

		// Dense layer
		denseWeight = new Matrix(nDense, secondReshape -> size);
		denseLayer = new MatVecMul(denseWeight, secondConv);

		// Linear regression
		W = new Vector(denseLayer -> size);
		predict = new InnerProduct(W, denseLayer);
		target = new Vector(1);
		sql = new SquaredLoss(predict, target);

		// +-------------------+
		// | Computation graph |
		// +-------------------+
		
		graph = new GraphFlow();

		// Original feature
		graph -> add(wl_feature, MATRIX);

		// First convolution layer
		graph -> add(firstInput, SHUFFLEMATRIX);
		graph -> add(firstFilter, TENSOR3D);
		graph -> add(firstBias, VECTOR);
		graph -> add(firstConv, CONV1D);
		graph -> add(firstReLU, LEAKYRELU);
		graph -> add(firstReshape, RESHAPE2D);

		// Second convolution layer
		graph -> add(secondInput, SHUFFLEMATRIX);
		graph -> add(secondFilter, TENSOR3D);
		graph -> add(secondBias, VECTOR);
		graph -> add(secondConv, CONV1D);
		graph -> add(secondReLU, LEAKYRELU);
		graph -> add(secondReshape, RESHAPE2D);

		// Dense layer
		graph -> add(denseWeight, MATRIX);
		graph -> add(denseLayer, MATVECMUL);

		// Linear regression
		graph -> add(W, VECTOR);
		graph -> add(predict, INNERPRODUCT);
		graph -> add(sql, SQUAREDLOSS);

		// +-----------------------------+
		// | Stochastic Gradient Descent |
		// +-----------------------------+

		sgd = new Momentum(momentum_param);
		sgd -> add(firstFilter);
		sgd -> add(firstBias);
		sgd -> add(secondFilter);
		sgd -> add(secondBias);
		sgd -> add(denseWeight);
		sgd -> add(W);

		// Sum gradients
		sum_gradients = new SumGradients();
		for (int i = 0; i < sgd -> params.size(); ++i) {
			sum_gradients -> add(sgd -> params[i]);
		}

		// Cache parameters
		cache_parameters = new CacheParameters();
		for (int i = 0; i < sgd -> params.size(); ++i) {
			cache_parameters -> add(sgd -> params[i]);
		}

		// Adjacency matrix
		adj = new int* [nVertices];
		for (int i = 0; i < nVertices; ++i) {
			adj[i] = new int [nVertices];
		}

		// Shortest Paths by Floyd-Warshall algorithm
		shortest_paths = new int* [nVertices];
		for (int i = 0; i < nVertices; ++i) {
			shortest_paths[i] = new int [nVertices];
		}

		// Histogram WL
		histogram = new float* [nVertices];
		for (int i = 0; i < nVertices; ++i) {
			histogram[i] = new float [nVertices * (nDepth + 1)];
		}

		// Order of vertices
		order = new int [nVertices];

		// Rank of vertices
		rank = new int [nVertices];
	}

	void weights_initialization() {
		for (int i = 0; i < sgd -> params.size(); ++i) {
			graph -> uniform_init(sgd -> params[i]);
		}
	}	

	void update_adjacency(DenseGraph *molecule) {
		// Empty graph
		for (int i = 0; i < nVertices; ++i) {
			for (int j = 0; j < nVertices; ++j) {
				adj[i][j] = 0;
			}
		}

		// Get the original graph
		for (int i = 0; i < molecule -> nVertices; ++i) {
			for (int j = 0; j < molecule -> nVertices; ++j) {
				adj[i][j] = molecule -> adj[i][j];
			}
		}

		// Dummy nodes
		/*
		for (int i = molecule -> nVertices; i < nVertices; ++i) {
			for (int j = 0; j < nVertices; ++j) {
				adj[i][j] = 1;
			}
		}
		*/
	}

	void update_feature(DenseGraph *molecule) {
		// Get the original graph feature
		for (int v = 0; v < molecule -> nVertices; ++v) {
			for (int f = 0; f < nFeatures; ++f) {
				feature -> value[feature -> index(v, f)] = molecule -> feature[v][f];
			}
		}

		// Dummy nodes
		for (int v = molecule -> nVertices; v < nVertices; ++v) {
			for (int f = 0; f < nFeatures; ++f) {
				feature -> value[feature -> index(v, f)] = 0.0;
			}
		}
	}

	void floyd_warshall() {
		for (int i = 0; i < nVertices; ++i) {
			for (int j = 0; j < nVertices; ++j) {
				shortest_paths[i][j] = INF;
				if (i == j) {
					shortest_paths[i][j] = 0;
				} else {
					if (adj[i][j] > 0) {
						shortest_paths[i][j] = 1;
						shortest_paths[j][i] = 1;
					}
				}
			}
		}

		for (int k = 0; k < nVertices; ++k) {
			for (int i = 0; i < nVertices; ++i) {
				for (int j = 0; j < nVertices; ++j) {
					shortest_paths[i][j] = min(shortest_paths[i][j], shortest_paths[i][k] + shortest_paths[k][j]);
				}
			}
		}
	}

	void weisfeiler_lehman() {
		for (int v = 0; v < nVertices; ++v) {
			for (int f = 0; f < nFeatures * (nDepth + 1); ++f) {
				histogram[v][f] = 0.0;
			}

			for (int d = 0; d <= nDepth; ++d) {
				for (int u = 0; u < nVertices; ++u) {
					if (shortest_paths[u][v] == d) {
						for (int f = 0; f < nFeatures; ++f) {
							histogram[v][d * nFeatures + f] += feature -> value[feature -> index(u, f)];
						}
					}
				}
			}
		}

		for (int v = 0; v < nVertices; ++v) {
			for (int f = 0; f < nFeatures * (nDepth + 1); ++f) {
				wl_feature -> value[wl_feature -> index(v, f)] = histogram[v][f];
			}
		}
	}

	int compare_vertices(int u, int v) {
		for (int f = 0; f < nFeatures * (nDepth + 1); ++f) {
			if (histogram[u][f] < histogram[v][f]) {
				return -1;
			}
			if (histogram[u][f] > histogram[v][f]) {
				return 1;
			}
		}
		return 0;
	}

	void rank_vertices() {
		for (int v = 0; v < nVertices; ++v) {
			order[v] = v;
		}

		for (int i = 0; i < nVertices; ++i) {
			for (int j = i + 1; j < nVertices; ++j) {
				if (compare_vertices(order[i], order[j]) < 0) {
					swap(order[i], order[j]);
				}
			}
		}

		for (int i = 0; i < nVertices; ++i) {
			rank[order[i]] = i;
		}
	}

	void find_sequence(DenseGraph *molecule) {
		for (int i = 0; i < nVertices; ++i) {
			bool stop = false;
			int j = 0;
			for (int d = 0; d < nVertices; ++d) {
				for (int v = 0; v < nVertices; ++v) {
					if ((shortest_paths[order[i]][order[v]] == d) && (order[v] < molecule -> nVertices)) {
						int pos = nNeighbors * i + j;
						sequence -> value[pos] = order[v];
						++j;
						if (j == nNeighbors) {
							stop = true;
							break;
						}
					}
				}
				if (stop == true) {
					break;
				}
			}
			while (j < nNeighbors) {
				int pos = nNeighbors * i + j;
				sequence -> value[pos] = molecule -> nVertices;
				++j;
			}
		}
	}

	void complete_computation_graph(DenseGraph *molecule) {
		assert(molecule -> nFeatures == nFeatures);
		assert(molecule -> nVertices <= nVertices);

		// Update the adjacency matrix
		update_adjacency(molecule);

		// Update the feature (with dummy nodes)
		update_feature(molecule);

		// Finding the shortest-paths by Floyd-Warshall algorithm
		floyd_warshall();
		
		// Get the feature vector for each vertex
		weisfeiler_lehman();

		// Find the optimal order of vertices
		rank_vertices();

		// Find the vertex sequence
		find_sequence(molecule);
	}

	float getLoss(int nBatch, DenseGraph **molecule, float *target) {
		float total_loss = 0.0;
		for (int i = 0; i < nBatch; ++i) {
			complete_computation_graph(molecule[i]);
			this -> target -> value[0] = target[i];
			graph -> forward();
			total_loss += sql -> getLoss();
		}
		return total_loss;
	}

	pair < float, float > BatchLearn(int nBatch, DenseGraph **molecule, float *target, float learning_rate) {
		assert(nBatch > 0);
		assert(molecule[0] -> nVertices <= nVertices);
		assert(molecule[0] -> nFeatures == nFeatures);

		pair < float, float > ret;
		ret.first = getLoss(nBatch, molecule, target);

		sum_gradients -> reset_sum_gradients();

		for (int i = 0; i < nBatch; ++i) {
			complete_computation_graph(molecule[i]);
			this -> target -> value[0] = target[i];

			graph -> forward();
			graph -> backward();

			sum_gradients -> cache_gradients();
		}

		sum_gradients -> get_sum_gradients();
		sgd -> Learn(learning_rate, nBatch);

		ret.second = getLoss(nBatch, molecule, target);
		return ret;
	}

	pair < float, float > BatchLearn(int nBatch, DenseGraph **molecule, float *target, int nIterations, float learning_rate, float epsilon) {
		assert(nBatch > 0);
		assert(molecule[0] -> nVertices <= nVertices);
		assert(molecule[0] -> nFeatures == nFeatures);

		cache_parameters -> cache_parameters();

		pair<float, float> ret;
		ret.first = getLoss(nBatch, molecule, target);
		ret.second = ret.first;

		float decay_lr = 0.5;
		float min_lr = 1e-6;

		for (int iter = 0; iter < nIterations; ++iter) {
			sum_gradients -> reset_sum_gradients();

			for (int i = 0; i < nBatch; ++i) {
				complete_computation_graph(molecule[i]);
				this -> target -> value[0] = target[i];

				graph -> forward();
				graph -> backward();

				sum_gradients -> cache_gradients();
			}

			sum_gradients -> get_sum_gradients();
			sgd -> Learn(learning_rate, nBatch);

			float loss = getLoss(nBatch, molecule, target);

			if (loss > ret.second) {
				cache_parameters -> restore_parameters();
				learning_rate *= decay_lr;
				if (learning_rate < min_lr) {
					break;
				}
			} else {
				ret.second = loss;
				cache_parameters -> cache_parameters();
			}
		}

		return ret;
	}

	pair < float, float > Learn(DenseGraph *molecule, float target, int nIterations, float learning_rate, float epsilon) {
		assert(molecule -> nVertices <= nVertices);
		assert(molecule -> nFeatures == nFeatures);

		complete_computation_graph(molecule);
		this -> target -> value[0] = target;

		cache_parameters -> cache_parameters();

		graph -> forward();
		float best_error = sql -> getLoss();

		if (best_error < epsilon) {
			return make_pair(best_error, best_error);
		}

		pair<float, float> ret;
		ret.first = best_error;

		float decay_lr = 0.5;
		float min_lr = 1e-6;

		for (int iter = 0; iter < nIterations; ++iter) {
			graph -> forward();
			graph -> backward();
			sgd -> Learn(learning_rate);

			graph -> forward();
			float error = sql -> getLoss();

			if (error < epsilon) {
				break;
			}

			if (error >= best_error) {
				cache_parameters -> restore_parameters();
				learning_rate *= decay_lr;
				learning_rate = max(learning_rate, min_lr);
			} else {
				best_error = error;
				cache_parameters -> cache_parameters();
			}
		}

		ret.second = best_error;
		return ret;
	}

	float Predict(DenseGraph *molecule) {
		complete_computation_graph(molecule);

		graph -> forward();

		return predict -> value[0];
	}

	vector<float> Feature(DenseGraph *molecule) {
		complete_computation_graph(molecule);

		graph -> forward();

		vector<float> vect;
		vect.clear();
		for (int i = 0; i < denseLayer -> size; ++i) {
			vect.push_back(denseLayer -> value[i]);
		}
		return vect;
	}

	void save_model(string filename) {
		ofstream file(filename.c_str(), ios::out);

		for (int i = 0; i < sgd -> params.size(); ++i) {
			for (int j = 0; j < sgd -> params[i] -> size; ++j) {
				file << sgd -> params[i] -> value[j] << " ";
			}
		}

		file.close();
	}

	void load_model(string filename) {
		ifstream file(filename.c_str(), ios::in);
		
		for (int i = 0; i < sgd -> params.size(); ++i) {
			for (int j = 0; j < sgd -> params[i] -> size; ++j) {
				file >> sgd -> params[i] -> value[j];
			}
		}

		file.close();
	}

	// Dynamic computation graph
	GraphFlow *graph;

	// Vertex sequence
	Vector *sequence;

	// Original feature
	Matrix *feature;

	// Histogram WL feature
	Matrix *wl_feature;

	// First convolutional layer
	ShuffleMatrix *firstInput;
	Tensor3D *firstFilter;
	Vector *firstBias;
	Conv1D *firstConv;
	LeakyReLU *firstReLU;
	Reshape2D *firstReshape;

	// Second convolutional layer
	ShuffleMatrix *secondInput;
	Tensor3D *secondFilter;
	Vector *secondBias;
	Conv1D *secondConv;
	LeakyReLU *secondReLU;
	Reshape2D *secondReshape;

	// Dense layer
	Matrix *denseWeight;
	MatVecMul *denseLayer;

	// Top layer as a linear regression
	Vector *W;

	Vector *predict;
	Vector *target;
	SquaredLoss *sql;

	// Stochastic Gradient Descent
	Momentum *sgd;

	// Sum gradients
	SumGradients *sum_gradients;

	// Cache parameters
	CacheParameters *cache_parameters;

	int nLevels;
	int nVertices;
	int nFeatures;
	int nNeighbors;
	int nDepth;
	int nChanels1;
	int nChanels2;
	int nDense;
	float momentum_param;

	// Adjacency matrix
	int **adj;

	// Histogram WL 
	float **histogram;

	// Floyd-Warshall algorithm
	int **shortest_paths;

	// Order of vertices
	int *order;

	// Rank of vertices
	int *rank;

	~LCNN() {
		delete sequence;
		delete feature;
		delete wl_feature;
		delete firstInput;
		delete firstFilter;
		delete firstBias;
		delete firstConv;
		delete firstReLU;
		delete firstReshape;
		delete secondInput;
		delete secondFilter;
		delete secondBias;
		delete secondConv;
		delete secondReLU;
		delete secondReshape;
		delete denseWeight;
		delete denseLayer;
		delete W;
		delete predict;
		delete target;
		delete sql;
		delete[] adj;
		delete[] histogram;
		delete[] shortest_paths;
		delete order;
		delete rank;
	}
};

#endif
