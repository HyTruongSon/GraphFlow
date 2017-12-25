// Framework: GraphFlow
// Class: Graph Convolution Network - Max Welling
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __GCN_MW_H_INCLUDED__
#define __GCN_MW_H_INCLUDED__

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
#include "Momentum.h"
#include "GraphFlow.h"

using namespace std;

const int INF = 1e9;

class GCN_MW {
public:
	GCN_MW(int nLevels, int max_nVertices, int nFeatures, int nHiddens, int nDepth, float momentum_param) {
		this -> nLevels = nLevels;
		this -> max_nVertices = max_nVertices;
		this -> nFeatures = nFeatures;
		this -> nHiddens = nHiddens;
		this -> nDepth = nDepth;
		this -> momentum_param = momentum_param;

		computation_graph();
		weights_initialization();
	}

	void computation_graph() {
		graph = new GraphFlow();

		feature = NULL;

		// Computation graph for each level
		level = new Level* [nLevels + 1];
		for (int l = 0; l <= nLevels; ++l) {
			level[l] = new Level();
			
			level[l] -> product1 = NULL;
			level[l] -> product2 = NULL;
			level[l] -> activation = NULL;
			level[l] -> hidden = NULL;

			// Learning parameters
			if (l == 0) {
				level[l] -> W = new Matrix(nFeatures * (nDepth + 1), nHiddens);
				level[l] -> prev_W = new Matrix(nFeatures * (nDepth + 1), nHiddens);
				level[l] -> sum_gradient_W = new Matrix(nFeatures * (nDepth + 1), nHiddens);
			} else {
				level[l] -> W = new Matrix(nHiddens, nHiddens);
				level[l] -> prev_W = new Matrix(nHiddens, nHiddens);
				level[l] -> sum_gradient_W = new Matrix(nHiddens, nHiddens);
			}
		}

		// Computation graph for the top level as a linear regression
		W = new Vector(nHiddens);
		prev_W = new Vector(nHiddens);
		sum_gradient_W = new Vector(nHiddens);

		final_feature = NULL;
		predict = NULL;
		target = new Vector(1);
		sql = NULL;

		// Stochastic Gradient Descent
		sgd = new Momentum(momentum_param);
		for (int l = 0; l <= nLevels; ++l) {
			sgd -> add(level[l] -> W);
		}
		sgd -> add(W);

		// Shortest Paths by Floyd-Warshall algorithm
		shortest_paths = new int* [max_nVertices];
		for (int i = 0; i < max_nVertices; ++i) {
			shortest_paths[i] = new int [max_nVertices];
		}
	}

	void weights_initialization() {
		for (int l = 0; l <= nLevels; ++l) {
			graph -> uniform_init(level[l] -> W);
		}

		graph -> uniform_init(W);
	}

	void copy(Vector *input, Vector *output) {
		for (int i = 0; i < input -> size; ++i) {
			output -> value[i] = input -> value[i];
		}
	}

	void copy(Matrix *input, Matrix *output) {
		for (int i = 0; i < input -> size; ++i) {
			output -> value[i] = input -> value[i];
		}
	}

	void cache_parameters() {
		for (int l = 0; l <= nLevels; ++l) {
			copy(level[l] -> W, level[l] -> prev_W);
		}

		copy(W, prev_W);
	}

	void restore_parameters() {
		for (int l = 0; l <= nLevels; ++l) {
			copy(level[l] -> prev_W, level[l] -> W);
		}

		copy(prev_W, W);
	}

	void complete_computation_graph(DenseGraph *molecule) {
		assert(molecule -> nFeatures == nFeatures);

		// Build a new computation graph
		graph -> clear();

		if (feature != NULL) {
			delete feature;
		}

		for (int l = 0; l <= nLevels; ++l) {	
			if (level[l] -> product1 != NULL) {
				delete level[l] -> product1;
			}		
			if (level[l] -> product2 != NULL) {
				delete level[l] -> product2;
			}	
			if (level[l] -> activation != NULL) {
				delete level[l] -> activation;
			}	
			if (level[l] -> hidden != NULL) {
				delete level[l] -> hidden;
			}	
		}

		if (final_feature != NULL) {
			delete final_feature;
		}

		if (predict != NULL) {
			delete predict;
		}

		if (sql != NULL) {
			delete sql;
		}

		// Finding the shortest-paths by Floyd-Warshall algorithm
		for (int i = 0; i < molecule -> nVertices; ++i) {
			for (int j = 0; j < molecule -> nVertices; ++j) {
				shortest_paths[i][j] = INF;
				if (i == j) {
					shortest_paths[i][j] = 0;
				} else {
					if (molecule -> adj[i][j] > 0) {
						shortest_paths[i][j] = 1;
						shortest_paths[j][i] = 1;
					}
				}
			}
		}

		for (int k = 0; k < molecule -> nVertices; ++k) {
			for (int i = 0; i < molecule -> nVertices; ++i) {
				for (int j = 0; j < molecule -> nVertices; ++j) {
					shortest_paths[i][j] = min(shortest_paths[i][j], shortest_paths[i][k] + shortest_paths[k][j]);
				}
			}
		}

		// Get the feature vector for each vertex
		feature = new Matrix (molecule -> nVertices, nFeatures * (nDepth + 1));

		for (int v = 0; v < molecule -> nVertices; ++v) {
			for (int f = 0; f < nFeatures * (nDepth + 1); ++f) {
				feature -> value[feature -> index(v, f)] = 0.0;
			}

			for (int d = 0; d <= nDepth; ++d) {
				for (int u = 0; u < molecule -> nVertices; ++u) {
					if (shortest_paths[u][v] == d) {
						for (int f = 0; f < nFeatures; ++f) {
							feature -> value[feature -> index(v, d * nFeatures + f)] += molecule -> feature[u][f];
						}
					}
				}
			}
		}

		// For each level
		for (int l = 0; l <= nLevels; ++l) {	
			// Learning parameters
			if (l == 0) {
				level[l] -> product1 = new MatMul(molecule -> norm_adj, feature);
			} else {
				level[l] -> product1 = new MatMul(molecule -> norm_adj, level[l - 1] -> hidden);
			}

			level[l] -> product2 = new MatMul(level[l] -> product1, level[l] -> W);
			level[l] -> activation = new LeakyReLU(level[l] -> product2);
			level[l] -> hidden = new Reshape2D(level[l] -> activation, molecule -> nVertices, nHiddens);
		}

		// Linear regression
		final_feature = new SumRows(level[nLevels] -> hidden);
		predict = new InnerProduct(W, final_feature);
		sql = new SquaredLoss(predict, target);

		// Complete the computation graph
		for (int l = 0; l <= nLevels; ++l) {
			graph -> add(level[l] -> product1, MATMUL);
			graph -> add(level[l] -> W, MATRIX);
			graph -> add(level[l] -> product2, MATMUL);
			graph -> add(level[l] -> activation, LEAKYRELU);
			graph -> add(level[l] -> hidden, RESHAPE2D);
		}

		graph -> add(final_feature, SUMROWS);
		graph -> add(W, MATRIX);
		graph -> add(predict, INNERPRODUCT);
		graph -> add(sql, SQUAREDLOSS);
	}

	void init_sum_gradients() {
		for (int l = 0; l <= nLevels; ++l) {
			for (int i = 0; i < level[l] -> W -> size; ++i) {
				level[l] -> sum_gradient_W -> gradient[i] = 0.0;
			}
		}

		for (int i = 0; i < W -> size; ++i) {
			sum_gradient_W -> gradient[i] = 0.0;
		}
	}

	void update_sum_gradients() {
		for (int l = 0; l <= nLevels; ++l) {
			for (int i = 0; i < level[l] -> W -> size; ++i) {
				level[l] -> sum_gradient_W -> gradient[i] += level[l] -> W -> gradient[i];
			}
		}

		for (int i = 0; i < W -> size; ++i) {
			sum_gradient_W -> gradient[i] += W -> gradient[i];
		}
	}

	void get_sum_gradients() {
		for (int l = 0; l <= nLevels; ++l) {
			for (int i = 0; i < level[l] -> W -> size; ++i) {
				level[l] -> W -> gradient[i] = level[l] -> sum_gradient_W -> gradient[i];
			}
		}

		for (int i = 0; i < W -> size; ++i) {
			W -> gradient[i] = sum_gradient_W -> gradient[i];
		}
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
		assert(molecule[0] -> nVertices <= max_nVertices);
		assert(molecule[0] -> nFeatures == nFeatures);

		pair < float, float > ret;
		ret.first = getLoss(nBatch, molecule, target);

		init_sum_gradients();

		for (int i = 0; i < nBatch; ++i) {
			complete_computation_graph(molecule[i]);
			this -> target -> value[0] = target[i];

			graph -> forward();
			graph -> backward();

			update_sum_gradients();
		}

		get_sum_gradients();
		sgd -> Learn(learning_rate, nBatch);

		ret.second = getLoss(nBatch, molecule, target);
		return ret;
	}

	pair<float, float> BatchLearn(int nBatch, DenseGraph **molecule, float *target, int nIterations, float learning_rate, float epsilon) {
		assert(nBatch > 0);
		assert(molecule[0] -> nVertices <= max_nVertices);
		assert(molecule[0] -> nFeatures == nFeatures);

		cache_parameters();

		pair<float, float> ret;
		ret.first = getLoss(nBatch, molecule, target);
		ret.second = ret.first;

		float decay_lr = 0.5;
		float min_lr = 1e-6;

		for (int iter = 0; iter < nIterations; ++iter) {
			init_sum_gradients();

			for (int i = 0; i < nBatch; ++i) {
				complete_computation_graph(molecule[i]);
				this -> target -> value[0] = target[i];

				graph -> forward();
				graph -> backward();

				update_sum_gradients();
			}

			get_sum_gradients();
			sgd -> Learn(learning_rate, nBatch);

			float loss = getLoss(nBatch, molecule, target);

			if (loss > ret.second) {
				restore_parameters();
				learning_rate *= decay_lr;
				if (learning_rate < min_lr) {
					break;
				}
			} else {
				ret.second = loss;
				cache_parameters();
			}
		}

		return ret;
	}

	pair<float, float> Learn(DenseGraph *molecule, float target, int nIterations, float learning_rate, float epsilon) {
		complete_computation_graph(molecule);
		this -> target -> value[0] = target;

		// Stochastic Gradient Descent
		cache_parameters();

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
				restore_parameters();
				learning_rate *= decay_lr;
				learning_rate = max(learning_rate, min_lr);
			} else {
				best_error = error;
				cache_parameters();
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
		for (int i = 0; i < final_feature -> size; ++i) {
			vect.push_back(final_feature -> value[i]);
		}
		return vect;
	}

	void save_model(string filename) {
		ofstream file(filename.c_str(), ios::out);
		
		for (int l = 0; l <= nLevels; ++l) {
			for (int i = 0; i < level[l] -> W -> size; ++i) {
				file << level[l] -> W -> value[i] << " ";
			}
			file << endl;
		}

		for (int i = 0; i < W -> size; ++i) {
			file << W -> value[i] << " ";
		}

		file.close();
	}

	void load_model(string filename) {
		ifstream file(filename.c_str(), ios::in);
		
		for (int l = 0; l <= nLevels; ++l) {
			for (int i = 0; i < level[l] -> W -> size; ++i) {
				file >> level[l] -> W -> value[i];
			}
		}

		for (int i = 0; i < W -> size; ++i) {
			file >> W -> value[i];
		}

		file.close();
	}

	struct Level {	
		Matrix *product1;
		Matrix *product2;
		Vector *activation;
		Matrix *hidden;

		Matrix *W;
		Matrix *prev_W;
		Matrix *sum_gradient_W;
	};

	Level **level;
	Matrix *feature;
	GraphFlow *graph;

	// Top layer as a linear regression
	Vector *W;
	Vector *prev_W;
	Vector *sum_gradient_W;

	SumRows *final_feature;
	Vector *predict;
	Vector *target;
	SquaredLoss *sql;

	// Stochastic Gradient Descent
	Momentum *sgd;

	int nLevels;
	int max_nVertices;
	int nFeatures;
	int nHiddens;
	int nDepth;
	float momentum_param;

	// Floyd-Warshall algorithm
	int **shortest_paths;

	~GCN_MW() {
		for (int l = 0; l <= nLevels; ++l) {
			delete level[l] -> W;
			delete level[l] -> prev_W;
			delete level[l] -> sum_gradient_W;
		}
		delete[] level;
		delete W;
		delete prev_W;
		delete sum_gradient_W;
		delete target;
		delete[] shortest_paths;
	}
};

#endif
