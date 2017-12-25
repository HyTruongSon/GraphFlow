// Framework: GraphFlow
// Class: Generalized Steerable Convolutional Networks (First-order representation) and Gated Recurrent Unit
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __GRU_GCN_1D_H_INCLUDED__
#define __GRU_GCN_1D_H_INCLUDED__

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

#include "SGD.h"
#include "Momentum.h"
#include "Adam.h"
#include "AdaMax.h"
#include "AdaDelta.h"

#include "SumGradients.h"
#include "CacheParameters.h"

#include "GraphFlow.h"

using namespace std;

const int INF = 1e9;

class GRU_GCN_1D {
public:
	GRU_GCN_1D(int nLevels, int max_nVertices, int nFeatures, int nHiddens, int nDepth, int max_Radius, double momentum_param) {
		this -> nLevels = nLevels;
		this -> max_nVertices = max_nVertices;
		this -> nFeatures = nFeatures;
		this -> nHiddens = nHiddens;
		this -> nDepth = nDepth;
		this -> max_Radius = max_Radius;
		this -> momentum_param = momentum_param;

		computation_graph();
		weights_initialization();
	}

	void computation_graph() {
		graph = new GraphFlow();

		// Vector 1
		one = new Vector(nHiddens);
		for (int i = 0; i < nHiddens; ++i) {
			one -> value[i] = 1.0;
		}

		// Feature vector for each vertex
		feature = new Vector* [max_nVertices];
		for (int v = 0; v < max_nVertices; ++v) {
			feature[v] = new Vector(nFeatures * (nDepth + 1));
		}

		// Parameters
		W = new Matrix(nHiddens, nFeatures * (nDepth + 1));
		W_z = new Matrix(nHiddens, nHiddens);
		U_z = new Matrix(nHiddens, nHiddens);
		W_r = new Matrix(nHiddens, nHiddens);
		U_r = new Matrix(nHiddens, nHiddens);
		W_h = new Matrix(nHiddens, nHiddens);
		U_h = new Matrix(nHiddens, nHiddens);
		W_g = new Matrix(nHiddens, nHiddens);
		U_g = new Matrix(nHiddens, nHiddens);
		U = new Vector(nHiddens);

		// Computation graph for each level
		level = new Level* [nLevels + 1];
		for (int l = 0; l <= nLevels; ++l) {
			level[l] = new Level();
			
			level[l] -> hidden = new Vector* [max_nVertices];

			if (l == 0) {
				level[l] -> map = new Vector* [max_nVertices];
				for (int v = 0; v < max_nVertices; ++v) {
					level[l] -> map[v] = new MatVecMul(W, feature[v]);
					level[l] -> hidden[v] = new Softmax(level[l] -> map[v]);
				}
			} else {
				// Representation
				level[l] -> neighbor = new RisiLayer1D* [max_nVertices];

				// Update gate
				level[l] -> update_part1 = new MatVecMul* [max_nVertices];
				level[l] -> update_part2 = new MatVecMul* [max_nVertices]; 
				level[l] -> update_sum = new Add* [max_nVertices];
				level[l] -> update = new Sigmoid* [max_nVertices];

				// Reset gate
				level[l] -> reset_part1 = new MatVecMul* [max_nVertices];
				level[l] -> reset_part2 = new MatVecMul* [max_nVertices]; 
				level[l] -> reset_sum = new Add* [max_nVertices];
				level[l] -> reset = new Sigmoid* [max_nVertices];

				// Candidate hidden
				level[l] -> candidate_prev = new Multiply* [max_nVertices];
				level[l] -> candidate_part1 = new MatVecMul* [max_nVertices];
				level[l] -> candidate_part2 = new MatVecMul* [max_nVertices];
				level[l] -> candidate_sum = new Add* [max_nVertices];
				level[l] -> candidate = new Tanh* [max_nVertices];

				// Hidden
				level[l] -> hidden_sub = new Subtract* [max_nVertices];
				level[l] -> hidden_part1 = new Multiply* [max_nVertices];
				level[l] -> hidden_part2 = new Multiply* [max_nVertices];

				for (int v = 0; v < max_nVertices; ++v) {
					// Representation
					level[l] -> neighbor[v] = new RisiLayer1D(nHiddens);

					// Update gate
					level[l] -> update_part1[v] = new MatVecMul(W_z, level[l] -> neighbor[v]);
					level[l] -> update_part2[v] = new MatVecMul(U_z, level[l - 1] -> hidden[v]);
					level[l] -> update_sum[v] = new Add(level[l] -> update_part1[v], level[l] -> update_part2[v]);
					level[l] -> update[v] = new Sigmoid(level[l] -> update_sum[v]);

					// Reset gate
					level[l] -> reset_part1[v] = new MatVecMul(W_r, level[l] -> neighbor[v]);
					level[l] -> reset_part2[v] = new MatVecMul(U_r, level[l - 1] -> hidden[v]);
					level[l] -> reset_sum[v] = new Add(level[l] -> reset_part1[v], level[l] -> reset_part2[v]);
					level[l] -> reset[v] = new Sigmoid(level[l] -> reset_sum[v]);

					// Candidate hidden
					level[l] -> candidate_prev[v] = new Multiply(level[l] -> reset[v], level[l - 1] -> hidden[v]);
					level[l] -> candidate_part1[v] = new MatVecMul(W_h, level[l] -> neighbor[v]);
					level[l] -> candidate_part2[v] = new MatVecMul(U_h, level[l] -> candidate_prev[v]);
					level[l] -> candidate_sum[v] = new Add(level[l] -> candidate_part1[v], level[l] -> candidate_part2[v]);
					level[l] -> candidate[v] = new Tanh(level[l] -> candidate_sum[v]);

					// Hidden
					level[l] -> hidden_sub[v] = new Subtract(one, level[l] -> update[v]);
					level[l] -> hidden_part1[v] = new Multiply(level[l] -> hidden_sub[v], level[l - 1] -> hidden[v]);
					level[l] -> hidden_part2[v] = new Multiply(level[l] -> update[v], level[l] -> candidate[v]);
					level[l] -> hidden[v] = new Add(level[l] -> hidden_part1[v], level[l] -> hidden_part2[v]);
				}
			}
		}

		// Computation graph for the top level 
		vertex_map1 = new MatVecMul* [max_nVertices];
		vertex_part1 = new Sigmoid* [max_nVertices];

		vertex_map2 = new MatVecMul* [max_nVertices];
		vertex_part2 = new Tanh* [max_nVertices];

		vertex_feature = new Multiply* [max_nVertices];

		for (int v = 0; v < max_nVertices; ++v) {
			vertex_map1[v] = new MatVecMul(W_g, level[nLevels] -> hidden[v]);
			vertex_part1[v] = new Sigmoid(vertex_map1[v]);

			vertex_map2[v] = new MatVecMul(U_g, level[nLevels] -> hidden[v]);
			vertex_part2[v] = new Tanh(vertex_map2[v]);

			vertex_feature[v] = new Multiply(vertex_part1[v], vertex_part2[v]);
		}

		graph_sum = new SumVectors(nHiddens);
		graph_feature = new Tanh(graph_sum);

		predict = new InnerProduct(U, graph_feature);
		target = new Vector(1);
		sql = new SquaredLoss(predict, target);

		// Stochastic Gradient Descent
		sgd = new Momentum(momentum_param);
		sgd -> add(W);
		sgd -> add(W_z);
		sgd -> add(U_z);
		sgd -> add(W_r);
		sgd -> add(U_r);
		sgd -> add(W_h);
		sgd -> add(U_h);
		sgd -> add(W_g);
		sgd -> add(U_g);
		sgd -> add(U);

		// Sum Gradients for Mini-batching
		sum_grad = new SumGradients();
		for (int i = 0; i < sgd -> params.size(); ++i) {
			sum_grad -> add(sgd -> params[i]);
		}

		// Cache Parameters
		cache_param = new CacheParameters();
		for (int i = 0; i < sgd -> params.size(); ++i) {
			cache_param -> add(sgd -> params[i]);
		}

		// Shortest Paths by Floyd-Warshall algorithm
		shortest_paths = new int* [max_nVertices];
		for (int i = 0; i < max_nVertices; ++i) {
			shortest_paths[i] = new int [max_nVertices];
		}
	}

	void weights_initialization() {
		for (int i = 0; i < sgd -> params.size(); ++i) {
			graph -> uniform_init(sgd -> params[i]);
		}
	}

	void complete_computation_graph(DenseGraph *molecule) {
		assert(molecule -> nFeatures == nFeatures);

		// Build a new computation graph
		graph -> clear();

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
		for (int v = 0; v < molecule -> nVertices; ++v) {
			for (int f = 0; f < nFeatures * (nDepth + 1); ++f) {
				this -> feature[v] -> value[f] = 0.0;
			}

			for (int d = 0; d <= nDepth; ++d) {
				for (int u = 0; u < molecule -> nVertices; ++u) {
					if (shortest_paths[u][v] == d) {
						for (int f = 0; f < nFeatures; ++f) {
							this -> feature[v] -> value[d * nFeatures + f] += molecule -> feature[u][f];
						}
					}
				}
			}
		}

		// Complete the computation graph
		graph -> add(W,	MATRIX);
		graph -> add(W_z, MATRIX);
		graph -> add(U_z, MATRIX);
		graph -> add(W_r, MATRIX);
		graph -> add(U_r, MATRIX);
		graph -> add(W_h, MATRIX);
		graph -> add(U_h, MATRIX);
		graph -> add(W_g, MATRIX);
		graph -> add(U_g, MATRIX);
		graph -> add(U, VECTOR);

		for (int l = 0; l <= nLevels; ++l) {
			if (l == 0) {
				for (int v = 0; v < molecule -> nVertices; ++v) {
					graph -> add(level[l] -> map[v], MATVECMUL);
					graph -> add(level[l] -> hidden[v], SOFTMAX);
				}
			} else {
				for (int v = 0; v < molecule -> nVertices; ++v) {
					// Representation
					level[l] -> neighbor[v] -> clear();
					for (int u = 0; u < molecule -> nVertices; ++u) {
						if (shortest_paths[v][u] <= min(l, max_Radius)) {
							level[l] -> neighbor[v] -> add_vector(level[l - 1] -> hidden[u]);
						}
					}
					graph -> add(level[l] -> neighbor[v], RISILAYER1D);

					// Update gate
					graph -> add(level[l] -> update_part1[v], MATVECMUL);
					graph -> add(level[l] -> update_part2[v], MATVECMUL);
					graph -> add(level[l] -> update_sum[v], ADD);
					graph -> add(level[l] -> update[v], SIGMOID);

					// Reset gate
					graph -> add(level[l] -> reset_part1[v], MATVECMUL);
					graph -> add(level[l] -> reset_part2[v], MATVECMUL);
					graph -> add(level[l] -> reset_sum[v], ADD);
					graph -> add(level[l] -> reset[v], SIGMOID);

					// Candidate hidden
					graph -> add(level[l] -> candidate_prev[v], MULTIPLY);
					graph -> add(level[l] -> candidate_part1[v], MATVECMUL);
					graph -> add(level[l] -> candidate_part2[v], MATVECMUL);
					graph -> add(level[l] -> candidate_sum[v], ADD);
					graph -> add(level[l] -> candidate[v], TANH);

					// Hidden
					graph -> add(level[l] -> hidden_sub[v], SUBTRACT);
					graph -> add(level[l] -> hidden_part1[v], MULTIPLY);
					graph -> add(level[l] -> hidden_part2[v], MULTIPLY);
					graph -> add(level[l] -> hidden[v], ADD);
				}
			}
		}

		// Computation graph for the top layer
		for (int v = 0; v < molecule -> nVertices; ++v) {
			graph -> add(vertex_map1[v], MATVECMUL);
			graph -> add(vertex_part1[v], SIGMOID);

			graph -> add(vertex_map2[v], MATVECMUL);
			graph -> add(vertex_part2[v], TANH);

			graph -> add(vertex_feature[v], MULTIPLY);
		}

		graph_sum -> clear();
		for (int v = 0; v < molecule -> nVertices; ++v) {
			graph_sum -> add_vector(vertex_feature[v]);
		}
		graph -> add(graph_sum, SUMVECTORS);

		graph -> add(graph_feature, TANH);
		graph -> add(predict, INNERPRODUCT);
		graph -> add(sql, SQUAREDLOSS);
	}

	double getLoss(int nBatch, DenseGraph **molecule, double *target) {
		double total_loss = 0.0;
		for (int i = 0; i < nBatch; ++i) {
			complete_computation_graph(molecule[i]);
			this -> target -> value[0] = target[i];
			graph -> forward();
			total_loss += sql -> getLoss();
		}
		return total_loss;
	}

	pair < double, double > BatchLearn(int nBatch, DenseGraph **molecule, double *target, double learning_rate) {
		assert(nBatch > 0);
		assert(molecule[0] -> nVertices <= max_nVertices);
		assert(molecule[0] -> nFeatures == nFeatures);

		pair < double, double > ret;
		ret.first = getLoss(nBatch, molecule, target);

		sum_grad -> reset_sum_gradients();

		for (int i = 0; i < nBatch; ++i) {
			complete_computation_graph(molecule[i]);
			this -> target -> value[0] = target[i];

			graph -> forward();
			graph -> backward();

			sum_grad -> cache_gradients();
		}

		sum_grad -> get_sum_gradients();
		sgd -> Learn(learning_rate, nBatch);

		ret.second = getLoss(nBatch, molecule, target);
		return ret;
	}

	pair<double, double> BatchLearn(int nBatch, DenseGraph **molecule, double *target, int nIterations, double learning_rate, double epsilon) {
		assert(nBatch > 0);
		assert(molecule[0] -> nVertices <= max_nVertices);
		assert(molecule[0] -> nFeatures == nFeatures);

		cache_param -> cache_parameters();

		pair<double, double> ret;
		ret.first = getLoss(nBatch, molecule, target);
		ret.second = ret.first;

		double decay_lr = 0.5;
		double min_lr = 1e-6;

		for (int iter = 0; iter < nIterations; ++iter) {
			sum_grad -> reset_sum_gradients();

			for (int i = 0; i < nBatch; ++i) {
				complete_computation_graph(molecule[i]);
				this -> target -> value[0] = target[i];

				graph -> forward();
				graph -> backward();

				sum_grad -> cache_gradients();
			}

			sum_grad -> get_sum_gradients();
			sgd -> Learn(learning_rate, nBatch);

			double loss = getLoss(nBatch, molecule, target);

			if (loss > ret.second) {
				cache_param -> restore_parameters();
				learning_rate *= decay_lr;
				if (learning_rate < min_lr) {
					break;
				}
			} else {
				ret.second = loss;
				cache_param -> cache_parameters();
			}
		}

		return ret;
	}

	pair<double, double> Learn(DenseGraph *molecule, double target, int nIterations, double learning_rate, double epsilon) {
		complete_computation_graph(molecule);
		this -> target -> value[0] = target;

		// Stochastic Gradient Descent
		cache_param -> cache_parameters();

		graph -> forward();
		double best_error = sql -> getLoss();

		if (best_error < epsilon) {
			return make_pair(best_error, best_error);
		}

		pair<double, double> ret;
		ret.first = best_error;

		double decay_lr = 0.5;
		double min_lr = 1e-6;

		for (int iter = 0; iter < nIterations; ++iter) {
			graph -> forward();
			graph -> backward();
			sgd -> Learn(learning_rate);

			graph -> forward();
			double error = sql -> getLoss();

			if (error < epsilon) {
				break;
			}

			if (error >= best_error) {
				cache_param -> restore_parameters();
				learning_rate *= decay_lr;
				learning_rate = max(learning_rate, min_lr);
			} else {
				best_error = error;
				cache_param -> cache_parameters();
			}
		}

		ret.second = best_error;
		return ret;
	}

	double Predict(DenseGraph *molecule) {
		complete_computation_graph(molecule);

		graph -> forward();

		return predict -> value[0];
	}

	vector<double> Feature(DenseGraph *molecule) {
		complete_computation_graph(molecule);

		graph -> forward();

		vector<double> vect;
		vect.clear();
		for (int i = 0; i < graph_feature -> size; ++i) {
			vect.push_back(graph_feature -> value[i]);
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

	struct Level {
		Vector **hidden;
		Vector **map;

		// Representation
		RisiLayer1D **neighbor;

		// Update gate
		MatVecMul **update_part1;
		MatVecMul **update_part2;
		Add **update_sum;
		Sigmoid **update;

		// Reset gate
		MatVecMul **reset_part1;
		MatVecMul **reset_part2;
		Add **reset_sum;
		Sigmoid **reset;

		// Candidate hidden
		Multiply **candidate_prev;
		MatVecMul **candidate_part1;
		MatVecMul **candidate_part2;
		Add **candidate_sum;
		Tanh **candidate;

		// Hidden
		Subtract **hidden_sub;
		Multiply **hidden_part1;
		Multiply **hidden_part2;
	};

	Level **level;
	Vector **feature;
	GraphFlow *graph;

	// Parameters
	Matrix *W;
	Matrix *W_z;
	Matrix *U_z;
	Matrix *W_r;
	Matrix *U_r;
	Matrix *W_h;
	Matrix *U_h;
	Matrix *W_g;
	Matrix *U_g;
	Vector *U;

	Vector *one;

	MatVecMul **vertex_map1;
	Sigmoid **vertex_part1;

	MatVecMul **vertex_map2;
	Tanh **vertex_part2;

	Multiply **vertex_feature;
	SumVectors *graph_sum;
	Tanh *graph_feature;

	Vector *predict;
	Vector *target;
	SquaredLoss *sql;

	// Stochastic Gradient Descent
	Momentum *sgd;

	// Sum Gradients for Mini-batching
	SumGradients *sum_grad;

	// Cache Parameters
	CacheParameters *cache_param;

	int nLevels;
	int max_nVertices;
	int nFeatures;
	int nHiddens;
	int nDepth;
	int max_Radius;
	double momentum_param;

	// Floyd-Warshall algorithm
	int **shortest_paths;

	~GRU_GCN_1D() {
		for (int l = 0; l <= nLevels; ++l) {
			delete[] level[l] -> hidden;
			delete[] level[l] -> map;
			delete[] level[l] -> neighbor;
			delete[] level[l] -> update_part1;
			delete[] level[l] -> update_part2;
			delete[] level[l] -> update_sum;
			delete[] level[l] -> update;
			delete[] level[l] -> reset_part1;
			delete[] level[l] -> reset_part2;
			delete[] level[l] -> reset_sum;
			delete[] level[l] -> reset;
			delete[] level[l] -> candidate_prev;
			delete[] level[l] -> candidate_part1;
			delete[] level[l] -> candidate_part2;
			delete[] level[l] -> candidate_sum;
			delete[] level[l] -> candidate;
			delete[] level[l] -> hidden_sub;
			delete[] level[l] -> hidden_part1;
			delete[] level[l] -> hidden_part2;
		}

		delete[] level;
		delete[] feature;
		
		delete W;
		delete W_z;
		delete U_z;
		delete W_r;
		delete U_r;
		delete W_h;
		delete U_h;
		delete W_g;
		delete U_g;
		delete U;

		delete one;
		delete vertex_map1;
		delete vertex_part1;
		delete vertex_map2;
		delete vertex_part2;
		delete vertex_feature;
		delete graph_sum;
		delete graph_feature;
		delete predict;
		delete target;
		delete sql;

		delete sgd;
		delete sum_grad;
		delete cache_param;
		delete[] shortest_paths;
	}
};

#endif