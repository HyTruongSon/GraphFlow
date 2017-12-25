// Framework: GraphFlow
// Class: Generalized Steerable Convolutional Networks (Third-order representation) for Kernel Learning
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __GCN_3D_KERNEL_H_INCLUDED__
#define __GCN_3D_KERNEL_H_INCLUDED__

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
#include "CacheParameters.h"
#include "SumGradients.h"
#include "GraphFlow.h"

using namespace std;

const int INF = 1e9;

class GCN_3D_Kernel {
public:
	GCN_3D_Kernel(int nLevels, int max_nVertices, int nFeatures, int nHiddens, int nDepth, int max_Radius, float momentum_param) {
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

		// Feature vector for each vertex
		feature_X = new Vector* [max_nVertices];
		feature_Y = new Vector* [max_nVertices];

		for (int v = 0; v < max_nVertices; ++v) {
			feature_X[v] = new Vector(nFeatures * (nDepth + 1));
			feature_Y[v] = new Vector(nFeatures * (nDepth + 1));
		}

		// Computation graph for each level
		level = new Level* [nLevels + 1];
		for (int l = 0; l <= nLevels; ++l) {
			level[l] = new Level();
			
			// Learning parameters
			level[l] -> W1 = new Matrix(nHiddens, nFeatures * (nDepth + 1));
			
			level[l] -> sum_X = new Vector* [max_nVertices];
			level[l] -> hidden_X = new Vector* [max_nVertices];

			level[l] -> sum_Y = new Vector* [max_nVertices];
			level[l] -> hidden_Y = new Vector* [max_nVertices];

			// For computation graph of level 0
			if (l == 0) {
				for (int v = 0; v < max_nVertices; ++v) {
					level[l] -> sum_X[v] = new MatVecMul(level[l] -> W1, feature_X[v]);
					level[l] -> hidden_X[v] = new Softmax(level[l] -> sum_X[v]);

					level[l] -> sum_Y[v] = new MatVecMul(level[l] -> W1, feature_Y[v]);
					level[l] -> hidden_Y[v] = new Softmax(level[l] -> sum_Y[v]);
				}
			} else {
				// For computation graph of level > 0
				level[l] -> W2 = new Matrix(nHiddens, nHiddens);

				level[l] -> neighbor_X = new RisiLayer3D* [max_nVertices];
				level[l] -> pooling_X = new KMax* [max_nVertices];
				level[l] -> part1_X = new MatVecMul* [max_nVertices];
				level[l] -> part2_X = new MatVecMul* [max_nVertices];

				level[l] -> neighbor_Y = new RisiLayer3D* [max_nVertices];
				level[l] -> pooling_Y = new KMax* [max_nVertices];
				level[l] -> part1_Y = new MatVecMul* [max_nVertices];
				level[l] -> part2_Y = new MatVecMul* [max_nVertices];

				for (int v = 0; v < max_nVertices; ++v) {
					level[l] -> part1_X[v] = new MatVecMul(level[l] -> W1, feature_X[v]);
					level[l] -> neighbor_X[v] = new RisiLayer3D(nHiddens);
					level[l] -> pooling_X[v] = new KMax(level[l] -> neighbor_X[v], nHiddens);
					level[l] -> part2_X[v] = new MatVecMul(level[l] -> W2, level[l] -> pooling_X[v]);
					level[l] -> sum_X[v] = new Add(level[l] -> part1_X[v], level[l] -> part2_X[v]);
					level[l] -> hidden_X[v] = new Softmax(level[l] -> sum_X[v]);

					level[l] -> part1_Y[v] = new MatVecMul(level[l] -> W1, feature_Y[v]);
					level[l] -> neighbor_Y[v] = new RisiLayer3D(nHiddens);
					level[l] -> pooling_Y[v] = new KMax(level[l] -> neighbor_Y[v], nHiddens);
					level[l] -> part2_Y[v] = new MatVecMul(level[l] -> W2, level[l] -> pooling_Y[v]);
					level[l] -> sum_Y[v] = new Add(level[l] -> part1_Y[v], level[l] -> part2_Y[v]);
					level[l] -> hidden_Y[v] = new Softmax(level[l] -> sum_Y[v]);
				}
			}
		}

		// Computation graph for the top level as a linear regression
		W = new Vector(2 * nHiddens);

		top_feature_X = new RisiLayer1D(nHiddens);
		top_feature_Y = new RisiLayer1D(nHiddens);

		final_feature = new ConCat(top_feature_X, top_feature_Y);
		predict = new InnerProduct(W, final_feature);
		target = new Vector(1);
		sql = new SquaredLoss(predict, target);

		// Stochastic Gradient Descent
		sgd = new Momentum(momentum_param);
		for (int l = 0; l <= nLevels; ++l) {
			sgd -> add(level[l] -> W1);

			if (l > 0) {
				sgd -> add(level[l] -> W2);
			}
		}
		sgd -> add(W);

		// Cache Parameters
		cache_param = new CacheParameters();
		for (int i = 0; i < sgd -> params.size(); ++i) {
			cache_param -> add(sgd -> params[i]);
		}

		// Sum Gradients
		sum_grad = new SumGradients();
		for (int i = 0; i < sgd -> params.size(); ++i) {
			sum_grad -> add(sgd -> params[i]);
		}

		// Shortest Paths by Floyd-Warshall algorithm
		shortest_paths_X = new int* [max_nVertices];
		shortest_paths_Y = new int* [max_nVertices];

		for (int i = 0; i < max_nVertices; ++i) {
			shortest_paths_X[i] = new int [max_nVertices];
			shortest_paths_Y[i] = new int [max_nVertices];
		}
	}

	void weights_initialization() {
		for (int l = 0; l <= nLevels; ++l) {
			graph -> uniform_init(level[l] -> W1);

			if (l > 0) {
				graph -> uniform_init(level[l] -> W2);
			}
		}

		graph -> uniform_init(W);
	}

	void Floyd_Warshall(DenseGraph *molecule, int **shortest_paths) {
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
	}

	void Weisfeiler_Lehman(DenseGraph *molecule, int **shortest_paths, Vector **feature) {
		for (int v = 0; v < molecule -> nVertices; ++v) {
			for (int f = 0; f < nFeatures * (nDepth + 1); ++f) {
				feature[v] -> value[f] = 0.0;
			}

			for (int d = 0; d <= nDepth; ++d) {
				for (int u = 0; u < molecule -> nVertices; ++u) {
					if (shortest_paths[u][v] == d) {
						for (int f = 0; f < nFeatures; ++f) {
							feature[v] -> value[d * nFeatures + f] += molecule -> feature[u][f];
						}
					}
				}
			}
		}
	}

	void complete_computation_graph(DenseGraph *molecule_X, DenseGraph *molecule_Y) {
		assert(molecule_X -> nFeatures == nFeatures);
		assert(molecule_Y -> nFeatures == nFeatures);

		// Build a new computation graph
		graph -> clear();

		// Floyd-Warshall algorithm
		Floyd_Warshall(molecule_X, shortest_paths_X);
		Floyd_Warshall(molecule_Y, shortest_paths_Y);

		// Get the feature vector for each vertex
		Weisfeiler_Lehman(molecule_X, shortest_paths_X, feature_X);
		Weisfeiler_Lehman(molecule_Y, shortest_paths_Y, feature_Y);

		// Complete the computation graph
		for (int l = 0; l <= nLevels; ++l) {
			graph -> add(level[l] -> W1, MATRIX);

			if (l == 0) {
				for (int v = 0; v < molecule_X -> nVertices; ++v) {
					graph -> add(level[l] -> sum_X[v], MATVECMUL);
					graph -> add(level[l] -> hidden_X[v], SOFTMAX);
				}

				for (int v = 0; v < molecule_Y -> nVertices; ++v) {
					graph -> add(level[l] -> sum_Y[v], MATVECMUL);
					graph -> add(level[l] -> hidden_Y[v], SOFTMAX);
				}
			} else {
				graph -> add(level[l] -> W2, MATRIX);

				for (int v = 0; v < molecule_X -> nVertices; ++v) {
					graph -> add(level[l] -> part1_X[v], MATVECMUL);

					level[l] -> neighbor_X[v] -> clear();
					for (int u = 0; u < molecule_X -> nVertices; ++u) {
						if (shortest_paths_X[v][u] <= min(l, max_Radius)) {
							level[l] -> neighbor_X[v] -> add_vector(level[l - 1] -> hidden_X[u]);
						}
					}

					graph -> add(level[l] -> neighbor_X[v], RISILAYER3D);
					graph -> add(level[l] -> pooling_X[v], KMAX);
					graph -> add(level[l] -> part2_X[v], MATVECMUL);
					graph -> add(level[l] -> sum_X[v], ADD);
					graph -> add(level[l] -> hidden_X[v], SOFTMAX);
				}

				for (int v = 0; v < molecule_Y -> nVertices; ++v) {
					graph -> add(level[l] -> part1_Y[v], MATVECMUL);

					level[l] -> neighbor_Y[v] -> clear();
					for (int u = 0; u < molecule_Y -> nVertices; ++u) {
						if (shortest_paths_Y[v][u] <= min(l, max_Radius)) {
							level[l] -> neighbor_Y[v] -> add_vector(level[l - 1] -> hidden_Y[u]);
						}
					}

					graph -> add(level[l] -> neighbor_Y[v], RISILAYER3D);
					graph -> add(level[l] -> pooling_Y[v], KMAX);
					graph -> add(level[l] -> part2_Y[v], MATVECMUL);
					graph -> add(level[l] -> sum_Y[v], ADD);
					graph -> add(level[l] -> hidden_Y[v], SOFTMAX);
				}
			}
		}

		// Computation graph for the top layer as a linear regression
		top_feature_X -> clear();
		for (int v = 0; v < molecule_X -> nVertices; ++v) {
			top_feature_X -> add_vector(level[nLevels] -> hidden_X[v]);
		}
		graph -> add(top_feature_X, RISILAYER1D);

		top_feature_Y -> clear();
		for (int v = 0; v < molecule_Y -> nVertices; ++v) {
			top_feature_Y -> add_vector(level[nLevels] -> hidden_Y[v]);
		}
		graph -> add(top_feature_Y, RISILAYER1D);

		graph -> add(final_feature, CONCAT);
		graph -> add(W, VECTOR);
		graph -> add(predict, INNERPRODUCT);
		graph -> add(sql, SQUAREDLOSS);
	}

	float getLoss(int nBatch, DenseGraph **molecule_X, DenseGraph **molecule_Y, float *target) {
		float total_loss = 0.0;
		for (int i = 0; i < nBatch; ++i) {
			complete_computation_graph(molecule_X[i], molecule_Y[i]);
			this -> target -> value[0] = target[i];
			graph -> forward();
			total_loss += sql -> getLoss();
		}
		return total_loss;
	}

	pair < float, float > BatchLearn(int nBatch, DenseGraph **molecule_X, DenseGraph **molecule_Y, float *target, float learning_rate) {
		assert(nBatch > 0);

		pair < float, float > ret;
		ret.first = getLoss(nBatch, molecule_X, molecule_Y, target);

		sum_grad -> reset_sum_gradients();

		for (int i = 0; i < nBatch; ++i) {
			complete_computation_graph(molecule_X[i], molecule_Y[i]);
			this -> target -> value[0] = target[i];

			graph -> forward();
			graph -> backward();

			sum_grad -> cache_gradients();
		}

		sum_grad -> get_sum_gradients();
		sgd -> Learn(learning_rate, nBatch);

		ret.second = getLoss(nBatch, molecule_X, molecule_Y, target);
		return ret;
	}

	pair<float, float> BatchLearn(int nBatch, DenseGraph **molecule_X, DenseGraph **molecule_Y, float *target, int nIterations, float learning_rate, float epsilon) {
		assert(nBatch > 0);

		cache_param -> cache_parameters();

		pair<float, float> ret;
		ret.first = getLoss(nBatch, molecule_X, molecule_Y, target);
		ret.second = ret.first;

		float decay_lr = 0.5;
		float min_lr = 1e-6;

		for (int iter = 0; iter < nIterations; ++iter) {
			sum_grad -> reset_sum_gradients();

			for (int i = 0; i < nBatch; ++i) {
				complete_computation_graph(molecule_X[i], molecule_Y[i]);
				this -> target -> value[0] = target[i];

				graph -> forward();
				graph -> backward();

				sum_grad -> cache_gradients();
			}

			sum_grad -> get_sum_gradients();
			sgd -> Learn(learning_rate, nBatch);

			float loss = getLoss(nBatch, molecule_X, molecule_Y, target);

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

	pair<float, float> Learn(DenseGraph *molecule_X, DenseGraph *molecule_Y, float target, int nIterations, float learning_rate, float epsilon) {
		complete_computation_graph(molecule_X, molecule_Y);
		this -> target -> value[0] = target;

		// Stochastic Gradient Descent
		cache_param -> cache_parameters();

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

	float Predict(DenseGraph *molecule_X, DenseGraph *molecule_Y) {
		complete_computation_graph(molecule_X, molecule_Y);

		graph -> forward();

		return predict -> value[0];
	}

	vector<float> Feature(DenseGraph *molecule_X, DenseGraph *molecule_Y) {
		complete_computation_graph(molecule_X, molecule_Y);

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
			for (int i = 0; i < level[l] -> W1 -> size; ++i) {
				file << level[l] -> W1 -> value[i] << " ";
			}
			file << endl;

			if (l > 0) {
				for (int i = 0; i < level[l] -> W2 -> size; ++i) {
					file << level[l] -> W2 -> value[i] << " ";
				}
				file << endl;
			}
		}

		for (int i = 0; i < W -> size; ++i) {
			file << W -> value[i] << " ";
		}

		file.close();
	}

	void load_model(string filename) {
		ifstream file(filename.c_str(), ios::in);
		
		for (int l = 0; l <= nLevels; ++l) {
			for (int i = 0; i < level[l] -> W1 -> size; ++i) {
				file >> level[l] -> W1 -> value[i];
			}

			if (l > 0) {
				for (int i = 0; i < level[l] -> W2 -> size; ++i) {
					file >> level[l] -> W2 -> value[i];
				}
			}
		}

		for (int i = 0; i < W -> size; ++i) {
			file >> W -> value[i];
		}

		file.close();
	}

	struct Level {
		Vector **hidden_X;
		Vector **sum_X;

		RisiLayer3D **neighbor_X;
		KMax **pooling_X;
		MatVecMul **part1_X;
		MatVecMul **part2_X;

		Vector **hidden_Y;
		Vector **sum_Y;

		RisiLayer3D **neighbor_Y;
		KMax **pooling_Y;
		MatVecMul **part1_Y;
		MatVecMul **part2_Y;
		
		Matrix *W1;
		Matrix *W2;
	};
	Level **level;
	
	Vector **feature_X;
	Vector **feature_Y;

	// Top layer as a linear regression
	RisiLayer1D *top_feature_X;
	RisiLayer1D *top_feature_Y;
	ConCat *final_feature;

	Vector *W;
	Vector *predict;
	Vector *target;
	SquaredLoss *sql;

	// Stochastic Gradient Descent
	Momentum *sgd;

	// Computation graph
	GraphFlow *graph;

	int nLevels;
	int max_nVertices;
	int nFeatures;
	int nHiddens;
	int nDepth;
	int max_Radius;
	float momentum_param;

	// Floyd-Warshall algorithm
	int **shortest_paths_X;
	int **shortest_paths_Y;

	// Cache parameters
	CacheParameters *cache_param;

	// Sum Gradients
	SumGradients *sum_grad;

	~GCN_3D_Kernel() {
		for (int l = 0; l <= nLevels; ++l) {
			delete[] level[l] -> hidden_X;
			delete[] level[l] -> sum_X;
			delete[] level[l] -> neighbor_X;
			delete[] level[l] -> pooling_X;
			delete[] level[l] -> part1_X;
			delete[] level[l] -> part2_X;

			delete[] level[l] -> hidden_Y;
			delete[] level[l] -> sum_Y;
			delete[] level[l] -> neighbor_Y;
			delete[] level[l] -> pooling_Y;
			delete[] level[l] -> part1_Y;
			delete[] level[l] -> part2_Y;

			delete level[l] -> W1;
			delete level[l] -> W2;
		}
		delete[] level;
		delete[] feature_X;
		delete[] feature_Y;
		delete W;
		delete top_feature_X;
		delete top_feature_Y;
		delete final_feature;
		delete predict;
		delete target;
		delete sql;
		delete[] shortest_paths_X;
		delete[] shortest_paths_Y;
	}
};

#endif
