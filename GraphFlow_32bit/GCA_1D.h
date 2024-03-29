// Framework: GraphFlow
// Class: Generalized Steerable Convolutional Autoencoder (First-order representation)
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __GCA_1D_H_INCLUDED__
#define __GCA_1D_H_INCLUDED__

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

class GCA_1D {
public:
	GCA_1D(int nLevels, int max_nVertices, int nFeatures, int nHiddens, int nDepth, int max_Radius, float momentum_param) {
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
		feature = new Vector* [max_nVertices];
		for (int v = 0; v < max_nVertices; ++v) {
			feature[v] = new Vector(nFeatures * (nDepth + 1));
		}

		// Computation graph for each level
		level = new Level* [nLevels + 1];
		for (int l = 0; l <= nLevels; ++l) {
			level[l] = new Level();
			
			// Learning parameters
			level[l] -> W1 = new Matrix(nHiddens, nFeatures * (nDepth + 1));
			level[l] -> prev_W1 = new Matrix(nHiddens, nFeatures * (nDepth + 1));
			level[l] -> sum_gradient_W1 = new Matrix(nHiddens, nFeatures * (nDepth + 1));

			level[l] -> sum = new Vector* [max_nVertices];
			level[l] -> hidden = new Vector* [max_nVertices];

			// For computation graph of level 0
			if (l == 0) {
				for (int v = 0; v < max_nVertices; ++v) {
					level[l] -> sum[v] = new MatVecMul(level[l] -> W1, feature[v]);
					level[l] -> hidden[v] = new Softmax(level[l] -> sum[v]);
				}
			} else {
				// For computation graph of level > 0
				level[l] -> W2 = new Matrix(nHiddens, nHiddens);
				level[l] -> prev_W2 = new Matrix(nHiddens, nHiddens);
				level[l] -> sum_gradient_W2 = new Matrix(nHiddens, nHiddens);

				level[l] -> neighbor = new RisiLayer1D* [max_nVertices];
				level[l] -> part1 = new MatVecMul* [max_nVertices];
				level[l] -> part2 = new MatVecMul* [max_nVertices];

				for (int v = 0; v < max_nVertices; ++v) {
					level[l] -> part1[v] = new MatVecMul(level[l] -> W1, feature[v]);
					level[l] -> neighbor[v] = new RisiLayer1D(nHiddens);
					level[l] -> part2[v] = new MatVecMul(level[l] -> W2, level[l] -> neighbor[v]);
					level[l] -> sum[v] = new Add(level[l] -> part1[v], level[l] -> part2[v]);
					level[l] -> hidden[v] = new Softmax(level[l] -> sum[v]);
				}
			}
		}

		// Computation graph for the top level
		predict = NULL;
		target = NULL;
		sql = NULL;

		// Stochastic Gradient Descent
		sgd = new Momentum(momentum_param);
		for (int l = 0; l <= nLevels; ++l) {
			sgd -> add(level[l] -> W1);

			if (l > 0) {
				sgd -> add(level[l] -> W2);
			}
		}

		// Shortest Paths by Floyd-Warshall algorithm
		shortest_paths = new int* [max_nVertices];
		for (int i = 0; i < max_nVertices; ++i) {
			shortest_paths[i] = new int [max_nVertices];
		}
	}

	void weights_initialization() {
		for (int l = 0; l <= nLevels; ++l) {
			graph -> uniform_init(level[l] -> W1);

			if (l > 0) {
				graph -> uniform_init(level[l] -> W2);
			}
		}
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
			copy(level[l] -> W1, level[l] -> prev_W1);

			if (l > 0) {
				copy(level[l] -> W2, level[l] -> prev_W2);
			}
		}
	}

	void restore_parameters() {
		for (int l = 0; l <= nLevels; ++l) {
			copy(level[l] -> prev_W1, level[l] -> W1);

			if (l > 0) {
				copy(level[l] -> prev_W2, level[l] -> W2);
			}
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
		for (int l = 0; l <= nLevels; ++l) {
			graph -> add(level[l] -> W1, MATRIX);

			if (l == 0) {
				for (int v = 0; v < molecule -> nVertices; ++v) {
					graph -> add(level[l] -> sum[v], MATVECMUL);
					graph -> add(level[l] -> hidden[v], SOFTMAX);
				}
			} else {
				graph -> add(level[l] -> W2, MATRIX);

				for (int v = 0; v < molecule -> nVertices; ++v) {
					graph -> add(level[l] -> part1[v], MATVECMUL);

					level[l] -> neighbor[v] -> clear();
					for (int u = 0; u < molecule -> nVertices; ++u) {
						if (shortest_paths[v][u] <= min(l, max_Radius)) {
							level[l] -> neighbor[v] -> add_vector(level[l - 1] -> hidden[u]);
						}
					}

					graph -> add(level[l] -> neighbor[v], RISILAYER1D);
					graph -> add(level[l] -> part2[v], MATVECMUL);
					graph -> add(level[l] -> sum[v], ADD);
					graph -> add(level[l] -> hidden[v], SOFTMAX);
				}
			}
		}

		// Computation graph for the top layer
		if (predict != NULL) {
			delete predict;
		}
		if (target != NULL) {
			delete target;
		}
		if (sql != NULL) {
			delete sql;
		}

		predict = new LinearGram(molecule -> nVertices);
		for (int v = 0; v < molecule -> nVertices; ++v) {
			predict -> add_vector(level[nLevels] -> hidden[v]);
		}

		target = new Matrix(molecule -> nVertices, molecule -> nVertices);
		for (int x = 0; x < molecule -> nVertices; ++x) {
			for (int y = 0; y < molecule -> nVertices; ++y) {
				int i = target -> index(x, y);
				target -> value[i] = molecule -> adj[x][y];
			}
		}

		sql = new SquaredLoss(predict, target);

		graph -> add(predict, LINEARGRAM);
		graph -> add(sql, SQUAREDLOSS);
	}

	void init_sum_gradients() {
		for (int l = 0; l <= nLevels; ++l) {
			for (int i = 0; i < level[l] -> W1 -> size; ++i) {
				level[l] -> sum_gradient_W1 -> gradient[i] = 0.0;
			}

			if (l > 0) {
				for (int i = 0; i < level[l] -> W2 -> size; ++i) {
					level[l] -> sum_gradient_W2 -> gradient[i] = 0.0;
				}
			}
		}
	}

	void update_sum_gradients() {
		for (int l = 0; l <= nLevels; ++l) {
			for (int i = 0; i < level[l] -> W1 -> size; ++i) {
				level[l] -> sum_gradient_W1 -> gradient[i] += level[l] -> W1 -> gradient[i];
			}

			if (l > 0) {
				for (int i = 0; i < level[l] -> W2 -> size; ++i) {
					level[l] -> sum_gradient_W2 -> gradient[i] += level[l] -> W2 -> gradient[i];
				}
			}
		}
	}

	void get_sum_gradients() {
		for (int l = 0; l <= nLevels; ++l) {
			for (int i = 0; i < level[l] -> W1 -> size; ++i) {
				level[l] -> W1 -> gradient[i] = level[l] -> sum_gradient_W1 -> gradient[i];
			}

			if (l > 0) {
				for (int i = 0; i < level[l] -> W2 -> size; ++i) {
					level[l] -> W2 -> gradient[i] = level[l] -> sum_gradient_W2 -> gradient[i];
				}
			}
		}
	}

	float getLoss(int nBatch, DenseGraph **molecule) {
		float total_loss = 0.0;
		for (int i = 0; i < nBatch; ++i) {
			complete_computation_graph(molecule[i]);
			graph -> forward();
			total_loss += sql -> getLoss();
		}
		return total_loss;
	}

	pair < float, float > BatchLearn(int nBatch, DenseGraph **molecule, float learning_rate) {
		assert(nBatch > 0);
		assert(molecule[0] -> nVertices <= max_nVertices);
		assert(molecule[0] -> nFeatures == nFeatures);

		pair < float, float > ret;
		ret.first = getLoss(nBatch, molecule);

		init_sum_gradients();

		for (int i = 0; i < nBatch; ++i) {
			complete_computation_graph(molecule[i]);

			graph -> forward();
			graph -> backward();

			update_sum_gradients();
		}

		get_sum_gradients();
		sgd -> Learn(learning_rate, nBatch);

		ret.second = getLoss(nBatch, molecule);
		return ret;
	}

	pair<float, float> BatchLearn(int nBatch, DenseGraph **molecule, int nIterations, float learning_rate, float epsilon) {
		assert(nBatch > 0);
		assert(molecule[0] -> nVertices <= max_nVertices);
		assert(molecule[0] -> nFeatures == nFeatures);

		cache_parameters();

		pair<float, float> ret;
		ret.first = getLoss(nBatch, molecule);
		ret.second = ret.first;

		float decay_lr = 0.5;
		float min_lr = 1e-6;

		for (int iter = 0; iter < nIterations; ++iter) {
			init_sum_gradients();

			for (int i = 0; i < nBatch; ++i) {
				complete_computation_graph(molecule[i]);

				graph -> forward();
				graph -> backward();

				update_sum_gradients();
			}

			get_sum_gradients();
			sgd -> Learn(learning_rate, nBatch);

			float loss = getLoss(nBatch, molecule);

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

	pair<float, float> Learn(DenseGraph *molecule, int nIterations, float learning_rate, float epsilon) {
		complete_computation_graph(molecule);

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

	void Predict(DenseGraph *molecule, float **mat) {
		complete_computation_graph(molecule);

		graph -> forward();

		for (int x = 0; x < molecule -> nVertices; ++x) {
			for (int y = 0; y < molecule -> nVertices; ++y) {
				int i = predict -> index(x, y);
				mat[x][y] = predict -> value[i];
			}
		}
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

		file.close();
	}

	struct Level {
		Vector **hidden;
		Vector **sum;

		RisiLayer1D **neighbor;
		MatVecMul **part1;
		MatVecMul **part2;
		
		Matrix *W1;
		Matrix *W2;

		Matrix *prev_W1;
		Matrix *prev_W2;

		Matrix *sum_gradient_W1;
		Matrix *sum_gradient_W2;
	};

	Level **level;
	Vector **feature;
	GraphFlow *graph;

	// Top layer
	LinearGram *predict;
	Matrix *target;
	SquaredLoss *sql;

	// Stochastic Gradient Descent
	Momentum *sgd;

	int nLevels;
	int max_nVertices;
	int nFeatures;
	int nHiddens;
	int nDepth;
	int max_Radius;
	float momentum_param;

	// Floyd-Warshall algorithm
	int **shortest_paths;

	~GCA_1D() {
		for (int l = 0; l <= nLevels; ++l) {
			delete[] level[l] -> hidden;
			delete[] level[l] -> sum;
			delete[] level[l] -> neighbor;
			delete[] level[l] -> part1;
			delete[] level[l] -> part2;
			delete level[l] -> W1;
			delete level[l] -> W2;
			delete level[l] -> prev_W1;
			delete level[l] -> prev_W2;
			delete level[l] -> sum_gradient_W1;
			delete level[l] -> sum_gradient_W2;
		}
		delete[] level;
		delete[] feature;
		delete predict;
		delete target;
		delete sql;
		delete[] shortest_paths;
	}
};

#endif
