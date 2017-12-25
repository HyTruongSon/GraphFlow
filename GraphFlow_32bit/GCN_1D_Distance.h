// Framework: GraphFlow
// Class: Generalized Steerable Convolutional Networks (First-order representation)
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __GCN_1D_DISTANCE_H_INCLUDED__
#define __GCN_1D_DISTANCE_H_INCLUDED__

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

class GCN_1D_Distance {
public:
	GCN_1D_Distance(int nLevels, int max_nVertices, int nFeatures, int nHiddens, int nDepth, int max_Radius, float momentum_param) {
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

		// +----------------------------+
		// | Chanel for vertex features |
		// +----------------------------+

		// Feature vector for each vertex
		feature = new Vector* [max_nVertices];
		for (int v = 0; v < max_nVertices; ++v) {
			feature[v] = new Vector(nFeatures * (nDepth + 1));
		}

		// Computation graph for each level
		chanel_vertex = new Level* [nLevels + 1];
		for (int l = 0; l <= nLevels; ++l) {
			chanel_vertex[l] = new Level();
			
			// Learning parameters
			chanel_vertex[l] -> W1 = new Matrix(nHiddens, nFeatures * (nDepth + 1));
			chanel_vertex[l] -> prev_W1 = new Matrix(nHiddens, nFeatures * (nDepth + 1));
			chanel_vertex[l] -> sum_gradient_W1 = new Matrix(nHiddens, nFeatures * (nDepth + 1));

			chanel_vertex[l] -> sum = new Vector* [max_nVertices];
			chanel_vertex[l] -> hidden = new Vector* [max_nVertices];

			// For computation graph of level 0
			if (l == 0) {
				for (int v = 0; v < max_nVertices; ++v) {
					chanel_vertex[l] -> sum[v] = new MatVecMul(chanel_vertex[l] -> W1, feature[v]);
					chanel_vertex[l] -> hidden[v] = new Softmax(chanel_vertex[l] -> sum[v]);
				}
			} else {
				// For computation graph of level > 0
				chanel_vertex[l] -> W2 = new Matrix(nHiddens, nHiddens);
				chanel_vertex[l] -> prev_W2 = new Matrix(nHiddens, nHiddens);
				chanel_vertex[l] -> sum_gradient_W2 = new Matrix(nHiddens, nHiddens);

				chanel_vertex[l] -> neighbor = new SumVectors* [max_nVertices];
				chanel_vertex[l] -> part1 = new MatVecMul* [max_nVertices];
				chanel_vertex[l] -> part2 = new MatVecMul* [max_nVertices];

				for (int v = 0; v < max_nVertices; ++v) {
					chanel_vertex[l] -> part1[v] = new MatVecMul(chanel_vertex[l] -> W1, feature[v]);
					chanel_vertex[l] -> neighbor[v] = new SumVectors(nHiddens);
					chanel_vertex[l] -> part2[v] = new MatVecMul(chanel_vertex[l] -> W2, chanel_vertex[l] -> neighbor[v]);
					chanel_vertex[l] -> sum[v] = new Add(chanel_vertex[l] -> part1[v], chanel_vertex[l] -> part2[v]);
					chanel_vertex[l] -> hidden[v] = new Softmax(chanel_vertex[l] -> sum[v]);
				}
			}
		}

		final_vertex = new SumVectors(nHiddens);

		// +---------------------+
		// | Chanel for distance |
		// +---------------------+

		// Distance matrix
		distance = new Vector* [max_nVertices];
		distance_sorted = new Sort* [max_nVertices]; 
		
		for (int v = 0; v < max_nVertices; ++v) {
			distance[v] = new Vector(max_nVertices);
			distance_sorted[v] = new Sort(distance[v]);
		}

		// Computation graph for each level
		chanel_distance = new Level* [nLevels + 1];
		for (int l = 0; l <= nLevels; ++l) {
			chanel_distance[l] = new Level();
			
			// Learning parameters
			chanel_distance[l] -> W1 = new Matrix(nHiddens, max_nVertices);
			chanel_distance[l] -> prev_W1 = new Matrix(nHiddens, max_nVertices);
			chanel_distance[l] -> sum_gradient_W1 = new Matrix(nHiddens, max_nVertices);

			chanel_distance[l] -> sum = new Vector* [max_nVertices];
			chanel_distance[l] -> hidden = new Vector* [max_nVertices];

			// For computation graph of level 0
			if (l == 0) {
				for (int v = 0; v < max_nVertices; ++v) {
					chanel_distance[l] -> sum[v] = new MatVecMul(chanel_distance[l] -> W1, distance_sorted[v]);
					chanel_distance[l] -> hidden[v] = new Softmax(chanel_distance[l] -> sum[v]);
				}
			} else {
				// For computation graph of level > 0
				chanel_distance[l] -> W2 = new Matrix(nHiddens, nHiddens);
				chanel_distance[l] -> prev_W2 = new Matrix(nHiddens, nHiddens);
				chanel_distance[l] -> sum_gradient_W2 = new Matrix(nHiddens, nHiddens);

				chanel_distance[l] -> neighbor = new SumVectors* [max_nVertices];
				chanel_distance[l] -> part1 = new MatVecMul* [max_nVertices];
				chanel_distance[l] -> part2 = new MatVecMul* [max_nVertices];

				for (int v = 0; v < max_nVertices; ++v) {
					chanel_distance[l] -> part1[v] = new MatVecMul(chanel_distance[l] -> W1, distance_sorted[v]);
					chanel_distance[l] -> neighbor[v] = new SumVectors(nHiddens);
					chanel_distance[l] -> part2[v] = new MatVecMul(chanel_distance[l] -> W2, chanel_distance[l] -> neighbor[v]);
					chanel_distance[l] -> sum[v] = new Add(chanel_distance[l] -> part1[v], chanel_distance[l] -> part2[v]);
					chanel_distance[l] -> hidden[v] = new Softmax(chanel_distance[l] -> sum[v]);
				}
			}
		}

		final_distance = new SumVectors(nHiddens);

		// +----------------------+
		// | Fully-Connect Layers |
		// +----------------------+

		// Computation graph for the top level as a linear regression
		W = new Vector(2 * nHiddens);
		prev_W = new Vector(2 * nHiddens);
		sum_gradient_W = new Vector(2 * nHiddens);

		final = new ConCat(final_vertex, final_distance);
		predict = new InnerProduct(W, final);
		target = new Vector(1);
		sql = new SquaredLoss(predict, target);

		// Stochastic Gradient Descent
		sgd = new Momentum(momentum_param);
		for (int l = 0; l <= nLevels; ++l) {
			sgd -> add(chanel_vertex[l] -> W1);
			sgd -> add(chanel_distance[l] -> W1);

			if (l > 0) {
				sgd -> add(chanel_vertex[l] -> W2);
				sgd -> add(chanel_distance[l] -> W2);
			}
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
			graph -> uniform_init(chanel_vertex[l] -> W1);
			graph -> uniform_init(chanel_distance[l] -> W1);

			if (l > 0) {
				graph -> uniform_init(chanel_vertex[l] -> W2);
				graph -> uniform_init(chanel_distance[l] -> W2);
			}
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
			copy(chanel_vertex[l] -> W1, chanel_vertex[l] -> prev_W1);
			copy(chanel_distance[l] -> W1, chanel_distance[l] -> prev_W1);

			if (l > 0) {
				copy(chanel_vertex[l] -> W2, chanel_vertex[l] -> prev_W2);
				copy(chanel_distance[l] -> W2, chanel_distance[l] -> prev_W2);
			}
		}
		copy(W, prev_W);
	}

	void restore_parameters() {
		for (int l = 0; l <= nLevels; ++l) {
			copy(chanel_vertex[l] -> prev_W1, chanel_vertex[l] -> W1);
			copy(chanel_distance[l] -> prev_W1, chanel_distance[l] -> W1);

			if (l > 0) {
				copy(chanel_vertex[l] -> prev_W2, chanel_vertex[l] -> W2);
				copy(chanel_distance[l] -> prev_W2, chanel_distance[l] -> W2);
			}
		}
		copy(prev_W, W);
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

		// Distance
		for (int v = 0; v < molecule -> nVertices; ++v) {
			for (int u = 0; u < max_nVertices; ++u) {
				distance[v] -> value[u] = 0.0;
			}
			for (int u = 0; u < molecule -> nVertices; ++u) {
				distance[v] -> value[u] = molecule -> distance[u][v];
			}
		}

		// +--------------------------------------------------+
		// | Complete the computation graph for chanel vertex |
		// +--------------------------------------------------+

		for (int l = 0; l <= nLevels; ++l) {
			graph -> add(chanel_vertex[l] -> W1, MATRIX);

			if (l == 0) {
				for (int v = 0; v < molecule -> nVertices; ++v) {
					graph -> add(chanel_vertex[l] -> sum[v], MATVECMUL);
					graph -> add(chanel_vertex[l] -> hidden[v], SOFTMAX);
				}
			} else {
				graph -> add(chanel_vertex[l] -> W2, MATRIX);

				for (int v = 0; v < molecule -> nVertices; ++v) {
					graph -> add(chanel_vertex[l] -> part1[v], MATVECMUL);

					chanel_vertex[l] -> neighbor[v] -> clear();
					for (int u = 0; u < molecule -> nVertices; ++u) {
						if (shortest_paths[v][u] <= min(l, max_Radius)) {
							chanel_vertex[l] -> neighbor[v] -> add_vector(chanel_vertex[l - 1] -> hidden[u]);
						}
					}

					graph -> add(chanel_vertex[l] -> neighbor[v], SUMVECTORS);
					graph -> add(chanel_vertex[l] -> part2[v], MATVECMUL);
					graph -> add(chanel_vertex[l] -> sum[v], ADD);
					graph -> add(chanel_vertex[l] -> hidden[v], SOFTMAX);
				}
			}
		}

		final_vertex -> clear();
		for (int v = 0; v < molecule -> nVertices; ++v) {
			final_vertex -> add_vector(chanel_vertex[nLevels] -> hidden[v]);
		}
		graph -> add(final_vertex, SUMVECTORS);
		
		// +----------------------------------------------------+
		// | Complete the computation graph for chanel distance |
		// +----------------------------------------------------+

		for (int v = 0; v < molecule -> nVertices; ++v) {
			graph -> add(distance_sorted[v], SORT);
		}

		for (int l = 0; l <= nLevels; ++l) {
			graph -> add(chanel_distance[l] -> W1, MATRIX);

			if (l == 0) {
				for (int v = 0; v < molecule -> nVertices; ++v) {
					graph -> add(chanel_distance[l] -> sum[v], MATVECMUL);
					graph -> add(chanel_distance[l] -> hidden[v], SOFTMAX);
				}
			} else {
				graph -> add(chanel_distance[l] -> W2, MATRIX);

				for (int v = 0; v < molecule -> nVertices; ++v) {
					graph -> add(chanel_distance[l] -> part1[v], MATVECMUL);

					chanel_distance[l] -> neighbor[v] -> clear();
					for (int u = 0; u < molecule -> nVertices; ++u) {
						if (shortest_paths[v][u] <= min(l, max_Radius)) {
							chanel_distance[l] -> neighbor[v] -> add_vector(chanel_distance[l - 1] -> hidden[u]);
						}
					}

					graph -> add(chanel_distance[l] -> neighbor[v], SUMVECTORS);
					graph -> add(chanel_distance[l] -> part2[v], MATVECMUL);
					graph -> add(chanel_distance[l] -> sum[v], ADD);
					graph -> add(chanel_distance[l] -> hidden[v], SOFTMAX);
				}
			}
		}

		final_distance -> clear();
		for (int v = 0; v < molecule -> nVertices; ++v) {
			final_distance -> add_vector(chanel_distance[nLevels] -> hidden[v]);
		}
		graph -> add(final_distance, SUMVECTORS);

		// Top layer
		graph -> add(final, CONCAT);
		graph -> add(W, VECTOR);
		graph -> add(predict, INNERPRODUCT);
		graph -> add(sql, SQUAREDLOSS);
	}

	void init_sum_gradients() {
		for (int l = 0; l <= nLevels; ++l) {
			for (int i = 0; i < chanel_vertex[l] -> W1 -> size; ++i) {
				chanel_vertex[l] -> sum_gradient_W1 -> gradient[i] = 0.0;
			}

			if (l > 0) {
				for (int i = 0; i < chanel_vertex[l] -> W2 -> size; ++i) {
					chanel_vertex[l] -> sum_gradient_W2 -> gradient[i] = 0.0;
				}
			}
		}

		for (int l = 0; l <= nLevels; ++l) {
			for (int i = 0; i < chanel_distance[l] -> W1 -> size; ++i) {
				chanel_distance[l] -> sum_gradient_W1 -> gradient[i] = 0.0;
			}

			if (l > 0) {
				for (int i = 0; i < chanel_distance[l] -> W2 -> size; ++i) {
					chanel_distance[l] -> sum_gradient_W2 -> gradient[i] = 0.0;
				}
			}
		}

		for (int i = 0; i < W -> size; ++i) {
			sum_gradient_W -> gradient[i] = 0.0;
		}
	}

	void update_sum_gradients() {
		for (int l = 0; l <= nLevels; ++l) {
			for (int i = 0; i < chanel_vertex[l] -> W1 -> size; ++i) {
				chanel_vertex[l] -> sum_gradient_W1 -> gradient[i] += chanel_vertex[l] -> W1 -> gradient[i];
			}

			if (l > 0) {
				for (int i = 0; i < chanel_vertex[l] -> W2 -> size; ++i) {
					chanel_vertex[l] -> sum_gradient_W2 -> gradient[i] += chanel_vertex[l] -> W2 -> gradient[i];
				}
			}
		}

		for (int l = 0; l <= nLevels; ++l) {
			for (int i = 0; i < chanel_distance[l] -> W1 -> size; ++i) {
				chanel_distance[l] -> sum_gradient_W1 -> gradient[i] += chanel_distance[l] -> W1 -> gradient[i];
			}

			if (l > 0) {
				for (int i = 0; i < chanel_distance[l] -> W2 -> size; ++i) {
					chanel_distance[l] -> sum_gradient_W2 -> gradient[i] += chanel_distance[l] -> W2 -> gradient[i];
				}
			}
		}

		for (int i = 0; i < W -> size; ++i) {
			sum_gradient_W -> gradient[i] += W -> gradient[i];
		}
	}

	void get_sum_gradients() {
		for (int l = 0; l <= nLevels; ++l) {
			for (int i = 0; i < chanel_vertex[l] -> W1 -> size; ++i) {
				chanel_vertex[l] -> W1 -> gradient[i] = chanel_vertex[l] -> sum_gradient_W1 -> gradient[i];
			}

			if (l > 0) {
				for (int i = 0; i < chanel_vertex[l] -> W2 -> size; ++i) {
					chanel_vertex[l] -> W2 -> gradient[i] = chanel_vertex[l] -> sum_gradient_W2 -> gradient[i];
				}
			}
		}

		for (int l = 0; l <= nLevels; ++l) {
			for (int i = 0; i < chanel_distance[l] -> W1 -> size; ++i) {
				chanel_distance[l] -> W1 -> gradient[i] = chanel_distance[l] -> sum_gradient_W1 -> gradient[i];
			}

			if (l > 0) {
				for (int i = 0; i < chanel_distance[l] -> W2 -> size; ++i) {
					chanel_distance[l] -> W2 -> gradient[i] = chanel_distance[l] -> sum_gradient_W2 -> gradient[i];
				}
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
		for (int i = 0; i < final -> size; ++i) {
			vect.push_back(final -> value[i]);
		}
		return vect;
	}

	void save_model(string filename) {
		ofstream file(filename.c_str(), ios::out);
		
		for (int l = 0; l <= nLevels; ++l) {
			for (int i = 0; i < chanel_vertex[l] -> W1 -> size; ++i) {
				file << chanel_vertex[l] -> W1 -> value[i] << " ";
			}
			file << endl;

			if (l > 0) {
				for (int i = 0; i < chanel_vertex[l] -> W2 -> size; ++i) {
					file << chanel_vertex[l] -> W2 -> value[i] << " ";
				}
				file << endl;
			}
		}

		for (int l = 0; l <= nLevels; ++l) {
			for (int i = 0; i < chanel_distance[l] -> W1 -> size; ++i) {
				file << chanel_distance[l] -> W1 -> value[i] << " ";
			}
			file << endl;

			if (l > 0) {
				for (int i = 0; i < chanel_distance[l] -> W2 -> size; ++i) {
					file << chanel_distance[l] -> W2 -> value[i] << " ";
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
			for (int i = 0; i < chanel_vertex[l] -> W1 -> size; ++i) {
				file >> chanel_vertex[l] -> W1 -> value[i];
			}

			if (l > 0) {
				for (int i = 0; i < chanel_vertex[l] -> W2 -> size; ++i) {
					file >> chanel_vertex[l] -> W2 -> value[i];
				}
			}
		}

		for (int l = 0; l <= nLevels; ++l) {
			for (int i = 0; i < chanel_distance[l] -> W1 -> size; ++i) {
				file >> chanel_distance[l] -> W1 -> value[i];
			}

			if (l > 0) {
				for (int i = 0; i < chanel_distance[l] -> W2 -> size; ++i) {
					file >> chanel_distance[l] -> W2 -> value[i];
				}
			}
		}

		for (int i = 0; i < W -> size; ++i) {
			file >> W -> value[i];
		}

		file.close();
	}

	struct Level {
		Vector **hidden;
		Vector **sum;

		SumVectors **neighbor;
		MatVecMul **part1;
		MatVecMul **part2;
		
		Matrix *W1;
		Matrix *W2;

		Matrix *prev_W1;
		Matrix *prev_W2;

		Matrix *sum_gradient_W1;
		Matrix *sum_gradient_W2;
	};

	// Chanel 1 for vertex features
	Level **chanel_vertex;

	// Chanel 2 for distance 
	Level **chanel_distance;

	Vector **feature;
	Vector **distance;
	Sort **distance_sorted;

	GraphFlow *graph;

	// Top layer as a linear regression
	Vector *W;
	Vector *prev_W;
	Vector *sum_gradient_W;

	SumVectors *final_vertex;
	SumVectors *final_distance;

	ConCat *final;
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
	int max_Radius;
	float momentum_param;

	// Floyd-Warshall algorithm
	int **shortest_paths;

	~GCN_1D_Distance() {
		for (int l = 0; l <= nLevels; ++l) {
			delete[] chanel_vertex[l] -> hidden;
			delete[] chanel_vertex[l] -> sum;
			delete[] chanel_vertex[l] -> neighbor;
			delete[] chanel_vertex[l] -> part1;
			delete[] chanel_vertex[l] -> part2;
			delete chanel_vertex[l] -> W1;
			delete chanel_vertex[l] -> W2;
			delete chanel_vertex[l] -> prev_W1;
			delete chanel_vertex[l] -> prev_W2;
			delete chanel_vertex[l] -> sum_gradient_W1;
			delete chanel_vertex[l] -> sum_gradient_W2;

			delete[] chanel_distance[l] -> hidden;
			delete[] chanel_distance[l] -> sum;
			delete[] chanel_distance[l] -> neighbor;
			delete[] chanel_distance[l] -> part1;
			delete[] chanel_distance[l] -> part2;
			delete chanel_distance[l] -> W1;
			delete chanel_distance[l] -> W2;
			delete chanel_distance[l] -> prev_W1;
			delete chanel_distance[l] -> prev_W2;
			delete chanel_distance[l] -> sum_gradient_W1;
			delete chanel_distance[l] -> sum_gradient_W2;
		}
		delete[] chanel_vertex;
		delete[] chanel_distance;
		delete[] feature;
		delete[] distance;
		delete[] distance_sorted;
		delete final_vertex;
		delete final_distance;
		delete W;
		delete prev_W;
		delete sum_gradient_W;
		// delete final;
		delete predict;
		delete target;
		delete sql;
		delete[] shortest_paths;
	}
};

#endif
