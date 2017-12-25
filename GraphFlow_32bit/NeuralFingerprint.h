// Framework: GraphFlow
// Class: NeuralFingerprint
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __NEURALFINGERPRINT_H_INCLUDED__
#define __NEURALFINGERPRINT_H_INCLUDED__

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

class NeuralFingerprint {
public:
	NeuralFingerprint(int nLevels, int max_nVertices, int nFeatures, int nHiddens, float momentum_param) {
		this -> nLevels = nLevels;
		this -> max_nVertices = max_nVertices;
		this -> nFeatures = nFeatures;
		this -> nHiddens = nHiddens;
		this -> momentum_param = momentum_param;

		computation_graph();
		weights_initialization();
	}

	void computation_graph() {
		graph = new GraphFlow();

		// Feature vector for each vertex
		feature = new Vector* [max_nVertices];
		for (int v = 0; v < max_nVertices; ++v) {
			feature[v] = new Vector(nFeatures);
		}

		// Computation graph for each level
		level = new Level* [nLevels + 1];
		for (int l = 0; l <= nLevels; ++l) {
			level[l] = new Level();
			
			// Learning parameters
			level[l] -> W1 = new Matrix(nHiddens, nFeatures);
			level[l] -> prev_W1 = new Matrix(nHiddens, nFeatures);
			level[l] -> sum_gradient_W1 = new Matrix(nHiddens, nFeatures);

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

				level[l] -> neighbor = new SumVectors* [max_nVertices];
				level[l] -> part1 = new MatVecMul* [max_nVertices];
				level[l] -> part2 = new MatVecMul* [max_nVertices];

				for (int v = 0; v < max_nVertices; ++v) {
					level[l] -> part1[v] = new MatVecMul(level[l] -> W1, feature[v]);
					level[l] -> neighbor[v] = new SumVectors(nHiddens);
					level[l] -> part2[v] = new MatVecMul(level[l] -> W2, level[l] -> neighbor[v]);
					level[l] -> sum[v] = new Add(level[l] -> part1[v], level[l] -> part2[v]);
					level[l] -> hidden[v] = new Softmax(level[l] -> sum[v]);
				}
			}
		}

		// Computation graph for the top level as a linear regression
		W = new Vector(nHiddens);
		prev_W = new Vector(nHiddens);
		sum_gradient_W = new Vector(nHiddens);

		final_feature = new SumVectors(nHiddens);
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

		copy(W, prev_W);
	}

	void restore_parameters() {
		for (int l = 0; l <= nLevels; ++l) {
			copy(level[l] -> prev_W1, level[l] -> W1);

			if (l > 0) {
				copy(level[l] -> prev_W2, level[l] -> W2);
			}
		}

		copy(prev_W, W);
	}

	void complete_computation_graph(DenseGraph *molecule) {
		assert(molecule -> nFeatures == nFeatures);

		// Build a new computation graph
		graph -> clear();

		// Get the feature vector for each vertex
		for (int v = 0; v < molecule -> nVertices; ++v) {
			for (int f = 0; f < nFeatures; ++f) {
				this -> feature[v] -> value[f] = molecule -> feature[v][f];
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
						if (molecule -> adj[v][u] > 0) {
							level[l] -> neighbor[v] -> add_vector(level[l - 1] -> hidden[u]);
						}
					}

					graph -> add(level[l] -> neighbor[v], SUMVECTORS);
					graph -> add(level[l] -> part2[v], MATVECMUL);
					graph -> add(level[l] -> sum[v], ADD);
					graph -> add(level[l] -> hidden[v], SOFTMAX);
				}
			}
		}

		// Computation graph for the top layer as a linear regression
		final_feature -> clear();
		for (int v = 0; v < molecule -> nVertices; ++v) {
			final_feature -> add_vector(level[nLevels] -> hidden[v]);
		}

		graph -> add(final_feature, SUMVECTORS);
		graph -> add(W, VECTOR);
		graph -> add(predict, INNERPRODUCT);
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

		for (int i = 0; i < W -> size; ++i) {
			sum_gradient_W -> gradient[i] = 0.0;
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

		for (int i = 0; i < W -> size; ++i) {
			sum_gradient_W -> gradient[i] += W -> gradient[i];
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
	
	float Predict(DenseGraph *molecule) {
		complete_computation_graph(molecule);

		graph -> forward();

		return predict -> value[0];
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

	Level **level;
	Vector **feature;
	GraphFlow *graph;

	// Top layer as a linear regression
	Vector *W;
	Vector *prev_W;
	Vector *sum_gradient_W;

	SumVectors *final_feature;
	Vector *predict;
	Vector *target;
	SquaredLoss *sql;

	// Stochastic Gradient Descent
	Momentum *sgd;

	int nLevels;
	int max_nVertices;
	int nFeatures;
	int nHiddens;
	float momentum_param;

	~NeuralFingerprint() {
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
		delete W;
		delete prev_W;
		delete sum_gradient_W;
		delete final_feature;
		delete predict;
		delete target;
		delete sql;
	}
};

#endif
