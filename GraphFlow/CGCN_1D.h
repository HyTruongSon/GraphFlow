// Framework: GraphFlow
// Class: Covariant Graph Convolution Networks (First-order representation)
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __CGCN_1D_H_INCLUDED__
#define __CGCN_1D_H_INCLUDED__

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

class CGCN_1D {
public:
	CGCN_1D(int nLevels, int max_nVertices, int nFeatures, int nDepth, double momentum_param) {
		this -> nLevels = nLevels;
		this -> max_nVertices = max_nVertices;
		this -> nFeatures = nFeatures;
		this -> nDepth = nDepth;
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

		// Learnable hash function
		H = new Vector(nFeatures * (nDepth + 1));
		prev_H = new Vector(nFeatures * (nDepth + 1));
		sum_gradient_H = new Vector(nFeatures * (nDepth + 1));

		// Computation graph for each level
		level = new Level* [nLevels + 1];
		for (int l = 0; l <= nLevels; ++l) {
			level[l] = new Level();

			level[l] -> representation = new Vector* [max_nVertices];

			// Initialize the 0/1 mask for the receptive field
			level[l] -> receptive_field = new Vector* [max_nVertices];
			for (int v = 0; v < max_nVertices; ++v) {
				level[l] -> receptive_field[v] = new Vector(max_nVertices);
			}

			// For computation graph of level 0
			if (l == 0) {
				for (int v = 0; v < max_nVertices; ++v) {
					level[l] -> representation[v] = new VertexRepresentation(max_nVertices, feature[v], H, v);
				}
			} else {
				// Learning parameters
				level[l] -> F = new Matrix(max_nVertices, max_nVertices);
				level[l] -> prev_F = new Matrix(max_nVertices, max_nVertices);
				level[l] -> sum_gradient_F = new Matrix(max_nVertices, max_nVertices);

				// For computation graph of level > 0
				level[l] -> neighbor = new RisiLayer1D* [max_nVertices];
				level[l] -> linear = new MatVecMul* [max_nVertices];
				level[l] -> masking = new Masking* [max_nVertices];

				for (int v = 0; v < max_nVertices; ++v) {
					level[l] -> neighbor[v] = new RisiLayer1D(max_nVertices);
					level[l] -> linear[v] = new MatVecMul(level[l] -> F, level[l] -> neighbor[v]);
					level[l] -> masking[v] = new Masking(level[l] -> linear[v], level[l] -> receptive_field[v]);
					level[l] -> representation[v] = new LeakyReLU(level[l] -> masking[v]);
				}
			}
		}

		// Computation graph for the top level as a linear regression
		sum_representations = new SumVectors(max_nVertices);
		predict = new SumComponents(sum_representations);
		target = new Vector(1);
		sql = new SquaredLoss(predict, target);

		// Stochastic Gradient Descent
		sgd = new Momentum(momentum_param);
		sgd -> add(H);
		for (int l = 1; l <= nLevels; ++l) {
			sgd -> add(level[l] -> F);
		}

		// Shortest Paths by Floyd-Warshall algorithm
		shortest_paths = new int* [max_nVertices];
		for (int i = 0; i < max_nVertices; ++i) {
			shortest_paths[i] = new int [max_nVertices];
		}
	}

	void weights_initialization() {
		graph -> uniform_init(H);

		for (int l = 1; l <= nLevels; ++l) {
			graph -> uniform_init(level[l] -> F);
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
		copy(H, prev_H);
		
		for (int l = 1; l <= nLevels; ++l) {
			copy(level[l] -> F, level[l] -> prev_F);
		}
	}

	void restore_parameters() {
		copy(prev_H, H);

		for (int l = 1; l <= nLevels; ++l) {
			copy(level[l] -> prev_F, level[l] -> F);
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

		// Initialize the 0/1 mask for the receptive field
		for (int l = 0; l <= nLevels; ++l) {
			for (int v = 0; v < max_nVertices; ++v) {
				for (int u = 0; u < max_nVertices; ++u) {
					level[l] -> receptive_field[v] -> value[u] = 0.0;
				}
			}

			for (int v = 0; v < molecule -> nVertices; ++v) {
				for (int u = 0; u < molecule -> nVertices; ++u) {
					if (shortest_paths[u][v] <= l) {
						level[l] -> receptive_field[v] -> value[u] = 1.0;
					}
				}
			}
		}

		// Complete the computation graph
		graph -> add(H, VECTOR);

		for (int l = 0; l <= nLevels; ++l) {
			if (l == 0) {
				for (int v = 0; v < molecule -> nVertices; ++v) {
					graph -> add(level[l] -> representation[v], VERTEXREPRESENTATION);
				}
			} else {
				graph -> add(level[l] -> F, MATRIX);

				for (int v = 0; v < molecule -> nVertices; ++v) {
					level[l] -> neighbor[v] -> clear();
					for (int u = 0; u < molecule -> nVertices; ++u) {
						if (molecule -> adj[u][v] > 0) {
							level[l] -> neighbor[v] -> add_vector(level[l - 1] -> representation[u]);
						}
					}

					graph -> add(level[l] -> neighbor[v], RISILAYER1D);
					graph -> add(level[l] -> linear[v], MATVECMUL);
					graph -> add(level[l] -> masking[v], MASKING);
					graph -> add(level[l] -> representation[v], LEAKYRELU);
				}
			}
		}

		// Computation graph for the top layer as a scalar
		sum_representations -> clear();
		for (int v = 0; v < molecule -> nVertices; ++v) {
			sum_representations -> add_vector(level[nLevels] -> representation[v]);
		}

		graph -> add(sum_representations, SUMVECTORS);
		graph -> add(predict, SUMCOMPONENTS);
		graph -> add(sql, SQUAREDLOSS);
	}

	void init_sum_gradients() {
		for (int i = 0; i < H -> size; ++i) {
			sum_gradient_H -> gradient[i] = 0.0;
		}

		for (int l = 1; l <= nLevels; ++l) {
			for (int i = 0; i < level[l] -> F -> size; ++i) {
				level[l] -> sum_gradient_F -> gradient[i] = 0.0;
			}
		}
	}

	void update_sum_gradients() {
		for (int i = 0; i < H -> size; ++i) {
			sum_gradient_H -> gradient[i] += H -> gradient[i];
		}

		for (int l = 1; l <= nLevels; ++l) {
			for (int i = 0; i < level[l] -> F -> size; ++i) {
				level[l] -> sum_gradient_F -> gradient[i] += level[l] -> F -> gradient[i];
			}
		}
	}

	void get_sum_gradients() {
		for (int i = 0; i < H -> size; ++i) {
			H -> gradient[i] = sum_gradient_H -> gradient[i];
		}

		for (int l = 1; l <= nLevels; ++l) {
			for (int i = 0; i < level[l] -> F -> size; ++i) {
				level[l] -> F -> gradient[i] = level[l] -> sum_gradient_F -> gradient[i];
			}
		}
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

	double Predict(DenseGraph *molecule) {
		complete_computation_graph(molecule);

		graph -> forward();

		return predict -> value[0];
	}

	void save_model(string filename) {
		ofstream file(filename.c_str(), ios::out);
		
		for (int i = 0; i < H -> size; ++i) {
			file << H -> value[i] << " ";
		}

		for (int l = 1; l <= nLevels; ++l) {
			for (int i = 0; i < level[l] -> F -> size; ++i) {
				file << level[l] -> F -> value[i] << " ";
			}
			file << endl;
		}

		file.close();
	}

	void load_model(string filename) {
		ifstream file(filename.c_str(), ios::in);
		
		for (int i = 0; i < H -> size; ++i) {
			file >> H -> value[i];
		}

		for (int l = 1; l <= nLevels; ++l) {
			for (int i = 0; i < level[l] -> F -> size; ++i) {
				file >> level[l] -> F -> value[i];
			}
		}

		file.close();
	}

	Vector *H;
	Vector *prev_H;
	Vector *sum_gradient_H;

	struct Level {
		Vector **representation;
		Vector **receptive_field;

		RisiLayer1D **neighbor;
		MatVecMul **linear;
		Masking **masking;
		
		Matrix *F;
		Matrix *prev_F;
		Matrix *sum_gradient_F;
	};

	Level **level;
	Vector **feature;
	GraphFlow *graph;

	SumVectors *sum_representations;
	SumComponents *predict;
	Vector *target;
	SquaredLoss *sql;

	// Stochastic Gradient Descent
	Momentum *sgd;

	int nLevels;
	int max_nVertices;
	int nFeatures;
	int nDepth;
	double momentum_param;

	// Floyd-Warshall algorithm
	int **shortest_paths;

	~CGCN_1D() {
		for (int l = 0; l <= nLevels; ++l) {
			delete[] level[l] -> representation;
			delete[] level[l] -> neighbor;
			delete[] level[l] -> linear;
			delete level[l] -> F;
			delete level[l] -> prev_F;
			delete level[l] -> sum_gradient_F;
		}
		delete[] level;
		delete[] feature;
		delete H;
		delete prev_H;
		delete sum_gradient_H;
		// delete sum_representations;
		delete predict;
		delete target;
		delete sql;
		delete[] shortest_paths;
	}
};

#endif