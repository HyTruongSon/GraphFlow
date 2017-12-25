// Framework: GraphFlow
// Class: First-order Steerable Message Passing (version 3) For Classifcation
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __SMP_1D_VER3_CLASSIFICATION_H_INCLUDED__
#define __SMP_1D_VER3_CLASSIFICATION_H_INCLUDED__

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

class SMP_1D_ver3_classification {
public:
	SMP_1D_ver3_classification(int nClass, int max_nVertices, int nLevels, int nChanels, int nFeatures, int nDepth, float momentum_param) {
		this -> nClass = nClass;
		this -> max_nVertices = max_nVertices;
		this -> nLevels = nLevels;
		this -> nChanels = nChanels;
		this -> nFeatures = nFeatures;
		this -> nDepth = nDepth;
		this -> momentum_param = momentum_param;
		this -> has_WL_ordering = true;

		computation_graph();
		weights_initialization();
	}

	SMP_1D_ver3_classification(int nClass, int max_nVertices, int nLevels, int nChanels, int nFeatures, int nDepth, float momentum_param, bool has_WL_ordering) {
		this -> nClass = nClass;
		this -> max_nVertices = max_nVertices;
		this -> nLevels = nLevels;
		this -> nChanels = nChanels;
		this -> nFeatures = nFeatures;
		this -> nDepth = nDepth;
		this -> momentum_param = momentum_param;
		this -> has_WL_ordering = has_WL_ordering;

		computation_graph();
		weights_initialization();
	}

	void computation_graph() {
		// +--------------------------+
		// | Component initialization |
		// +--------------------------+

		// Synthesized graph features
		feature = new Matrix* [max_nVertices];
		for (int v = 0; v < max_nVertices; ++v) {
			feature[v] = new Matrix(nFeatures * (nDepth + 1), 1);
		}

		// Mapping from the original synthesized graph features into chanels
		H = new Matrix(nChanels, nFeatures * (nDepth + 1));
		
		// Set of identity matrices
		eye = new Matrix* [max_nVertices + 1];
		for (int size = 1; size <= max_nVertices; ++size) {
			eye[size] = new Matrix(size, size);
			for (int i = 0; i < size; ++i) {
				for (int j = 0; j < size; ++j) {
					int index = eye[size] -> index(i, j);
					if (i == j) {
						eye[size] -> value[index] = 1;
					} else {
						eye[size] -> value[index] = 0;
					}
				}
			}
		}

		// Set of one matrices
		one = new Matrix* [max_nVertices + 1];
		for (int size = 1; size <= max_nVertices; ++size) {
			one[size] = new Matrix(size, size);
			for (int i = 0; i < size; ++i) {
				for (int j = 0; j < size; ++j) {
					int index = one[size] -> index(i, j);
					one[size] -> value[index] = 1;
				}
			}
		}

		// Multiple levels
		level = new Level* [nLevels + 1];

		// Each level
		for (int l = 0; l <= nLevels; ++l) {
			// Create the level
			level[l] = new Level();

			// Receptive fields
			level[l] -> phi = new vector<int> [max_nVertices];

			// Level 0
			if (l == 0) {
				// Number of chanels in this level
				level[l] -> nChanels = nChanels;
				int C = level[l] -> nChanels;

				level[l] -> affine = new MatMul* [max_nVertices];
				level[l] -> f_transpose = new Transpose* [max_nVertices];
				level[l] -> f = new LeakyReLU2D* [max_nVertices];

				for (int v = 0; v < max_nVertices; ++v) {
					level[l] -> affine[v] = new MatMul(C, C);
					level[l] -> f_transpose[v] = new Transpose(C, C);
					level[l] -> f[v] = new LeakyReLU2D(C, C);
				}
			} 

			// Level from 1 to nLevels
			if (l > 0) {
				// Double number of chanels in this level
				level[l] -> nChanels = level[l - 1] -> nChanels * 2;

				// Number of chanels in this level
				int C = level[l] -> nChanels;

				// Number of chanels in the previous level
				int prevC = level[l - 1] -> nChanels;

				// Steerable Filters and Bias
				level[l] -> lambda1 = new Vector* [max_nVertices + 1];
				level[l] -> lambda2 = new Vector* [max_nVertices + 1];

				level[l] -> K_eye = new Matrix(prevC, prevC);
				level[l] -> K_one = new Matrix(prevC, prevC);

				level[l] -> W_eye = new ScalarMatMul* [max_nVertices + 1];
				level[l] -> W_one = new ScalarMatMul* [max_nVertices + 1]; 

				level[l] -> b = new Vector* [max_nVertices + 1];

				for (int size = 1; size <= max_nVertices; ++size) {
					level[l] -> lambda1[size] = new Vector(1);
					level[l] -> lambda2[size] = new Vector(1);

					level[l] -> W_eye[size] = new ScalarMatMul(level[l] -> lambda1[size], eye[size]);
					level[l] -> W_one[size] = new ScalarMatMul(level[l] -> lambda2[size], one[size]);

					level[l] -> b[size] = new Vector(C);
				}

				// For computations
				level[l] -> affine = new MatMul* [max_nVertices];
				level[l] -> affine2 = new MatMul* [max_nVertices];
				level[l] -> filtered = new MatMul* [max_nVertices];
				level[l] -> filtered2 = new MatMul* [max_nVertices];
				level[l] -> matcon = new MatrixConcat* [max_nVertices];
				level[l] -> sum = new SumMatrices* [max_nVertices];
				level[l] -> add = new VectorAddMatrix* [max_nVertices];
				level[l] -> f = new LeakyReLU2D* [max_nVertices];

				for (int v = 0; v < max_nVertices; ++v) {
					level[l] -> affine[v] = new MatMul(max_nVertices, prevC);
					level[l] -> affine2[v] = new MatMul(max_nVertices, prevC);
					level[l] -> filtered[v] = new MatMul(max_nVertices, prevC);
					level[l] -> filtered2[v] = new MatMul(max_nVertices, prevC);
					level[l] -> matcon[v] = new MatrixConcat(max_nVertices, C);
					level[l] -> sum[v] = new SumMatrices(max_nVertices, prevC);
					level[l] -> add[v] = new VectorAddMatrix(max_nVertices, C);
					level[l] -> f[v] = new LeakyReLU2D(max_nVertices, C);
				}

				// For permutation matrices
				level[l] -> X = new Matrix** [max_nVertices];
				level[l] -> permuted_f = new MatMul** [max_nVertices];

				for (int v = 0; v < max_nVertices; ++v) {
					level[l] -> X[v] = new Matrix* [max_nVertices];
					level[l] -> permuted_f[v] = new MatMul* [max_nVertices];

					for (int w = 0; w < max_nVertices; ++w) {
						level[l] -> X[v][w] = new Matrix(max_nVertices, max_nVertices);
						level[l] -> permuted_f[v][w] = new MatMul(max_nVertices, prevC);
					}
				}
			}
		}

		// Number of chanels on the top level
		int C = level[nLevels] -> nChanels;

		// On top of everything
		shrinked = new ShrinkMatrix* [max_nVertices];
		vertex_feature = new LeakyReLU* [max_nVertices];

		for (int v = 0; v < max_nVertices; ++v) {
			shrinked[v] = new ShrinkMatrix(C);
			vertex_feature[v] = new LeakyReLU(C);
		}

		graph_feature = new SumVectors(C);

		// Dense Layer
		W = new Matrix(nClass, C);

		// Predictions
		predict = new MatVecMul(nClass);

		// Target
		target = new Vector(1);

		// Loss function
		logl = new LogLoss(predict, target);

		// +-------------------+
		// | Computation graph |
		// +-------------------+
		
		graph = new GraphFlow();

		// +-----------------------------+
		// | Stochastic Gradient Descent |
		// +-----------------------------+

		sgd = new Momentum(momentum_param);
		sgd -> add(H);
		for (int l = 1; l <= nLevels; ++l) {
			for (int size = 1; size <= max_nVertices; ++size) {
				sgd -> add(level[l] -> lambda1[size]);
				sgd -> add(level[l] -> lambda2[size]);
				sgd -> add(level[l] -> b[size]);
			}
			sgd -> add(level[l] -> K_eye);
			sgd -> add(level[l] -> K_one);
		}
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
		adj = new int* [max_nVertices];
		for (int i = 0; i < max_nVertices; ++i) {
			adj[i] = new int [max_nVertices];
		}

		// Shortest Paths by Floyd-Warshall algorithm
		shortest_paths = new int* [max_nVertices];
		for (int i = 0; i < max_nVertices; ++i) {
			shortest_paths[i] = new int [max_nVertices];
		}

		// Histogram WL
		histogram = new float* [max_nVertices];
		for (int i = 0; i < max_nVertices; ++i) {
			histogram[i] = new float [max_nVertices * (nDepth + 1)];
		}

		// Order of vertices
		order = new int [max_nVertices];

		// Rank of vertices
		rank = new int [max_nVertices];
	}

	void weights_initialization() {
		for (int i = 0; i < sgd -> params.size(); ++i) {
			graph -> uniform_init(sgd -> params[i]);
		}
	}	

	void update_adjacency(DenseGraph *molecule) {
		// Get the original graph
		for (int i = 0; i < molecule -> nVertices; ++i) {
			for (int j = 0; j < molecule -> nVertices; ++j) {
				adj[i][j] = molecule -> adj[i][j];
			}
		}
	}

	void update_feature(DenseGraph *molecule) {
		// Get the original graph feature
		for (int v = 0; v < molecule -> nVertices; ++v) {
			for (int f = 0; f < nFeatures; ++f) {
				feature[v] -> value[feature[v] -> index(f, 0)] = molecule -> feature[v][f];
			}
		}
	}

	void floyd_warshall(DenseGraph *molecule) {
		for (int i = 0; i < molecule -> nVertices; ++i) {
			for (int j = 0; j < molecule -> nVertices; ++j) {
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

		for (int k = 0; k < molecule -> nVertices; ++k) {
			for (int i = 0; i < molecule -> nVertices; ++i) {
				for (int j = 0; j < molecule -> nVertices; ++j) {
					shortest_paths[i][j] = min(shortest_paths[i][j], shortest_paths[i][k] + shortest_paths[k][j]);
				}
			}
		}
	}

	void weisfeiler_lehman(DenseGraph *molecule) {
		for (int v = 0; v < molecule -> nVertices; ++v) {
			for (int f = 0; f < nFeatures * (nDepth + 1); ++f) {
				histogram[v][f] = 0.0;
			}

			for (int d = 0; d <= nDepth; ++d) {
				for (int u = 0; u < molecule -> nVertices; ++u) {
					if (shortest_paths[u][v] == d) {
						for (int f = 0; f < nFeatures; ++f) {
							histogram[v][d * nFeatures + f] += feature[u] -> value[feature[u] -> index(f, 0)];
						}
					}
				}
			}
		}

		for (int v = 0; v < molecule -> nVertices; ++v) {
			for (int f = 0; f < nFeatures * (nDepth + 1); ++f) {
				feature[v] -> value[feature[v] -> index(f, 0)] = histogram[v][f];
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

	void rank_vertices(DenseGraph *molecule) {
		for (int v = 0; v < molecule -> nVertices; ++v) {
			order[v] = v;
		}

		for (int i = 0; i < molecule -> nVertices; ++i) {
			for (int j = i + 1; j < molecule -> nVertices; ++j) {
				if (compare_vertices(order[i], order[j]) < 0) {
					swap(order[i], order[j]);
				}
			}
		}

		for (int i = 0; i < molecule -> nVertices; ++i) {
			rank[order[i]] = i;
		}
	}

	void union_set(vector<int> &A, vector<int> &B) {
		for (int i = 0; i < B.size(); ++i) {
			bool found = false;
			for (int j = 0; j < A.size(); ++j) {
				if (A[j] == B[i]) {
					found = true;
					break;
				}
			}
			if (!found) {
				A.push_back(B[i]);
			}
		}
	}

	void sort(vector<int> &A) {
		for (int i = 0; i < A.size(); ++i) {
			for (int j = i + 1; j < A.size(); ++j) {
				if (rank[A[i]] > rank[A[j]]) {
					swap(A[i], A[j]);
				}
			}
		}
	}

	void init_permutation_matrix(Matrix *X, vector<int> &phi1, vector<int> &phi2) {
		assert(X -> nRows == phi1.size());
		assert(X -> nColumns == phi2.size());

		for (int i = 0; i < phi1.size(); ++i) {
			for (int j = 0; j < phi2.size(); ++j) {
				int index = X -> index(i, j);
				X -> value[index] = 0;
				if (phi1[i] == phi2[j]) {
					X -> value[index] = 1;
				}
			}
		}
	} 

	void init_receptive_field_permutation_matrix(DenseGraph *molecule) {
		// Constructing the receptive fields
		for (int l = 0; l <= nLevels; ++l) {
			if (l == 0) {
				for (int v = 0; v < molecule -> nVertices; ++v) {
					level[l] -> phi[v].clear();
					level[l] -> phi[v].push_back(v);
				}
			} else {
				for (int v = 0; v < molecule -> nVertices; ++v) {
					level[l] -> phi[v].clear();

					for (int u = 0; u < molecule -> nVertices; ++u) {
						if (shortest_paths[u][v] <= 1) {
							union_set(level[l] -> phi[v], level[l - 1] -> phi[u]);
						}
					}

					// Weisfeiler-Lehman ordering
					if (has_WL_ordering) {
						sort(level[l] -> phi[v]);
					}
				}
			}
		}
 
		// Constructing the permutation matrices
		for (int l = 1; l <= nLevels; ++l) {
			for (int v = 0; v < molecule -> nVertices; ++v) {
				for (int w = 0; w < molecule -> nVertices; ++w) {
					if (shortest_paths[v][w] <= 1) {
						level[l] -> X[v][w] -> setParameter(level[l] -> phi[v].size(), level[l - 1] -> phi[w].size());
						init_permutation_matrix(level[l] -> X[v][w], level[l] -> phi[v], level[l - 1] -> phi[w]);
					}
				}
			}
		}
	}

	void complete_computation_graph(DenseGraph *molecule) {
		assert(molecule -> nFeatures == nFeatures);
		assert(molecule -> nVertices <= max_nVertices);

		// Update the adjacency matrix
		update_adjacency(molecule);

		// Update the feature (with dummy nodes)
		update_feature(molecule);

		// Finding the shortest-paths by Floyd-Warshall algorithm
		floyd_warshall(molecule);
		
		// Get the feature vector for each vertex
		weisfeiler_lehman(molecule);

		// Find the optimal order of vertices
		rank_vertices(molecule);

		// Initialize the receptive fields and permutation matrices
		init_receptive_field_permutation_matrix(molecule);

		// Constructing the dynamic computation graph
		graph -> clear();
		graph -> add(H, MATRIX);
		for (int l = 1; l <= nLevels; ++l) {
			for (int size = 1; size <= max_nVertices; ++size) {
				graph -> add(level[l] -> lambda1[size], VECTOR);
				graph -> add(level[l] -> lambda2[size], VECTOR);
				graph -> add(level[l] -> b[size], VECTOR);
			}
			graph -> add(level[l] -> K_eye, MATRIX);
			graph -> add(level[l] -> K_one, MATRIX);
		}
		graph -> add(W, VECTOR);

		for (int l = 0; l <= nLevels; ++l) {
			if (l == 0) {
				for (int v = 0; v < molecule -> nVertices; ++v) {
					level[l] -> affine[v] -> setParameter(H, feature[v]);
					graph -> add(level[l] -> affine[v], MATMUL);

					level[l] -> f_transpose[v] -> setParameter(level[l] -> affine[v]);
					graph -> add(level[l] -> f_transpose[v], TRANSPOSE);

					level[l] -> f[v] -> setParameter(level[l] -> f_transpose[v], 0);
					graph -> add(level[l] -> f[v], LEAKYRELU2D);
				}
			}

			if (l > 0) {
				// Number of chanels in the previous level
				int prevC = level[l - 1] -> nChanels;

				for (int v = 0; v < molecule -> nVertices; ++v) {
					level[l] -> sum[v] -> setParameter(level[l] -> phi[v].size(), prevC);
					level[l] -> sum[v] -> clear();

					for (int w = 0; w < molecule -> nVertices; ++w) {
						if (shortest_paths[v][w] <= 1) {
							level[l] -> permuted_f[v][w] -> setParameter(level[l] -> X[v][w], level[l - 1] -> f[w]);
							graph -> add(level[l] -> permuted_f[v][w], MATMUL);

							level[l] -> sum[v] -> add_matrix(level[l] -> permuted_f[v][w]);
						}
					}

					graph -> add(level[l] -> sum[v], SUMMATRICES);

					int size = level[l] -> phi[v].size();

					graph -> add(level[l] -> W_eye[size], SCALARMATMUL);
					graph -> add(level[l] -> W_one[size], SCALARMATMUL);

					// Affine transformation
					level[l] -> affine[v] -> setParameter(level[l] -> W_eye[size], level[l] -> sum[v]);
					graph -> add(level[l] -> affine[v], MATMUL);

					level[l] -> affine2[v] -> setParameter(level[l] -> W_one[size], level[l] -> sum[v]);
					graph -> add(level[l] -> affine2[v], MATMUL);

					level[l] -> filtered[v] -> setParameter(level[l] -> affine[v], level[l] -> K_eye);
					graph -> add(level[l] -> filtered[v], MATMUL);

					level[l] -> filtered2[v] -> setParameter(level[l] -> affine2[v], level[l] -> K_one);
					graph -> add(level[l] -> filtered2[v], MATMUL);

					// Chanels concatenation
					level[l] -> matcon[v] -> setParameter(level[l] -> filtered[v], level[l] -> filtered2[v]);
					graph -> add(level[l] -> matcon[v], MATRIXCONCAT);

					level[l] -> add[v] -> setParameter(level[l] -> b[size], level[l] -> matcon[v]);
					graph -> add(level[l] -> add[v], VECTORADDMATRIX);

					level[l] -> f[v] -> setParameter(level[l] -> add[v], 0);
					graph -> add(level[l] -> f[v], LEAKYRELU2D);
				}
			}
		}

		graph_feature -> clear();

		for (int v = 0; v < molecule -> nVertices; ++v) {
			shrinked[v] -> setParameter(level[nLevels] -> f[v], 0);
			graph -> add(shrinked[v], SHRINKMATRIX);

			vertex_feature[v] -> setParameter(shrinked[v]);
			graph -> add(vertex_feature[v], LEAKYRELU);

			graph_feature -> add_vector(vertex_feature[v]);
		}

		graph -> add(graph_feature, SUMVECTORS);

		predict -> setParameter(W, graph_feature);
		graph -> add(predict, MATVECMUL);

		graph -> add(logl, LOGLOSS);
	}

	void free_memory_for_dynamic_graphs(DenseGraph *molecule) {
		for (int l = 0; l <= nLevels; ++l) {
			if (l == 0) {
				for (int v = 0; v < molecule -> nVertices; ++v) {
					delete level[l] -> affine[v];
					level[l] -> affine[v] = NULL;

					delete level[l] -> f_transpose[v];
					level[l] -> f_transpose[v] = NULL;

					delete level[l] -> f[v];
					level[l] -> f[v] = NULL;
				}
			}

			if (l > 0) {
				for (int v = 0; v < molecule -> nVertices; ++v) {
					for (int w = 0; w < molecule -> nVertices; ++w) {
						if (adj[v][w] > 0) {
							delete level[l] -> permuted_f[v][w];
							level[l] -> permuted_f[v][w] = NULL;

							delete level[l] -> X[v][w];
							level[l] -> X[v][w] = NULL;
						}
					}
					delete level[l] -> sum[v];
					level[l] -> sum[v] = NULL;

					delete level[l] -> affine[v];
					level[l] -> affine[v] = NULL;

					delete level[l] -> affine2[v];
					level[l] -> affine2[v] = NULL;

					delete level[l] -> matcon[v];
					level[l] -> matcon[v] = NULL;

					delete level[l] -> add[v];
					level[l] -> add[v] = NULL;

					delete level[l] -> f[v];
					level[l] -> f[v] = NULL;
				}
			}
		}

		for (int v = 0; v < molecule -> nVertices; ++v) {
			delete shrinked[v];
			shrinked[v] = NULL;

			delete vertex_feature[v];
			vertex_feature[v] = NULL;
		}

		delete graph_feature;
		graph_feature = NULL;

		delete predict;
		predict = NULL;

		delete logl;
		logl = NULL;
	}

	float getLoss(int nBatch, DenseGraph **molecule, float *target) {
		float total_loss = 0.0;
		for (int i = 0; i < nBatch; ++i) {
			complete_computation_graph(molecule[i]);
			this -> target -> value[0] = target[i];
			graph -> forward();
			total_loss += logl -> getLoss();
		}
		return total_loss;
	}

	pair < float, float > BatchLearn(int nBatch, DenseGraph **molecule, float *target, float learning_rate) {
		assert(nBatch > 0);
		assert(molecule[0] -> nVertices <= max_nVertices);
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
		assert(molecule[0] -> nVertices <= max_nVertices);
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
		assert(molecule -> nVertices <= max_nVertices);
		assert(molecule -> nFeatures == nFeatures);

		complete_computation_graph(molecule);
		this -> target -> value[0] = target;

		cache_parameters -> cache_parameters();

		graph -> forward();
		float best_error = logl -> getLoss();

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
			float error = logl -> getLoss();

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
		assert(molecule -> nVertices <= max_nVertices);

		complete_computation_graph(molecule);

		graph -> forward();

		int predict_label = 0;
		for (int c = 0; c < nClass; ++c) {
			if (predict -> value[c] > predict -> value[predict_label]) {
				predict_label = c;
			}
		}
		return predict_label;
	}

	vector<float> Feature(DenseGraph *molecule) {
		assert(molecule -> nVertices <= max_nVertices);
		complete_computation_graph(molecule);

		graph -> forward();

		vector<float> vect;
		vect.clear();
		for (int i = 0; i < graph_feature -> size; ++i) {
			vect.push_back(graph_feature -> value[i]);
		}
		return vect;
	}

	vector<float> ForDebugging(DenseGraph *molecule) {
		assert(molecule -> nVertices <= max_nVertices);
		complete_computation_graph(molecule);

		graph -> forward();

		cout << endl;
		for (int l = 1; l <= nLevels; ++l) {
			cout << "**** Level " << l << endl;
			for (int v = 0; v < molecule -> nVertices; ++v) {
				cout << "Vertex " << v << endl;
				for (int i = 0; i < level[l] -> add[v] -> size; ++i) {
					cout << level[l] -> add[v] -> value[i] << " ";
				}
				cout << endl;
			}
		}

		cout << "**** Shrinked" <<  endl;
		for (int v = 0; v < molecule -> nVertices; ++v) {
			cout << "Vertex " << v << endl;
			for (int i = 0; i < shrinked[v] -> size; ++i) {
				cout << shrinked[v] -> value[i] << " ";
			}
			cout << endl;
		}

		vector<float> vect;
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

	// Dynamic computation graph
	GraphFlow *graph;

	// Number of classes
	int nClass;

	// Original synthesized graph features
	Matrix **feature;

	// Mapping from the original synthesized graph features into chanels
	Matrix *H;

	// Each level
	struct Level {
		// Number of chanels in this level
		int nChanels;

		// Steerable Filters
		Vector **lambda1;
		Vector **lambda2;

		ScalarMatMul **W_eye;
		ScalarMatMul **W_one;

		Matrix *K_eye;
		Matrix *K_one;

		// Bias
		Vector **b;

		// LeakyReLU2D
		LeakyReLU2D **f;

		// Transpose (if necessary)
		Transpose **f_transpose;

		// Addition
		VectorAddMatrix **add;

		// Affine transformation
		MatMul **affine;
		MatMul **affine2;

		MatMul **filtered;
		MatMul **filtered2;

		// Chanels concatenation
		MatrixConcat **matcon;

		// Sum of Matrices
		SumMatrices **sum;

		// Permutation matrices
		Matrix ***X;

		// Permuted features
		MatMul ***permuted_f;

		// Indices for the receptive fields
		vector<int> *phi;
	};

	// Multiple levels
	Level **level;

	// Shrink from matrices to vectors
	ShrinkMatrix **shrinked;

	// Vertex features
	LeakyReLU **vertex_feature;

	// Graph feature
	SumVectors *graph_feature;

	// Dense Layer
	Matrix *W;

	// Prediction
	MatVecMul *predict;

	// Target
	Vector *target;

	// Log Loss
	LogLoss *logl;

	// Set of identity matrices
	Matrix **eye;

	// Set of one matrices
	Matrix **one;

	// Stochastic Gradient Descent
	Momentum *sgd;

	// Sum gradients
	SumGradients *sum_gradients;

	// Cache parameters
	CacheParameters *cache_parameters;

	// Maximum number of vertices
	int max_nVertices;

	// Number of levels
	int nLevels;

	// Number of chanels
	int nChanels;

	// Number of original vertex features
	int nFeatures;

	// The depth of Weisfeiler-Lehman
	int nDepth;

	// Momentum parameter
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

	// Weisfeiler-Lehman ordering
	bool has_WL_ordering;

	~SMP_1D_ver3_classification() {
		for (int l = 0; l <= nLevels; ++l) {
			if (l == 0) {
				delete[] level[l] -> f;
				delete[] level[l] -> f_transpose;
				delete[] level[l] -> affine;
			} else {
				delete[] level[l] -> f;
				delete[] level[l] -> add;
				delete[] level[l] -> affine;
				delete[] level[l] -> affine2;
				delete[] level[l] -> matcon;
				delete[] level[l] -> sum;
				delete[] level[l] -> X;
				delete[] level[l] -> permuted_f;
				delete[] level[l] -> phi;
			}
		}
		delete[] shrinked;
		delete[] vertex_feature;
		delete graph_feature;
		delete W;
		delete predict;
		delete target;
		delete logl;
		delete sgd;
		delete sum_gradients;
		delete cache_parameters;
		delete[] adj;
		delete[] histogram;
		delete[] shortest_paths;
		delete[] order;
		delete[] rank;
	}
};

#endif
