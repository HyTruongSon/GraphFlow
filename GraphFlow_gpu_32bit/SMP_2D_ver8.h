// Framework: GraphFlow
// Class: Second-order Steerable Message Passing (version 8)
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __SMP_2D_VER8_H_INCLUDED__
#define __SMP_2D_VER8_H_INCLUDED__

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

class SMP_2D_ver8 {
public:
	SMP_2D_ver8(int max_nVertices, int nLevels, int nChanels, int nFeatures, int nDepth, float momentum_param) {
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

	SMP_2D_ver8(int max_nVertices, int nLevels, int nChanels, int nFeatures, int nDepth, float momentum_param, bool has_WL_ordering) {
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
				level[l] -> matmul = new MatMul* [max_nVertices];
				level[l] -> f_reshape = new Reshape3D* [max_nVertices];
				level[l] -> f = new LeakyReLU3D* [max_nVertices];

				for (int v = 0; v < max_nVertices; ++v) {
					level[l] -> matmul[v] = new MatMul(nChanels, nChanels);
					level[l] -> f_reshape[v] = new Reshape3D(nChanels, nChanels, nChanels);
					level[l] -> f[v] = new LeakyReLU3D(nChanels, nChanels, nChanels);
				}
			} 

			// Level from 1 to nLevels
			if (l > 0) {
				// Bias
				level[l] -> b = new Vector(nChanels);

				// For reducing the number of chanels
				level[l] -> K = new Matrix(nChanels, nContractions * nChanels);

				// For computations
				level[l] -> f = new LeakyReLU3D* [max_nVertices];
				level[l] -> contract = new RisiContraction_18* [max_nVertices];
				level[l] -> adj = new Matrix* [max_nVertices];
				level[l] -> represent = new CustomMatMulTensor* [max_nVertices];
				level[l] -> add = new VectorAddTensor* [max_nVertices];

				for (int v = 0; v < max_nVertices; ++v) {
					level[l] -> f[v] = new LeakyReLU3D(max_nVertices, max_nVertices, nChanels);
					level[l] -> contract[v] = new RisiContraction_18(max_nVertices, max_nVertices, nContractions * nChanels);
					level[l] -> adj[v] = new Matrix(max_nVertices, max_nVertices);
					level[l] -> represent[v] = new CustomMatMulTensor(max_nVertices, max_nVertices, nChanels);
					level[l] -> add[v] = new VectorAddTensor(max_nVertices, max_nVertices, nChanels); 
				}

				level[l] -> X_mul_f = new MatTensorMul** [max_nVertices];
				level[l] -> quadratic = new TensorMatMul** [max_nVertices];

				for (int v = 0; v < max_nVertices; ++v) {
					level[l] -> X_mul_f[v] = new MatTensorMul* [max_nVertices];
					level[l] -> quadratic[v] = new TensorMatMul* [max_nVertices];
					
					for (int w = 0; w < max_nVertices; ++w) {
						level[l] -> X_mul_f[v][w] = new MatTensorMul(max_nVertices, max_nVertices, nChanels);
						level[l] -> quadratic[v][w] = new TensorMatMul(max_nVertices, max_nVertices, nChanels);
					}
				}

				// For permutation matrices
				level[l] -> X = new Matrix** [max_nVertices];
				level[l] -> X_transpose = new Matrix** [max_nVertices];

				for (int v = 0; v < max_nVertices; ++v) {
					level[l] -> X[v] = new Matrix* [max_nVertices];
					level[l] -> X_transpose[v] = new Matrix* [max_nVertices];

					for (int w = 0; w < max_nVertices; ++w) {
						level[l] -> X[v][w] = new Matrix(max_nVertices, max_nVertices);
						level[l] -> X_transpose[v][w] = new Matrix(max_nVertices, max_nVertices);
					}
				}
			}
		}

		// On top of everything
		shrinked = new ShrinkTensor* [max_nVertices];
		vertex_feature = new LeakyReLU* [max_nVertices];

		for (int v = 0; v < max_nVertices; ++v) {
			shrinked[v] = new ShrinkTensor(nChanels);
			vertex_feature[v] = new LeakyReLU(nChanels);
		}

		graph_feature = new SumVectors(nChanels);
		predict = new InnerProduct();
		sql = new SquaredLoss();

		// Linear regression
		W = new Vector(nChanels);

		// Target
		target = new Vector(1);

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
			sgd -> add(level[l] -> K);
			sgd -> add(level[l] -> b);
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

	void init_receptive_field_permutation_matrix_reduced_adj(DenseGraph *molecule) {
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
				for (int i = 0; i < level[l] -> phi[v].size(); ++i) {
					int w = level[l] -> phi[v][i];
						
					level[l] -> X[v][w] -> setParameter(level[l] -> phi[v].size(), level[l - 1] -> phi[w].size());
					init_permutation_matrix(level[l] -> X[v][w], level[l] -> phi[v], level[l - 1] -> phi[w]);

					level[l] -> X_transpose[v][w] -> setParameter(level[l - 1] -> phi[w].size(), level[l] -> phi[v].size());
					init_permutation_matrix(level[l] -> X_transpose[v][w], level[l - 1] -> phi[w], level[l] -> phi[v]);
				}
			}
		}

		// Constructing the reduced adjacency matrices
		for (int l = 1; l <= nLevels; ++l) {
			for (int v = 0; v < molecule -> nVertices; ++v) {
				level[l] -> adj[v] -> setParameter(level[l] -> phi[v].size(), level[l] -> phi[v].size());

				for (int i = 0; i < level[l] -> phi[v].size(); ++i) {
					int vertex1 = level[l] -> phi[v][i];

					for (int j = 0; j < level[l] -> phi[v].size(); ++j) {
						int vertex2 = level[l] -> phi[v][j];
						int ind = level[l] -> adj[v] -> index(i, j);

						if (vertex1 == vertex2) {
							level[l] -> adj[v] -> value[ind] = 1.0;
						} else {
							level[l] -> adj[v] -> value[ind] = adj[vertex1][vertex2];
						}
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

		// Update the feature 
		update_feature(molecule);

		// Finding the shortest-paths by Floyd-Warshall algorithm
		floyd_warshall(molecule);
		
		// Get the feature vector for each vertex
		weisfeiler_lehman(molecule);

		// Find the optimal order of vertices
		rank_vertices(molecule);

		// Initialize the receptive fields, permutation matrices and reduced adjacency matrices
		init_receptive_field_permutation_matrix_reduced_adj(molecule);

		// Constructing the dynamic computation graph
		graph -> clear();
		graph -> add(H, MATRIX);
		for (int l = 1; l <= nLevels; ++l) {
			graph -> add(level[l] -> K, MATRIX);
			graph -> add(level[l] -> b, VECTOR);
		}
		graph -> add(W, VECTOR);

		for (int l = 0; l <= nLevels; ++l) {
			if (l == 0) {
				for (int v = 0; v < molecule -> nVertices; ++v) {
					level[l] -> matmul[v] -> setParameter(H, feature[v]);
					graph -> add(level[l] -> matmul[v], MATMUL);

					level[l] -> f_reshape[v] -> setParameter(level[l] -> matmul[v], 1, 1, nChanels);
					graph -> add(level[l] -> f_reshape[v], RESHAPE3D);

					level[l] -> f[v] -> setParameter(level[l] -> f_reshape[v]);
					graph -> add(level[l] -> f[v], LEAKYRELU3D);
				}
			}

			if (l > 0) {
				for (int v = 0; v < molecule -> nVertices; ++v) {
					// Size of the receptive field
					int size = level[l] -> phi[v].size();

					// Contraction
					level[l] -> contract[v] -> setParameter(size, nChanels);
					level[l] -> contract[v] -> clear();

					for (int i = 0; i < size; ++i) {
						int w = level[l] -> phi[v][i];

						level[l] -> X_mul_f[v][w] -> setParameter(level[l] -> X[v][w], level[l - 1] -> f[w]);
						graph -> add(level[l] -> X_mul_f[v][w], MATTENSORMUL);

						level[l] -> quadratic[v][w] -> setParameter(level[l] -> X_mul_f[v][w], level[l] -> X_transpose[v][w]);
						graph -> add(level[l] -> quadratic[v][w], TENSORMATMUL);

						level[l] -> contract[v] -> add_tensor(level[l] -> quadratic[v][w]);
					}

					level[l] -> contract[v] -> set_adjacency(level[l] -> adj[v]);
					graph -> add(level[l] -> contract[v], RISICONTRACTION_18);

					// Cut the number of chanels
					level[l] -> represent[v] -> setParameter(level[l] -> K, level[l] -> contract[v]);
					graph -> add(level[l] -> represent[v], CUSTOMMATMULTENSOR);

					// Add the bias
					level[l] -> add[v] -> setParameter(level[l] -> b, level[l] -> represent[v]);
					graph -> add(level[l] -> add[v], VECTORADDTENSOR);

					// Non-linearity
					level[l] -> f[v] -> setParameter(level[l] -> add[v]);
					graph -> add(level[l] -> f[v], LEAKYRELU3D);
				}
			}
		}

		graph_feature -> clear();

		for (int v = 0; v < molecule -> nVertices; ++v) {
			shrinked[v] -> setParameter(level[nLevels] -> f[v]);
			graph -> add(shrinked[v], SHRINKTENSOR);

			vertex_feature[v] -> setParameter(shrinked[v]);
			graph -> add(vertex_feature[v], LEAKYRELU);

			graph_feature -> add_vector(vertex_feature[v]);
		}

		graph -> add(graph_feature, SUMVECTORS);

		predict -> setParameter(graph_feature, W);
		graph -> add(predict, INNERPRODUCT);

		sql -> setParameter(predict, target);
		graph -> add(sql, SQUAREDLOSS);
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
		assert(molecule -> nVertices <= max_nVertices);

		complete_computation_graph(molecule);

		graph -> forward();

		return predict -> value[0];
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

	// Number of contractions in this class
	static const int nContractions = RisiContraction_18::nContractions;

	// Original synthesized graph features
	Matrix **feature;

	// Set of identity matrices
	Matrix **eye;

	// Set of one matrices
	Matrix **one;

	// Mapping from the original synthesized graph features into chanels
	Matrix *H;

	// Each level
	struct Level {
		// For reducing the number of chanels
		Matrix *K;

		// Bias
		Vector *b;

		// For WL features
		MatMul **matmul;

		// LeakyReLU3D
		LeakyReLU3D **f;

		// Reshape (if necessary)
		Reshape3D **f_reshape;

		// MatTensorMul
		MatTensorMul ***X_mul_f;

		// TensorMatMul
		TensorMatMul ***quadratic;

		// Reduced adjacency matrix
		Matrix **adj;

		// RisiContraction_18
		RisiContraction_18 **contract;

		// Cut number of chanels 
		CustomMatMulTensor **represent;

		// Addition with bias
		VectorAddTensor **add;

		// Permutation matrices
		Matrix ***X;

		// Permutation matrices transpose
		Matrix ***X_transpose;

		// Indices for the receptive fields
		vector<int> *phi;
	};

	// Multiple levels
	Level **level;

	// Shrink from tensors to vectors
	ShrinkTensor **shrinked;

	// Vertex features
	LeakyReLU **vertex_feature;

	// Graph feature
	SumVectors *graph_feature;

	// Linear Regression
	Vector *W;

	// Prediction
	InnerProduct *predict;

	// Target
	Vector *target;

	// Squared loss
	SquaredLoss *sql;

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

	~SMP_2D_ver8() {
		delete[] feature;
		for (int l = 0; l <= nLevels; ++l) {
			if (l == 0) {
				delete[] level[l] -> matmul;
				delete[] level[l] -> f_reshape;
				delete[] level[l] -> f;
			} else {
				delete[] level[l] -> f;
				delete[] level[l] -> contract;
				delete[] level[l] -> represent;
				delete[] level[l] -> add;
				delete[] level[l] -> phi;
				delete[] level[l] -> quadratic;
				delete[] level[l] -> X_mul_f;
				delete[] level[l] -> X;
				delete[] level[l] -> X_transpose;
			}
		}
		delete[] shrinked;
		delete[] vertex_feature;
		delete graph_feature;
		delete W;
		delete predict;
		delete target;
		delete sql;
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
