// Framework: GraphFlow
// Class: Second-order Steerable Message Passing (Omega version)
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __SMP_OMEGA_PHYSICS_H_INCLUDED__
#define __SMP_OMEGA_PHYSICS_H_INCLUDED__

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <thread>
#include <assert.h>

#include "DenseGraph.h"
#include "SumGradients.h"
#include "CacheParameters.h"
#include "Adam.h"
#include "GraphFlow.h"

using namespace std;

class SMP_omega_physics {
public:
	SMP_omega_physics(int max_nVertices, int max_receptive_field, int nLevels, int nChanels, int nFeatures) {
		assert(max_receptive_field <= max_nVertices);

		this -> use_coulomb = false;
		this -> max_nVertices = max_nVertices;
		this -> max_receptive_field = max_receptive_field;
		this -> nLevels = nLevels;
		this -> nChanels = nChanels;
		this -> nFeatures = nFeatures;

		// Multi-threading mode
		this -> nThreads = 0;
		this -> multi_threaded = false;

		computation_graph();
		weights_initialization();
	}

	SMP_omega_physics(bool use_coulomb, int max_nVertices, int max_receptive_field, int nLevels, int nChanels, int nFeatures) {
		assert(max_receptive_field <= max_nVertices);

		this -> use_coulomb = use_coulomb;
		this -> max_nVertices = max_nVertices;
		this -> max_receptive_field = max_receptive_field;
		this -> nLevels = nLevels;
		this -> nChanels = nChanels;
		this -> nFeatures = nFeatures;

		// Multi-threading mode
		this -> nThreads = 0;
		this -> multi_threaded = false;

		computation_graph();
		weights_initialization();
	}

	// +-------------------------+
	// | Multi-threading (Begin) |
	// +-------------------------+

	void init_multi_threads(int nThreads) {
		assert(nThreads > 1);

		this -> nThreads = nThreads;
		multi_threaded = true;

		// Initialize all instances
		instance = new SMP_omega_physics* [this -> nThreads];
		for (int i = 0; i < nThreads; ++i) {
			instance[i] = new SMP_omega_physics(use_coulomb, max_nVertices, max_receptive_field, nLevels, nChanels, nFeatures);
		}

		// Initialize multi-threaded jobs
		job = new std::thread [this -> nThreads];
	}

	// +-----------------------+
	// | Multi-threading (End) |
	// +-----------------------+

	void computation_graph() {
		// +--------------------------+
		// | Component initialization |
		// +--------------------------+

		// Synthesized graph features
		feature = new Matrix* [max_nVertices];
		for (int v = 0; v < max_nVertices; ++v) {
			feature[v] = new Matrix(nFeatures, 1);
		}

		// Mapping from the original synthesized graph features into chanels
		H = new Matrix(nChanels, nFeatures);

		// Maximum size of the receptive field
		int N = max_receptive_field;

		// Multiple levels
		level = new Level* [nLevels + 1];

		// Number of chanels
		int C, prevC;

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
				C = level[l] -> nChanels;

				level[l] -> matmul = new MatMul* [max_nVertices];
				level[l] -> f_reshape = new Reshape3D* [max_nVertices];
				level[l] -> f = new LeakyReLU3D* [max_nVertices];

				for (int v = 0; v < max_nVertices; ++v) {
					level[l] -> matmul[v] = new MatMul(C, 1);
					level[l] -> f_reshape[v] = new Reshape3D(1, 1, C);
					level[l] -> f[v] = new LeakyReLU3D(1, 1, C);
				}
			} 

			// Level from 1 to nLevels
			if (l > 0) {
				// Number of chanels in this level
				level[l] -> nChanels = level[l - 1] -> nChanels / 2;
				if (level[l] -> nChanels == 0) {
					level[l] -> nChanels = 1;
				}

				C = level[l] -> nChanels;

				// Number of chanels in the previous level
				prevC = level[l - 1] -> nChanels;

				// Bias
				level[l] -> b = new Vector(C);

				// For reducing the number of chanels
				level[l] -> K = new Matrix(nContractions * prevC, C);

				// For computations
				level[l] -> f = new LeakyReLU3D* [max_nVertices];
				level[l] -> contract = new RisiContraction_18* [max_nVertices];
				level[l] -> adj = new Matrix* [max_nVertices];
				level[l] -> reshape2D = new Reshape2D* [max_nVertices];
				level[l] -> represent = new MatMul* [max_nVertices];
				level[l] -> reshape3D = new Reshape3D* [max_nVertices];
				level[l] -> add = new VectorAddTensor* [max_nVertices];

				for (int v = 0; v < max_nVertices; ++v) {
					level[l] -> f[v] = new LeakyReLU3D(N, N, C);
					level[l] -> contract[v] = new RisiContraction_18(N, N, nContractions * prevC);
					level[l] -> adj[v] = new Matrix(N, N);
					level[l] -> reshape2D[v] = new Reshape2D(N * N, nContractions * prevC);
					level[l] -> represent[v] = new MatMul(N * N, nContractions * prevC, nContractions * prevC, C);
					level[l] -> reshape3D[v] = new Reshape3D(N, N, C);
					level[l] -> add[v] = new VectorAddTensor(N, N, C); 
				}

				level[l] -> X_mul_f = new MatTensorMul** [max_nVertices];
				level[l] -> quadratic = new TensorMatMul** [max_nVertices];

				for (int v = 0; v < max_nVertices; ++v) {
					level[l] -> X_mul_f[v] = new MatTensorMul* [max_nVertices];
					level[l] -> quadratic[v] = new TensorMatMul* [max_nVertices];
					
					for (int w = 0; w < max_nVertices; ++w) {
						level[l] -> X_mul_f[v][w] = new MatTensorMul(N, N, prevC);
						level[l] -> quadratic[v][w] = new TensorMatMul(N, N, prevC);
					}
				}

				// For permutation matrices
				level[l] -> X = new Matrix** [max_nVertices];
				level[l] -> X_transpose = new Matrix** [max_nVertices];

				for (int v = 0; v < max_nVertices; ++v) {
					level[l] -> X[v] = new Matrix* [max_nVertices];
					level[l] -> X_transpose[v] = new Matrix* [max_nVertices];

					for (int w = 0; w < max_nVertices; ++w) {
						level[l] -> X[v][w] = new Matrix(N, N);
						level[l] -> X_transpose[v][w] = new Matrix(N, N);
					}
				}
			}
		}

		// On top of everything
		shrinked = new ShrinkTensor** [nLevels + 1];
		vertex_feature = new LeakyReLU** [nLevels + 1];
		level_feature = new SumVectors* [nLevels + 1];

		int nTotalFeatures = 0;

		for (int l = 0; l <= nLevels; ++l) {
			// Number of chanels in this level
			C = level[l] -> nChanels;

			shrinked[l] = new ShrinkTensor* [max_nVertices];
			vertex_feature[l] = new LeakyReLU* [max_nVertices];
			level_feature[l] = new SumVectors(C);

			for (int v = 0; v < max_nVertices; ++v) {
				shrinked[l][v] = new ShrinkTensor(C);
				vertex_feature[l][v] = new LeakyReLU(C);
			}

			nTotalFeatures += C;
		}

		// Fully-connected layers
		graph_feature = new ConcatVectors(nTotalFeatures);

		int nHidden = nTotalFeatures / 2;
		W1 = new Matrix(nHidden, nTotalFeatures);
		W2 = new Vector(nHidden);

		hidden = new MatVecMul(nHidden);
		hidden_activation = new LeakyReLU(hidden);

		predict = new InnerProduct();
		sql = new SquaredLoss();

		// Target
		target = new Vector(1);

		// +-------------------+
		// | Computation graph |
		// +-------------------+
		
		graph = new GraphFlow();

		// +-----------------------------+
		// | Stochastic Gradient Descent |
		// +-----------------------------+

		sgd = new Adam();
		sgd -> add(H);
		for (int l = 1; l <= nLevels; ++l) {
			sgd -> add(level[l] -> K);
			sgd -> add(level[l] -> b);
		}
		sgd -> add(W1);
		sgd -> add(W2);

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

	void limit_receptive_field(int v, vector<int> &A) {
		for (int i = 0; i < A.size(); ++i) {
			for (int j = i + 1; j < A.size(); ++j) {
				if (shortest_paths[v][A[i]] > shortest_paths[v][A[j]]) {
					swap(A[i], A[j]);
				}
			}
		}

		int u, d;
		while (A.size() > max_receptive_field) {
			d = shortest_paths[v][A[A.size() - 1]];
			while (true) {
				u = A[A.size() - 1];
				if (shortest_paths[v][u] == d) {
					A.pop_back();
				} else {
					break;
				}
			}
		}
		
		assert(A.size() <= max_receptive_field);
		assert(A.size() > 0);
		assert(A[0] == v);
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

					// Limit the size of the receptive field
					if (level[l] -> phi[v].size() > max_receptive_field) {
						limit_receptive_field(v, level[l] -> phi[v]);
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

						// Check if we use the Coulomb matrix
						if (!use_coulomb) {
							if (vertex1 == vertex2) {
								level[l] -> adj[v] -> value[ind] = 1.0;
							} else {
								level[l] -> adj[v] -> value[ind] = adj[vertex1][vertex2];
							}
						} else {
							// Coulomb distance
							level[l] -> adj[v] -> value[ind] = molecule -> coulomb[vertex1][vertex2];
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

		// Initialize the receptive fields, permutation matrices and reduced adjacency matrices
		init_receptive_field_permutation_matrix_reduced_adj(molecule);

		// Constructing the dynamic computation graph
		graph -> clear();
		graph -> add(H, MATRIX);
		for (int l = 1; l <= nLevels; ++l) {
			graph -> add(level[l] -> K, MATRIX);
			graph -> add(level[l] -> b, VECTOR);
		}
		graph -> add(W1, MATRIX);
		graph -> add(W2, VECTOR);

		// Number of chanels
		int C, prevC;

		for (int l = 0; l <= nLevels; ++l) {
			if (l == 0) {
				// Number of chanels in this level
				C = level[l] -> nChanels;

				for (int v = 0; v < molecule -> nVertices; ++v) {
					level[l] -> matmul[v] -> setParameter(H, feature[v]);
					graph -> add(level[l] -> matmul[v], MATMUL);

					level[l] -> f_reshape[v] -> setParameter(level[l] -> matmul[v], 1, 1, C);
					graph -> add(level[l] -> f_reshape[v], RESHAPE3D);

					level[l] -> f[v] -> setParameter(level[l] -> f_reshape[v]);
					graph -> add(level[l] -> f[v], LEAKYRELU3D);
				}
			}

			if (l > 0) {
				// Number of chanels in this level
				C = level[l] -> nChanels;

				// Number of chanels in the previous level
				prevC = level[l - 1] -> nChanels;

				for (int v = 0; v < molecule -> nVertices; ++v) {
					// Size of the receptive field
					int size = level[l] -> phi[v].size();

					// Contraction
					level[l] -> contract[v] -> setParameter(size, prevC);
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
					level[l] -> reshape2D[v] -> setParameter(level[l] -> contract[v], size * size, nContractions * prevC);
					graph -> add(level[l] -> reshape2D[v], RESHAPE2D);

					level[l] -> represent[v] -> setParameter(level[l] -> reshape2D[v], level[l] -> K);
					graph -> add(level[l] -> represent[v], MATMUL);

					level[l] -> reshape3D[v] -> setParameter(level[l] -> represent[v], size, size, C);
					graph -> add(level[l] -> reshape3D[v], RESHAPE3D);

					// Add the bias
					level[l] -> add[v] -> setParameter(level[l] -> b, level[l] -> reshape3D[v]);
					graph -> add(level[l] -> add[v], VECTORADDTENSOR);

					// Non-linearity
					level[l] -> f[v] -> setParameter(level[l] -> add[v]);
					graph -> add(level[l] -> f[v], LEAKYRELU3D);
				}
			}
		}

		// Concatenate every level feature
		graph_feature -> clear();

		for (int l = 0; l <= nLevels; ++l) {
			level_feature[l] -> clear();

			for (int v = 0; v < molecule -> nVertices; ++v) {
				shrinked[l][v] -> setParameter(level[l] -> f[v]);
				graph -> add(shrinked[l][v], SHRINKTENSOR);

				vertex_feature[l][v] -> setParameter(shrinked[l][v]);
				graph -> add(vertex_feature[l][v], LEAKYRELU);

				level_feature[l] -> add_vector(vertex_feature[l][v]);
			}

			graph -> add(level_feature[l], SUMVECTORS);

			graph_feature -> add_vector(level_feature[l]);
		}

		graph -> add(graph_feature, CONCATVECTORS);
		
		// Fully-connected layers
		hidden -> setParameter(W1, graph_feature);
		graph -> add(hidden, MATVECMUL);

		hidden_activation -> setParameter(hidden);
		graph -> add(hidden_activation, LEAKYRELU);

		predict -> setParameter(hidden_activation, W2);
		graph -> add(predict, INNERPRODUCT);

		sql -> setParameter(predict, target);
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

	// +-------------------------+
	// | Multi-threading (Begin) |
	// +-------------------------+

	void copy_value(Adam *from, Adam *to) {
		assert(from -> params.size() == to -> params.size());

		for (int i = 0; i < from -> params.size(); ++i) {
			assert(from -> params[i] -> size == to -> params[i] -> size);

			for (int v = 0; v < from -> params[i] -> size; ++v) {
				to -> params[i] -> value[v] = from -> params[i] -> value[v];
			}
		}
	}

	void clean_gradient(Adam *sgd) {
		for (int i = 0; i < sgd -> params.size(); ++i) {
			for (int v = 0; v < sgd -> params[i] -> size; ++v) {
				sgd -> params[i] -> gradient[v] = 0.0;
			}
		}
	}

	void add_gradient(Adam *from, Adam *to) {
		assert(from -> params.size() == to -> params.size());

		for (int i = 0; i < from -> params.size(); ++i) {
			assert(from -> params[i] -> size == to -> params[i] -> size);

			for (int v = 0; v < from -> params[i] -> size; ++v) {
				to -> params[i] -> gradient[v] += from -> params[i] -> gradient[v];
			}
		}
	}

	static void compute_gradient_job(SMP_omega_physics *instance, DenseGraph *molecule, double target) {
		instance -> complete_computation_graph(molecule);
		instance -> target -> value[0] = target;

		instance -> graph -> forward();
		instance -> graph -> backward();
	}

	void Threaded_BatchLearn(int nBatch, DenseGraph **molecule, double *target, double learning_rate) {
		assert(multi_threaded == true);
		assert(nThreads > 1);

		assert(nBatch > 0);
		for (int i = 0; i < nBatch; ++i) {
			assert(molecule[i] -> nVertices <= max_nVertices);
			assert(molecule[i] -> nFeatures == nFeatures);
		}

		clean_gradient(sgd);

		int start = 0;
		while (start < nBatch) {
			int finish = start + nThreads - 1;
			if (finish >= nBatch) {
				finish = nBatch - 1;
			}

			int nRuns = finish - start + 1;

			for (int t = 0; t < nRuns; ++t) {
				copy_value(sgd, instance[t] -> sgd);
			}

			for (int i = start; i <= finish; ++i) {
				int t = i - start;
				job[t] = std::thread(compute_gradient_job, instance[t], molecule[i], target[i]);
			}

			for (int t = 0; t < nRuns; ++t) {
				job[t].join();
			}

			for (int t = 0; t < nRuns; ++t) {
				add_gradient(instance[t] -> sgd, sgd);
			}

			start = finish + 1;
		}

		sgd -> Learn(learning_rate, nBatch);
	}

	// +-----------------------+
	// | Multi-threading (End) |
	// +-----------------------+

	pair < double, double > BatchLearn(int nBatch, DenseGraph **molecule, double *target, double learning_rate) {
		assert(nBatch > 0);
		for (int i = 0; i < nBatch; ++i) {
			assert(molecule[i] -> nVertices <= max_nVertices);
			assert(molecule[i] -> nFeatures == nFeatures);
		}

		pair < double, double > ret;
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

	pair < double, double > BatchLearn(int nBatch, DenseGraph **molecule, double *target, int nIterations, double learning_rate, double epsilon) {
		assert(nBatch > 0);
		for (int i = 0; i < nBatch; ++i) {
			assert(molecule[i] -> nVertices <= max_nVertices);
			assert(molecule[i] -> nFeatures == nFeatures);
		}

		cache_parameters -> cache_parameters();

		pair<double, double> ret;
		ret.first = getLoss(nBatch, molecule, target);
		ret.second = ret.first;

		double decay_lr = 0.5;
		double min_lr = 1e-6;

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

			double loss = getLoss(nBatch, molecule, target);

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

	pair < double, double > Learn(DenseGraph *molecule, double target, int nIterations, double learning_rate, double epsilon) {
		assert(molecule -> nVertices <= max_nVertices);
		assert(molecule -> nFeatures == nFeatures);

		complete_computation_graph(molecule);
		this -> target -> value[0] = target;

		cache_parameters -> cache_parameters();

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

	double Predict(DenseGraph *molecule) {
		assert(molecule -> nVertices <= max_nVertices);

		complete_computation_graph(molecule);

		graph -> forward();

		return predict -> value[0];
	}

	// +------------------------+
	// | Multi-threaded (Begin) |
	// +------------------------+

	static void predict_job(SMP_omega_physics *instance, DenseGraph *molecule, double *predict, int position) {
		instance -> complete_computation_graph(molecule);
		instance -> graph -> forward();
		predict[position] = instance -> predict -> value[0];
	}

	void Threaded_Predict(int nBatch, DenseGraph **molecule, double *predict) {
		assert(multi_threaded == true);
		assert(nThreads > 1);

		assert(nBatch > 0);
		for (int i = 0; i < nBatch; ++i) {
			assert(molecule[i] -> nVertices <= max_nVertices);
			assert(molecule[i] -> nFeatures == nFeatures);
		}

		for (int t = 0; t < nThreads; ++t) {
			copy_value(sgd, instance[t] -> sgd);
		}

		int start = 0;
		while (start < nBatch) {
			int finish = start + nThreads - 1;
			if (finish >= nBatch) {
				finish = nBatch - 1;
			}

			int nRuns = finish - start + 1;

			for (int i = start; i <= finish; ++i) {
				int t = i - start;
				job[t] = std::thread(predict_job, instance[t], molecule[i], predict, i);
			}

			for (int t = 0; t < nRuns; ++t) {
				job[t].join();
			}

			start = finish + 1;
		}
	}

	// +----------------------+
	// | Multi-threaded (End) |
	// +----------------------+

	vector<double> Feature(DenseGraph *molecule) {
		assert(molecule -> nVertices <= max_nVertices);
		complete_computation_graph(molecule);

		graph -> forward();

		vector<double> vect;
		vect.clear();
		for (int i = 0; i < graph_feature -> size; ++i) {
			vect.push_back(graph_feature -> value[i]);
		}
		return vect;
	}

	vector<double> ForDebugging(DenseGraph *molecule) {
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

	// Dynamic computation graph
	GraphFlow *graph;

	// Number of contractions in this class
	static const int nContractions = RisiContraction_18::nContractions;

	// Infinity
	static const int INF = 1e9;

	// Number of threads
	int nThreads;

	// Multi-threading mode
	bool multi_threaded;

	// Instances for multi-threads
	SMP_omega_physics **instance;

	// Multi-threaded jobs
	std::thread *job;

	// Original synthesized graph features
	Matrix **feature;

	// Mapping from the original synthesized graph features into chanels
	Matrix *H;

	// Each level
	struct Level {
		// Number of chanels in this level
		int nChanels;

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
		Reshape2D **reshape2D;
		MatMul **represent;
		Reshape3D **reshape3D;

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
	ShrinkTensor ***shrinked;

	// Vertex features
	LeakyReLU ***vertex_feature;

	// Level feature
	SumVectors **level_feature;

	// Graph feature
	ConcatVectors *graph_feature;

	// Fully-connected layer
	Matrix *W1;
	Vector *W2;
	MatVecMul *hidden;
	LeakyReLU *hidden_activation;

	// Prediction
	InnerProduct *predict;

	// Target
	Vector *target;

	// Squared loss
	SquaredLoss *sql;

	// Stochastic Gradient Descent
	Adam *sgd;

	// Sum gradients
	SumGradients *sum_gradients;

	// Cache parameters
	CacheParameters *cache_parameters;

	// Maximum number of vertices
	int max_nVertices;

	// Maximum size of the receptive field
	int max_receptive_field;

	// Number of levels
	int nLevels;

	// Number of chanels
	int nChanels;

	// Number of original vertex features
	int nFeatures;

	// Adjacency matrix
	int **adj;

	// Floyd-Warshall algorithm
	int **shortest_paths;

	// For the usage of Coulomb matrix in Tensor Contractions
	bool use_coulomb;

	~SMP_omega_physics() {
		delete[] feature;
		for (int l = 0; l <= nLevels; ++l) {
			if (l == 0) {
				delete[] level[l] -> matmul;
				delete[] level[l] -> f_reshape;
				delete[] level[l] -> f;
			} else {
				delete[] level[l] -> f;
				delete[] level[l] -> contract;
				delete[] level[l] -> reshape2D;
				delete[] level[l] -> represent;
				delete[] level[l] -> reshape3D;
				delete[] level[l] -> add;
				delete[] level[l] -> phi;
				delete[] level[l] -> quadratic;
				delete[] level[l] -> X_mul_f;
				delete[] level[l] -> X;
				delete[] level[l] -> X_transpose;
			}
			delete[] shrinked[l];
			delete[] vertex_feature[l];
		}
		delete graph_feature;
		delete W1;
		delete W2;
		delete hidden;
		delete hidden_activation;
		delete predict;
		delete target;
		delete sql;
		delete sgd;
		delete sum_gradients;
		delete cache_parameters;
		delete[] adj;
		delete[] shortest_paths;
	}
};

#endif