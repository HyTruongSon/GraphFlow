// Framework: GraphFlow
// Class: GRU
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

// Based on Deep Learning tutorial: http://deeplearning.net/tutorial/lstm.html

#ifndef __GRU_H_INCLUDED__
#define __GRU_H_INCLUDED__

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <assert.h>

#include "CacheParameters.h"
#include "Momentum.h"
#include "GraphFlow.h"

using namespace std;

const float gradient_clipping_threshold = 1.0;

class GRU {
public:
	GRU(int nFeatures, int nHiddens, int nClasses, int max_nLevels, float momentum_param)  {
		this -> nFeatures = nFeatures;
		this -> nHiddens = nHiddens;
		this -> nClasses = nClasses;
		this -> max_nLevels = max_nLevels;
		this -> momentum_param = momentum_param;

		init_learning_parameters();
		init_computation_graph();
		init_weight_matrices();
	}

	void save_model(string model_fn) {
		ofstream file(model_fn.c_str(), ios::out);
		for (int i = 0; i < sgd -> params.size(); ++i) {
			for (int j = 0; j < sgd -> params[i] -> size; ++j) {
				file << sgd -> params[i] -> value[j] << " ";
			}
			file << endl;
		}
		file.close();
	}

	void load_model(string model_fn) {
		ifstream file(model_fn.c_str(), ios::in);
		for (int i = 0; i < sgd -> params.size(); ++i) {
			for (int j = 0; j < sgd -> params[i] -> size; ++j) {
				file >> sgd -> params[i] -> value[j];
			}
		}
		file.close();
	}

	float norm1(Vector *V) {
		float ret = 0.0;
		for (int i = 0; i < V -> size; ++i) {
			ret += abs(V -> value[i]);
		}
		return ret;
	}

	void gradient_clipping(Vector *V) {
		float V_norm = norm1(V);
		if (V_norm > gradient_clipping_threshold) {
			for (int i = 0; i < V -> size; ++i) {
				V -> gradient[i] = gradient_clipping_threshold / V_norm * V -> gradient[i];
			}
		}
	}

	void gradient_descent(float learning_rate) {
		for (int i = 0; i < sgd -> params.size(); ++i) {
			gradient_clipping(sgd -> params[i]);
		}

		sgd -> Learn(learning_rate);
	}

	float getLoss(int nLevels) {
		float total = 0.0;
		for (int l = 0; l < nLevels; ++l) {
			total += level[l] -> logl -> getLoss();
		}
		return total;
	}

	pair<float, float> Learn(int nLevels, float **x_sequence, int *target_sequence, int nIterations, float learning_rate) {
		// Computation graph
		complete_computation_graph(nLevels, x_sequence);

		// Add target
		for (int l = 0; l < nLevels; ++l) {
			level[l] -> target -> value[0] = target_sequence[l];
		}

		// Stochastic Gradient Descent
		cache_param -> cache_parameters();

		graph -> forward();
		float best_logl = getLoss(nLevels);

		pair<float, float> ret;
		ret.first = best_logl;

		float decay_lr = 0.5;
		float min_lr = 1e-20;

		for (int iter = 0; iter < nIterations; ++iter) {
			graph -> forward();
			graph -> backward();
			gradient_descent(learning_rate);

			graph -> forward();
			float logl = getLoss(nLevels);

			cout << "Iteration " << (iter + 1) << ": Log-Likelihood = " << logl << endl;

			if (logl <= best_logl) {
				cache_param -> restore_parameters();

				if (learning_rate <= min_lr) {
					break;
				}
				learning_rate *= decay_lr;
			} else {
				best_logl = logl;
				cache_param -> cache_parameters();
			}
		}

		ret.second = best_logl;
		return ret;
	}

	void Predict(int nLevels, float **x_sequence, int *predict_sequence) {
		// Computation graph
		complete_computation_graph(nLevels, x_sequence);

		graph -> forward();
		
		for (int l = 0; l < nLevels; ++l) {
			float max_prob = 0.0;
			for (int i = 0; i < level[l] -> softmax -> size; ++i) {
				if (level[l] -> softmax -> value[i] > max_prob) {
					predict_sequence[l] = i;
					max_prob = level[l] -> softmax -> value[i];
				}
			}
		}
	}

	void complete_computation_graph(int nLevels, float **x_sequence) {
		graph -> clear();

		// Update gate
		graph -> add(W_z, MATRIX);
		graph -> add(U_z, MATRIX);
		graph -> add(b_z, VECTOR);

		// Reset gate
		graph -> add(W_r, MATRIX);
		graph -> add(U_r, MATRIX);
		graph -> add(b_r, VECTOR);

		// Candidate value
		graph -> add(W_h, MATRIX);
		graph -> add(U_h, MATRIX);
		graph -> add(b_h, VECTOR);

		// Softmax matrix
		graph -> add(theta, MATRIX);

		for (int l = 0; l < nLevels; ++l) {
			for (int i = 0; i < nFeatures; ++i) {
				level[l] -> x -> value[i] = x_sequence[l][i];
			}
		}

		for (int l = 0; l < nLevels; ++l) {
			// Update gate
			graph -> add(level[l] -> update_part1, MATVECMUL);
			if (l > 0) {
				graph -> add(level[l] -> update_part2, MATVECMUL);
			}
			graph -> add(level[l] -> update_sum, SUMVECTORS);
			graph -> add(level[l] -> update, SIGMOID);

			// Reset gate
			graph -> add(level[l] -> reset_part1, MATVECMUL);
			if (l > 0) {
				graph -> add(level[l] -> reset_part2, MATVECMUL);
			}
			graph -> add(level[l] -> reset_sum, SUMVECTORS);
			graph -> add(level[l] -> reset, SIGMOID);

			// Candidate value
			graph -> add(level[l] -> candidate_part1, MATVECMUL);
			if (l > 0) {
				graph -> add(level[l] -> candidate_prev, MULTIPLY);
				graph -> add(level[l] -> candidate_part2, MATVECMUL);
			}
			graph -> add(level[l] -> candidate_sum, SUMVECTORS);
			graph -> add(level[l] -> candidate, SIGMOID);

			// Hidden
			graph -> add(level[l] -> hidden_part1, MULTIPLY);
			if (l > 0) {
				graph -> add(level[l] -> hidden_sub, SUBTRACT);
				graph -> add(level[l] -> hidden_part2, MULTIPLY);
			}
			graph -> add(level[l] -> hidden, SUMVECTORS);

			// LogLoss
			graph -> add(level[l] -> average_pool, AVERAGEVECTORS);
			graph -> add(level[l] -> linear, MATVECMUL);
			graph -> add(level[l] -> softmax, SOFTMAX);
			graph -> add(level[l] -> logl, LOGLOSS);
		}
	}

	void init_computation_graph() {
		// GraphFlow
		graph = new GraphFlow();

		// Levels
		level = new Level* [max_nLevels];

		for (int l = 0; l < max_nLevels; ++l) {
			level[l] = new Level();

			// Input
			level[l] -> x = new Vector(nFeatures);

			// Target
			level[l] -> target = new Vector(1);

			// Update gate
			level[l] -> update_part1 = new MatVecMul(W_z, level[l] -> x);

			level[l] -> update_sum = new SumVectors(nHiddens);
			level[l] -> update_sum -> add_vector(level[l] -> update_part1);
			level[l] -> update_sum -> add_vector(b_z);

			if (l > 0) {
				level[l] -> update_part2 = new MatVecMul(U_z, level[l - 1] -> hidden);
				level[l] -> update_sum -> add_vector(level[l] -> update_part2);
			}

			level[l] -> update = new Sigmoid(level[l] -> update_sum);

			// Reset gate
			level[l] -> reset_part1 = new MatVecMul(W_r, level[l] -> x);

			level[l] -> reset_sum = new SumVectors(nHiddens);
			level[l] -> reset_sum -> add_vector(level[l] -> reset_part1);
			level[l] -> reset_sum -> add_vector(b_r);

			if (l > 0) {
				level[l] -> reset_part2 = new MatVecMul(U_r, level[l - 1] -> hidden);
				level[l] -> reset_sum -> add_vector(level[l] -> reset_part2);
			}

			level[l] -> reset = new Sigmoid(level[l] -> reset_sum);

			// Candidate value
			level[l] -> candidate_part1 = new MatVecMul(W_h, level[l] -> x);

			level[l] -> candidate_sum = new SumVectors(nHiddens);
			level[l] -> candidate_sum -> add_vector(level[l] -> candidate_part1);
			level[l] -> candidate_sum -> add_vector(b_h);

			if (l > 0) {
				level[l] -> candidate_prev = new Multiply(level[l] -> reset, level[l - 1] -> hidden);
				level[l] -> candidate_part2 = new MatVecMul(U_h, level[l] -> candidate_prev);
				level[l] -> candidate_sum -> add_vector(level[l] -> candidate_part2);
			}

			level[l] -> candidate = new Tanh(level[l] -> candidate_sum);

			// Hidden
			level[l] -> hidden_part1 = new Multiply(level[l] -> update, level[l] -> candidate);
			
			level[l] -> hidden = new SumVectors(nHiddens);
			level[l] -> hidden -> add_vector(level[l] -> hidden_part1);
			
			if (l > 0) {
				level[l] -> hidden_sub = new Subtract(one, level[l] -> update);
				level[l] -> hidden_part2 = new Multiply(level[l] -> hidden_sub, level[l - 1] -> hidden);
				level[l] -> hidden -> add_vector(level[l] -> hidden_part2);
			}

			// Average Pooling
			level[l] -> average_pool = new AverageVectors(nHiddens);
			for (int t = 0; t <= l; ++t) {
				level[l] -> average_pool -> add_vector(level[t] -> hidden);
			}

			// Softmax
			level[l] -> linear = new MatVecMul(theta, level[l] -> average_pool);
			level[l] -> softmax = new Softmax(level[l] -> linear);

			// LogLoss
			level[l] -> logl = new LogLoss(level[l] -> softmax, level[l] -> target);
		}

		// Stochastic Gradient Descent
		sgd = new Momentum(momentum_param);

		// Update gate
		sgd -> add(W_z);
		sgd -> add(U_z);		
		sgd -> add(b_z);

		// Reset gate
		sgd -> add(W_r);
		sgd -> add(U_r);		
		sgd -> add(b_r);

		// Candidate value
		sgd -> add(W_h);
		sgd -> add(U_h);
		sgd -> add(b_h);

		// Softmax matrix
		sgd -> add(theta);
	}

	void init_weight_matrices() {
		// Update gate
		graph -> uniform_init(W_z);
		graph -> uniform_init(U_z);		
		graph -> uniform_init(b_z);

		// Reset gate
		graph -> uniform_init(W_r);
		graph -> uniform_init(U_r);		
		graph -> uniform_init(b_r);

		// Candidate value
		graph -> uniform_init(W_h);
		graph -> uniform_init(U_h);
		graph -> uniform_init(b_h);

		// Softmax matrix
		graph -> uniform_init(theta);
	}

	void init_learning_parameters() {
		// Update gate
		W_z = new Matrix(nHiddens, nFeatures);
		U_z = new Matrix(nHiddens, nHiddens);
		b_z = new Vector(nHiddens);

		// Reset gate
		W_r = new Matrix(nHiddens, nFeatures);
		U_r = new Matrix(nHiddens, nHiddens);
		b_r = new Vector(nHiddens);

		// Candidate value
		W_h = new Matrix(nHiddens, nFeatures);
		U_h = new Matrix(nHiddens, nHiddens);
		b_h = new Vector(nHiddens);

		// Softmax matrix
		theta = new Matrix(nClasses, nHiddens);

		// Vector one
		one = new Vector(nHiddens);
		for (int i = 0; i < one -> size; ++i) {
			one -> value[i] = 1.0;
		}

		// Cache parameters
		cache_param = new CacheParameters();
		cache_param -> add(W_z);
		cache_param -> add(U_z);
		cache_param -> add(b_z);
		cache_param -> add(W_r);
		cache_param -> add(U_r);
		cache_param -> add(b_r);
		cache_param -> add(W_h);
		cache_param -> add(U_h);
		cache_param -> add(b_h);
		cache_param -> add(theta);
	}

	struct Level {
		// Input
		Vector *x;

		// Target
		Vector *target;

		// Update gate
		MatVecMul *update_part1;
		MatVecMul *update_part2;
		SumVectors *update_sum;
		Sigmoid *update;

		// Reset gate
		MatVecMul *reset_part1;
		MatVecMul *reset_part2;
		SumVectors *reset_sum;
		Sigmoid *reset;

		// Candidate
		MatVecMul *candidate_part1;
		Multiply *candidate_prev;
		MatVecMul *candidate_part2;
		SumVectors *candidate_sum;
		Tanh *candidate;

		// Hidden
		Subtract *hidden_sub;
		Multiply *hidden_part1;
		Multiply *hidden_part2;
		SumVectors *hidden;

		// Softmax
		AverageVectors *average_pool;
		MatVecMul *linear;
		Softmax *softmax;
		LogLoss *logl;
	};
	
	Level **level;

	// Vector 1
	Vector *one;

	// Input gate
	Matrix *W_z;
	Matrix *U_z;
	Vector *b_z;

	// Reset gate
	Matrix *W_r;
	Matrix *U_r;
	Vector *b_r;
 
 	// Candidate value
	Matrix *W_h;
	Matrix *U_h;
	Vector *b_h;

	// Softmax matrix
	Matrix *theta;

	// Cache Parameters
	CacheParameters *cache_param;

	// GraphFlow
	GraphFlow *graph;

	// Stochastic Gradient Descent
	Momentum *sgd;

	int nFeatures;
	int nHiddens;
	int nClasses;
	int max_nLevels;
	float momentum_param;

	~GRU() {
		for (int l = 0; l <= max_nLevels; ++l) {
 			delete level[l] -> x;
			delete level[l] -> target;
			delete level[l] -> update_part1;
			delete level[l] -> update_part2;
			delete level[l] -> update_sum;
			delete level[l] -> update;
			delete level[l] -> reset_part1;
			delete level[l] -> reset_part2;
			delete level[l] -> reset_sum;
			delete level[l] -> reset;
			delete level[l] -> candidate_part1;
			delete level[l] -> candidate_prev;
			delete level[l] -> candidate_part2;
			delete level[l] -> candidate_sum;
			delete level[l] -> candidate;
			delete level[l] -> hidden_sub;
			delete level[l] -> hidden_part1;
			delete level[l] -> hidden_part2;
			delete level[l] -> hidden;
			delete level[l] -> average_pool;
			delete level[l] -> linear;
			delete level[l] -> softmax;
			delete level[l] -> logl;
		}
		delete[] level;
		delete one;
		delete W_z;
		delete U_z;
		delete b_z;
		delete W_r;
		delete U_r;
		delete b_r;
	 	delete W_h;
		delete U_h;
		delete b_h;
		delete theta;
		delete cache_param;
		delete graph;
	}
};

#endif
