// Framework: GraphFlow
// Class: LSTM
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

// Based on Deep Learning tutorial: http://deeplearning.net/tutorial/lstm.html

#ifndef __LSTM_H_INCLUDED__
#define __LSTM_H_INCLUDED__

#include <iostream>
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

const double gradient_clipping_threshold = 1.0;

class LSTM {
public:
	LSTM(int nFeatures, int nHiddens, int nClasses, int max_nLevels, double momenum_param)  {
		this -> nFeatures = nFeatures;
		this -> nHiddens = nHiddens;
		this -> nClasses = nClasses;
		this -> max_nLevels = max_nLevels;
		this -> momenum_param = momenum_param;

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

	double norm1(Vector *V) {
		double ret = 0.0;
		for (int i = 0; i < V -> size; ++i) {
			ret += abs(V -> value[i]);
		}
		return ret;
	}

	void gradient_clipping(Vector *V) {
		double V_norm = norm1(V);
		if (V_norm > gradient_clipping_threshold) {
			for (int i = 0; i < V -> size; ++i) {
				V -> gradient[i] = gradient_clipping_threshold / V_norm * V -> gradient[i];
			}
		}
	}

	void gradient_descent(double learning_rate) {
		for (int i = 0; i < sgd -> params.size(); ++i) {
			gradient_clipping(sgd -> params[i]);
		}

		sgd -> Learn(learning_rate);
	}

	double getLoss(int nLevels) {
		double total = 0.0;
		for (int l = 0; l < nLevels; ++l) {
			total += level[l] -> logl -> getLoss();
		}
		return total;
	}

	pair<double, double> Learn(int nLevels, double **x_sequence, int *target_sequence, int nIterations, double learning_rate) {
		// Computation graph
		complete_computation_graph(nLevels, x_sequence);

		// Add target
		for (int l = 0; l < nLevels; ++l) {
			level[l] -> target -> value[0] = target_sequence[l];
		}

		// Stochastic Gradient Descent
		cache_params -> cache_parameters();

		graph -> forward();
		double best_logl = getLoss(nLevels);

		pair<double, double> ret;
		ret.first = best_logl;

		double decay_lr = 0.5;
		double min_lr = 1e-20;

		for (int iter = 0; iter < nIterations; ++iter) {
			graph -> forward();
			graph -> backward();
			gradient_descent(learning_rate);

			graph -> forward();
			double logl = getLoss(nLevels);

			cout << "Iteration " << (iter + 1) << ": Log-Likelihood = " << logl << endl;

			if (logl <= best_logl) {
				cache_params -> restore_parameters();

				if (learning_rate <= min_lr) {
					break;
				}
				learning_rate *= decay_lr;
			} else {
				best_logl = logl;
				cache_params -> cache_parameters();
			}
		}

		ret.second = best_logl;
		return ret;
	}

	void Predict(int nLevels, double **x_sequence, int *predict_sequence) {
		// Computation graph
		complete_computation_graph(nLevels, x_sequence);

		graph -> forward();
		
		for (int l = 0; l < nLevels; ++l) {
			double max_prob = 0.0;
			for (int i = 0; i < level[l] -> softmax -> size; ++i) {
				if (level[l] -> softmax -> value[i] > max_prob) {
					predict_sequence[l] = i;
					max_prob = level[l] -> softmax -> value[i];
				}
			}
		}
	}

	void complete_computation_graph(int nLevels, double **x_sequence) {
		graph -> clear();

		// Input gate
		graph -> add(Wi, MATRIX);
		graph -> add(Ui, MATRIX);
		graph -> add(bi, VECTOR);

		// Candidate value
		graph -> add(Wc, MATRIX);
		graph -> add(Uc, MATRIX);
		graph -> add(bc, VECTOR);

		// Forget gate
		graph -> add(Wf, MATRIX);
		graph -> add(Uf, MATRIX);
		graph -> add(bf, VECTOR);

		// Output gate
		graph -> add(Wo, MATRIX);
		graph -> add(Uo, MATRIX);
		graph -> add(Vo, MATRIX);
		graph -> add(bo, VECTOR);

		// Softmax matrix
		graph -> add(theta, MATRIX);

		for (int l = 0; l < nLevels; ++l) {
			for (int i = 0; i < nFeatures; ++i) {
				level[l] -> x -> value[i] = x_sequence[l][i];
			}
		}

		for (int l = 0; l < nLevels; ++l) {
			// Input gate
			graph -> add(level[l] -> input_Wx, MATVECMUL);
			if (l > 0) {
				graph -> add(level[l] -> input_Uh, MATVECMUL);
			}
			graph -> add(level[l] -> input_sum, SUMVECTORS);
			graph -> add(level[l] -> input, SIGMOID);

			// Candidate value
			graph -> add(level[l] -> candidate_Wx, MATVECMUL);
			if (l > 0) {
				graph -> add(level[l] -> candidate_Uh, MATVECMUL);
			}
			graph -> add(level[l] -> candidate_sum, SUMVECTORS);
			graph -> add(level[l] -> candidate, TANH);

			// Forget gate
			graph -> add(level[l] -> forget_Wx, MATVECMUL);
			if (l > 0) {
				graph -> add(level[l] -> forget_Uh, MATVECMUL);
			}
			graph -> add(level[l] -> forget_sum, SUMVECTORS);
			graph -> add(level[l] -> forget, SIGMOID);

			// Memory cell
			graph -> add(level[l] -> memory_new, MULTIPLY);
			if (l > 0) {
				graph -> add(level[l] -> memory_old, MULTIPLY);
			}
			graph -> add(level[l] -> memory, SUMVECTORS);

			// Output gate
			graph -> add(level[l] -> output_Wx, MATVECMUL);
			graph -> add(level[l] -> output_Vmemory, MATVECMUL);
			if (l > 0) {
				graph -> add(level[l] -> output_Uh, MATVECMUL);
			}
			graph -> add(level[l] -> output_sum, SUMVECTORS);
			graph -> add(level[l] -> output, SIGMOID);

			// Hidden
			graph -> add(level[l] -> tanh_memory, TANH);
			graph -> add(level[l] -> hidden, MULTIPLY);

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

			// Input gate
			level[l] -> input_Wx = new MatVecMul(Wi, level[l] -> x);

			level[l] -> input_sum = new SumVectors(nHiddens);
			level[l] -> input_sum -> add_vector(level[l] -> input_Wx);
			level[l] -> input_sum -> add_vector(bi);

			if (l > 0) {
				level[l] -> input_Uh = new MatVecMul(Ui, level[l - 1] -> hidden);
				level[l] -> input_sum -> add_vector(level[l] -> input_Uh);
			}

			level[l] -> input = new Sigmoid(level[l] -> input_sum);

			// Candidate value
			level[l] -> candidate_Wx = new MatVecMul(Wc, level[l] -> x);

			level[l] -> candidate_sum = new SumVectors(nHiddens);
			level[l] -> candidate_sum -> add_vector(level[l] -> candidate_Wx);
			level[l] -> candidate_sum -> add_vector(bc);

			if (l > 0) {
				level[l] -> candidate_Uh = new MatVecMul(Uc, level[l - 1] -> hidden);
				level[l] -> candidate_sum -> add_vector(level[l] -> candidate_Uh);
			}

			level[l] -> candidate = new Tanh(level[l] -> candidate_sum);

			// Forget gate
			level[l] -> forget_Wx = new MatVecMul(Wf, level[l] -> x);

			level[l] -> forget_sum = new SumVectors(nHiddens);
			level[l] -> forget_sum -> add_vector(level[l] -> forget_Wx);
			level[l] -> forget_sum -> add_vector(bf);

			if (l > 0) {
				level[l] -> forget_Uh = new MatVecMul(Uf, level[l - 1] -> hidden);
				level[l] -> forget_sum -> add_vector(level[l] -> forget_Uh);
			}

			level[l] -> forget = new Sigmoid(level[l] -> forget_sum);

			// Memory cell
			level[l] -> memory_new = new Multiply(level[l] -> input, level[l] -> candidate);

			level[l] -> memory = new SumVectors(nHiddens);
			level[l] -> memory -> add_vector(level[l] -> memory_new);

			if (l > 0) {
				level[l] -> memory_old = new Multiply(level[l] -> forget, level[l - 1] -> memory);
				level[l] -> memory -> add_vector(level[l] -> memory_old);
			}

			// Output gate
			level[l] -> output_Wx = new MatVecMul(Wo, level[l] -> x);
			level[l] -> output_Vmemory = new MatVecMul(Vo, level[l] -> memory);

			level[l] -> output_sum = new SumVectors(nHiddens);
			level[l] -> output_sum -> add_vector(level[l] -> output_Wx);
			level[l] -> output_sum -> add_vector(level[l] -> output_Vmemory);
			level[l] -> output_sum -> add_vector(bo);

			if (l > 0) {
				level[l] -> output_Uh = new MatVecMul(Uo, level[l - 1] -> hidden);
				level[l] -> output_sum -> add_vector(level[l] -> output_Uh);
			}

			level[l] -> output = new Sigmoid(level[l] -> output_sum);

			// Hidden
			level[l] -> tanh_memory = new Tanh(level[l] -> memory);
			level[l] -> hidden = new Multiply(level[l] -> output, level[l] -> tanh_memory);

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
		sgd = new Momentum(momenum_param);

		// Input gate
		sgd -> add(Wi);
		sgd -> add(Ui);
		sgd -> add(bi);

		// Candidate value
		sgd -> add(Wc);
		sgd -> add(Uc);
		sgd -> add(bc);

		// Forget gate
		sgd -> add(Wf);
		sgd -> add(Uf);		
		sgd -> add(bf);

		// Output gate
		sgd -> add(Wo);
		sgd -> add(Uo);
		sgd -> add(Vo);
		sgd -> add(bo);

		// Softmax matrix
		sgd -> add(theta);

		// Cache Parameters
		cache_params = new CacheParameters();

		// Input gate
		cache_params -> add(Wi);
		cache_params -> add(Ui);
		cache_params -> add(bi);

		// Candidate value
		cache_params -> add(Wc);
		cache_params -> add(Uc);
		cache_params -> add(bc);

		// Forget gate
		cache_params -> add(Wf);
		cache_params -> add(Uf);		
		cache_params -> add(bf);

		// Output gate
		cache_params -> add(Wo);
		cache_params -> add(Uo);
		cache_params -> add(Vo);
		cache_params -> add(bo);

		// Softmax matrix
		cache_params -> add(theta);
	}

	void init_weight_matrices() {
		// Input gate
		graph -> uniform_init(Wi);
		graph -> uniform_init(Ui);
		
		graph -> uniform_init(bi);
		// graph -> assign_init(bi, 1.0);

		// Candidate value
		graph -> uniform_init(Wc);
		graph -> uniform_init(Uc);
		
		graph -> uniform_init(bc);
		// graph -> assign_init(bc, 1.0);

		// Forget gate
		graph -> uniform_init(Wf);
		graph -> uniform_init(Uf);
		
		// graph -> uniform_init(bf);
		graph -> assign_init(bf, 1.0);

		// Output gate
		graph -> uniform_init(Wo);
		graph -> uniform_init(Uo);
		graph -> uniform_init(Vo);
		
		graph -> uniform_init(bo);
		// graph -> assign_init(bo, 1.0);

		// Softmax matrix
		graph -> uniform_init(theta);
	}

	void init_learning_parameters() {
		// Input gate
		Wi = new Matrix(nHiddens, nFeatures);
		Ui = new Matrix(nHiddens, nHiddens);
		bi = new Vector(nHiddens);

		// Candidate value
		Wc = new Matrix(nHiddens, nFeatures);
		Uc = new Matrix(nHiddens, nHiddens);
		bc = new Vector(nHiddens);

		// Forget gate
		Wf = new Matrix(nHiddens, nFeatures);
		Uf = new Matrix(nHiddens, nHiddens);
		bf = new Vector(nHiddens);

		// Output gate
		Wo = new Matrix(nHiddens, nFeatures);
		Uo = new Matrix(nHiddens, nHiddens);
		Vo = new Matrix(nHiddens, nHiddens);
		bo = new Vector(nHiddens);

		// Softmax matrix
		theta = new Matrix(nClasses, nHiddens);
	}

	struct Level {
		// Input
		Vector *x;

		// Target
		Vector *target;

		// Input gate
		MatVecMul *input_Wx;
		MatVecMul *input_Uh;
		SumVectors *input_sum;
		Sigmoid *input;

		// Candiate value
		MatVecMul *candidate_Wx;
		MatVecMul *candidate_Uh;
		SumVectors *candidate_sum;
		Tanh *candidate;

		// Forget gate
		MatVecMul *forget_Wx;
		MatVecMul *forget_Uh;
		SumVectors *forget_sum;
		Sigmoid *forget;

		// Memory cell
		Multiply *memory_new;
		Multiply *memory_old;
		SumVectors *memory;

		// Output gate
		MatVecMul *output_Wx;
		MatVecMul *output_Uh;
		MatVecMul *output_Vmemory;
		SumVectors *output_sum;
		Sigmoid *output;

		// Hidden
		Tanh *tanh_memory;
		Multiply *hidden;

		// Softmax
		AverageVectors *average_pool;
		MatVecMul *linear;
		Softmax *softmax;
		LogLoss *logl;
	};
	
	Level **level;

	// Input gate
	Matrix *Wi;
	Matrix *Ui;
	Vector *bi;

	// Candidate value
	Matrix *Wc;
	Matrix *Uc;
	Vector *bc;

	// Forget gate
	Matrix *Wf;
	Matrix *Uf;
	Vector *bf;

	// Output gate
	Matrix *Wo;
	Matrix *Uo;
	Matrix *Vo;
	Vector *bo;

	// Softmax matrix
	Matrix *theta;

	// GraphFlow
	GraphFlow *graph;

	// Stochastic Gradient Descent
	Momentum *sgd;

	// Cache parameters
	CacheParameters *cache_params;

	int nFeatures;
	int nHiddens;
	int nClasses;
	int max_nLevels;
	double momenum_param;

	~LSTM() {
		for (int l = 0; l <= max_nLevels; ++l) {
			delete level[l] -> x;
			delete level[l] -> target;
			delete level[l] -> input_Wx;
			delete level[l] -> input_Uh;
			delete level[l] -> input_sum;
			delete level[l] -> input;
			delete level[l] -> candidate_Wx;
			delete level[l] -> candidate_Uh;
			delete level[l] -> candidate_sum;
			delete level[l] -> candidate;
			delete level[l] -> forget_Wx;
			delete level[l] -> forget_Uh;
			delete level[l] -> forget_sum;
			delete level[l] -> forget;
			delete level[l] -> memory_new;
			delete level[l] -> memory_old;
			delete level[l] -> memory;
			delete level[l] -> output_Wx;
			delete level[l] -> output_Uh;
			delete level[l] -> output_Vmemory;
			delete level[l] -> output_sum;
			delete level[l] -> output;
			delete level[l] -> tanh_memory;
			delete level[l] -> hidden;
			delete level[l] -> average_pool;
			delete level[l] -> linear;
			delete level[l] -> softmax;
			delete level[l] -> logl;
		}
		delete[] level;
		delete Wi;
		delete Ui;
		delete bi;
		delete Wc;
		delete Uc;
		delete bc;
		delete Wf;
		delete Uf;
		delete bf;
		delete Wo;
		delete Uo;
		delete Vo;
		delete bo;
		delete theta;
		delete graph;
	}
};

#endif