// Framework: GraphFlow
// Author: Machine Learning Group of UChicago
// Main Contributor: Hy Truong Son
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <time.h>

#include "../GraphFlow/GRU.h"

const int nFeatures = 20;
const int nHiddens = 50;
const int nClasses = 2;
const int max_nLevels = 10;

const int nIterations = 100;
const double learning_rate = 1.0;
const double momentum_param = 0.9;

const string model_fn = "model-GRU.dat";

int nLevels;

GRU *train_network = new GRU(nFeatures, nHiddens, nClasses, max_nLevels, momentum_param);
GRU *test_network = new GRU(nFeatures, nHiddens, nClasses, max_nLevels, momentum_param);

int main(int argc, char **argv) {
	srand(time(NULL));
	
	double **x_sequence = new double* [max_nLevels];

	for (int i = 0; i < max_nLevels; ++i) {
		x_sequence[i] = new double [nFeatures];
		for (int j = 0; j < nFeatures; ++j) {
			int value = rand() % 2;
			x_sequence[i][j] = value;
		}
	}
 
	int *target_sequence = new int [max_nLevels];
	for (int i = 0; i < max_nLevels; ++i) {
		target_sequence[i] = 0;
		for (int j = 0; j < nFeatures; ++j) {
			if (i + 1 < max_nLevels) {
				target_sequence[i] += (int)(x_sequence[i + 1][j]);
			} else {
				target_sequence[i] += (int)(x_sequence[i][j]);
			}
		}
		target_sequence[i] %= 2;
	} 

	pair<double, double> info = train_network -> Learn(max_nLevels, x_sequence, target_sequence, nIterations, learning_rate);

	cout << "Before training log-likelihood: " << info.first << endl;
	cout << "After training log-likelihood: " << info.second << endl;

	train_network -> save_model(model_fn);

	test_network -> load_model(model_fn);

	int *predict_sequence = new int [max_nLevels];

	test_network -> Predict(max_nLevels, x_sequence, predict_sequence);

	for (int i = 0; i < max_nLevels; ++i) {
		cout << "Sequence: ";
		for (int j = 0; j < nFeatures; ++j) {
			cout << x_sequence[i][j] << " ";
		}
		cout << "Predict: " << predict_sequence[i] << " ";
		cout << "Target: " << target_sequence[i] << " ";
		if (predict_sequence[i] == target_sequence[i]) {
			cout << "YES" << endl;
		} else {
			cout << "NO" << endl;
		}
	}

	return 0;
}