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
#include <array>
#include <time.h>
#include <sys/time.h>

#include "../GraphFlow/SMP_theta_physics.h"
#include "../kaggle_utils/MoleculeBuilder.cpp"

using namespace std;

const int max_nVertices = 100;
const int max_receptive_field = 7;
const int nChanels = 16;
const int nLevels = 20;
const int nFeatures = 5;

const int nThreads = 4;

const double learning_rate = 0.001;
const int nEpochs = 1024;

string model_fn = "SMP_theta_physics-model.dat";

SMP_theta_physics train_network(max_nVertices, max_receptive_field, nLevels, nChanels, nFeatures);
SMP_theta_physics test_network(max_nVertices, max_receptive_field, nLevels, nChanels, nFeatures);

// Get the millisecond
void time_ms(long int &ms) {
	struct timeval tp;
	gettimeofday(&tp, NULL);
	ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;
}

// Difference in milliseconds
long int difftime_ms(long int &end, long int &start) {
	return end - start;
}

Molecule **molecule;

int main(int argc, char **argv) {
	// Measuring time
	long int start, end;

	cout << "Molecule builder init..." << std::endl;
	MoleculeBuilder* moleculeBuilder = new MoleculeBuilder();

	molecule = moleculeBuilder->getMolecules();
	int nMolecules = moleculeBuilder->getNumberOfMolecules();

	// Starting time
	time_ms(start);

	cout << "--- Learning ------------------------------" << endl;

	DenseGraph **graphs = new DenseGraph* [nMolecules];
	
	double** targets = new double*[nMolecules];
	for (int i = 0; i < nMolecules; i++){
		targets[i] = new double[1461];
	}

	double **predict = new double* [nMolecules];
	for (int i = 0; i < nMolecules; i++){
		predict[i] = new double[1461];
	}

	for (int i = 0; i < nMolecules; ++i) {
		graphs[i] = molecule[i] -> graph;
		targets[i] = molecule[i] -> target;
	}

	cout << "Multi-threadings" << endl;
	train_network.init_multi_threads(nThreads);


	for (int j = 0; j < nEpochs; ++j) {
		train_network.Threaded_BatchLearn(nMolecules, graphs, targets, learning_rate);
		cout << "Done epoch " << j + 1 << " / " << nEpochs << endl;

		train_network.Threaded_Predict(nMolecules, graphs, predict);
		for (int i = 0; i < nMolecules; ++i) {
			cout << "Molecule " << (i + 1) << ": ";
			cout << "Target = {" << targets[i][0] << " " << targets[i][1] << " " << targets[i][2] << "}, Predict = {" \
			<< predict[i][0] << " " << predict[i][1] << " " << predict[i][2] << "}" << endl;
		}
		cout << endl;
	}

	// Save model to file
	train_network.save_model(model_fn);

	cout << endl << "--- Predicting ----------------------------" << endl;

	// Load model from file
	test_network.load_model(model_fn);

	for (int i = 0; i < nMolecules; ++i) {
		cout << "Molecule " << (i + 1) << ": ";

		double* predict = test_network.Predict(molecule[i] -> graph);
		
		cout << "Target = {" << molecule[i] -> target[0] << " " << molecule[i] -> target[1] << " " << molecule[i] -> target[2] << "}, Predict = {" \
		<< predict[0] << " " << predict[1] << " " << predict[2] << "}" << endl;
	}

	// Ending time
	time_ms(end);

	cout << endl << difftime_ms(end, start) << " ms" << endl;

	return 0;
}