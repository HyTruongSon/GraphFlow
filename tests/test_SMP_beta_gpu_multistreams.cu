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
#include <fstream>

#include "../GraphFlow_gpu/SMP_beta_gpu_multistreams.h"
#include "../kaggle_utils/MoleculeBuilder.cpp"

using namespace std;

const int max_nVertices = 25;
const int nChanels = 4;
const int nLevels = 5;
const int nFeatures = 5;

const int targetSize = 1461;

const int nThreads = 16;

const double learning_rate = 1e-4;
const int nEpochs = 10;

string model_fn = "SMP_omega_gpu_multistreams.dat";

SMP_beta_gpu_multistreams train_network(max_nVertices, nLevels, nChanels, nFeatures, true);
SMP_beta_gpu_multistreams test_network(max_nVertices, nLevels, nChanels, nFeatures, true);

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
		targets[i] = new double[targetSize];
	}

	double **predict = new double* [nMolecules];
	for (int i = 0; i < nMolecules; i++){
		predict[i] = new double[targetSize];
	}

	for (int i = 0; i < nMolecules; ++i) {
		graphs[i] = molecule[i] -> graph;
		targets[i] = molecule[i] -> target;
	}

	cout << "Multi-threadings" << endl;
	train_network.init_multi_threads(nThreads);


	for (int j = 0; j < nEpochs; ++j) {
		for (int batch = 0; batch < nMolecules % 1000; ++batch){
			DenseGraph** _graphs = new DenseGraph*[nMolecules % 1000];
			double** _targets = new double*[nMolecules % 1000];
			for(int ind = 0; ind < nMolecules % 1000; ++ind){
				_graphs[ind] = graphs[batch * (nMolecules % 1000) + ind];
				_targets[ind] = targets[batch * (nMolecules % 1000) + ind];
			}
			train_network.Threaded_BatchLearn(nMolecules % 1000, _graphs, _targets, learning_rate);
		}

		train_network.Threaded_Predict(nMolecules, graphs, predict);
		double totalLoss = 0.;
		for(int ind = 0; ind < nMolecules; ++ind){
			double loss = train_network.getLoss(nMolecules, graphs, targets[ind]);
			totalLoss += loss;
		}
		cout << "Done epoch " << j + 1 << " / " << nEpochs << "\tLoss : " << totalLoss << endl;
	}

	// Save model to file
	train_network.save_model(model_fn);

	cout << endl << "--- Predicting ----------------------------" << endl;

	// Load model from file
	test_network.load_model(model_fn);

	for (int i = 0; i < nMolecules; ++i) {
		cout << "Molecule " << (i + 1) << "\n";

		double* predict = test_network.Predict(molecule[i] -> graph);
		ofstream outfile("predictions/molecule_" + std::to_string(i + 1), std::ios::out);
		for(int ind = 0; ind < targetSize; ++ind){
			outfile << predict[ind];
			outfile << "\n";
		}
		outfile.close();
	}

	// Ending time
	time_ms(end);

	cout << endl << difftime_ms(end, start) << " ms" << endl;

	return 0;
}