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

#include "../GraphFlow/SMP_omega_physics.h"

using namespace std;

const int max_nVertices = 10;
const int max_receptive_field = 4;
const int nChanels = 16;
const int nLevels = 2;
const int nFeatures = 4;

const int nThreads = 4;

const double learning_rate = 0.001;
const int nEpochs = 1024;

const int nMolecules = 4;

string model_fn = "SMP_omega_physics-model.dat";

SMP_omega_physics train_network(max_nVertices, max_receptive_field, nLevels, nChanels, nFeatures);
SMP_omega_physics test_network(max_nVertices, max_receptive_field, nLevels, nChanels, nFeatures);

struct Molecule {
	DenseGraph *graph;
	double target;
	vector< pair<int, int> > edge;
	vector< string > label;

	void build() {
		for (int i = 0; i < edge.size(); ++i) {
			int u = edge[i].first;
			int v = edge[i].second;
			graph -> adj[u][v] = 1;
			graph -> adj[v][u] = 1;
		}

		for (int v = 0; v < graph -> nVertices; ++v) {
			if (label[v] == "C") {
				graph -> feature[v][0] = 1.0;
			}
			if (label[v] == "H") {
				graph -> feature[v][1] = 1.0;
			}
			if (label[v] == "N") {
				graph -> feature[v][2] = 1.0;
			}
			if (label[v] == "O") {
				graph -> feature[v][3] = 1.0;
			}
		}
	}
};
Molecule **molecule;

void init(Molecule *mol, string name) {
	if (name == "CH4") {
		mol -> graph = new DenseGraph(5, nFeatures);
		mol -> target = mol -> graph -> nVertices;

		mol -> edge.clear();
		mol -> edge.push_back(make_pair(0, 1));
		mol -> edge.push_back(make_pair(0, 2));
		mol -> edge.push_back(make_pair(0, 3));
		mol -> edge.push_back(make_pair(0, 4));

		mol -> label.clear();
		mol -> label.push_back("C");
		mol -> label.push_back("H");
		mol -> label.push_back("H");
		mol -> label.push_back("H");
		mol -> label.push_back("H");

		mol -> build();
	}

	if (name == "NH3") {
		mol -> graph = new DenseGraph(4, nFeatures);
		mol -> target = mol -> graph -> nVertices;

		mol -> edge.clear();
		mol -> edge.push_back(make_pair(0, 1));
		mol -> edge.push_back(make_pair(0, 2));
		mol -> edge.push_back(make_pair(0, 3));

		mol -> label.clear();
		mol -> label.push_back("N");
		mol -> label.push_back("H");
		mol -> label.push_back("H");
		mol -> label.push_back("H");

		mol -> build();
	}

	if (name == "H2O") {
		mol -> graph = new DenseGraph(3, nFeatures);
		mol -> target = mol -> graph -> nVertices;

		mol -> edge.clear();
		mol -> edge.push_back(make_pair(0, 1));
		mol -> edge.push_back(make_pair(0, 2));

		mol -> label.clear();
		mol -> label.push_back("O");
		mol -> label.push_back("H");
		mol -> label.push_back("H");

		mol -> build();
	}

	if (name == "C2H4") {
		mol -> graph = new DenseGraph(6, nFeatures);
		mol -> target = mol -> graph -> nVertices;

		mol -> edge.clear();
		mol -> edge.push_back(make_pair(0, 1));
		mol -> edge.push_back(make_pair(0, 2));
		mol -> edge.push_back(make_pair(0, 3));
		mol -> edge.push_back(make_pair(3, 4));
		mol -> edge.push_back(make_pair(3, 5));

		mol -> label.clear();
		mol -> label.push_back("C");
		mol -> label.push_back("H");
		mol -> label.push_back("H");
		mol -> label.push_back("C");
		mol -> label.push_back("H");
		mol -> label.push_back("H");

		mol -> build();
	}
}

int main(int argc, char **argv) {
	// Measuring time
	time_t start, end;

	// Starting time
	time(&start);

	molecule = new Molecule* [nMolecules];
	for (int i = 0; i < nMolecules; ++i) {
		molecule[i] = new Molecule();
	}

	init(molecule[0], "CH4");
	init(molecule[1], "NH3");
	init(molecule[2], "H2O");
	init(molecule[3], "C2H4");

	cout << "--- Learning ------------------------------" << endl;

	DenseGraph **graphs = new DenseGraph* [nMolecules];
	double *targets = new double [nMolecules];
	double *predict = new double [nMolecules];

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
			cout << "Target = " << targets[i] << ", Predict = " << predict[i] << endl;
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

		double predict = test_network.Predict(molecule[i] -> graph);
		
		cout << "Target = " << molecule[i] -> target << ", Predict = " << predict << endl;
	}

	// Ending time
	time(&end);

	cout << endl << difftime(end, start) << " seconds" << endl;

	return 0;
}