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

#include "../GraphFlow/SMP_2D_classification.h"

using namespace std;

const int max_nVertices = 10;
const int nClass = max_nVertices + 1;
const int nChanels = 10;
const int nLevels = 2;
const int nFeatures = 4;
const int nDepth = 5;
const double momentum_param = 0.9;

const double learning_rate = 0.001;
const int nEpochs = 100;

const int nMolecules = 4;

string model_fn = "SMP_2D_classification-model.dat";

SMP_2D_classification train_network(nClass, max_nVertices, nLevels, nChanels, nFeatures, nDepth, momentum_param);
SMP_2D_classification test_network(nClass, max_nVertices, nLevels, nChanels, nFeatures, nDepth, momentum_param);

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
	for (int i = 0; i < nMolecules; ++i) {
		graphs[i] = molecule[i] -> graph;
		targets[i] = molecule[i] -> target;
	}

	for (int j = 0; j < nEpochs; ++j) {
		train_network.BatchLearn(nMolecules, graphs, targets, learning_rate);
		cout << "Done epoch " << j + 1 << " / " << nEpochs << endl;
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
}