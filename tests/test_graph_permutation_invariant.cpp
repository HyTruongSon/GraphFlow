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
#include <algorithm>

#include "../GraphFlow/SMP_theta_physics.h"

using namespace std;

const int nVertices = 20;
const int max_receptive_field = 10;
const int nChanels = 16;
const int nLevels = 6;
const int nFeatures = 6;

SMP_theta_physics network(nVertices, max_receptive_field, nLevels, nChanels, nFeatures);

// SMP_beta_physics network(nVertices, nLevels, nChanels, nFeatures);

DenseGraph* Erdos_Renyi(int nVertices, int nFeatures) {
	DenseGraph *graph = new DenseGraph(nVertices, nFeatures);

	// Random the adjacency matrix
	for (int i = 0; i < nVertices; ++i) {
		for (int j = i + 1; j < nVertices; ++j) {
			graph -> adj[i][j] = rand() % 2;
			graph -> adj[j][i] = graph -> adj[i][j];
		}
	}

	// Random the vertex features
	for (int i = 0; i < nVertices; ++i) {
		for (int j = 0; j < nFeatures; ++j) {
			graph -> feature[i][j] = rand() % 2;
		}
	}

	return graph;
}

pair< DenseGraph*, vector<int> > Permute(DenseGraph *graph) {
	DenseGraph* another = new DenseGraph(graph -> nVertices, graph -> nFeatures);

	// Create a permutation
	vector<int> perm;
	perm.clear();
	for (int i = 0; i < graph -> nVertices; ++i) {
		perm.push_back(i);
	}
	for (int i = 0; i < perm.size(); ++i) {
		int j = rand() % perm.size();
		swap(perm[i], perm[j]);
	}

	// Create the permuted adjacency
	for (int i = 0; i < another -> nVertices; ++i) {
		for (int j = 0; j < another -> nVertices; ++j) {
			int vertex_i = perm[i];
			int vertex_j = perm[j];
			another -> adj[i][j] = graph -> adj[vertex_i][vertex_j];
		}
	}

	// Create the permuted vertex features
	for (int i = 0; i < another -> nVertices; ++i) {
		int vertex_i = perm[i];
		for (int j = 0; j < another -> nFeatures; ++j) {
			another -> feature[i][j] = graph -> feature[vertex_i][j];
		}
	}

	return make_pair(another, perm);
}

void print(DenseGraph *graph) {
	cout << "Adjaceny matrix:" << endl;
	for (int i = 0; i < graph -> nVertices; ++i) {
		cout << "Vertex " << i << ": ";
		for (int j = 0; j < graph -> nVertices; ++j) {
			cout << graph -> adj[i][j] << " ";
		}
		cout << endl;
	}

	cout << endl << "Vertex features:" << endl;
	for (int i = 0; i < graph -> nVertices; ++i) {
		cout << "Vertex " << i << ": ";
		for (int j = 0; j < graph -> nFeatures; ++j) {
			cout << graph -> feature[i][j] << " ";
		}
		cout << endl;
	}
}

int main(int argc, char **argv) {
	srand(0);

	cout << "Number of vertices: " << nVertices << endl;
	cout << "Number of chanels (at level 0): " << nChanels << endl;
	cout << "Number of levels: " << nLevels << endl;
	cout << "Number of vertex features: " << nFeatures << endl;

	// Create a Erdos Renyi random graph
	DenseGraph *graph1 = Erdos_Renyi(nVertices, nFeatures);

	// Permute the graph
	pair < DenseGraph*, vector<int> > info = Permute(graph1);

	// Write the information 
	DenseGraph *graph2 = info.first;
	vector<int> perm = info.second;

	cout << "-----------------------------------------" << endl;

	cout << "Graph 1" << endl;
	print(graph1);

	cout << "-----------------------------------------" << endl;

	cout << "Graph 2:" << endl;
	print(graph2);

	cout << "-----------------------------------------" << endl;
	
	cout << "Permutation: ";
	for (int i = 0; i < perm.size(); ++i) {
		cout << perm[i] << " ";
	}
	cout << endl;

	cout << "-----------------------------------------" << endl;
	
	vector<double> graph_feature_1 = network.Feature(graph1);
	cout << "Graph feature 1:" << endl;
	for (int i = 0; i < graph_feature_1.size(); ++i) {
		cout << graph_feature_1[i] << " ";
	}
	cout << endl;
	cout << "Number of features: " << graph_feature_1.size() << endl;

	cout << "-----------------------------------------" << endl;
	
	vector<double> graph_feature_2 = network.Feature(graph2);
	cout << "Graph feature 2:" << endl;
	for (int i = 0; i < graph_feature_2.size(); ++i) {
		cout << graph_feature_2[i] << " ";
	}
	cout << endl;
	cout << "Number of features: " << graph_feature_2.size() << endl;

	cout << "-----------------------------------------" << endl;

	double difference = 0.0;
	for (int i = 0; i < graph_feature_1.size(); ++i) {
		difference += abs(graph_feature_1[i] - graph_feature_2[i]);
	}	
	cout << "Difference in norm l1: " << difference << endl;

	return 0;
}