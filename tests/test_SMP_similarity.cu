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
#include <sys/time.h>

#include "../GraphFlow_gpu_32bit/SMP_omega.h"
#include "../GraphFlow_gpu_32bit/SMP_omega_gpu.h"

using namespace std;

const int nVertices = 20;
const int max_receptive_field = 15;
const int nChanels = 10;
const int nLevels = 6;
const int nFeatures = 6;
const int nDepth = 5;

const int RANDOM_SEED = 123456789;

SMP_omega *network_cpu; 
SMP_omega_gpu *network_gpu;

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

int main(int argc, char **argv) {
	long int start;
	long int end;

	srand(RANDOM_SEED);
	network_cpu = new SMP_omega(nVertices, max_receptive_field, nLevels, nChanels, nFeatures, nDepth, false);

	srand(RANDOM_SEED);
	network_gpu = new SMP_omega_gpu(nVertices, max_receptive_field, nLevels, nChanels, nFeatures, nDepth, false);

	cout << "Number of vertices: " << nVertices << endl;
	cout << "Number of chanels (at level 0): " << nChanels << endl;
	cout << "Number of levels: " << nLevels << endl;
	cout << "Number of vertex features: " << nFeatures << endl;
	cout << "Depth of the Histogram-alignment Weisfeiler-Lehman: " << nDepth << endl;

	// Create a Erdos Renyi random graph
	DenseGraph *graph = Erdos_Renyi(nVertices, nFeatures);

	cout << "-----------------------------------------" << endl;

	cout << "Graph" << endl;
	print(graph);

	cout << "-----------------------------------------" << endl;
	
	time_ms(start);
	vector<float> graph_feature_gpu = network_gpu -> Feature(graph);
	time_ms(end);

	cout << "GPU forward time: " << difftime_ms(end, start) << " ms" << endl;

	cout << "Graph feature (GPU):" << endl;
	for (int i = 0; i < graph_feature_gpu.size(); ++i) {
		cout << graph_feature_gpu[i] << " ";
	}
	cout << endl;
	cout << "Number of features (GPU): " << graph_feature_gpu.size() << endl;

	cout << "-----------------------------------------" << endl;

	time_ms(start);
	vector<float> graph_feature_cpu = network_cpu -> Feature(graph);
	time_ms(end);

	cout << "CPU forward time: " << difftime_ms(end, start) << " ms" << endl;

	cout << "Graph feature (CPU):" << endl;
	for (int i = 0; i < graph_feature_cpu.size(); ++i) {
		cout << graph_feature_cpu[i] << " ";
	}
	cout << endl;
	cout << "Number of features (CPU): " << graph_feature_cpu.size() << endl;

	cout << "-----------------------------------------" << endl;

	float difference = 0.0;
	for (int i = 0; i < graph_feature_cpu.size(); ++i) {
		difference += abs(graph_feature_cpu[i] - graph_feature_gpu[i]);
	}	
	cout << "Difference in norm l1: " << difference << endl;

	return 0;
}