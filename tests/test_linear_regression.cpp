// Framework: GraphFlow
// Author: Machine Learning Group of UChicago
// Main Contributor: Hy Truong Son
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../GraphFlow/Vector.h"
#include "../GraphFlow/InnerProduct.h"
#include "../GraphFlow/SquaredLoss.h"
#include "../GraphFlow/GraphFlow.h"

using namespace std;

int main(int argc, char **argv) {
	// +-----------------------------+
	// | Neural Network Architecture |
	// +-----------------------------+

	int nDimensions = 10;

	GraphFlow *graph = new GraphFlow();

	Vector *x = new Vector(nDimensions);
	graph -> add(x, VECTOR);

	Vector *w = new Vector(nDimensions);
	graph -> add(w, VECTOR);

	InnerProduct *predict = new InnerProduct(x, w);
	graph -> add(predict, INNERPRODUCT);

	Vector *target = new Vector(1);

	SquaredLoss *sql = new SquaredLoss(predict, target);
	graph -> add(sql, SQUAREDLOSS);

	// +----------------+
	// | Make some data |
	// +----------------+

	for (int i = 0; i < nDimensions; ++i) {
		x -> value[i] = (double)(i);
		w -> value[i] = 0.0;
	}
	target -> value[0] = 10;

	cout << "Initial input:" << endl;
	for (int i = 0; i < nDimensions; ++i) {
		cout << x -> value[i] << " ";
	}
	cout << endl;

	cout << "Initial weight:" << endl;
	for (int i = 0; i < nDimensions; ++i) {
		cout << w -> value[i] << " ";
	}
	cout << endl;

	cout << "Learning target: " << target -> value[0] << endl;

	// +------------------+
	// | Training process |
	// +------------------+

	graph -> forward();

	cout << "Initial prediction: " << predict -> value[0] << endl;
	cout << "Initial loss: " << sql -> getLoss() << endl;

	double learning_rate = 0.001;
	int nIterations = 20;

	for (int iter = 0; iter < nIterations; ++iter) {
		graph -> forward();
		
		cout << "--- Iteration " << (iter + 1) << " --------------------------------------------" << endl;
		cout << "    Before training loss: " << sql -> getLoss() << endl;

		graph -> backward();

		for (int i = 0; i < nDimensions; ++i) {
			w -> value[i] -= learning_rate * w -> gradient[i];
		}

		graph -> forward();

		cout << "    After training loss: " << sql -> getLoss() << endl;
	}

	// +-----------------------+
	// | Get the trained model |
	// +-----------------------+

	cout << "--- Summary -------------------------------------------------" << endl;
	cout << "Input:" << endl;
	for (int i = 0; i < nDimensions; ++i) {
		cout << x -> value[i] << " ";
	}
	cout << endl;

	cout << "Weight:" << endl;
	for (int i = 0; i < nDimensions; ++i) {
		cout << w -> value[i] << " ";
	}
	cout << endl;

	graph -> forward();

	cout << "Prediction: " << predict -> value[0] << endl;
	cout << "Squared loss: " << sql -> getLoss() << endl;

	return 0;
}