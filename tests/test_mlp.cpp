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

#include "../GraphFlow/SGD.h"
#include "../GraphFlow/Momentum.h"
#include "../GraphFlow/GraphFlow.h"

using namespace std;

// +-----------+
// | Constants |
// +-----------+

const int nInputs = 784;
const int nHiddens = 128;
const int nOutputs = 10;

// +---------------------+
// | Training Parameters |
// +---------------------+

const int Epochs = 10;
const int nIterations = 20;
const double learning_rate = 1e-2;
const double epsilon = 1e-3;

// +---------+
// | Dataset |
// +---------+

const int nTrain = 60000;
const int nTest = 10000;

const string train_fn = "MNIST/train-images.idx3-ubyte";
const string trainLabel_fn = "MNIST/train-labels.idx1-ubyte";
const string test_fn = "MNIST/t10k-images.idx3-ubyte";
const string testLabel_fn = "MNIST/t10k-labels.idx1-ubyte";

double **train;
int *trainLabel;
double **test;
int *testLabel;

// +------------------------------+
// | Computation Graph Components |
// +------------------------------+

GraphFlow *graph;
Vector *input;
Matrix *W1;
Vector *hidden_sum;
Vector *hidden_act;
Matrix *W2;
Vector *output_sum;
Vector *predict;
Vector *target;
SquaredLoss *sql;
Momentum *sgd;

// +-----------------------------+
// | Neural Network Architecture |
// +-----------------------------+

void network_architecture() {
	graph = new GraphFlow();
	
	input = new Vector(nInputs);
	graph -> add(input, VECTOR);

	W1 = new Matrix(nHiddens, nInputs);
	graph -> add(W1, MATRIX);

	hidden_sum = new MatVecMul(W1, input);
	graph -> add(hidden_sum, MATVECMUL);

	hidden_act = new Sigmoid(hidden_sum);
	graph -> add(hidden_act, SIGMOID);

	W2 = new Matrix(nOutputs, nHiddens);
	graph -> add(W2, MATRIX);

	output_sum = new MatVecMul(W2, hidden_act);
	graph -> add(output_sum, MATVECMUL);

	predict = new Sigmoid(output_sum);
	graph -> add(predict, SIGMOID);

	target = new Vector(nOutputs);

	sql = new SquaredLoss(predict, target);
	graph -> add(sql, SQUAREDLOSS);

	sgd = new Momentum();
	sgd -> add(W1);
	sgd -> add(W2);
}

// +------------------------+
// | Weights Initialization |
// +------------------------+

void weights_initialization() {
	graph -> uniform_init(W1);
	graph -> uniform_init(W2);
}

// +------------------+
// | Loading datasets |
// +------------------+

void load_images(const string filename, double **matrix, int nSamples, int dimension) {
	ifstream file(filename.c_str(), ios::in | ios::binary);
	char number;
	for (int i = 0; i < 16; ++i) {
		file.read(&number, sizeof(char));
	}
	for (int sample = 0; sample < nSamples; ++sample) {
		for (int i = 0; i < dimension; ++i) {
			file.read(&number, sizeof(char));
			if (number == 0) {
				matrix[sample][i] = 0.0;
			} else {
				matrix[sample][i] = 1.0;
			}
		}
	}
	file.close();
}

void load_labels(const string filename, int *vector, int nSamples) {
	ifstream file(filename.c_str(), ios::in | ios::binary);
	char number;
	for (int i = 0; i < 8; ++i) {
		file.read(&number, sizeof(char));
	}
	for (int sample = 0; sample < nSamples; ++sample) {
		file.read(&number, sizeof(char));
		vector[sample] = (int)(number);
	}
	file.close();
}

void load_data() {
	train = new double* [nTrain];
	trainLabel = new int [nTrain];
	test = new double* [nTest];
	testLabel = new int [nTest];

	for (int i = 0; i < nTrain; ++i) {
		train[i] = new double [nInputs];
	}

	for (int i = 0; i < nTest; ++i) {
		test[i] = new double [nInputs];
	}

	load_images(train_fn, train, nTrain, nInputs);
	load_labels(trainLabel_fn, trainLabel, nTrain);

	load_images(test_fn, test, nTest, nInputs);
	load_labels(testLabel_fn, testLabel, nTest);
}

// +------------------------------+
// | Training and Testing Process |
// +------------------------------+

void training() {
	for (int sample = 0; sample < nTrain; ++sample) {
		// cout << "Example " << (sample + 1) << endl;

		// Load input
		for (int i = 0; i < nInputs; ++i) {
			input -> value[i] = train[sample][i];
		}		

		// Load target
		for (int i = 0; i < nOutputs; ++i) {
			target -> value[i] = 0.0;
		}
		target -> value[trainLabel[sample]] = 1.0;

		// Gradient descent
		// graph -> forward();
		// cout << "    Before training loss: " << sql -> getLoss() << endl;

		for (int iter = 0; iter < nIterations; ++iter) {
			graph -> forward();

			if (sql -> getLoss() < epsilon) {
				break;
			}

			graph -> backward();


			sgd -> Learn(learning_rate);

			/*
			for (int i = 0; i < W1 -> size; ++i) {
				W1 -> value[i] -= learning_rate * W1 -> gradient[i];
			}

			for (int i = 0; i < W2 -> size; ++i) {
				W2 -> value[i] -= learning_rate * W2 -> gradient[i];
			}
			*/
		}

		// graph -> forward();
		// cout << "    After training loss: " << sql -> getLoss() << endl;

		if ((sample + 1) % 1000 == 0) {
			cout << "Done training " << (sample + 1) << " examples" << endl;
		}
	}
}

int find_max(Vector *vect) {
	int index = 0;
	for (int i = 1; i < vect -> size; ++i) {
		if (vect -> value[i] > vect -> value[index]) {
			index = i;
		}
	}
	return index;
}

void testing() {
	int nCorrect = 0;
	for (int sample = 0; sample < nTest; ++sample) {
		// cout << "Example " << (sample + 1) << endl;

		// Load input
		for (int i = 0; i < nInputs; ++i) {
			input -> value[i] = test[sample][i];
		}

		graph -> forward();
		int predict_label = find_max(predict);

		if (predict_label == testLabel[sample]) {
			++nCorrect;
		}
	}
	double accuracy = (double)(nCorrect) / nTest * 100.0;
	cout << endl << "Accuracy: " << accuracy << endl;
}

void process() {
	for (int epoch = 0; epoch < Epochs; ++epoch) {
		cout << "--- Epoch " << (epoch + 1) << " -----------------------------" << endl;

		training();
		testing();
	}
}

// +--------------+
// | Main Program |
// +--------------+

int main(int argc, char **argv) {
	network_architecture();
	weights_initialization();
	load_data();
	process();
	return 0;
}