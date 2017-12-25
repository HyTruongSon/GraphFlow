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
#include "../GraphFlow/Adam.h"
#include "../GraphFlow/AdaDelta.h"
#include "../GraphFlow/GraphFlow.h"

using namespace std;

// +-----------+
// | Constants |
// +-----------+

const int width = 28;
const int height = 28;
const int nInputs = width * height;
const int nInputChanels = 1;
const int nOutputs = 10;

// +---------------------+
// | Training Parameters |
// +---------------------+

const int Epochs = 10;
const double learning_rate = 1e-2;
const double lambda = 0.001;

const bool printOutput = false;

// +---------+
// | Dataset |
// +---------+

const int nTrain = 60000;
const int nTest = 10000;

const string train_fn = "MNIST/train-images.idx3-ubyte";
const string trainLabel_fn = "MNIST/train-labels.idx1-ubyte";
const string test_fn = "MNIST/t10k-images.idx3-ubyte";
const string testLabel_fn = "MNIST/t10k-labels.idx1-ubyte";

const string model_fn = "model_CNN_MNIST_maxpool.dat";

double **train;
int *trainLabel;
double **test;
int *testLabel;

// +------------------------------+
// | Computation Graph Components |
// +------------------------------+

// Dynamic computation graph
GraphFlow *graph;

// Input as a 3D tensor
Tensor3D *input;

// Convolution layer 1
Tensor4D *filter1;
Matrix *bias1;
Tensor3D *conv1;
Vector *conv1_relu;
Tensor3D *conv1_reshape;

// Pooling layer 1
Tensor3D *pool1;

// Convolution layer 2
Tensor4D *filter2;
Matrix *bias2;
Tensor3D *conv2;
Vector *conv2_relu;
Tensor3D *conv2_reshape;

// Pooling layer 2
Tensor3D *pool2;

// Fully-connected layer - Softmax
Matrix *W;
Vector *bias;
Vector *linear;
Vector *score;
Vector *target;
LogLoss *logLoss;

// L2-Regularization
L2Regularization *l2reg;

// Stochastic Gradient Descent
SGD *sgd;

// +-----------------------------+
// | Neural Network Architecture |
// +-----------------------------+

void network_architecture() {
	// Input as a 3D tensor
	input = new Tensor3D(height, width, nInputChanels);

	// Convolution layer 1
	filter1 = new Tensor4D(5, 5, input -> nDepth, 8);
	bias1 = new Matrix(filter1 -> nChanels1, filter1 -> nChanels2);
	conv1 = new Conv2D(input, filter1, bias1, 1, 2);
	conv1_relu = new LeakyReLU(conv1);
	conv1_reshape = new Reshape3D(conv1_relu, conv1 -> nRows, conv1 -> nColumns, conv1 -> nDepth);

	// Pooling layer 1
	pool1 = new MaxPool2D(conv1_reshape, 2, 2);

	// Convolution layer 2
	filter2 = new Tensor4D(5, 5, pool1 -> nDepth, 16);
	bias2 = new Matrix(filter2 -> nChanels1, filter2 -> nChanels2);
	conv2 = new Conv2D(pool1, filter2, bias2, 1, 2);
	conv2_relu = new LeakyReLU(conv2);
	conv2_reshape = new Reshape3D(conv2_relu, conv2 -> nRows, conv2 -> nColumns, conv2 -> nDepth);

	// Pooling layer 2
	pool2 = new MaxPool2D(conv2_reshape, 2, 2);

	// Fully-connected layer - Softmax
	W = new Matrix(nOutputs, pool2 -> size);
	linear = new MatVecMul(W, pool2);
	bias = new Vector(nOutputs);
	score = new Add(linear, bias);
	target = new Vector(1);
	logLoss = new LogLoss(score, target);

	// L2-Regularization
	l2reg = new L2Regularization(lambda);
	l2reg -> add(filter1);
	l2reg -> add(filter2);
	l2reg -> add(W);

	// Computation Graph
	graph = new GraphFlow();
	graph -> clear();

	graph -> add(input, TENSOR3D);

	graph -> add(filter1, TENSOR4D);
	graph -> add(bias1, MATRIX);
	graph -> add(conv1, CONV2D);
	graph -> add(conv1_relu, LEAKYRELU);
	graph -> add(conv1_reshape, RESHAPE3D);
	graph -> add(pool1, MAXPOOL2D);

	graph -> add(filter2, TENSOR4D);
	graph -> add(bias2, MATRIX);
	graph -> add(conv2, CONV2D);
	graph -> add(conv2_relu, LEAKYRELU);
	graph -> add(conv2_reshape, RESHAPE3D);
	graph -> add(pool2, MAXPOOL2D);

	graph -> add(W, MATRIX);
	graph -> add(linear, MATVECMUL);
	graph -> add(bias, VECTOR);
	graph -> add(score, ADD);

	graph -> add(logLoss, LOGLOSS);
	graph -> add(l2reg, L2REGULARIZATION);

	// Weights initialization
	graph -> Xavier_init(filter1);
	graph -> Xavier_init(bias1);
	graph -> Xavier_init(filter2);
	graph -> Xavier_init(bias2);
	graph -> Xavier_init(W);
	graph -> Xavier_init(bias);

	// Stochastic Gradient Descent
	sgd = new SGD();
	sgd -> add(filter1);
	sgd -> add(bias1);
	sgd -> add(filter2);
	sgd -> add(bias2);
	sgd -> add(W);
	sgd -> add(bias);
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

int find_max(Vector *vect) {
	int index = 0;
	for (int i = 1; i < vect -> size; ++i) {
		if (vect -> value[i] > vect -> value[index]) {
			index = i;
		}
	}
	return index;
}

void save_model() {
	ofstream file(model_fn.c_str(), ios::out);

	for (int i = 0; i < sgd -> params.size(); ++i) {
		for (int j = 0; j < sgd -> params[i] -> size; ++j) {
			file << sgd -> params[i] -> value[j] << " ";
		}
		file << endl;
	}

	file.close();
}

void load_model() {
	ifstream file(model_fn.c_str(), ios::in);

	for (int i = 0; i < sgd -> params.size(); ++i) {
		for (int j = 0; j < sgd -> params[i] -> size; ++j) {
			file >> sgd -> params[i] -> value[j];
		}
	}

	file.close();
}

void training() {
	for (int sample = 0; sample < nTrain; ++sample) {
		if (printOutput) {
			cout << endl << "Example " << (sample + 1) << ":" << endl;
		}

		// Load input
		for (int r = 0; r < height; ++r) {
			for (int c = 0; c < width; ++c) {
				int i = input -> index(r, c, 0);
				input -> value[i] = train[sample][r * width + c];
			}
		}		

		// Load target
		target -> value[0] = trainLabel[sample];

		double totalLoss;

		graph -> forward();

		if (printOutput) {
			cout << "    Log-loss (before): " << logLoss -> getLoss() << endl;
			cout << "    L2-regularization (before): " << l2reg -> getLoss() << endl;
		}

		graph -> backward();
		sgd -> Learn(learning_rate);

		if (printOutput) {
			graph -> forward();
			cout << "    Log-loss (after): " << logLoss -> getLoss() << endl;
			cout << "    L2-regularization (after): " << l2reg -> getLoss() << endl;
		}

		if ((sample + 1) % 1000 == 0) {
			cout << "Done training " << (sample + 1) << " examples" << endl;
			save_model();
		}
	}

	save_model();
}

void testing() {
	// Load the model from file
	load_model();

	int nCorrect = 0;
	for (int sample = 0; sample < nTest; ++sample) {
		// Load input
		for (int r = 0; r < height; ++r) {
			for (int c = 0; c < width; ++c) {
				int i = input -> index(r, c, 0);
				input -> value[i] = test[sample][r * width + c];
			}
		}		

		graph -> forward();
		int predict_label = find_max(score);

		if (predict_label == testLabel[sample]) {
			++nCorrect;
		}

		if ((sample + 1) % 1000 == 0) {
			cout << "Done testing " << (sample + 1) << " examples" << endl;
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
	load_data();
	process();
	return 0;
}