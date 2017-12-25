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
#include "../GraphFlow/SumGradients.h"
#include "../GraphFlow/GraphFlow.h"

using namespace std;

// +-----------+
// | Constants |
// +-----------+

const int width = 32;
const int height = 32;
const int nInputs = width * height;
const int nInputChanels = 3;
const int nOutputs = 10;

const int nInputBytes = nInputs * nInputChanels;

// +---------------------+
// | Training Parameters |
// +---------------------+

const int Epochs = 100;
const double lambda = 1e-4;
const int nBatch = 50;

const double initial_lr = 1e-2;
const double final_lr = 1e-5;
const int tau = 100000;
double learning_rate;
int nIterations;

const int nTrain = 50000;
const int nTest = 10000;
const bool printOutput = false;

// +----------+
// | Datasets |
// +----------+

const int nData = 10000;
const int nTrainBatch = 5;
const int nTestBatch = 1;

const string train_fn[nTrainBatch] = {
	"CIFAR-10/cifar-10-batches-bin/data_batch_1.bin",
	"CIFAR-10/cifar-10-batches-bin/data_batch_2.bin",
	"CIFAR-10/cifar-10-batches-bin/data_batch_3.bin",
	"CIFAR-10/cifar-10-batches-bin/data_batch_4.bin",
	"CIFAR-10/cifar-10-batches-bin/data_batch_5.bin"
};

const string test_fn[nTestBatch] = {
	"CIFAR-10/cifar-10-batches-bin/test_batch.bin",
};

const string model_fn = "CNN_CIFAR10_large.model";
const string report_fn = "CNN_CIFAR10_large.report";

ofstream report;

// +--------------+
// | Data storage |
// +--------------+

struct Example {
	double image[width][height][nInputChanels];
	int label;
};

vector < Example* > train;
vector < Example* > test;

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

// Convolution layer 3
Tensor4D *filter3;
Matrix *bias3;
Tensor3D *conv3;
Vector *conv3_relu;
Tensor3D *conv3_reshape;

// Pooling layer 3
Tensor3D *pool3;

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
	filter1 = new Tensor4D(5, 5, input -> nDepth, 32);
	bias1 = new Matrix(filter1 -> nChanels1, filter1 -> nChanels2);
	conv1 = new Conv2D(input, filter1, bias1, 1, 2);
	conv1_relu = new LeakyReLU(conv1);
	conv1_reshape = new Reshape3D(conv1_relu, conv1 -> nRows, conv1 -> nColumns, conv1 -> nDepth);

	// Pooling layer 1
	pool1 = new MaxPool2D(conv1_reshape, 2, 2);

	// Convolution layer 2
	filter2 = new Tensor4D(5, 5, pool1 -> nDepth, 64);
	bias2 = new Matrix(filter2 -> nChanels1, filter2 -> nChanels2);
	conv2 = new Conv2D(pool1, filter2, bias2, 1, 2);
	conv2_relu = new LeakyReLU(conv2);
	conv2_reshape = new Reshape3D(conv2_relu, conv2 -> nRows, conv2 -> nColumns, conv2 -> nDepth);

	// Pooling layer 2
	pool2 = new MaxPool2D(conv2_reshape, 2, 2);

	// Convolution layer 3
	filter3 = new Tensor4D(5, 5, pool2 -> nDepth, 128);
	bias3 = new Matrix(filter3 -> nChanels1, filter3 -> nChanels2);
	conv3 = new Conv2D(pool2, filter3, bias3, 1, 2);
	conv3_relu = new LeakyReLU(conv3);
	conv3_reshape = new Reshape3D(conv3_relu, conv3 -> nRows, conv3 -> nColumns, conv3 -> nDepth);

	// Pooling layer 3
	pool3 = new MaxPool2D(conv3_reshape, 2, 2);

	// Fully-connected layer - Softmax
	W = new Matrix(nOutputs, pool3 -> size);
	linear = new MatVecMul(W, pool3);
	bias = new Vector(nOutputs);
	score = new Add(linear, bias);
	target = new Vector(1);
	logLoss = new LogLoss(score, target);

	// L2-Regularization
	l2reg = new L2Regularization(lambda);
	l2reg -> add(filter1);
	l2reg -> add(filter2);
	l2reg -> add(filter3);
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

	graph -> add(filter3, TENSOR4D);
	graph -> add(bias3, MATRIX);
	graph -> add(conv3, CONV2D);
	graph -> add(conv3_relu, LEAKYRELU);
	graph -> add(conv3_reshape, RESHAPE3D);
	graph -> add(pool3, MAXPOOL2D);

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
	graph -> Xavier_init(filter3);
	graph -> Xavier_init(bias3);
	graph -> Xavier_init(W);
	graph -> Xavier_init(bias);

	// Stochastic Gradient Descent
	sgd = new SGD();
	sgd -> add(filter1);
	sgd -> add(bias1);
	sgd -> add(filter2);
	sgd -> add(bias2);
	sgd -> add(filter3);
	sgd -> add(bias3);
	sgd -> add(W);
	sgd -> add(bias);
}

// +--------------------+
// | Loading train data |
// +--------------------+

void load_train_data() {
	train.clear();
	for (int i = 0; i < nTrainBatch; ++i) {
		string filename = train_fn[i];
		cout << "Load train data '" << filename << "'" << endl;

		ifstream file(filename.c_str(), ios::in | ios::binary);

		char number;
		for (int sample = 0; sample < nData; ++sample) {
			Example *example = new Example();

			file.read(&number, sizeof(char));
			example -> label = int(number);

			for (int d = 0; d < nInputChanels; ++d) {
				for (int r = 0; r < height; ++r) {
					for (int c = 0; c < width; ++c) {
						file.read(&number, sizeof(char));
						example -> image[r][c][d] = double(number);
					}
				}
			}

			train.push_back(example);
		}

		file.close();
	}
	cout << "Number of training examples: " << train.size() << endl;

	/*
	for (int d = 0; d < nInputChanels; ++d) {
		for (int r = 0; r < height; ++r) {
			for (int c = 0; c < width; ++c) {
				cout << train[1] -> image[r][c][d] << " ";
			}
			cout << endl;
		}
	}
	*/
}

// +-------------------+
// | Loading test data |
// +-------------------+

void load_test_data() {
	test.clear();
	for (int i = 0; i < nTestBatch; ++i) {
		string filename = test_fn[i];
		cout << "Load test data '" << filename << "'" << endl;

		ifstream file(filename.c_str(), ios::in | ios::binary);

		char number;
		for (int sample = 0; sample < nData; ++sample) {
			Example *example = new Example();

			file.read(&number, sizeof(char));
			example -> label = int(number);

			for (int d = 0; d < nInputChanels; ++d) {
				for (int r = 0; r < height; ++r) {
					for (int c = 0; c < width; ++c) {
						file.read(&number, sizeof(char));
						example -> image[r][c][d] = double(number);
					}
				}
			}

			test.push_back(example);
		}

		file.close();
	}
	cout << "Number of testing examples: " << test.size() << endl;
}

// +---------------------------------+
// | Finding the maximum probability |
// +---------------------------------+

int find_max(Vector *vect) {
	int index = 0;
	for (int i = 1; i < vect -> size; ++i) {
		if (vect -> value[i] > vect -> value[index]) {
			index = i;
		}
	}
	return index;
}

// +------------------------+
// | Save the model to file |
// +------------------------+

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

// +--------------------------+
// | Load the model from file |
// +--------------------------+

void load_model() {
	ifstream file(model_fn.c_str(), ios::in);

	for (int i = 0; i < sgd -> params.size(); ++i) {
		for (int j = 0; j < sgd -> params[i] -> size; ++j) {
			file >> sgd -> params[i] -> value[j];
		}
	}

	file.close();
}

// +------------------+
// | Training process |
// +------------------+

void training() {
	SumGradients *cache = new SumGradients();
	for (int i = 0; i < sgd -> params.size(); ++i) {
		cache -> add(sgd -> params[i]);
	}

	int start = 0;
	while (start + nBatch - 1 < nTrain) {
		int finish = start + nBatch - 1;

		++nIterations;
		if (nIterations <= tau) {
			double p = double(nIterations) / double(tau);
			learning_rate = initial_lr * (1.0 - p) + p * final_lr;
		} 

		if (printOutput) {
			cout << "Minibatch of example " << (start + 1) << " - example " << (finish + 1) << endl;
			cout << "Learning rate: " << learning_rate << endl;
		}

		cache -> reset_sum_gradients();

		for (int sample = start; sample <= finish; ++sample) {
			for (int r = 0; r < height; ++r) {
				for (int c = 0; c < width; ++c) {
					for (int d = 0; d < nInputChanels; ++d) {
						int i = input -> index(r, c, d);
						input -> value[i] = train[sample] -> image[r][c][d];
					}
				}
			}		

			target -> value[0] = train[sample] -> label;

			graph -> forward();
			graph -> backward();

			cache -> cache_gradients();
		}

		cache -> get_sum_gradients();
		sgd -> Learn(learning_rate, nBatch);

		if ((finish + 1) % 1000 == 0) {
			cout << "Done training " << (finish + 1) << " examples" << endl;
			save_model();
		}
		start = finish + 1;
	}

	save_model();
}

// +-----------------+
// | Testing process |
// +-----------------+

void testing(int epoch) {
	int nCorrect = 0;
	for (int sample = 0; sample < nTest; ++sample) {
		// Load input
		for (int r = 0; r < height; ++r) {
			for (int c = 0; c < width; ++c) {
				for (int d = 0; d < nInputChanels; ++d) {
					int i = input -> index(r, c, d);
					input -> value[i] = test[sample] -> image[r][c][d];
				}
			}
		}		

		graph -> forward();
		int predict_label = find_max(score);

		if (predict_label == test[sample] -> label) {
			++nCorrect;
		}

		if ((sample + 1) % 1000 == 0) {
			cout << "Done testing " << (sample + 1) << " examples" << endl;
		}
	}
	double accuracy = (double)(nCorrect) / nTest * 100.0;
	cout << "Epoch " << epoch << ": Accuracy = " << accuracy << endl;

	report.open(report_fn.c_str(), ios::app);
	report << "Epoch " << epoch << ": Accuracy = " << accuracy << endl;
	report.close();
}

// +----------------------------+
// | Training / Testing Process |
// +----------------------------+

void process() {
	learning_rate = initial_lr;
	int nIterations = 0;

	report.open(report_fn.c_str(), ios::out);
	report.close();

	for (int epoch = 0; epoch < Epochs; ++epoch) {
		cout << "--- Epoch " << (epoch + 1) << " -----------------------------" << endl;

		training();
		testing(epoch);
	}
}

// +--------------+
// | Main Program |
// +--------------+

int main(int argc, char **argv) {
	network_architecture();
	load_train_data();
	load_test_data();
	process();
	return 0;
}