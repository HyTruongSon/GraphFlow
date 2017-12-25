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
const string test_fn = "MNIST/t10k-images.idx3-ubyte";
const string model_fn = "autoencoder-model.dat";

double **train;
double **test;

// +------------------------------+
// | Computation Graph Components |
// +------------------------------+

GraphFlow *graph;
Vector *input;
Matrix *W;
Vector *hidden_sum;
Vector *hidden_act;
Matrix *W_transpose;
Vector *output_sum;
Vector *predict;
Vector *target;
SquaredLoss *sql;
SGD *sgd;

// +-----------------------------+
// | Neural Network Architecture |
// +-----------------------------+

void network_architecture() {
	graph = new GraphFlow();
	
	input = new Vector(nInputs);
	graph -> add(input, VECTOR);

	W = new Matrix(nHiddens, nInputs);
	graph -> add(W, MATRIX);

	hidden_sum = new MatVecMul(W, input);
	graph -> add(hidden_sum, MATVECMUL);

	hidden_act = new Sigmoid(hidden_sum);
	graph -> add(hidden_act, SIGMOID);

	W_transpose = new Transpose(W);
	graph -> add(W_transpose, TRANSPOSE);

	output_sum = new MatVecMul(W_transpose, hidden_act);
	graph -> add(output_sum, MATVECMUL);

	predict = new Sigmoid(output_sum);
	graph -> add(predict, SIGMOID);

	target = new Vector(nInputs);

	sql = new SquaredLoss(predict, target);
	graph -> add(sql, SQUAREDLOSS);

	sgd = new SGD();
	sgd -> add(W);
}

// +------------------------+
// | Weights Initialization |
// +------------------------+

void weights_initialization() {
	graph -> uniform_init(W);
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

void load_data() {
	train = new double* [nTrain];
	test = new double* [nTest];

	for (int i = 0; i < nTrain; ++i) {
		train[i] = new double [nInputs];
	}

	for (int i = 0; i < nTest; ++i) {
		test[i] = new double [nInputs];
	}

	load_images(train_fn, train, nTrain, nInputs);
	load_images(test_fn, test, nTest, nInputs);
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
		for (int i = 0; i < nInputs; ++i) {
			target -> value[i] = input -> value[i];
		}

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

void testing() {
	int average_error = 0;
	for (int sample = 0; sample < nTest; ++sample) {
		// cout << "Example " << (sample + 1) << endl;

		// Load input
		for (int i = 0; i < nInputs; ++i) {
			input -> value[i] = test[sample][i];
		}

		graph -> forward();
		
		for (int i = 0; i < predict -> size; ++i) {
			if ((predict -> value[i] >= 0.5) && (input -> value[i] < 0.5)) {
				++average_error;
			}
			if ((predict -> value[i] < 0.5) && (input -> value[i] >= 0.5)) {
				++average_error;
			}
		}
	}

	cout << "Average error: " << (double)(average_error) / (nTest * input -> size) << " wrong bits" << endl;
}

void save_model() {
	cout << "Save the autoencoder to file " << model_fn << endl;
	ofstream file(model_fn.c_str(), ios::out);
	for (int i = 0; i < nHiddens; ++i) {
		for (int j = 0; j < nInputs; ++j) {
			file << W -> value[i * nInputs + j] << " ";
		}
		file << endl;
	}
	file.close();
}

void process() {
	for (int epoch = 0; epoch < Epochs; ++epoch) {
		cout << "--- Epoch " << (epoch + 1) << " -----------------------------" << endl;

		training();
		testing();
		save_model();
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
	save_model();
	return 0;
}