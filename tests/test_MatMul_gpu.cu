// Framework: GraphFlow
// Author: Machine Learning Group of UChicago
// Main Contributor: Hy Truong Son
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#include <iostream>
#include <cstring>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <ctime>
#include <sys/time.h>

#include "../GraphFlow_gpu/GraphFlow.h"

using namespace std;

const int A_nRows = 1600;
const int A_nColumns = 720;

const int B_nRows = 720;
const int B_nColumns = 40; 

const double RANDOM_SEED = 123456789;

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

int get_random(int number) {
	return rand() % number;
}

int main(int argc, char **argv) {
	long int start, end;

	srand(RANDOM_SEED);

	Matrix *A = new Matrix(A_nRows, A_nColumns);
	Matrix *B = new Matrix(B_nRows, B_nColumns);

	for (int i = 0; i < A -> size; ++i) {
		A -> value[i] = get_random(100);
	}

	for (int i = 0; i < B -> size; ++i) {
		B -> value[i] = get_random(100);
	}

	MatMul_gpu *obj = new MatMul_gpu(A, B);
	
	// Another way to initialize obj
	// MatMul_gpu *obj = new MatMul_gpu(A_nRows, A_nColumns, B_nRows, B_nColumns);
	// obj -> setParameter(A, B);

	MatMul *ground_truth = new MatMul(A, B);

	// Forward

	time_ms(start);
	obj -> forward();
	time_ms(end);

	cout << "GPU forward time: " << difftime_ms(end, start) << " ms" << endl;

	time_ms(start);
	ground_truth -> forward();
	time_ms(end);

	cout << "CPU forward time: " << difftime_ms(end, start) << " ms" << endl;

	assert(obj -> size == ground_truth -> size);
	assert(obj -> nRows == ground_truth -> nRows);
	assert(obj -> nColumns == ground_truth -> nColumns);

	double forward_error = 0.0;
	for (int i = 0; i < obj -> size; ++i) {
		forward_error += abs(obj -> value[i] - ground_truth -> value[i]);
	}

	cout << "Forward error: " << forward_error << endl << endl;

	// Backward

	double *A_gradient = new double [A -> size];
	double *B_gradient = new double [B -> size];

	double *A_init_gradient = new double [A -> size];
	double *B_init_gradient = new double [B -> size];

	for (int i = 0; i < obj -> size; ++i) {
		obj -> gradient[i] = get_random(100);
		ground_truth -> gradient[i] = obj -> gradient[i];
	}

	for (int i = 0; i < A -> size; ++i) {
		A_init_gradient[i] = get_random(100);
		A -> gradient[i] = A_init_gradient[i];
	}

	for (int i = 0; i < B -> size; ++i) {
		B_init_gradient[i] = get_random(100);
		B -> gradient[i] = B_init_gradient[i];
	}

	time_ms(start);
	obj -> backward();
	time_ms(end);

	cout << "GPU backward time: " << difftime_ms(end, start) << " ms" << endl;

	for (int i = 0; i < A -> size; ++i) {
		A_gradient[i] = A -> gradient[i];
	}

	for (int i = 0; i < B -> size; ++i) {
		B_gradient[i] = B -> gradient[i];
	}

	for (int i = 0; i < A -> size; ++i) {
		A -> gradient[i] = A_init_gradient[i];
	}

	for (int i = 0; i < B -> size; ++i) {
		B -> gradient[i] = B_init_gradient[i];
	}

	time_ms(start);
	ground_truth -> backward();
	time_ms(end);

	cout << "CPU backward time: " << difftime_ms(end, start) << " ms" << endl;

	double backward_error_A = 0.0;
	for (int i = 0; i < A -> size; ++i) {
		backward_error_A += abs(A_gradient[i] - A -> gradient[i]);
	}

	double backward_error_B = 0.0;
	for (int i = 0; i < B -> size; ++i) {
		backward_error_B += abs(B_gradient[i] - B -> gradient[i]);
	}

	cout << "Backward error for A: " << backward_error_A << endl;
	cout << "Backward error for B: " << backward_error_B << endl << endl;

	return 0;
}
