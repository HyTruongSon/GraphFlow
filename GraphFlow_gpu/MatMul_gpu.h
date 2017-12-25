// Framework: GraphFlow
// Class: MatMul_gpu
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __MATMUL_GPU_H_INCLUDED__
#define __MATMUL_GPU_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <assert.h>

#include "Matrix.h"

const int MATMUL_GPU_THREADS = 512;
const int MATMUL_GPU_BLOCK_SIZE = 22;

// +----------------------------------+
// | Kernel Function For Forward Pass |
// +----------------------------------+

__global__ void Matrix_Multiplication_GPU(double *A, double *B, double *C, int A_nRows, int A_nColumns, int B_nRows, int B_nColumns) {
	__shared__ double shared_A[MATMUL_GPU_BLOCK_SIZE][MATMUL_GPU_BLOCK_SIZE];
	__shared__ double shared_B[MATMUL_GPU_BLOCK_SIZE][MATMUL_GPU_BLOCK_SIZE];

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row = MATMUL_GPU_BLOCK_SIZE * blockIdx.x + tx;
	int column = MATMUL_GPU_BLOCK_SIZE * blockIdx.y + ty;

	int nStrides = (A_nColumns - 1) / MATMUL_GPU_BLOCK_SIZE + 1;

	double inner_product = 0.0;

	for (int stride = 0; stride < nStrides; ++stride) {
		if ((row < A_nRows) && (stride * MATMUL_GPU_BLOCK_SIZE + ty < A_nColumns)) {
			shared_A[tx][ty] = A[row * A_nColumns + stride * MATMUL_GPU_BLOCK_SIZE + ty];
		} else {
			shared_A[tx][ty] = 0.0;
		}

		if ((column < B_nColumns) && (stride * MATMUL_GPU_BLOCK_SIZE + tx < B_nRows)) {
			shared_B[tx][ty] = B[(stride * MATMUL_GPU_BLOCK_SIZE + tx) * B_nColumns + column];
		} else {
			shared_B[tx][ty] = 0.0;
		}

		__syncthreads();
		for (int i = 0; i < MATMUL_GPU_BLOCK_SIZE; ++i) {
			inner_product += shared_A[tx][i] * shared_B[i][ty];
		}
		__syncthreads();
	}

	if ((row < A_nRows) && (column < B_nColumns)) {
		C[row * B_nColumns + column] += inner_product;
	}
}

// +---------------------------------------------------------------+
// | Kernel Function For Backward Pass (Gradient For First Matrix) |
// +---------------------------------------------------------------+

__global__ void MatMul_backward_first(double *A_gradient, double *B, double *C_gradient, int A_nRows, int A_nColumns, int B_nRows, int B_nColumns) {
	__shared__ int C_nColumns;

	C_nColumns = B_nColumns;

	int global_threadId = blockDim.x * blockIdx.x + threadIdx.x;

	if (global_threadId < A_nRows * A_nColumns) {
		int i = global_threadId / A_nColumns;
		int k = global_threadId % A_nColumns;

		double sum = 0.0;
		for (int j = 0; j < C_nColumns; ++j) {
			sum += C_gradient[i * C_nColumns + j] * B[k * B_nColumns + j];
		}
		A_gradient[i * A_nColumns + k] += sum;
	}
} 

// +----------------------------------------------------------------+
// | Kernel Function For Backward Pass (Gradient For Second Matrix) |
// +----------------------------------------------------------------+

__global__ void MatMul_backward_second(double *A, double *B_gradient, double *C_gradient, int A_nRows, int A_nColumns, int B_nRows, int B_nColumns) {
	__shared__ int C_nColumns;

	C_nColumns = B_nColumns;

	int global_threadId = blockDim.x * blockIdx.x + threadIdx.x;

	if (global_threadId < B_nRows * B_nColumns) {
		int k = global_threadId / B_nColumns;
		int j = global_threadId % B_nColumns;

		double sum = 0.0;
		for (int i = 0; i < A_nRows; ++i) {
			sum += C_gradient[i * C_nColumns + j] * A[i * A_nColumns + k];
		}
		B_gradient[k * B_nColumns + j] += sum;
	}
}

class MatMul_gpu: public Matrix {
public:
	MatMul_gpu(int max_first_nRows, int max_first_nColumns, int max_second_nRows, int max_second_nColumns) : Matrix(max_first_nRows, max_second_nColumns) {
		cudaError err;

		assert(max_first_nColumns == max_second_nRows);

		size_first = max_first_nRows * max_first_nColumns * sizeof(double);
		size_second = max_second_nRows * max_second_nColumns * sizeof(double);
		size_this = size * sizeof(double);

		err = cudaMalloc((void **) &device_first, size_first);
		assert(err == cudaSuccess);

		err = cudaMalloc((void **) &device_second, size_second);
		assert(err == cudaSuccess);

		err = cudaMalloc((void **) &device_this, size_this);
		assert(err == cudaSuccess);

		// Memory for the DEPRECATED backward
		temp = new double [max(max_first_nRows * max_first_nColumns, max_second_nRows * max_second_nColumns)];

		use_gpu_stream = false;
	}

	MatMul_gpu(Matrix *first, Matrix *second) : Matrix(first -> nRows, second -> nColumns) {
		cudaError err;

		assert(first -> nColumns == second -> nRows);

		this -> first = first;
		this -> second = second;

		size_first = first -> nRows * first -> nColumns * sizeof(double);
		size_second = second -> nRows * second -> nColumns * sizeof(double);
		size_this = size * sizeof(double);

		err = cudaMalloc((void **) &device_first, size_first);
		assert(err == cudaSuccess);

		err = cudaMalloc((void **) &device_second, size_second);
		assert(err == cudaSuccess);

		err = cudaMalloc((void **) &device_this, size_this);
		assert(err == cudaSuccess);

		// Memory for the DEPRECATED backward
		temp = new double [max(first -> size, second -> size)];

		use_gpu_stream = false;
	}

	void setParameter(Matrix *first, Matrix *second) {
		assert(first -> nColumns == second -> nRows);
		
		this -> first = first;
		this -> second = second;

		nRows = first -> nRows;
		nColumns = second -> nColumns;
		size = nRows * nColumns;

		size_first = first -> nRows * first -> nColumns * sizeof(double);
		size_second = second -> nRows * second -> nColumns * sizeof(double);
		size_this = size * sizeof(double);
	}

	int rounded_division(int number1, int number2) {
		if (number1 % number2 == 0) {
			return number1 / number2;
		}
		return number1 / number2 + 1;
	}

	// Set GPU stream
	void set_gpu_stream(cudaStream_t stream) {
		use_gpu_stream = true;
		this -> stream = stream;
	}

	// Turn off GPU stream
	void turn_off_gpu_stream() {
		use_gpu_stream = false;
	}

	// +--------------+
	// | Forward Pass |
	// +--------------+

	void forward() {
		int complexity = first -> nRows * first -> nColumns * second -> nColumns;
		if (complexity <= COMPLEXITY_THRESHOLD) {
			forward_CPU();
		} else {
			forward_GPU();
		}
	}

	// +---------------+
	// | Backward Pass |
	// +---------------+

	void backward() {
		int complexity = first -> nRows * first -> nColumns * second -> nColumns;
		if (complexity <= COMPLEXITY_THRESHOLD) {
			backward_CPU();
		} else {
			backward_GPU();
		}
	}

	// +---------------------+
	// | Forward Pass in CPU |
	// +---------------------+

	void forward_CPU() {
		for (int i = 0; i < size; ++i) {
			value[i] = 0.0;
		}

		for (int i = 0; i < nRows; ++i) {
			for (int j = 0; j < nColumns; ++j) {
				int v = index(i, j);
				for (int k = 0; k < first -> nColumns; ++k) {
					int f = first -> index(i, k);
					int s = second -> index(k, j);
					value[v] += first -> value[f] * second -> value[s];
				}
			}
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}

	// +----------------------+
	// | Backward Pass in GPU |
	// +----------------------+

	void backward_CPU() {
		for (int i = 0; i < nRows; ++i) {
			for (int j = 0; j < nColumns; ++j) {
				int v = index(i, j);
				for (int k = 0; k < first -> nColumns; ++k) {
					int f = first -> index(i, k);
					int s = second -> index(k, j);

					first -> gradient[f] += gradient[v] * second -> value[s];
					second -> gradient[s] += gradient[v] * first -> value[f];
				}
			}
		}
	}

	// +---------------------+
	// | Forward Pass in GPU |
	// +---------------------+

	void forward_GPU() {
		for (int i = 0; i < size; ++i) {
			value[i] = 0.0;
		}

		int BLOCKS_PER_GRID_X = rounded_division(nRows, MATMUL_GPU_BLOCK_SIZE);
		int BLOCKS_PER_GRID_Y = rounded_division(nColumns, MATMUL_GPU_BLOCK_SIZE);

		dim3 dimGrid(BLOCKS_PER_GRID_X, BLOCKS_PER_GRID_Y);
		dim3 dimBlock(MATMUL_GPU_BLOCK_SIZE, MATMUL_GPU_BLOCK_SIZE);

		if (!use_gpu_stream) {

			// Default NULL stream

			cudaMemcpy(device_first, first -> value, size_first, cudaMemcpyHostToDevice);
			cudaMemcpy(device_second, second -> value, size_second, cudaMemcpyHostToDevice);
			cudaMemcpy(device_this, this -> value, size_this, cudaMemcpyHostToDevice);

			Matrix_Multiplication_GPU <<< dimGrid, dimBlock >>> 
				(device_first, device_second, device_this, first -> nRows, first -> nColumns, second -> nRows, second -> nColumns);

			cudaMemcpy(value, device_this, size_this, cudaMemcpyDeviceToHost);
		} else {

			// Multi-streams in GPU

			cudaMemcpyAsync(device_first, first -> value, size_first, cudaMemcpyHostToDevice, stream);
			cudaMemcpyAsync(device_second, second -> value, size_second, cudaMemcpyHostToDevice, stream);
			cudaMemcpyAsync(device_this, this -> value, size_this, cudaMemcpyHostToDevice, stream);

			Matrix_Multiplication_GPU <<< dimGrid, dimBlock, 0, stream >>> 
				(device_first, device_second, device_this, first -> nRows, first -> nColumns, second -> nRows, second -> nColumns);

			cudaMemcpyAsync(value, device_this, size_this, cudaMemcpyDeviceToHost, stream);
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}

	// +----------------------------------+
	// | DEPRECATED: Backward Pass in GPU |
	// +----------------------------------+

	void DEPRECATED_backward_GPU() {
		int BLOCKS_PER_GRID_X;
		int BLOCKS_PER_GRID_Y;

		BLOCKS_PER_GRID_X = rounded_division(first -> nRows, MATMUL_GPU_BLOCK_SIZE);
		BLOCKS_PER_GRID_Y = rounded_division(first -> nColumns, MATMUL_GPU_BLOCK_SIZE);

		dim3 dimGrid_first(BLOCKS_PER_GRID_X, BLOCKS_PER_GRID_Y);
		dim3 dimBlock_first(MATMUL_GPU_BLOCK_SIZE, MATMUL_GPU_BLOCK_SIZE);

		BLOCKS_PER_GRID_X = rounded_division(second -> nRows, MATMUL_GPU_BLOCK_SIZE);
		BLOCKS_PER_GRID_Y = rounded_division(second -> nColumns, MATMUL_GPU_BLOCK_SIZE);

		dim3 dimGrid_second(BLOCKS_PER_GRID_X, BLOCKS_PER_GRID_Y);
		dim3 dimBlock_second(MATMUL_GPU_BLOCK_SIZE, MATMUL_GPU_BLOCK_SIZE);

		if (!use_gpu_stream) {

			// Default NULL streams

			cudaMemcpy(device_this, this -> gradient, size_this, cudaMemcpyHostToDevice);

			/* Compute the gradient for the first matrix */
			for (int row = 0; row < second -> nRows; ++row) {
				for (int column = 0; column < second -> nColumns; ++column) {
					temp[column * second -> nRows + row] = second -> value[row * second -> nColumns + column];
				}
			}

			cudaMemcpy(device_first, first -> gradient, size_first, cudaMemcpyHostToDevice);
			cudaMemcpy(device_second, temp, size_second, cudaMemcpyHostToDevice);

			Matrix_Multiplication_GPU <<< dimGrid_first, dimBlock_first >>> 
				(device_this, device_second, device_first, this -> nRows, this -> nColumns, second -> nColumns, second -> nRows);

			cudaMemcpy(first -> gradient, device_first, size_first, cudaMemcpyDeviceToHost);

			/* Compute the gradient for the second matrix */
			for (int row = 0; row < first -> nRows; ++row) {
				for (int column = 0; column < first -> nColumns; ++column) {
					temp[column * first -> nRows + row] = first -> value[row * first -> nColumns + column];
				}
			}

			cudaMemcpy(device_first, temp, size_first, cudaMemcpyHostToDevice);
			cudaMemcpy(device_second, second -> gradient, size_second, cudaMemcpyHostToDevice);

			Matrix_Multiplication_GPU <<< dimGrid_second, dimBlock_second >>> 
				(device_first, device_this, device_second, first -> nColumns, first -> nRows, this -> nRows, this -> nColumns);
			
			cudaMemcpy(second -> gradient, device_second, size_second, cudaMemcpyDeviceToHost);
		} else {

			// Multi-streams in GPU

			cudaMemcpyAsync(device_this, this -> gradient, size_this, cudaMemcpyHostToDevice, stream);

			/* Compute the gradient for the first matrix */
			for (int row = 0; row < second -> nRows; ++row) {
				for (int column = 0; column < second -> nColumns; ++column) {
					temp[column * second -> nRows + row] = second -> value[row * second -> nColumns + column];
				}
			}

			cudaMemcpyAsync(device_first, first -> gradient, size_first, cudaMemcpyHostToDevice, stream);
			cudaMemcpyAsync(device_second, temp, size_second, cudaMemcpyHostToDevice, stream);

			Matrix_Multiplication_GPU <<< dimGrid_first, dimBlock_first, 0, stream >>> 
				(device_this, device_second, device_first, this -> nRows, this -> nColumns, second -> nColumns, second -> nRows);

			cudaMemcpyAsync(first -> gradient, device_first, size_first, cudaMemcpyDeviceToHost, stream);

			/* Compute the gradient for the second matrix */
			for (int row = 0; row < first -> nRows; ++row) {
				for (int column = 0; column < first -> nColumns; ++column) {
					temp[column * first -> nRows + row] = first -> value[row * first -> nColumns + column];
				}
			}

			cudaMemcpyAsync(device_first, temp, size_first, cudaMemcpyHostToDevice, stream);
			cudaMemcpyAsync(device_second, second -> gradient, size_second, cudaMemcpyHostToDevice, stream);

			Matrix_Multiplication_GPU <<< dimGrid_second, dimBlock_second, 0, stream >>> 
				(device_first, device_this, device_second, first -> nColumns, first -> nRows, this -> nRows, this -> nColumns);
			
			cudaMemcpyAsync(second -> gradient, device_second, size_second, cudaMemcpyDeviceToHost, stream);
		}
	}

	// +----------------------+
	// | Backward Pass in GPU |
	// +----------------------+

	void backward_GPU() {
		dim3 dimGrid_first(rounded_division(first -> size, MATMUL_GPU_THREADS));
		dim3 dimBlock_first(MATMUL_GPU_THREADS);

		dim3 dimGrid_second(rounded_division(second -> size, MATMUL_GPU_THREADS));
		dim3 dimBlock_second(MATMUL_GPU_THREADS);

		if (!use_gpu_stream) {

			// Default NULL stream

			cudaMemcpy(device_this, this -> gradient, size_this, cudaMemcpyHostToDevice);

			/* Compute the gradient for the first matrix */
			cudaMemcpy(device_first, first -> gradient, size_first, cudaMemcpyHostToDevice);
			cudaMemcpy(device_second, second -> value, size_second, cudaMemcpyHostToDevice);

			MatMul_backward_first <<< dimGrid_first, dimBlock_first >>> 
				(device_first, device_second, device_this, first -> nRows, first -> nColumns, second -> nRows, second -> nColumns);

			cudaMemcpy(first -> gradient, device_first, size_first, cudaMemcpyDeviceToHost);

			/* Compute the gradient for the second matrix */
			cudaMemcpy(device_first, first -> value, size_first, cudaMemcpyHostToDevice);
			cudaMemcpy(device_second, second -> gradient, size_second, cudaMemcpyHostToDevice);

			MatMul_backward_second <<< dimGrid_second, dimBlock_second >>> 
				(device_first, device_second, device_this, first -> nRows, first -> nColumns, second -> nRows, second -> nColumns);

			cudaMemcpy(second -> gradient, device_second, size_second, cudaMemcpyDeviceToHost);
		} else {

			// Multi-streams in GPU

			cudaMemcpyAsync(device_this, this -> gradient, size_this, cudaMemcpyHostToDevice, stream);

			/* Compute the gradient for the first matrix */
			cudaMemcpyAsync(device_first, first -> gradient, size_first, cudaMemcpyHostToDevice, stream);
			cudaMemcpyAsync(device_second, second -> value, size_second, cudaMemcpyHostToDevice, stream);

			MatMul_backward_first <<< dimGrid_first, dimBlock_first, 0, stream >>> 
				(device_first, device_second, device_this, first -> nRows, first -> nColumns, second -> nRows, second -> nColumns);

			cudaMemcpyAsync(first -> gradient, device_first, size_first, cudaMemcpyDeviceToHost, stream);

			/* Compute the gradient for the second matrix */
			cudaMemcpyAsync(device_first, first -> value, size_first, cudaMemcpyHostToDevice, stream);
			cudaMemcpyAsync(device_second, second -> gradient, size_second, cudaMemcpyHostToDevice, stream);

			MatMul_backward_second <<< dimGrid_second, dimBlock_second, 0, stream >>> 
				(device_first, device_second, device_this, first -> nRows, first -> nColumns, second -> nRows, second -> nColumns);

			cudaMemcpyAsync(second -> gradient, device_second, size_second, cudaMemcpyDeviceToHost, stream);
		}
	}

	// Threshold for complexity to decide GPU or CPU
	static const int COMPLEXITY_THRESHOLD = 1e6;

	Matrix *first;
	Matrix *second;

	size_t size_first;
	size_t size_second;
	size_t size_this;

	double *device_first;
	double *device_second;
	double *device_this;

	// Memory for the DEPRECATED backward
	double *temp;

	// Use GPU stream
	bool use_gpu_stream;

	// GPU stream
	cudaStream_t stream;

	void handleCudaError(cudaError err) {
		if (err != cudaSuccess) {
			printf("%s\n", cudaGetErrorString(err));
		}
	}

	~MatMul_gpu() {
		cudaFree(device_first);
		cudaFree(device_second);
		cudaFree(device_this);
		
		// Memory for the DEPRECATED backward
		delete[] temp;
	}
};

#endif