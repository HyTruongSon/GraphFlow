// Framework: GraphFlow
// Class: RisiContraction_18_gpu
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __RISICONTRACTION_18_GPU_H_INCLUDED__
#define __RISICONTRACTION_18_GPU_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <assert.h>
#include <stdio.h>

#include "Tensor3D.h"
#include "Tensor4D.h"

using namespace std;

// +-------------------------------------------+
// | Atomic Addition Operation For Double Type |
// +-------------------------------------------+ 

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
	static __inline__ __device__ double atomicAdd(double *address, double val) {
		unsigned long long int* address_as_ull = (unsigned long long int*) address;
		unsigned long long int old = *address_as_ull, assumed;
		if (val == 0.0) {
			return __longlong_as_double(old);
		}
		do {
			assumed = old;
			old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +__longlong_as_double(assumed)));
		} while (assumed != old);
		return __longlong_as_double(old);
	}
#endif

// +-------------------------------------+
// | Kernel Function For The Forward Job |
// +-------------------------------------+

__global__ void RisiContraction_18_forward_job(double *tensor, double *adj, double *value, int N, int nChanels) {
	__shared__ int nContractions;
	__shared__ int A;
	__shared__ int B;
	__shared__ int C;
	__shared__ int Y;

	nContractions = 18;

	int global_threadId = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (global_threadId < N * N * nChanels * nContractions) {	
		C = nChanels;
		B = N * C;
		A = N * B;

		Y = nChanels * nContractions;
		
		int f = (global_threadId % Y) % nChanels;
		int Case = (global_threadId % Y) / nChanels + 1;
		int y = (global_threadId / Y) % N;
		int x = (global_threadId / Y) / N;

		int a, b, c, d, e;
		double adj_value;

		double sum = 0.0;

		// +-----------+
		// | 1 + 1 + 1 |
		// +-----------+

		// Case 1 (1/50): Fix a, b. Contract c, d, e.
		if (Case == 1) {
			a = x;
			b = y;

			for (d = 0; d < N; ++d) {
				for (e = 0; e < N; ++e) {
					adj_value = adj[d * N + e];
					if (adj_value > 0) {
						for (c = 0; c < N; ++c) {
							sum += tensor[a * A + b * B + c * C + f] * adj_value;
						}
					}
				}
			}
		}
				
		// Case 2 (3/50): Fix a, d. Contract b, c, e.
		if (Case == 2) {		
			a = x;
			d = y;

			for (e = 0; e < N; ++e) {
				adj_value = adj[d * N + e];
				if (adj_value > 0) {
					for (b = 0; b < N; ++b) {
						for (c = 0; c < N; ++c) {
							sum += tensor[a * A + b * B + c * C + f] * adj_value;
						}
					}
				}
			}	
		}
		
		// Case 3 (5/50): Fix b, c. Contract a, d, e.
		if (Case == 3) {		
			b = x;
			c = y;

			for (d = 0; d < N; ++d) {
				for (e = 0; e < N; ++e) {
					adj_value = adj[d * N + e];
					if (adj_value > 0) {
						for (a = 0; a < N; ++a) {
							sum += tensor[a * A + b * B + c * C + f] * adj_value;
						}
					}
				}
			}	
		}

		// Case 4 (6/50): Fix b, d. Contract a, c, e.
		if (Case == 4) {
			b = x;
			d = y;

			for (e = 0; e < N; ++e) {
				adj_value = adj[d * N + e];
				if (adj_value > 0) {
					for (a = 0; a < N; ++a) {
						for (c = 0; c < N; ++c) {
							sum += tensor[a * A + b * B + c * C + f] * adj_value;
						}
					}
				}
			}
		}

		// Case 5 (10/50): Fix d, e. Contract a, b, c.
		if (Case == 5) {		
			d = x;
			e = y;

			adj_value = adj[d * N + e];
			if (adj_value > 0) {
				for (a = 0; a < N; ++a) {
					for (b = 0; b < N; ++b) {
						for (c = 0; c < N; ++c) {
							sum += tensor[a * A + b * B + c * C + f] * adj_value;
						}
					}
				}
			}
		}

		// +-------+
		// | 1 + 2 |
		// +-------+

		// Case 6 (11/50): (a, b). Contract (c, d). Singleton (e).
		if (Case == 6) {
			a = x;
			b = y;

			for (d = 0; d < N; ++d) {
				for (e = 0; e < N; ++e) {
					adj_value = adj[d * N + e];
					c = d;
					sum += tensor[a * A + b * B + c * C + f] * adj_value;
				}
			}
		}

		// Case 7 (13/50): (a, b). Contract (d, e). Singleton (c).
		if (Case == 7) {
			a = x;
			b = y;

			for (d = 0; d < N; ++d) {
				e = d;
				adj_value = adj[d * N + e];
				if (adj_value > 0) {
					for (c = 0; c < N; ++c) {
						sum += tensor[a * A + b * B + c * C + f] * adj_value;
					}
				}
			}
		}

		// Case 8 (17/50): (a, d). Contract (b, c). Singleton (e).
		if (Case == 8) {
			a = x;
			d = y;

			for (e = 0; e < N; ++e) {
				adj_value = adj[d * N + e];
				if (adj_value > 0) {
					for (b = 0; b < N; ++b) {
						c = b;
						sum += tensor[a * A + b * B + c * C + f] * adj_value;
					}
				}
			}
		}

		// Case 9 (18/50): (a, d). Contract (b, e). Singleton (c).
		if (Case == 9) {
			a = x;
			d = y;

			for (e = 0; e < N; ++e) {
				adj_value = adj[d * N + e];
				if (adj_value > 0) {
					b = e;
					for (c = 0; c < N; ++c) {
						sum += tensor[a * A + b * B + c * C + f] * adj_value;
					}
				}
			}
		}

		// Case 10 (23/50): (b, c). Contract (a, d). Singleton (e).
		if (Case == 10) {
			b = x;
			c = y;

			for (d = 0; d < N; ++d) {
				for (e = 0; e < N; ++e) {
					adj_value = adj[d * N + e];
					if (adj_value > 0) {
						a = d;
						sum += tensor[a * A + b * B + c * C + f] * adj_value;
					}
				}
			}
		}

		// Case 11 (26/50): (b, d). Contract (a, c). Singleton (e).
		if (Case == 11) {
			b = x;
			d = y;

			for (e = 0; e < N; ++e) {
				adj_value = adj[d * N + e];
				if (adj_value > 0) {
					for (a = 0; a < N; ++a) {
						c = a;
						sum += tensor[a * A + b * B + c * C + f] * adj_value;
					}
				}
			}
		}

		// Case 12 (27/50): (b, d). Contract (a, e). Singleton (c).
		if (Case == 12) {
			b = x;
			d = y;

			for (e = 0; e < N; ++e) {
				adj_value = adj[d * N + e];
				if (adj_value > 0) {
					a = e;
					for (int c = 0; c < N; ++c) {
						sum += tensor[a * A + b * B + c * C + f] * adj_value;
					}
				}
			}
		}

		// Case 13 (28/50): (b, d). Contract (c, e). Singleton (a).
		if (Case == 13) {
			b = x;
			d = y;

			for (e = 0; e < N; ++e) {
				adj_value = adj[d * N + e];
				if (adj_value > 0) {
					c = e;
					for (int a = 0; a < N; ++a) {
						sum += tensor[a * A + b * B + c * C + f] * adj_value;
					}
				}
			}
		}

		// Case 14 (38/50): (d, e). Contract (a, b). Singleton (c).
		if (Case == 14) {
			d = x;
			e = y;

			adj_value = adj[d * N + e];
			if (adj_value > 0) {
				for (int a = 0; a < N; ++a) {
					b = a;
					for (int c = 0; c < N; ++c) {
						sum += tensor[a * A + b * B + c * C + f] * adj_value;
					}
				}
			}
		}

		// Case 15 (40/50): (d, e). Contract (b, c). Singleton (a).
		if (Case == 15) {
			d = x;
			e = y;

			adj_value = adj[d * N + e];
			if (adj_value > 0) {
				for (int b = 0; b < N; ++b) {
					c = b;
					for (int a = 0; a < N; ++a) {
						sum += tensor[a * A + b * B + c * C + f] * adj_value;
					}
				}
			}
		}

		// +---+
		// | 3 |
		// +---+

		// Case 16 (43/50): (a, d). Contract (b, c, e).
		if (Case == 16) {
			a = x;
			d = y;

			for (int e = 0; e < N; ++e) {
				adj_value = adj[d * N + e];
				if (adj_value > 0) {
					b = e;
					c = e;
					sum += tensor[a * A + b * B + c * C + f] * adj_value;
				}
			}
		}	

		// Case 17 (46/50): (b, d). Contract (a, c, e).
		if (Case == 17) {
			b = x;
			d = y;

			for (int e = 0; e < N; ++e) {
				adj_value = adj[d * N + e];
				if (adj_value > 0) {
					a = e;
					c = e;
					sum += tensor[a * A + b * B + c * C + f] * adj_value;
				}
			}
		}

		// Case 18 (50/50): (d, e). Contract (a, b, c).
		if (Case == 18) {
			d = x;
			e = y;

			adj_value = adj[d * N + e];
			if (adj_value > 0) {
				for (int a = 0; a < N; ++a) {
					b = a;
					c = a;
					sum += tensor[a * A + b * B + c * C + f] * adj_value;
				}
			}
		}

		value[global_threadId] = sum;
	}
}

// +-------------------------------------------------+
// | DEPRECATED: Kernel Function For The Forward Job |
// +-------------------------------------------------+

__global__ void DEPRECATED_RisiContraction_18_forward_job(double *tensor, double *adj, double *value, int N, int nChanels) {
	__shared__ int d;
	__shared__ int e;
	__shared__ double adj_value;
	__shared__ int length;
	__shared__ int partition_size;
	__shared__ int nContractions;

	d = blockIdx.x;
	e = blockIdx.y;
	adj_value = adj[d * N + e];
	length = N * N * N * nChanels;
	partition_size = ceil((double)(length) / (double)(blockDim.x)); 
	nContractions = 18;

	if ((d < N) && (e < N) && (adj_value > 0.0) && (threadIdx.x < blockDim.x) && (threadIdx.x * partition_size < length)) {

		int from = threadIdx.x * partition_size;
		int to = (threadIdx.x + 1) * partition_size - 1;

		if (to >= length) {
			to = length - 1;
		}

		int a, b, c, f, i, ind;
		double product;
		int X = N * nChanels * nContractions;
		int Y = nChanels * nContractions;

		for (i = from; i <= to; ++i) {
			// Indexing: i = ((a * N + b) * N + c) * nChanels + f;

			f = i % nChanels;
			c = (i / nChanels) % N;
			b = ((i / nChanels) / N) % N;
			a = ((i / nChanels) / N) / N;

			product = tensor[i] * adj_value;

			// +-----------+
			// | 1 + 1 + 1 |
			// +-----------+

			// Case 1 (1/50): Fix a, b. Contract c, d, e.
			ind = a * X + b * Y + 0 * nChanels + f;
			atomicAdd(&value[ind], product);
			
			// Case 2 (3/50): Fix a, d. Contract b, c, e.
			ind = a * X + d * Y + 1 * nChanels + f;
			atomicAdd(&value[ind], product);

			// Case 3 (5/50): Fix b, c. Contract a, d, e.
			ind = b * X + c * Y + 2 * nChanels + f;
			atomicAdd(&value[ind], product);

			// Case 4 (6/50): Fix b, d. Contract a, c, e.
			ind = b * X + d * Y + 3 * nChanels + f;
			atomicAdd(&value[ind], product);

			// Case 5 (10/50): Fix d, e. Contract a, b, c.
			ind = d * X + e * Y + 4 * nChanels + f;
			atomicAdd(&value[ind], product);

			// +-------+
			// | 1 + 2 |
			// +-------+

			// Case 6 (11/50): (a, b). Contract (c, d). Singleton (e).
			if (c == d) {
				ind = a * X + b * Y + 5 * nChanels + f;
				atomicAdd(&value[ind], product);
			}

			// Case 7 (13/50): (a, b). Contract (d, e). Singleton (c).
			if (d == e) {
				ind = a * X + b * Y + 6 * nChanels + f;
				atomicAdd(&value[ind], product);
			}

			// Case 8 (17/50): (a, d). Contract (b, c). Singleton (e).
			if (b == c) {
				ind = a * X + d * Y + 7 * nChanels + f;
				atomicAdd(&value[ind], product);
			}

			// Case 9 (18/50): (a, d). Contract (b, e). Singleton (c).
			if (b == e) {
				ind = a * X + d * Y + 8 * nChanels + f;
				atomicAdd(&value[ind], product);
			}

			// Case 10 (23/50): (b, c). Contract (a, d). Singleton (e).
			if (a == d) {
				ind = b * X + c * Y + 9 * nChanels + f;
				atomicAdd(&value[ind], product);
			}

			// Case 11 (26/50): (b, d). Contract (a, c). Singleton (e).
			if (a == c) {
				ind = b * X + d * Y + 10 * nChanels + f;
				atomicAdd(&value[ind], product);
			}

			// Case 12 (27/50): (b, d). Contract (a, e). Singleton (c).
			if (a == e) {
				ind = b * X + d * Y + 11 * nChanels + f;
				atomicAdd(&value[ind], product);
			}

			// Case 13 (28/50): (b, d). Contract (c, e). Singleton (a).
			if (c == e) {
				ind = b * X + d * Y + 12 * nChanels + f;
				atomicAdd(&value[ind], product);
			}

			// Case 14 (38/50): (d, e). Contract (a, b). Singleton (c).
			if (a == b) {
				ind = d * X + e * Y + 13 * nChanels + f;
				atomicAdd(&value[ind], product);
			}

			// Case 15 (40/50): (d, e). Contract (b, c). Singleton (a).
			if (b == c) {
				ind = d * X + e * Y + 14 * nChanels + f;
				atomicAdd(&value[ind], product);
			}

			// +---+
			// | 3 |
			// +---+

			// Case 16 (43/50): (a, d). Contract (b, c, e).
			if ((b == c) && (c == e))  {
				ind = a * X + d * Y + 15 * nChanels + f;
				atomicAdd(&value[ind], product);
			}

			// Case 17 (46/50): (b, d). Contract (a, c, e).
			if ((a == c) && (c == e))  {
				ind = b * X + d * Y + 16 * nChanels + f;
				atomicAdd(&value[ind], product);
			}

			// Case 18 (50/50): (d, e). Contract (a, b, c).
			if ((a == b) && (b == c))  {
				ind = d * X + e * Y + 17 * nChanels + f;
				atomicAdd(&value[ind], product);
			}
		}
	}
}

// +--------------------------------------+
// | Kernel Function For The Backward Job |
// +--------------------------------------+

__global__ void RisiContraction_18_backward_job(double *tensor_gradient, double *adj, double *gradient, int N, int nChanels) {
	
	__shared__ int nContractions;
	__shared__ int X;
	__shared__ int Y;

	nContractions = 18;

	int global_threadId = blockIdx.x * blockDim.x + threadIdx.x;

	if (global_threadId < N * N * N * nChanels) {
		X = N * nChanels * nContractions;
		Y = nChanels * nContractions;

		int f = global_threadId % nChanels;
		int c = (global_threadId / nChanels) % N;
		int b = ((global_threadId / nChanels) / N) % N;
		int a = ((global_threadId / nChanels) / N) / N;

		double sum = 0.0;

		int ind;
		double adj_value;

		for (int d = 0; d < N; ++d) {
			for (int e = 0; e < N; ++e) {
				adj_value = adj[d * N + e];

				if (adj_value > 0) {
					// +-----------+
					// | 1 + 1 + 1 |
					// +-----------+

					// Case 1 (1/50): Fix a, b. Contract c, d, e.
					ind = a * X + b * Y + 0 * nChanels + f;
					sum += gradient[ind] * adj_value;

					// Case 2 (3/50): Fix a, d. Contract b, c, e.
					ind = a * X + d * Y + 1 * nChanels + f;
					sum += gradient[ind] * adj_value;

					// Case 3 (5/50): Fix b, c. Contract a, d, e.
					ind = b * X + c * Y + 2 * nChanels + f;
					sum += gradient[ind] * adj_value;

					// Case 4 (6/50): Fix b, d. Contract a, c, e.
					ind = b * X + d * Y + 3 * nChanels + f;
					sum += gradient[ind] * adj_value;

					// Case 5 (10/50): Fix d, e. Contract a, b, c.
					ind = d * X + e * Y + 4 * nChanels + f;
					sum += gradient[ind] * adj_value;

					// +-------+
					// | 1 + 2 |
					// +-------+

					// Case 6 (11/50): (a, b). Contract (c, d). Singleton (e).
					if (c == d) {
						ind = a * X + b * Y + 5 * nChanels + f;
						sum += gradient[ind] * adj_value;
					}

					// Case 7 (13/50): (a, b). Contract (d, e). Singleton (c).
					if (d == e) {
						ind = a * X + b * Y + 6 * nChanels + f;
						sum += gradient[ind] * adj_value;
					}

					// Case 8 (17/50): (a, d). Contract (b, c). Singleton (e).
					if (b == c) {
						ind = a * X + d * Y + 7 * nChanels + f;
						sum += gradient[ind] * adj_value;
					}

					// Case 9 (18/50): (a, d). Contract (b, e). Singleton (c).
					if (b == e) {
						ind = a * X + d * Y + 8 * nChanels + f;
						sum += gradient[ind] * adj_value;
					}

					// Case 10 (23/50): (b, c). Contract (a, d). Singleton (e).
					if (a == d) {
						ind = b * X + c * Y + 9 * nChanels + f;
						sum += gradient[ind] * adj_value;
					}

					// Case 11 (26/50): (b, d). Contract (a, c). Singleton (e).
					if (a == c) {
						ind = b * X + d * Y + 10 * nChanels + f;
						sum += gradient[ind] * adj_value;
					}

					// Case 12 (27/50): (b, d). Contract (a, e). Singleton (c).
					if (a == e) {
						ind = b * X + d * Y + 11 * nChanels + f;
						sum += gradient[ind] * adj_value;
					}

					// Case 13 (28/50): (b, d). Contract (c, e). Singleton (a).
					if (c == e) {
						ind = b * X + d * Y + 12 * nChanels + f;
						sum += gradient[ind] * adj_value;
					}

					// Case 14 (38/50): (d, e). Contract (a, b). Singleton (c).
					if (a == b) {
						ind = d * X + e * Y + 13 * nChanels + f;
						sum += gradient[ind] * adj_value;
					}

					// Case 15 (40/50): (d, e). Contract (b, c). Singleton (a).
					if (b == c) {
						ind = d * X + e * Y + 14 * nChanels + f;
						sum += gradient[ind] * adj_value;
					}

					// +---+
					// | 3 |
					// +---+

					// Case 16 (43/50): (a, d). Contract (b, c, e).
					if ((b == c) && (c == e))  {
						ind = a * X + d * Y + 15 * nChanels + f;
						sum += gradient[ind] * adj_value;
					}

					// Case 17 (46/50): (b, d). Contract (a, c, e).
					if ((a == c) && (c == e))  {
						ind = b * X + d * Y + 16 * nChanels + f;
						sum += gradient[ind] * adj_value;
					}

					// Case 18 (50/50): (d, e). Contract (a, b, c).
					if ((a == b) && (b == c))  {
						ind = d * X + e * Y + 17 * nChanels + f;
						sum += gradient[ind] * adj_value;
					}
				}
			}
		}
		
		tensor_gradient[global_threadId] += sum;
	}
}

// +----------------------------------------------+
// | DEPRECATED: Kernel Function For Backward Job |
// +----------------------------------------------+

__global__ void DEPRECATED_RisiContraction_18_backward_job(double *tensor_gradient, double *adj, double *gradient, int N, int nChanels) {
	__shared__ int d;
	__shared__ int e;
	__shared__ double adj_value;
	__shared__ int length;
	__shared__ int partition_size;
	__shared__ int nContractions;

	d = blockIdx.x;
	e = blockIdx.y;
	adj_value = adj[d * N + e];
	length = N * N * N * nChanels;
	partition_size = ceil((double)(length) / (double)(blockDim.x)); 
	nContractions = 18;

	if ((d < N) && (e < N) && (adj_value > 0.0) && (threadIdx.x < blockDim.x) && (threadIdx.x * partition_size < length)) {
	
		int from = threadIdx.x * partition_size;
		int to = (threadIdx.x + 1) * partition_size - 1;

		if (to >= length) {
			to = length - 1;
		}

		int a, b, c, f, i, ind;
		double sum;
		int X = N * nChanels * nContractions;
		int Y = nChanels * nContractions;

		for (i = from; i <= to; ++i) {
			// Indexing: i = ((a * N + b) * N + c) * nChanels + f;

			f = i % nChanels;
			c = (i / nChanels) % N;
			b = ((i / nChanels) / N) % N;
			a = ((i / nChanels) / N) / N;

			sum = 0.0;

			// +-----------+
			// | 1 + 1 + 1 |
			// +-----------+

			// Case 1 (1/50): Fix a, b. Contract c, d, e.
			ind = a * X + b * Y + 0 * nChanels + f;
			sum += gradient[ind] * adj_value;

			// Case 2 (3/50): Fix a, d. Contract b, c, e.
			ind = a * X + d * Y + 1 * nChanels + f;
			sum += gradient[ind] * adj_value;

			// Case 3 (5/50): Fix b, c. Contract a, d, e.
			ind = b * X + c * Y + 2 * nChanels + f;
			sum += gradient[ind] * adj_value;

			// Case 4 (6/50): Fix b, d. Contract a, c, e.
			ind = b * X + d * Y + 3 * nChanels + f;
			sum += gradient[ind] * adj_value;

			// Case 5 (10/50): Fix d, e. Contract a, b, c.
			ind = d * X + e * Y + 4 * nChanels + f;
			sum += gradient[ind] * adj_value;

			// +-------+
			// | 1 + 2 |
			// +-------+

			// Case 6 (11/50): (a, b). Contract (c, d). Singleton (e).
			if (c == d) {
				ind = a * X + b * Y + 5 * nChanels + f;
				sum += gradient[ind] * adj_value;
			}

			// Case 7 (13/50): (a, b). Contract (d, e). Singleton (c).
			if (d == e) {
				ind = a * X + b * Y + 6 * nChanels + f;
				sum += gradient[ind] * adj_value;
			}

			// Case 8 (17/50): (a, d). Contract (b, c). Singleton (e).
			if (b == c) {
				ind = a * X + d * Y + 7 * nChanels + f;
				sum += gradient[ind] * adj_value;
			}

			// Case 9 (18/50): (a, d). Contract (b, e). Singleton (c).
			if (b == e) {
				ind = a * X + d * Y + 8 * nChanels + f;
				sum += gradient[ind] * adj_value;
			}

			// Case 10 (23/50): (b, c). Contract (a, d). Singleton (e).
			if (a == d) {
				ind = b * X + c * Y + 9 * nChanels + f;
				sum += gradient[ind] * adj_value;
			}

			// Case 11 (26/50): (b, d). Contract (a, c). Singleton (e).
			if (a == c) {
				ind = b * X + d * Y + 10 * nChanels + f;
				sum += gradient[ind] * adj_value;
			}

			// Case 12 (27/50): (b, d). Contract (a, e). Singleton (c).
			if (a == e) {
				ind = b * X + d * Y + 11 * nChanels + f;
				sum += gradient[ind] * adj_value;
			}

			// Case 13 (28/50): (b, d). Contract (c, e). Singleton (a).
			if (c == e) {
				ind = b * X + d * Y + 12 * nChanels + f;
				sum += gradient[ind] * adj_value;
			}

			// Case 14 (38/50): (d, e). Contract (a, b). Singleton (c).
			if (a == b) {
				ind = d * X + e * Y + 13 * nChanels + f;
				sum += gradient[ind] * adj_value;
			}

			// Case 15 (40/50): (d, e). Contract (b, c). Singleton (a).
			if (b == c) {
				ind = d * X + e * Y + 14 * nChanels + f;
				sum += gradient[ind] * adj_value;
			}

			// +---+
			// | 3 |
			// +---+

			// Case 16 (43/50): (a, d). Contract (b, c, e).
			if ((b == c) && (c == e))  {
				ind = a * X + d * Y + 15 * nChanels + f;
				sum += gradient[ind] * adj_value;
			}

			// Case 17 (46/50): (b, d). Contract (a, c, e).
			if ((a == c) && (c == e))  {
				ind = b * X + d * Y + 16 * nChanels + f;
				sum += gradient[ind] * adj_value;
			}

			// Case 18 (50/50): (d, e). Contract (a, b, c).
			if ((a == b) && (b == c))  {
				ind = d * X + e * Y + 17 * nChanels + f;
				sum += gradient[ind] * adj_value;
			}


			atomicAdd(&tensor_gradient[i], sum);
		}
	}
}

class RisiContraction_18_gpu: public Tensor3D {
public:

	RisiContraction_18_gpu(int max_N, int max_nChanels) : Tensor3D(max_N, max_N, nContractions * max_nChanels) {
		cudaError err;

		// GPU device - Tensor of 4 dimensions
		tensor_size = max_N * max_N * max_N * max_nChanels * sizeof(double);
		
		err = cudaMalloc( &device_tensor_value, tensor_size);
		assert(err == cudaSuccess);

		err = cudaMalloc( &device_tensor_gradient, tensor_size);
		assert(err == cudaSuccess);

		// GPU device - The reduced adjacency matrix
		adj_size = max_N * max_N * sizeof(double);

		err = cudaMalloc( &device_adj_value, adj_size);
		assert(err == cudaSuccess);

		// GPU device - For this object
		this_size = max_N * max_N * max_nChanels * nContractions * sizeof(double);
		
		err = cudaMalloc( &device_value, this_size);
		assert(err == cudaSuccess);

		err = cudaMalloc( &device_gradient, this_size);
		assert(err == cudaSuccess);

		use_gpu_stream = false;
	}

	RisiContraction_18_gpu(Tensor4D *tensor, Matrix *adj) : Tensor3D(tensor -> nRows, tensor -> nRows, nContractions * tensor -> nChanels2) {
		this -> tensor = tensor;
		this -> adj = adj;

		N = tensor -> nRows;
		nChanels = tensor -> nChanels2;

		assert(N == tensor -> nColumns);
		assert(N == tensor -> nChanels1);
		assert(N == adj -> nRows);
		assert(N == adj -> nColumns);

		cudaError err;

		// GPU device - Tensor of 4 dimensions
		tensor_size = N * N * N * nChanels * sizeof(double);
		
		err = cudaMalloc((void **) &device_tensor_value, tensor_size);
		assert(err == cudaSuccess);

		err = cudaMalloc((void **) &device_tensor_gradient, tensor_size);
		assert(err == cudaSuccess);

		// GPU device - The reduced adjacency matrix
		adj_size = N * N * sizeof(double);

		err = cudaMalloc((void **) &device_adj_value, adj_size);
		assert(err == cudaSuccess);

		// GPU device - For this object
		this_size = N * N * nChanels * nContractions * sizeof(double);
		
		err = cudaMalloc((void **) &device_value, this_size);
		assert(err == cudaSuccess);

		err = cudaMalloc((void **) &device_gradient, this_size);
		assert(err == cudaSuccess);

		use_gpu_stream = false;
	}

	void setParameter(Tensor4D *tensor, Matrix *adj) {
		this -> tensor = tensor;
		this -> adj = adj;

		N = tensor -> nRows;
		nChanels = tensor -> nChanels2;

		assert(N == tensor -> nColumns);
		assert(N == tensor -> nChanels1);
		assert(N == adj -> nRows);
		assert(N == adj -> nColumns);

		nRows = N;
		nColumns = N;
		nDepth = nChanels * nContractions;
		size = nRows * nColumns * nDepth;
	}

	// Ceiling
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
		int complexity = N * N * N * N * N * nChanels * nContractions;
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
		int complexity = N * N * N * N * N * nChanels * nContractions;
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

		int ind;
		double delta;
		double adj_value;

		int a, b, c, d, e, f;

		for (d = 0; d < N; ++d) {
			for (e = 0; e < N; ++e) {
				adj_value = adj -> value[adj -> index(d, e)];

				if (adj_value > 0) {
					for (f = 0; f < nChanels; ++f) {
						for (a = 0; a < N; ++a) {
							for (b = 0; b < N; ++b) {
								for (c = 0; c < N; ++c) {
									
									delta = tensor -> value[tensor -> index(a, b, c, f)] * adj_value;

									// +-----------+
									// | 1 + 1 + 1 |
									// +-----------+

									// Case 1 (1/50): Fix a, b. Contract c, d, e.
									ind = index(a, b, 0 * nChanels + f);
									value[ind] += delta;

									// Case 2 (3/50): Fix a, d. Contract b, c, e.
									ind = index(a, d, 1 * nChanels + f);
									value[ind] += delta;

									// Case 3 (5/50): Fix b, c. Contract a, d, e.
									ind = index(b, c, 2 * nChanels + f);
									value[ind] += delta;

									// Case 4 (6/50): Fix b, d. Contract a, c, e.
									ind = index(b, d, 3 * nChanels + f);
									value[ind] += delta;

									// Case 5 (10/50): Fix d, e. Contract a, b, c.
									ind = index(d, e, 4 * nChanels + f);
									value[ind] += delta;
								}
							}
						}
					}

					c = d;
					for (f = 0; f < nChanels; ++f) {
						for (a = 0; a < N; ++a) {
							for (b = 0; b < N; ++b) {

								delta = tensor -> value[tensor -> index(a, b, c, f)] * adj_value;

								// Case 6 (11/50): (a, b). Contract (c, d). Singleton (e).
								ind = index(a, b, 5 * nChanels + f);
								// c == d
								value[ind] += delta;
							}
						}
					}

					if (d == e) {
						for (f = 0; f < nChanels; ++f) {
							for (a = 0; a < N; ++a) {
								for (b = 0; b < N; ++b) {
									for (c = 0; c < N; ++c) {
										
										delta = tensor -> value[tensor -> index(a, b, c, f)] * adj_value;

										// Case 7 (13/50): (a, b). Contract (d, e). Singleton (c).
										ind = index(a, b, 6 * nChanels + f);
										// d == e
										value[ind] += delta;
									}
								}
							}
						}
					}

					for (f = 0; f < nChanels; ++f) {
						for (a = 0; a < N; ++a) {
							for (b = 0; b < N; ++b) {
								c = b;
								delta = tensor -> value[tensor -> index(a, b, c, f)] * adj_value;

								// Case 8 (17/50): (a, d). Contract (b, c). Singleton (e).
								ind = index(a, d, 7 * nChanels + f);
								// b == c
								value[ind] += delta;
							}
						}
					}

					b = e;
					for (f = 0; f < nChanels; ++f) {
						for (a = 0; a < N; ++a) {
							for (c = 0; c < N; ++c) {

								delta = tensor -> value[tensor -> index(a, b, c, f)] * adj_value;

								// Case 9 (18/50): (a, d). Contract (b, e). Singleton (c).
								ind = index(a, d, 8 * nChanels + f);
								// b == e
								value[ind] += delta;
							}
						}
					}

					a = d;
					for (f = 0; f < nChanels; ++f) {
						for (b = 0; b < N; ++b) {
							for (c = 0; c < N; ++c) {

								delta = tensor -> value[tensor -> index(a, b, c, f)] * adj_value;
									
								// Case 10 (23/50): (b, c). Contract (a, d). Singleton (e).
								ind = index(b, c, 9 * nChanels + f);
								// a == d
								value[ind] += delta;
								
							}
						}
					}

					for (f = 0; f < nChanels; ++f) {
						for (a = 0; a < N; ++a) {
							for (b = 0; b < N; ++b) {
								
								c = a;
								delta = tensor -> value[tensor -> index(a, b, c, f)] * adj_value;

								// Case 11 (26/50): (b, d). Contract (a, c). Singleton (e).
								ind = index(b, d, 10 * nChanels + f);
								// a == c
								value[ind] += delta;
							}
						}
					}

					a = e;
					for (f = 0; f < nChanels; ++f) {
						for (b = 0; b < N; ++b) {
							for (c = 0; c < N; ++c) {

								delta = tensor -> value[tensor -> index(a, b, c, f)] * adj_value;

								// Case 12 (27/50): (b, d). Contract (a, e). Singleton (c).
								ind = index(b, d, 11 * nChanels + f);
								// a == e
								value[ind] += delta;
							}
						}
					}

					c = e;
					for (f = 0; f < nChanels; ++f) {
						for (a = 0; a < N; ++a) {
							for (b = 0; b < N; ++b) {

								delta = tensor -> value[tensor -> index(a, b, c, f)] * adj_value;

								// Case 13 (28/50): (b, d). Contract (c, e). Singleton (a).
								ind = index(b, d, 12 * nChanels + f);
								// c == e
								value[ind] += delta;
							}
						}
					}

					for (f = 0; f < nChanels; ++f) {
						for (a = 0; a < N; ++a) {
							for (c = 0; c < N; ++c) {
								
								b = a;
								delta = tensor -> value[tensor -> index(a, b, c, f)] * adj_value;
										
								// Case 14 (38/50): (d, e). Contract (a, b). Singleton (c).
								ind = index(d, e, 13 * nChanels + f);
								// a == b
								value[ind] += delta;
							}
						}
					}

					for (f = 0; f < nChanels; ++f) {
						for (a = 0; a < N; ++a) {
							for (b = 0; b < N; ++b) {
								
								c = b;
								delta = tensor -> value[tensor -> index(a, b, c, f)] * adj_value;

								// Case 15 (40/50): (d, e). Contract (b, c). Singleton (a).
								ind = index(d, e, 14 * nChanels + f);
								// b == c
								value[ind] += delta;
							}
						}
					}

					for (f = 0; f < nChanels; ++f) {
						for (a = 0; a < N; ++a) {
							
							b = e;
							c = e;
							delta = tensor -> value[tensor -> index(a, b, c, f)] * adj_value;

							// +---+
							// | 3 |
							// +---+

							// Case 16 (43/50): (a, d). Contract (b, c, e).
							ind = index(a, d, 15 * nChanels + f);
							// b == c && c == e
							value[ind] += delta;
						}
					}

					for (f = 0; f < nChanels; ++f) {
						for (b = 0; b < N; ++b) {
							
							a = e;
							c = e;
							delta = tensor -> value[tensor -> index(a, b, c, f)] * adj_value;

							// Case 17 (46/50): (b, d). Contract (a, c, e).
							ind = index(b, d, 16 * nChanels + f);
							// a == c && c == e
							value[ind] += delta;
						}
					}

					for (f = 0; f < nChanels; ++f) {
						for (a = 0; a < N; ++a) {
							
							b = a;
							c = a;
							delta = tensor -> value[tensor -> index(a, b, c, f)] * adj_value;

							// Case 18 (50/50): (d, e). Contract (a, b, c).
							ind = index(d, e, 17 * nChanels + f);
							// a == b && b == c
							value[ind] += delta;
						}
					}
				}
			}
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}

	// +----------------------+
	// | Backward Pass in CPU |
	// +----------------------+

	void backward_CPU() {
		int ind;
		double adj_value;

		int a, b, c, d, e, f;
		int t;

		for (d = 0; d < N; ++d) {
			for (e = 0; e < N; ++e) {
				adj_value = adj -> value[adj -> index(d, e)];

				if (adj_value > 0) {
					for (f = 0; f < nChanels; ++f) {
						for (a = 0; a < N; ++a) {
							for (b = 0; b < N; ++b) {
								for (c = 0; c < N; ++c) {

									// +-----------+
									// | 1 + 1 + 1 |
									// +-----------+

									t = tensor -> index(a, b, c, f);

									// Case 1 (1/50): Fix a, b. Contract c, d, e.
									ind = index(a, b, 0 * nChanels + f);
									tensor -> gradient[t] += gradient[ind] * adj_value;

									// Case 2 (3/50): Fix a, d. Contract b, c, e.
									ind = index(a, d, 1 * nChanels + f);
									tensor -> gradient[t] += gradient[ind] * adj_value;

									// Case 3 (5/50): Fix b, c. Contract a, d, e.
									ind = index(b, c, 2 * nChanels + f);
									tensor -> gradient[t] += gradient[ind] * adj_value;

									// Case 4 (6/50): Fix b, d. Contract a, c, e.
									ind = index(b, d, 3 * nChanels + f);
									tensor -> gradient[t] += gradient[ind] * adj_value;

									// Case 5 (10/50): Fix d, e. Contract a, b, c.
									ind = index(d, e, 4 * nChanels + f);
									tensor -> gradient[t] += gradient[ind] * adj_value;
								}
							}
						}
					}

					c = d;
					for (f = 0; f < nChanels; ++f) {
						for (a = 0; a < N; ++a) {
							for (b = 0; b < N; ++b) {

								t = tensor -> index(a, b, c, f);

								// Case 6 (11/50): (a, b). Contract (c, d). Singleton (e).
								ind = index(a, b, 5 * nChanels + f);
								// c == d
								tensor -> gradient[t] += gradient[ind] * adj_value;
							}
						}
					}

					if (d == e) {
						for (f = 0; f < nChanels; ++f) {
							for (a = 0; a < N; ++a) {
								for (b = 0; b < N; ++b) {
									for (c = 0; c < N; ++c) {
										
										t = tensor -> index(a, b, c, f);

										// Case 7 (13/50): (a, b). Contract (d, e). Singleton (c).
										ind = index(a, b, 6 * nChanels + f);
										// d == e
										tensor -> gradient[t] += gradient[ind] * adj_value;
									}
								}
							}
						}
					}

					for (f = 0; f < nChanels; ++f) {
						for (a = 0; a < N; ++a) {
							for (b = 0; b < N; ++b) {
								c = b;

								t = tensor -> index(a, b, c, f);

								// Case 8 (17/50): (a, d). Contract (b, c). Singleton (e).
								ind = index(a, d, 7 * nChanels + f);
								// b == c
								tensor -> gradient[t] += gradient[ind] * adj_value;
							}
						}
					}

					b = e;
					for (f = 0; f < nChanels; ++f) {
						for (a = 0; a < N; ++a) {
							for (c = 0; c < N; ++c) {

								t = tensor -> index(a, b, c, f);

								// Case 9 (18/50): (a, d). Contract (b, e). Singleton (c).
								ind = index(a, d, 8 * nChanels + f);
								// b == e
								tensor -> gradient[t] += gradient[ind] * adj_value;
							}
						}
					}

					a = d;
					for (f = 0; f < nChanels; ++f) {
						for (b = 0; b < N; ++b) {
							for (c = 0; c < N; ++c) {

								t = tensor -> index(a, b, c, f);

								// Case 10 (23/50): (b, c). Contract (a, d). Singleton (e).
								ind = index(b, c, 9 * nChanels + f);
								// a == d
								tensor -> gradient[t] += gradient[ind] * adj_value;
							}
						}
					}

					for (f = 0; f < nChanels; ++f) {
						for (a = 0; a < N; ++a) {
							for (b = 0; b < N; ++b) {
								
								c = a;

								t = tensor -> index(a, b, c, f);

								// Case 11 (26/50): (b, d). Contract (a, c). Singleton (e).
								ind = index(b, d, 10 * nChanels + f);
								// a == c
								tensor -> gradient[t] += gradient[ind] * adj_value;
							}
						}
					}

					a = e;
					for (f = 0; f < nChanels; ++f) {
						for (b = 0; b < N; ++b) {
							for (c = 0; c < N; ++c) {

								t = tensor -> index(a, b, c, f);

								// Case 12 (27/50): (b, d). Contract (a, e). Singleton (c).
								ind = index(b, d, 11 * nChanels + f);
								// a == e
								tensor -> gradient[t] += gradient[ind] * adj_value;
							}
						}
					}

					c = e;
					for (f = 0; f < nChanels; ++f) {
						for (a = 0; a < N; ++a) {
							for (b = 0; b < N; ++b) {

								t = tensor -> index(a, b, c, f);

								// Case 13 (28/50): (b, d). Contract (c, e). Singleton (a).
								ind = index(b, d, 12 * nChanels + f);
								// c == e
								tensor -> gradient[t] += gradient[ind] * adj_value;
							}
						}
					}

					for (f = 0; f < nChanels; ++f) {
						for (a = 0; a < N; ++a) {
							for (c = 0; c < N; ++c) {
								
								b = a;
										
								t = tensor -> index(a, b, c, f);

								// Case 14 (38/50): (d, e). Contract (a, b). Singleton (c).
								ind = index(d, e, 13 * nChanels + f);
								// a == b
								tensor -> gradient[t] += gradient[ind] * adj_value;
							}
						}
					}

					for (f = 0; f < nChanels; ++f) {
						for (a = 0; a < N; ++a) {
							for (b = 0; b < N; ++b) {
								
								c = b;

								t = tensor -> index(a, b, c, f);

								// Case 15 (40/50): (d, e). Contract (b, c). Singleton (a).
								ind = index(d, e, 14 * nChanels + f);
								// b == c
								tensor -> gradient[t] += gradient[ind] * adj_value;
							}
						}
					}

					for (f = 0; f < nChanels; ++f) {
						for (a = 0; a < N; ++a) {
							
							b = e;
							c = e;

							// +---+
							// | 3 |
							// +---+

							t = tensor -> index(a, b, c, f);

							// Case 16 (43/50): (a, d). Contract (b, c, e).
							ind = index(a, d, 15 * nChanels + f);
							// b == c && c == e
							tensor -> gradient[t] += gradient[ind] * adj_value;
						}
					}

					for (f = 0; f < nChanels; ++f) {
						for (b = 0; b < N; ++b) {
							
							a = e;
							c = e;

							t = tensor -> index(a, b, c, f);

							// Case 17 (46/50): (b, d). Contract (a, c, e).
							ind = index(b, d, 16 * nChanels + f);
							// a == c && c == e
							tensor -> gradient[t] += gradient[ind] * adj_value;
						}
					}

					for (f = 0; f < nChanels; ++f) {
						for (a = 0; a < N; ++a) {
							
							b = a;
							c = a;

							t = tensor -> index(a, b, c, f);

							// Case 18 (50/50): (d, e). Contract (a, b, c).
							ind = index(d, e, 17 * nChanels + f);
							// a == b && b == c
							tensor -> gradient[t] += gradient[ind] * adj_value;
						}
					}
				}
			}
		}
	}

	// +---------------------+
	// | Forward Pass in GPU |
	// +---------------------+

	void forward_GPU() {
		cudaError err;

		for (int i = 0; i < size; ++i) {
			value[i] = 0.0;
		}

		dim3 dimGrid(rounded_division(size, nThreads));
		dim3 dimBlock(nThreads);

		if (!use_gpu_stream) {

			// Default NULL stream

			err = cudaMemcpy(device_tensor_value, tensor -> value, tensor_size, cudaMemcpyHostToDevice);
			assert(err == cudaSuccess);

			err = cudaMemcpy(device_adj_value, adj -> value, adj_size, cudaMemcpyHostToDevice);
			assert(err == cudaSuccess);

			err = cudaMemcpy(device_value, value, this_size, cudaMemcpyHostToDevice);
			assert(err == cudaSuccess);

			// Kernel launch
			RisiContraction_18_forward_job <<< dimGrid, dimBlock >>> (device_tensor_value, device_adj_value, device_value, N, nChanels);

			// Thread synchronization
			// err = cudaThreadSynchronize();
			// assert(err == cudaSuccess); 

			err = cudaMemcpy(value, device_value, this_size, cudaMemcpyDeviceToHost);
			assert(err == cudaSuccess); 
		} else {

			// Multi-streams in GPU

			err = cudaMemcpyAsync(device_tensor_value, tensor -> value, tensor_size, cudaMemcpyHostToDevice, stream);
			assert(err == cudaSuccess);

			err = cudaMemcpyAsync(device_adj_value, adj -> value, adj_size, cudaMemcpyHostToDevice, stream);
			assert(err == cudaSuccess);

			err = cudaMemcpyAsync(device_value, value, this_size, cudaMemcpyHostToDevice, stream);
			assert(err == cudaSuccess);

			// Kernel launch
			RisiContraction_18_forward_job <<< dimGrid, dimBlock, 0, stream >>> (device_tensor_value, device_adj_value, device_value, N, nChanels);

			// Thread synchronization
			// err = cudaThreadSynchronize();
			// assert(err == cudaSuccess); 

			err = cudaMemcpyAsync(value, device_value, this_size, cudaMemcpyDeviceToHost, stream);
			assert(err == cudaSuccess); 
		}
		
		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}

	// +---------------------------------+
	// | DEPRECATED: Forward Pass in GPU |
	// +---------------------------------+

	void DEPRECATED_forward_GPU() {
		cudaError err;

		for (int i = 0; i < size; ++i) {
			value[i] = 0.0;
		}

		dim3 dimGrid(N, N);
		dim3 dimBlock(nThreads);

		if (!use_gpu_stream) {

			// Default NULL stream

			err = cudaMemcpy(device_tensor_value, tensor -> value, tensor_size, cudaMemcpyHostToDevice);
			assert(err == cudaSuccess);

			err = cudaMemcpy(device_adj_value, adj -> value, adj_size, cudaMemcpyHostToDevice);
			assert(err == cudaSuccess);

			err = cudaMemcpy(device_value, value, this_size, cudaMemcpyHostToDevice);
			assert(err == cudaSuccess);

			// Kernel launch
			DEPRECATED_RisiContraction_18_forward_job <<< dimGrid, dimBlock >>> (device_tensor_value, device_adj_value, device_value, N, nChanels);

			// Thread synchronization
			// err = cudaThreadSynchronize();
			// assert(err == cudaSuccess); 

			err = cudaMemcpy(value, device_value, this_size, cudaMemcpyDeviceToHost);
			assert(err == cudaSuccess); 
		} else {

			// Multi-streams in GPU

			err = cudaMemcpyAsync(device_tensor_value, tensor -> value, tensor_size, cudaMemcpyHostToDevice, stream);
			assert(err == cudaSuccess);

			err = cudaMemcpyAsync(device_adj_value, adj -> value, adj_size, cudaMemcpyHostToDevice, stream);
			assert(err == cudaSuccess);

			err = cudaMemcpyAsync(device_value, value, this_size, cudaMemcpyHostToDevice, stream);
			assert(err == cudaSuccess);

			// Kernel launch
			DEPRECATED_RisiContraction_18_forward_job <<< dimGrid, dimBlock, 0, stream >>> (device_tensor_value, device_adj_value, device_value, N, nChanels);

			// Thread synchronization
			// err = cudaThreadSynchronize();
			// assert(err == cudaSuccess); 

			err = cudaMemcpyAsync(value, device_value, this_size, cudaMemcpyDeviceToHost, stream);
			assert(err == cudaSuccess); 
		}
		
		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}

	// +----------------------+
	// | Backward Pass in GPU |
	// +----------------------+

	void backward_GPU() {
		cudaError err;

		dim3 dimGrid(rounded_division(tensor -> size, nThreads));
		dim3 dimBlock(nThreads);

		if (!use_gpu_stream) {

			// Default NULL stream

			err = cudaMemcpy(device_tensor_gradient, tensor -> gradient, tensor_size, cudaMemcpyHostToDevice);
			assert(err == cudaSuccess);

			err = cudaMemcpy(device_adj_value, adj -> value, adj_size, cudaMemcpyHostToDevice);
			assert(err == cudaSuccess);

			err = cudaMemcpy(device_gradient, gradient, this_size, cudaMemcpyHostToDevice);
			assert(err == cudaSuccess);

			// Kernel launch
			RisiContraction_18_backward_job <<< dimGrid, dimBlock >>> (device_tensor_gradient, device_adj_value, device_gradient, N, nChanels);

			// Thread synchronization
			// err = cudaThreadSynchronize();
			// assert(err == cudaSuccess); 

			err = cudaMemcpy(tensor -> gradient, device_tensor_gradient, tensor_size, cudaMemcpyDeviceToHost);
			assert(err == cudaSuccess);
		} else {

			// Multi-streams in GPU

			err = cudaMemcpyAsync(device_tensor_gradient, tensor -> gradient, tensor_size, cudaMemcpyHostToDevice, stream);
			assert(err == cudaSuccess);

			err = cudaMemcpyAsync(device_adj_value, adj -> value, adj_size, cudaMemcpyHostToDevice, stream);
			assert(err == cudaSuccess);

			err = cudaMemcpyAsync(device_gradient, gradient, this_size, cudaMemcpyHostToDevice, stream);
			assert(err == cudaSuccess);

			RisiContraction_18_backward_job <<< dimGrid, dimBlock, 0, stream >>> (device_tensor_gradient, device_adj_value, device_gradient, N, nChanels);

			// Thread synchronization
			// err = cudaThreadSynchronize();
			// assert(err == cudaSuccess); 

			err = cudaMemcpyAsync(tensor -> gradient, device_tensor_gradient, tensor_size, cudaMemcpyDeviceToHost, stream);
			assert(err == cudaSuccess);
		}
	}

	// +----------------------------------+
	// | DEPRECATED: Backward Pass in GPU |
	// +----------------------------------+

	void DEPRECATED_backward_GPU() {
		cudaError err;

		dim3 dimGrid(N, N);
		dim3 dimBlock(nThreads);

		if (!use_gpu_stream) {

			// Default NULL stream

			err = cudaMemcpy(device_tensor_gradient, tensor -> gradient, tensor_size, cudaMemcpyHostToDevice);
			assert(err == cudaSuccess);

			err = cudaMemcpy(device_adj_value, adj -> value, adj_size, cudaMemcpyHostToDevice);
			assert(err == cudaSuccess);

			err = cudaMemcpy(device_gradient, gradient, this_size, cudaMemcpyHostToDevice);
			assert(err == cudaSuccess);

			// Kernel launch
			DEPRECATED_RisiContraction_18_backward_job <<< dimGrid, dimBlock >>> (device_tensor_gradient, device_adj_value, device_gradient, N, nChanels);

			// Thread synchronization
			// err = cudaThreadSynchronize();
			// assert(err == cudaSuccess); 

			err = cudaMemcpy(tensor -> gradient, device_tensor_gradient, tensor_size, cudaMemcpyDeviceToHost);
			assert(err == cudaSuccess);
		} else {

			// Multi-streams in GPU

			err = cudaMemcpyAsync(device_tensor_gradient, tensor -> gradient, tensor_size, cudaMemcpyHostToDevice, stream);
			assert(err == cudaSuccess);

			err = cudaMemcpyAsync(device_adj_value, adj -> value, adj_size, cudaMemcpyHostToDevice, stream);
			assert(err == cudaSuccess);

			err = cudaMemcpyAsync(device_gradient, gradient, this_size, cudaMemcpyHostToDevice, stream);
			assert(err == cudaSuccess);

			// Kernel launch
			DEPRECATED_RisiContraction_18_backward_job <<< dimGrid, dimBlock, 0, stream >>> (device_tensor_gradient, device_adj_value, device_gradient, N, nChanels);

			// Thread synchronization
			// err = cudaThreadSynchronize();
			// assert(err == cudaSuccess); 

			err = cudaMemcpyAsync(tensor -> gradient, device_tensor_gradient, tensor_size, cudaMemcpyDeviceToHost, stream);
			assert(err == cudaSuccess);
		}
	}

	// Number of contractions implemented in this class
	static const int nContractions = 18;

	// Number of threads in a GPU block
	static const int nThreads = 512;

	// Threshold of complexity to decide GPU or CPU
	static const int COMPLEXITY_THRESHOLD = 1e6;

	// Using GPU streams
	bool use_gpu_stream;

	// GPU stream
	cudaStream_t stream;

	// The size of the receptive field
	int N;

	// Number of chanels
	int nChanels;

	// Tensor of 4 dimensions
	Tensor4D *tensor;

	// The reduced adjacency matrix
	Matrix *adj;

	// GPU device - Tensor of 4 dimensions
	double *device_tensor_value;
	double *device_tensor_gradient;

	// GPU device - The reduced adjacency matrix
	double *device_adj_value;

	// GPU device - For this object
	double *device_value;
	double *device_gradient;

	// Sizes
	size_t tensor_size;
	size_t adj_size;
	size_t this_size;

	void handleCudaError(cudaError err) {
		if (err != cudaSuccess) {
			printf("%s\n", cudaGetErrorString(err));
		}
	}

	~RisiContraction_18_gpu() {
		cudaFree(device_tensor_value);
		cudaFree(device_tensor_gradient);
		cudaFree(device_adj_value);
		cudaFree(device_value);
		cudaFree(device_gradient);
	}
};

#endif