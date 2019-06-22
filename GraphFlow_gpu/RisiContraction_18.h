// Framework: GraphFlow
// Class: RisiContraction_18
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __RISICONTRACTION_18_H_INCLUDED__
#define __RISICONTRACTION_18_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <assert.h>

#include "Tensor3D.h"

using namespace std;

class RisiContraction_18: public Tensor3D {
public:
	RisiContraction_18(int max_nRows, int max_nColumns, int max_nDepth) : Tensor3D(max_nRows, max_nColumns, max_nDepth) {
		tensors.clear();
	}

	RisiContraction_18(int N, int nChanels) : Tensor3D(N, N, nContractions * nChanels) {
		this -> N = N;
		this -> nChanels = nChanels;

		tensors.clear();
	}

	void setParameter(int N, int nChanels) {
		this -> N = N;
		this -> nChanels = nChanels;

		this -> nRows = N;
		this -> nColumns = N;
		this -> nDepth = nChanels * nContractions;
		size = nRows * nColumns * nDepth;

		tensors.clear();
	}

	void add_tensor(Tensor3D *tensor) {
		assert(tensor -> nRows == N);
		assert(tensor -> nColumns == N);
		assert(tensor -> nDepth == nChanels);
		tensors.push_back(tensor);
	}

	void set_adjacency(Matrix *adj) {
		assert(adj -> nRows == N);
		assert(adj -> nColumns == N);
		this -> adj = adj;
	}

	void clear() {
		tensors.clear();
	}

	double value_at(int a, int b, int c, int d, int e, int f) {
		return tensors[a] -> value[tensors[a] -> index(b, c, f)] * adj -> value[adj -> index(d, e)];
	}

	void set_gradient_for(int a, int b, int c, int d, int e, int f, double grad) {
		tensors[a] -> gradient[tensors[a] -> index(b, c, f)] += grad * adj -> value[adj -> index(d, e)];
	}

	void forward() {
		assert(tensors.size() == N);

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
									
									delta = tensors[a] -> value[tensors[a] -> index(b, c, f)] * adj_value;

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

								delta = tensors[a] -> value[tensors[a] -> index(b, c, f)] * adj_value;

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
										
										delta = tensors[a] -> value[tensors[a] -> index(b, c, f)] * adj_value;

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
								delta = tensors[a] -> value[tensors[a] -> index(b, c, f)] * adj_value;

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

								delta = tensors[a] -> value[tensors[a] -> index(b, c, f)] * adj_value;

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

								delta = tensors[a] -> value[tensors[a] -> index(b, c, f)] * adj_value;
									
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
								delta = tensors[a] -> value[tensors[a] -> index(b, c, f)] * adj_value;

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

								delta = tensors[a] -> value[tensors[a] -> index(b, c, f)] * adj_value;

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

								delta = tensors[a] -> value[tensors[a] -> index(b, c, f)] * adj_value;

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
								delta = tensors[a] -> value[tensors[a] -> index(b, c, f)] * adj_value;
										
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
								delta = tensors[a] -> value[tensors[a] -> index(b, c, f)] * adj_value;

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
							delta = tensors[a] -> value[tensors[a] -> index(b, c, f)] * adj_value;

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
							delta = tensors[a] -> value[tensors[a] -> index(b, c, f)] * adj_value;

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
							delta = tensors[a] -> value[tensors[a] -> index(b, c, f)] * adj_value;

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

	void backward() {
		assert(tensors.size() == N);

		int ind;
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

									// +-----------+
									// | 1 + 1 + 1 |
									// +-----------+

									// Case 1 (1/50): Fix a, b. Contract c, d, e.
									ind = index(a, b, 0 * nChanels + f);
									tensors[a] -> gradient[tensors[a] -> index(b, c, f)] += gradient[ind] * adj_value;

									// Case 2 (3/50): Fix a, d. Contract b, c, e.
									ind = index(a, d, 1 * nChanels + f);
									tensors[a] -> gradient[tensors[a] -> index(b, c, f)] += gradient[ind] * adj_value;

									// Case 3 (5/50): Fix b, c. Contract a, d, e.
									ind = index(b, c, 2 * nChanels + f);
									tensors[a] -> gradient[tensors[a] -> index(b, c, f)] += gradient[ind] * adj_value;

									// Case 4 (6/50): Fix b, d. Contract a, c, e.
									ind = index(b, d, 3 * nChanels + f);
									tensors[a] -> gradient[tensors[a] -> index(b, c, f)] += gradient[ind] * adj_value;

									// Case 5 (10/50): Fix d, e. Contract a, b, c.
									ind = index(d, e, 4 * nChanels + f);
									tensors[a] -> gradient[tensors[a] -> index(b, c, f)] += gradient[ind] * adj_value;
								}
							}
						}
					}

					c = d;
					for (f = 0; f < nChanels; ++f) {
						for (a = 0; a < N; ++a) {
							for (b = 0; b < N; ++b) {

								// Case 6 (11/50): (a, b). Contract (c, d). Singleton (e).
								ind = index(a, b, 5 * nChanels + f);
								// c == d
								tensors[a] -> gradient[tensors[a] -> index(b, c, f)] += gradient[ind] * adj_value;
							}
						}
					}

					if (d == e) {
						for (f = 0; f < nChanels; ++f) {
							for (a = 0; a < N; ++a) {
								for (b = 0; b < N; ++b) {
									for (c = 0; c < N; ++c) {
										
										// Case 7 (13/50): (a, b). Contract (d, e). Singleton (c).
										ind = index(a, b, 6 * nChanels + f);
										// d == e
										tensors[a] -> gradient[tensors[a] -> index(b, c, f)] += gradient[ind] * adj_value;
									}
								}
							}
						}
					}

					for (f = 0; f < nChanels; ++f) {
						for (a = 0; a < N; ++a) {
							for (b = 0; b < N; ++b) {
								c = b;

								// Case 8 (17/50): (a, d). Contract (b, c). Singleton (e).
								ind = index(a, d, 7 * nChanels + f);
								// b == c
								tensors[a] -> gradient[tensors[a] -> index(b, c, f)] += gradient[ind] * adj_value;
							}
						}
					}

					b = e;
					for (f = 0; f < nChanels; ++f) {
						for (a = 0; a < N; ++a) {
							for (c = 0; c < N; ++c) {

								// Case 9 (18/50): (a, d). Contract (b, e). Singleton (c).
								ind = index(a, d, 8 * nChanels + f);
								// b == e
								tensors[a] -> gradient[tensors[a] -> index(b, c, f)] += gradient[ind] * adj_value;
							}
						}
					}

					a = d;
					for (f = 0; f < nChanels; ++f) {
						for (b = 0; b < N; ++b) {
							for (c = 0; c < N; ++c) {

								// Case 10 (23/50): (b, c). Contract (a, d). Singleton (e).
								ind = index(b, c, 9 * nChanels + f);
								// a == d
								tensors[a] -> gradient[tensors[a] -> index(b, c, f)] += gradient[ind] * adj_value;
							}
						}
					}

					for (f = 0; f < nChanels; ++f) {
						for (a = 0; a < N; ++a) {
							for (b = 0; b < N; ++b) {
								
								c = a;

								// Case 11 (26/50): (b, d). Contract (a, c). Singleton (e).
								ind = index(b, d, 10 * nChanels + f);
								// a == c
								tensors[a] -> gradient[tensors[a] -> index(b, c, f)] += gradient[ind] * adj_value;
							}
						}
					}

					a = e;
					for (f = 0; f < nChanels; ++f) {
						for (b = 0; b < N; ++b) {
							for (c = 0; c < N; ++c) {

								// Case 12 (27/50): (b, d). Contract (a, e). Singleton (c).
								ind = index(b, d, 11 * nChanels + f);
								// a == e
								tensors[a] -> gradient[tensors[a] -> index(b, c, f)] += gradient[ind] * adj_value;
							}
						}
					}

					c = e;
					for (f = 0; f < nChanels; ++f) {
						for (a = 0; a < N; ++a) {
							for (b = 0; b < N; ++b) {

								// Case 13 (28/50): (b, d). Contract (c, e). Singleton (a).
								ind = index(b, d, 12 * nChanels + f);
								// c == e
								tensors[a] -> gradient[tensors[a] -> index(b, c, f)] += gradient[ind] * adj_value;
							}
						}
					}

					for (f = 0; f < nChanels; ++f) {
						for (a = 0; a < N; ++a) {
							for (c = 0; c < N; ++c) {
								
								b = a;
										
								// Case 14 (38/50): (d, e). Contract (a, b). Singleton (c).
								ind = index(d, e, 13 * nChanels + f);
								// a == b
								tensors[a] -> gradient[tensors[a] -> index(b, c, f)] += gradient[ind] * adj_value;
							}
						}
					}

					for (f = 0; f < nChanels; ++f) {
						for (a = 0; a < N; ++a) {
							for (b = 0; b < N; ++b) {
								
								c = b;

								// Case 15 (40/50): (d, e). Contract (b, c). Singleton (a).
								ind = index(d, e, 14 * nChanels + f);
								// b == c
								tensors[a] -> gradient[tensors[a] -> index(b, c, f)] += gradient[ind] * adj_value;
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

							// Case 16 (43/50): (a, d). Contract (b, c, e).
							ind = index(a, d, 15 * nChanels + f);
							// b == c && c == e
							tensors[a] -> gradient[tensors[a] -> index(b, c, f)] += gradient[ind] * adj_value;
						}
					}

					for (f = 0; f < nChanels; ++f) {
						for (b = 0; b < N; ++b) {
							
							a = e;
							c = e;

							// Case 17 (46/50): (b, d). Contract (a, c, e).
							ind = index(b, d, 16 * nChanels + f);
							// a == c && c == e
							tensors[a] -> gradient[tensors[a] -> index(b, c, f)] += gradient[ind] * adj_value;
						}
					}

					for (f = 0; f < nChanels; ++f) {
						for (a = 0; a < N; ++a) {
							
							b = a;
							c = a;

							// Case 18 (50/50): (d, e). Contract (a, b, c).
							ind = index(d, e, 17 * nChanels + f);
							// a == b && b == c
							tensors[a] -> gradient[tensors[a] -> index(b, c, f)] += gradient[ind] * adj_value;
						}
					}
				}
			}
		}
	}

	void DEPRECATED_forward() {
		assert(tensors.size() == N);

		for (int i = 0; i < size; ++i) {
			value[i] = 0.0;
		}

		int Case;
		int ind;

		for (int f = 0; f < nChanels; ++f) {
			for (int a = 0; a < N; ++a) {
				for (int b = 0; b < N; ++b) {
					for (int c = 0; c < N; ++c) {
						for (int d = 0; d < N; ++d) {
							for (int e = 0; e < N; ++e) {

								// +-----------+
								// | 1 + 1 + 1 |
								// +-----------+

								// Case 1 (1/50): Fix a, b. Contract c, d, e.
								Case = 1;
								ind = index(a, b, (Case - 1) * nChanels + f);
								value[ind] += value_at(a, b, c, d, e, f);

								// Case 2 (3/50): Fix a, d. Contract b, c, e.
								Case = 2;
								ind = index(a, d, (Case - 1) * nChanels + f);
								value[ind] += value_at(a, b, c, d, e, f);

								// Case 3 (5/50): Fix b, c. Contract a, d, e.
								Case = 3;
								ind = index(b, c, (Case - 1) * nChanels + f);
								value[ind] += value_at(a, b, c, d, e, f);

								// Case 4 (6/50): Fix b, d. Contract a, c, e.
								Case = 4;
								ind = index(b, d, (Case - 1) * nChanels + f);
								value[ind] += value_at(a, b, c, d, e, f);

								// Case 5 (10/50): Fix d, e. Contract a, b, c.
								Case = 5;
								ind = index(d, e, (Case - 1) * nChanels + f);
								value[ind] += value_at(a, b, c, d, e, f);

								// +-------+
								// | 1 + 2 |
								// +-------+

								// Case 6 (11/50): (a, b). Contract (c, d). Singleton (e).
								Case = 6;
								ind = index(a, b, (Case - 1) * nChanels + f);
								if (c == d) {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 7 (13/50): (a, b). Contract (d, e). Singleton (c).
								Case = 7;
								ind = index(a, b, (Case - 1) * nChanels + f);
								if (d == e) {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 8 (17/50): (a, d). Contract (b, c). Singleton (e).
								Case = 8;
								ind = index(a, d, (Case - 1) * nChanels + f);
								if (b == c) {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 9 (18/50): (a, d). Contract (b, e). Singleton (c).
								Case = 9;
								ind = index(a, d, (Case - 1) * nChanels + f);
								if (b == e) {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 10 (23/50): (b, c). Contract (a, d). Singleton (e).
								Case = 10;
								ind = index(b, c, (Case - 1) * nChanels + f);
								if (a == d) {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 11 (26/50): (b, d). Contract (a, c). Singleton (e).
								Case = 11;
								ind = index(b, d, (Case - 1) * nChanels + f);
								if (a == c) {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 12 (27/50): (b, d). Contract (a, e). Singleton (c).
								Case = 12;
								ind = index(b, d, (Case - 1) * nChanels + f);
								if (a == e) {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 13 (28/50): (b, d). Contract (c, e). Singleton (a).
								Case = 13;
								ind = index(b, d, (Case - 1) * nChanels + f);
								if (c == e) {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 14 (38/50): (d, e). Contract (a, b). Singleton (c).
								Case = 14;
								ind = index(d, e, (Case - 1) * nChanels + f);
								if (a == b) {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 15 (40/50): (d, e). Contract (b, c). Singleton (a).
								Case = 15;
								ind = index(d, e, (Case - 1) * nChanels + f);
								if (b == c) {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// +---+
								// | 3 |
								// +---+

								// Case 16 (43/50): (a, d). Contract (b, c, e).
								Case = 16;
								ind = index(a, d, (Case - 1) * nChanels + f);
								if ((b == c) && (c == e))  {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 17 (46/50): (b, d). Contract (a, c, e).
								Case = 17;
								ind = index(b, d, (Case - 1) * nChanels + f);
								if ((a == c) && (c == e))  {
									value[ind] += value_at(a, b, c, d, e, f);
								}

								// Case 18 (50/50): (d, e). Contract (a, b, c).
								Case = 18;
								ind = index(d, e, (Case - 1) * nChanels + f);
								if ((a == b) && (b == c))  {
									value[ind] += value_at(a, b, c, d, e, f);
								}
							}
						}
					}
				}
			}
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}

	void DEPRECATED_backward() {
		int Case;
		int ind;

		for (int f = 0; f < nChanels; ++f) {
			for (int a = 0; a < N; ++a) {
				for (int b = 0; b < N; ++b) {
					for (int c = 0; c < N; ++c) {
						for (int d = 0; d < N; ++d) {
							for (int e = 0; e < N; ++e) {

								// +-----------+
								// | 1 + 1 + 1 |
								// +-----------+

								// Case 1 (1/50): Fix a, b. Contract c, d, e.
								Case = 1;
								ind = index(a, b, (Case - 1) * nChanels + f);
								set_gradient_for(a, b, c, d, e, f, gradient[ind]);

								// Case 2 (3/50): Fix a, d. Contract b, c, e.
								Case = 2;
								ind = index(a, d, (Case - 1) * nChanels + f);
								set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								
								// Case 3 (5/50): Fix b, c. Contract a, d, e.
								Case = 3;
								ind = index(b, c, (Case - 1) * nChanels + f);
								set_gradient_for(a, b, c, d, e, f, gradient[ind]);

								// Case 4 (6/50): Fix b, d. Contract a, c, e.
								Case = 4;
								ind = index(b, d, (Case - 1) * nChanels + f);
								set_gradient_for(a, b, c, d, e, f, gradient[ind]);

								// Case 5 (10/50): Fix d, e. Contract a, b, c.
								Case = 5;
								ind = index(d, e, (Case - 1) * nChanels + f);
								set_gradient_for(a, b, c, d, e, f, gradient[ind]);

								// +-------+
								// | 1 + 2 |
								// +-------+

								// Case 6 (11/50): (a, b). Contract (c, d). Singleton (e).
								Case = 6;
								ind = index(a, b, (Case - 1) * nChanels + f);
								if (c == d) {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 7 (13/50): (a, b). Contract (d, e). Singleton (c).
								Case = 7;
								ind = index(a, b, (Case - 1) * nChanels + f);
								if (d == e) {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 8 (17/50): (a, d). Contract (b, c). Singleton (e).
								Case = 8;
								ind = index(a, d, (Case - 1) * nChanels + f);
								if (b == c) {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 9 (18/50): (a, d). Contract (b, e). Singleton (c).
								Case = 9;
								ind = index(a, d, (Case - 1) * nChanels + f);
								if (b == e) {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 10 (23/50): (b, c). Contract (a, d). Singleton (e).
								Case = 10;
								ind = index(b, c, (Case - 1) * nChanels + f);
								if (a == d) {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 11 (26/50): (b, d). Contract (a, c). Singleton (e).
								Case = 11;
								ind = index(b, d, (Case - 1) * nChanels + f);
								if (a == c) {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 12 (27/50): (b, d). Contract (a, e). Singleton (c).
								Case = 12;
								ind = index(b, d, (Case - 1) * nChanels + f);
								if (a == e) {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 13 (28/50): (b, d). Contract (c, e). Singleton (a).
								Case = 13;
								ind = index(b, d, (Case - 1) * nChanels + f);
								if (c == e) {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 14 (38/50): (d, e). Contract (a, b). Singleton (c).
								Case = 14;
								ind = index(d, e, (Case - 1) * nChanels + f);
								if (a == b) {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 15 (40/50): (d, e). Contract (b, c). Singleton (a).
								Case = 15;
								ind = index(d, e, (Case - 1) * nChanels + f);
								if (b == c) {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// +---+
								// | 3 |
								// +---+

								// Case 16 (43/50): (a, d). Contract (b, c, e).
								Case = 16;
								ind = index(a, d, (Case - 1) * nChanels + f);
								if ((b == c) && (c == e))  {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 17 (46/50): (b, d). Contract (a, c, e).
								Case = 17;
								ind = index(b, d, (Case - 1) * nChanels + f);
								if ((a == c) && (c == e))  {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}

								// Case 18 (50/50): (d, e). Contract (a, b, c).
								Case = 18;
								ind = index(d, e, (Case - 1) * nChanels + f);
								if ((a == b) && (b == c))  {
									set_gradient_for(a, b, c, d, e, f, gradient[ind]);
								}
							}
						}
					}
				}
			}
		}
	}

	// Number of contractions implemented in this class
	static const int nContractions = 3;

	// The size of the receptive field
	int N;

	// Number of chanels
	int nChanels;

	// Neighbors' representations
	vector < Tensor3D* > tensors;

	// The reduced adjacency matrix
	Matrix *adj;

	~RisiContraction_18() {
		tensors.clear();
	}
};

#endif