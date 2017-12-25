// Framework: GraphFlow
// Class: RisiContraction_18_thread
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __RISICONTRACTION_18_THREAD_H_INCLUDED__
#define __RISICONTRACTION_18_THREAD_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <assert.h>
#include <thread>

#include "Tensor3D.h"

using namespace std;

class RisiContraction_18_thread: public Tensor3D {
public:
	RisiContraction_18_thread(int max_nRows, int max_nColumns, int max_nDepth) : Tensor3D(max_nRows, max_nColumns, max_nDepth) {
		assert(DEPRECATED == false);

		tensors.clear();
	}

	RisiContraction_18_thread(int N, int nChanels) : Tensor3D(N, N, nContractions * nChanels) {
		assert(DEPRECATED == false);

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

	// Contraction 1 - Contraction 3
	static void forward_job_0(RisiContraction_18_thread *obj) {
		int nChanels = obj -> nChanels;
		int N = obj -> N;

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
								ind = obj -> index(a, b, (Case - 1) * nChanels + f);
								obj -> value[ind] += obj -> value_at(a, b, c, d, e, f);

								// Case 2 (3/50): Fix a, d. Contract b, c, e.
								Case = 2;
								ind = obj -> index(a, d, (Case - 1) * nChanels + f);
								obj -> value[ind] += obj -> value_at(a, b, c, d, e, f);

								// Case 3 (5/50): Fix b, c. Contract a, d, e.
								Case = 3;
								ind = obj -> index(b, c, (Case - 1) * nChanels + f);
								obj -> value[ind] += obj -> value_at(a, b, c, d, e, f);
							}
						}
					}
				}
			}
		}
	}

	// Contraction 4 - Contraction 6
	static void forward_job_1(RisiContraction_18_thread *obj) {
		int nChanels = obj -> nChanels;
		int N = obj -> N;

		int Case;
		int ind;

		for (int f = 0; f < nChanels; ++f) {
			for (int a = 0; a < N; ++a) {
				for (int b = 0; b < N; ++b) {
					for (int c = 0; c < N; ++c) {
						for (int d = 0; d < N; ++d) {
							for (int e = 0; e < N; ++e) {

								// Case 4 (6/50): Fix b, d. Contract a, c, e.
								Case = 4;
								ind = obj -> index(b, d, (Case - 1) * nChanels + f);
								obj -> value[ind] += obj -> value_at(a, b, c, d, e, f);

								// Case 5 (10/50): Fix d, e. Contract a, b, c.
								Case = 5;
								ind = obj -> index(d, e, (Case - 1) * nChanels + f);
								obj -> value[ind] += obj -> value_at(a, b, c, d, e, f);
							}
						}
					}
				}
			}
		}

		for (int f = 0; f < nChanels; ++f) {
			for (int a = 0; a < N; ++a) {
				for (int b = 0; b < N; ++b) {
					for (int c = 0; c < N; ++c) {
						for (int e = 0; e < N; ++e) {

							// +-------+
							// | 1 + 2 |
							// +-------+

							// Case 6 (11/50): (a, b). Contract (c, d). Singleton (e).
							Case = 6;
							ind = obj -> index(a, b, (Case - 1) * nChanels + f);
							// c == d
							obj -> value[ind] += obj -> value_at(a, b, c, c, e, f);
						}
					}
				}
			}
		}
	}

	// Contraction 7 - Contraction 9
	static void forward_job_2(RisiContraction_18_thread *obj) {
		int nChanels = obj -> nChanels;
		int N = obj -> N;

		int Case;
		int ind;

		for (int f = 0; f < nChanels; ++f) {
			for (int a = 0; a < N; ++a) {
				for (int b = 0; b < N; ++b) {
					for (int c = 0; c < N; ++c) {
						for (int d = 0; d < N; ++d) {
							
							// Case 7 (13/50): (a, b). Contract (d, e). Singleton (c).
							Case = 7;
							ind = obj -> index(a, b, (Case - 1) * nChanels + f);
							// d == e
							obj -> value[ind] += obj -> value_at(a, b, c, d, d, f);
						}
					}
				}
			}
		}

		for (int f = 0; f < nChanels; ++f) {
			for (int a = 0; a < N; ++a) {
				for (int b = 0; b < N; ++b) {
					for (int d = 0; d < N; ++d) {
						for (int e = 0; e < N; ++e) {

							// Case 8 (17/50): (a, d). Contract (b, c). Singleton (e).
							Case = 8;
							ind = obj -> index(a, d, (Case - 1) * nChanels + f);
							// b == c
							obj -> value[ind] += obj -> value_at(a, b, b, d, e, f);
						}
					}
				}
			}
		}

		for (int f = 0; f < nChanels; ++f) {
			for (int a = 0; a < N; ++a) {
				for (int b = 0; b < N; ++b) {
					for (int c = 0; c < N; ++c) {
						for (int d = 0; d < N; ++d) {

							// Case 9 (18/50): (a, d). Contract (b, e). Singleton (c).
							Case = 9;
							ind = obj -> index(a, d, (Case - 1) * nChanels + f);
							// b == e
							obj -> value[ind] += obj -> value_at(a, b, c, d, b, f);
						}
					}
				}
			}
		}
	}

	// Contraction 10 - Contraction 12
	static void forward_job_3(RisiContraction_18_thread *obj) {
		int nChanels = obj -> nChanels;
		int N = obj -> N;

		int Case;
		int ind;

		for (int f = 0; f < nChanels; ++f) {
			for (int a = 0; a < N; ++a) {
				for (int b = 0; b < N; ++b) {
					for (int c = 0; c < N; ++c) {
						for (int e = 0; e < N; ++e) {

							// Case 10 (23/50): (b, c). Contract (a, d). Singleton (e).
							Case = 10;
							ind = obj -> index(b, c, (Case - 1) * nChanels + f);
							// a == d
							obj -> value[ind] += obj -> value_at(a, b, c, a, e, f);
						}
					}
				}
			}
		}

		for (int f = 0; f < nChanels; ++f) {
			for (int a = 0; a < N; ++a) {
				for (int b = 0; b < N; ++b) {
					for (int d = 0; d < N; ++d) {
						for (int e = 0; e < N; ++e) {

							// Case 11 (26/50): (b, d). Contract (a, c). Singleton (e).
							Case = 11;
							ind = obj -> index(b, d, (Case - 1) * nChanels + f);
							// a == c
							obj -> value[ind] += obj -> value_at(a, b, a, d, e, f);
						}
					}
				}
			}
		}

		for (int f = 0; f < nChanels; ++f) {
			for (int a = 0; a < N; ++a) {
				for (int b = 0; b < N; ++b) {
					for (int c = 0; c < N; ++c) {
						for (int d = 0; d < N; ++d) {

							// Case 12 (27/50): (b, d). Contract (a, e). Singleton (c).
							Case = 12;
							ind = obj -> index(b, d, (Case - 1) * nChanels + f);
							// a == e
							obj -> value[ind] += obj -> value_at(a, b, c, d, a, f);
						}
					}
				}
			}
		}
	}

	// Contraction 13 - Contraction 15
	static void forward_job_4(RisiContraction_18_thread *obj) {
		int nChanels = obj -> nChanels;
		int N = obj -> N;

		int Case;
		int ind;

		for (int f = 0; f < nChanels; ++f) {
			for (int a = 0; a < N; ++a) {
				for (int b = 0; b < N; ++b) {
					for (int c = 0; c < N; ++c) {
						for (int d = 0; d < N; ++d) {

							// Case 13 (28/50): (b, d). Contract (c, e). Singleton (a).
							Case = 13;
							ind = obj -> index(b, d, (Case - 1) * nChanels + f);
							// c == e
							obj -> value[ind] += obj -> value_at(a, b, c, d, c, f);
						}
					}
				}
			}
		}

		for (int f = 0; f < nChanels; ++f) {
			for (int a = 0; a < N; ++a) {
				for (int c = 0; c < N; ++c) {
					for (int d = 0; d < N; ++d) {
						for (int e = 0; e < N; ++e) {

							// Case 14 (38/50): (d, e). Contract (a, b). Singleton (c).
							Case = 14;
							ind = obj -> index(d, e, (Case - 1) * nChanels + f);
							// a == b
							obj -> value[ind] += obj -> value_at(a, a, c, d, e, f);
						}
					}
				}
			}
		}

		for (int f = 0; f < nChanels; ++f) {
			for (int a = 0; a < N; ++a) {
				for (int b = 0; b < N; ++b) {
					for (int d = 0; d < N; ++d) {
						for (int e = 0; e < N; ++e) {

							// Case 15 (40/50): (d, e). Contract (b, c). Singleton (a).
							Case = 15;
							ind = obj -> index(d, e, (Case - 1) * nChanels + f);
							// b == c
							obj -> value[ind] += obj -> value_at(a, b, b, d, e, f);
						}
					}
				}
			}
		}
	}

	// Contraction 16 - Contraction 18
	static void forward_job_5(RisiContraction_18_thread *obj) {
		int nChanels = obj -> nChanels;
		int N = obj -> N;

		int Case;
		int ind;

		for (int f = 0; f < nChanels; ++f) {
			for (int a = 0; a < N; ++a) {
				for (int b = 0; b < N; ++b) {
					for (int d = 0; d < N; ++d) {

						// +---+
						// | 3 |
						// +---+

						// Case 16 (43/50): (a, d). Contract (b, c, e).
						Case = 16;
						ind = obj -> index(a, d, (Case - 1) * nChanels + f);
						// b == c && c == e
						obj -> value[ind] += obj -> value_at(a, b, b, d, b, f);
					}
				}
			}
		}

		for (int f = 0; f < nChanels; ++f) {
			for (int a = 0; a < N; ++a) {
				for (int b = 0; b < N; ++b) {
					for (int d = 0; d < N; ++d) {
								
						// Case 17 (46/50): (b, d). Contract (a, c, e).
						Case = 17;
						ind = obj -> index(b, d, (Case - 1) * nChanels + f);
						// a == c && c == e
						obj -> value[ind] += obj -> value_at(a, b, a, d, a, f);
					}
				}
			}
		}

		for (int f = 0; f < nChanels; ++f) {
			for (int a = 0; a < N; ++a) {
				for (int d = 0; d < N; ++d) {
					for (int e = 0; e < N; ++e) {

						// Case 18 (50/50): (d, e). Contract (a, b, c).
						Case = 18;
						ind = obj -> index(d, e, (Case - 1) * nChanels + f);
						// a == b && b == c
						obj -> value[ind] += obj -> value_at(a, a, a, d, e, f);
					}
				}
			}
		}
	}

	// Contraction 1 - Contraction 3
	static void backward_job_0(RisiContraction_18_thread *obj) {
		int nChanels = obj -> nChanels;
		int N = obj -> N;

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
								ind = obj -> index(a, b, (Case - 1) * nChanels + f);
								obj -> set_gradient_for(a, b, c, d, e, f, obj -> gradient[ind]);

								// Case 2 (3/50): Fix a, d. Contract b, c, e.
								Case = 2;
								ind = obj -> index(a, d, (Case - 1) * nChanels + f);
								obj -> set_gradient_for(a, b, c, d, e, f, obj -> gradient[ind]);

								// Case 3 (5/50): Fix b, c. Contract a, d, e.
								Case = 3;
								ind = obj -> index(b, c, (Case - 1) * nChanels + f);
								obj -> set_gradient_for(a, b, c, d, e, f, obj -> gradient[ind]);
							}
						}
					}
				}
			}
		}
	}

	// Contraction 4 - Contraction 6
	static void backward_job_1(RisiContraction_18_thread *obj) {
		int nChanels = obj -> nChanels;
		int N = obj -> N;

		int Case;
		int ind;

		for (int f = 0; f < nChanels; ++f) {
			for (int a = 0; a < N; ++a) {
				for (int b = 0; b < N; ++b) {
					for (int c = 0; c < N; ++c) {
						for (int d = 0; d < N; ++d) {
							for (int e = 0; e < N; ++e) {

								// Case 4 (6/50): Fix b, d. Contract a, c, e.
								Case = 4;
								ind = obj -> index(b, d, (Case - 1) * nChanels + f);
								obj -> set_gradient_for(a, b, c, d, e, f, obj -> gradient[ind]);

								// Case 5 (10/50): Fix d, e. Contract a, b, c.
								Case = 5;
								ind = obj -> index(d, e, (Case - 1) * nChanels + f);
								obj -> set_gradient_for(a, b, c, d, e, f, obj -> gradient[ind]);
							}
						}
					}
				}
			}
		}

		for (int f = 0; f < nChanels; ++f) {
			for (int a = 0; a < N; ++a) {
				for (int b = 0; b < N; ++b) {
					for (int c = 0; c < N; ++c) {
						for (int e = 0; e < N; ++e) {

							// +-------+
							// | 1 + 2 |
							// +-------+

							// Case 6 (11/50): (a, b). Contract (c, d). Singleton (e).
							Case = 6;
							ind = obj -> index(a, b, (Case - 1) * nChanels + f);
							// c == d
							obj -> set_gradient_for(a, b, c, c, e, f, obj -> gradient[ind]);
						}
					}
				}
			}
		}
	}

	// Contraction 7 - Contraction 9
	static void backward_job_2(RisiContraction_18_thread *obj) {
		int nChanels = obj -> nChanels;
		int N = obj -> N;

		int Case;
		int ind;

		for (int f = 0; f < nChanels; ++f) {
			for (int a = 0; a < N; ++a) {
				for (int b = 0; b < N; ++b) {
					for (int c = 0; c < N; ++c) {
						for (int d = 0; d < N; ++d) {
							
							// Case 7 (13/50): (a, b). Contract (d, e). Singleton (c).
							Case = 7;
							ind = obj -> index(a, b, (Case - 1) * nChanels + f);
							// d == e
							obj -> set_gradient_for(a, b, c, d, d, f, obj -> gradient[ind]);
						}
					}
				}
			}
		}

		for (int f = 0; f < nChanels; ++f) {
			for (int a = 0; a < N; ++a) {
				for (int b = 0; b < N; ++b) {
					for (int d = 0; d < N; ++d) {
						for (int e = 0; e < N; ++e) {

							// Case 8 (17/50): (a, d). Contract (b, c). Singleton (e).
							Case = 8;
							ind = obj -> index(a, d, (Case - 1) * nChanels + f);
							// b == c
							obj -> set_gradient_for(a, b, b, d, e, f, obj -> gradient[ind]);
						}
					}
				}
			}
		}

		for (int f = 0; f < nChanels; ++f) {
			for (int a = 0; a < N; ++a) {
				for (int b = 0; b < N; ++b) {
					for (int c = 0; c < N; ++c) {
						for (int d = 0; d < N; ++d) {

							// Case 9 (18/50): (a, d). Contract (b, e). Singleton (c).
							Case = 9;
							ind = obj -> index(a, d, (Case - 1) * nChanels + f);
							// b == e
							obj -> set_gradient_for(a, b, c, d, b, f, obj -> gradient[ind]);
						}
					}
				}
			}
		}
	}

	// Contraction 10 - Contraction 12
	static void backward_job_3(RisiContraction_18_thread *obj) {
		int nChanels = obj -> nChanels;
		int N = obj -> N;

		int Case;
		int ind;

		for (int f = 0; f < nChanels; ++f) {
			for (int a = 0; a < N; ++a) {
				for (int b = 0; b < N; ++b) {
					for (int c = 0; c < N; ++c) {
						for (int e = 0; e < N; ++e) {

							// Case 10 (23/50): (b, c). Contract (a, d). Singleton (e).
							Case = 10;
							ind = obj -> index(b, c, (Case - 1) * nChanels + f);
							// a == d
							obj -> set_gradient_for(a, b, c, a, e, f, obj -> gradient[ind]);
						}
					}
				}
			}
		}

		for (int f = 0; f < nChanels; ++f) {
			for (int a = 0; a < N; ++a) {
				for (int b = 0; b < N; ++b) {
					for (int d = 0; d < N; ++d) {
						for (int e = 0; e < N; ++e) {

							// Case 11 (26/50): (b, d). Contract (a, c). Singleton (e).
							Case = 11;
							ind = obj -> index(b, d, (Case - 1) * nChanels + f);
							// a == c
							obj -> set_gradient_for(a, b, a, d, e, f, obj -> gradient[ind]);
						}
					}
				}
			}
		}

		for (int f = 0; f < nChanels; ++f) {
			for (int a = 0; a < N; ++a) {
				for (int b = 0; b < N; ++b) {
					for (int c = 0; c < N; ++c) {
						for (int d = 0; d < N; ++d) {

							// Case 12 (27/50): (b, d). Contract (a, e). Singleton (c).
							Case = 12;
							ind = obj -> index(b, d, (Case - 1) * nChanels + f);
							// a == e
							obj -> set_gradient_for(a, b, c, d, a, f, obj -> gradient[ind]);
						}
					}
				}
			}
		}
	}

	// Contraction 13 - Contraction 15
	static void backward_job_4(RisiContraction_18_thread *obj) {
		int nChanels = obj -> nChanels;
		int N = obj -> N;

		int Case;
		int ind;

		for (int f = 0; f < nChanels; ++f) {
			for (int a = 0; a < N; ++a) {
				for (int b = 0; b < N; ++b) {
					for (int c = 0; c < N; ++c) {
						for (int d = 0; d < N; ++d) {

							// Case 13 (28/50): (b, d). Contract (c, e). Singleton (a).
							Case = 13;
							ind = obj -> index(b, d, (Case - 1) * nChanels + f);
							// c == e
							obj -> set_gradient_for(a, b, c, d, c, f, obj -> gradient[ind]);
						}
					}
				}
			}
		}

		for (int f = 0; f < nChanels; ++f) {
			for (int a = 0; a < N; ++a) {
				for (int c = 0; c < N; ++c) {
					for (int d = 0; d < N; ++d) {
						for (int e = 0; e < N; ++e) {

							// Case 14 (38/50): (d, e). Contract (a, b). Singleton (c).
							Case = 14;
							ind = obj -> index(d, e, (Case - 1) * nChanels + f);
							// a == b
							obj -> set_gradient_for(a, a, c, d, e, f, obj -> gradient[ind]);
						}
					}
				}
			}
		}

		for (int f = 0; f < nChanels; ++f) {
			for (int a = 0; a < N; ++a) {
				for (int b = 0; b < N; ++b) {
					for (int d = 0; d < N; ++d) {
						for (int e = 0; e < N; ++e) {

							// Case 15 (40/50): (d, e). Contract (b, c). Singleton (a).
							Case = 15;
							ind = obj -> index(d, e, (Case - 1) * nChanels + f);
							// b == c
							obj -> set_gradient_for(a, b, b, d, e, f, obj -> gradient[ind]);
						}
					}
				}
			}
		}
	}

	// Contraction 16 - Contraction 18
	static void backward_job_5(RisiContraction_18_thread *obj) {
		int nChanels = obj -> nChanels;
		int N = obj -> N;

		int Case;
		int ind;

		for (int f = 0; f < nChanels; ++f) {
			for (int a = 0; a < N; ++a) {
				for (int b = 0; b < N; ++b) {
					for (int d = 0; d < N; ++d) {

						// +---+
						// | 3 |
						// +---+

						// Case 16 (43/50): (a, d). Contract (b, c, e).
						Case = 16;
						ind = obj -> index(a, d, (Case - 1) * nChanels + f);
						// b == c && c == e
						obj -> set_gradient_for(a, b, b, d, b, f, obj -> gradient[ind]);
					}
				}
			}
		}

		for (int f = 0; f < nChanels; ++f) {
			for (int a = 0; a < N; ++a) {
				for (int b = 0; b < N; ++b) {
					for (int d = 0; d < N; ++d) {
								
						// Case 17 (46/50): (b, d). Contract (a, c, e).
						Case = 17;
						ind = obj -> index(b, d, (Case - 1) * nChanels + f);
						// a == c && c == e
						obj -> set_gradient_for(a, b, a, d, a, f, obj -> gradient[ind]);
					}
				}
			}
		}

		for (int f = 0; f < nChanels; ++f) {
			for (int a = 0; a < N; ++a) {
				for (int d = 0; d < N; ++d) {
					for (int e = 0; e < N; ++e) {

						// Case 18 (50/50): (d, e). Contract (a, b, c).
						Case = 18;
						ind = obj -> index(d, e, (Case - 1) * nChanels + f);
						// a == b && b == c
						obj -> set_gradient_for(a, a, a, d, e, f, obj -> gradient[ind]);
					}
				}
			}
		}
	}


	void forward() {
		assert(tensors.size() == N);
		assert(nThreads == 6);

		for (int i = 0; i < size; ++i) {
			value[i] = 0.0;
		}

		forward_thread[0] = std::thread(forward_job_0, this);
		forward_thread[1] = std::thread(forward_job_1, this);
		forward_thread[2] = std::thread(forward_job_2, this);
		forward_thread[3] = std::thread(forward_job_3, this);
		forward_thread[4] = std::thread(forward_job_4, this);
		forward_thread[5] = std::thread(forward_job_5, this);

		for (int t = 0; t < nThreads; ++t) {
			forward_thread[t].join();
		}

		for (int i = 0; i < size; ++i) {
			gradient[i] = 0.0;
		}
	}

	void backward() {
		assert(tensors.size() == N);
		assert(nThreads == 6);

		backward_thread[0] = std::thread(backward_job_0, this);
		backward_thread[1] = std::thread(backward_job_1, this);
		backward_thread[2] = std::thread(backward_job_2, this);
		backward_thread[3] = std::thread(backward_job_3, this);
		backward_thread[4] = std::thread(backward_job_4, this);
		backward_thread[5] = std::thread(backward_job_5, this);

		for (int t = 0; t < nThreads; ++t) {
			backward_thread[t].join();
		}
	}

	// DEPRECATED
	static const bool DEPRECATED = true;

	// Number of contractions implemented in this class
	static const int nContractions = 18;

	// Number of threads
	static const int nThreads = 6;

	// Threads for Forward
	std::thread forward_thread[nThreads];

	// Threads for Backward
	std::thread backward_thread[nThreads];

	// The size of the receptive field
	int N;

	// Number of chanels
	int nChanels;

	// Neighbors' representations
	vector < Tensor3D* > tensors;

	// The reduced adjacency matrix
	Matrix *adj;

	~RisiContraction_18_thread() {
		tensors.clear();
	}
};

#endif