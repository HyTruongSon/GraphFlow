// Framework: GraphFlow
// Class: GraphFlow
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __GraphFlow_H_INCLUDED__
#define __GraphFlow_H_INCLUDED__

#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

#include "Entity.h"
#include "Vector.h"
#include "Matrix.h"
#include "Tensor3D.h"
#include "Tensor4D.h"
#include "Reshape2D.h"
#include "Reshape3D.h"
#include "Reshape4D.h"
#include "Identity.h"
#include "Sigmoid.h"
#include "Tanh.h"
#include "ReLU.h"
#include "LeakyReLU.h"
#include "LeakyReLU2D.h"
#include "LeakyReLU3D.h"
#include "InnerProduct.h"
#include "OuterProduct.h"
#include "Add.h"
#include "VectorAddMatrix.h"
#include "VectorAddTensor.h"
#include "Subtract.h"
#include "Multiply.h"
#include "Transpose.h"
#include "ScalarMatMul.h"
#include "MatVecMul.h"
#include "MatMul.h"
#include "SumComponents.h"
#include "SumVectors.h"
#include "SumMatrices.h"
#include "SumTensor3D.h"
#include "AverageVectors.h"
#include "SumRows.h"
#include "VertexRepresentation.h"
#include "RisiLayer1D.h"
#include "RisiLayer2D.h"
#include "RisiLayer3D.h"
#include "Conv1D.h"
#include "Conv2D.h"
#include "MaxPool2D.h"
#include "AveragePool2D.h"
#include "Masking.h"
#include "DropOut.h"
#include "Norm3D.h"
#include "KMax.h"
#include "Sort.h"
#include "Softmax.h"
#include "Softmax2D.h"
#include "Softmax3D.h"
#include "ConCat.h"
#include "ShuffleMatrix.h"
#include "LinearGram.h"
#include "SquaredLoss.h"
#include "LogLoss.h"
#include "L1Regularization.h"
#include "L2Regularization.h"
#include "ShrinkMatrix.h"
#include "ShrinkTensor.h"
#include "VectorBroadcastMat.h"
#include "MatBroadcastMat.h"
#include "MatTensorMul.h"
#include "TensorMatMul.h"
#include "TensorMul.h"
#include "Tensor4DTensor3DMul.h"
#include "MatrixConcat.h"
#include "Tensor3DConcat.h"
#include "Tensor4DConcat.h"
#include "StackTensor3D.h"
#include "StackTensor3D_thread.h"
#include "CustomMatMulTensor.h"
#include "RisiContraction_4.h"
#include "RisiContraction_4_thread.h"
#include "RisiContraction_10.h"
#include "RisiContraction_50.h"
#include "RisiContraction_18.h"
#include "RisiContraction_18_thread.h"
#include "RisiContraction_18_dropout.h"
#include "ConcatVectors.h"

using namespace std;

const int ENTITY 						= 0;
const int VECTOR 						= 1;
const int MATRIX 						= 2;
const int TENSOR3D 						= 3;
const int TENSOR4D 						= 4;
const int RESHAPE2D 					= 5;
const int RESHAPE3D 					= 6;
const int RESHAPE4D 					= 7;
const int IDENTITY 						= 8;
const int SIGMOID 						= 9;
const int TANH 							= 10;
const int RELU 							= 11;
const int LEAKYRELU 					= 12;
const int LEAKYRELU2D 					= 13;
const int LEAKYRELU3D 					= 14;
const int INNERPRODUCT 					= 15;
const int OUTERPRODUCT 					= 16;
const int ADD 							= 17;
const int VECTORADDMATRIX 				= 18;
const int VECTORADDTENSOR 				= 19;
const int SUBTRACT 						= 20;
const int MULTIPLY 						= 21;
const int TRANSPOSE 					= 22;
const int SCALARMATMUL 					= 23;
const int MATVECMUL 					= 24;
const int MATMUL 						= 25;
const int SUMCOMPONENTS 				= 26;
const int SUMVECTORS 					= 27;
const int SUMMATRICES 					= 28;
const int SUMTENSOR3D					= 29;
const int AVERAGEVECTORS 				= 30;
const int SUMROWS 						= 31;
const int VERTEXREPRESENTATION 			= 32;
const int RISILAYER1D 					= 33;
const int RISILAYER2D 					= 34;
const int RISILAYER3D 					= 35;
const int CONV1D 						= 36;
const int CONV2D 						= 37;
const int MAXPOOL2D 					= 38;
const int AVERAGEPOOL2D 				= 39;
const int MASKING 						= 40;
const int DROPOUT 						= 41;
const int NORM3D 						= 42;
const int KMAX 							= 43;
const int SORT 							= 44;
const int SOFTMAX 						= 45;
const int SOFTMAX2D 					= 46;
const int SOFTMAX3D 					= 47;
const int CONCAT						= 48;
const int SHUFFLEMATRIX 				= 49;
const int LINEARGRAM 					= 50;
const int SQUAREDLOSS 					= 51;
const int LOGLOSS 						= 52;
const int L1REGULARIZATION 				= 53;
const int L2REGULARIZATION 				= 54;
const int SHRINKMATRIX 					= 55;
const int SHRINKTENSOR 					= 56;
const int VECTORBROADCASTMAT 			= 57;
const int MATBROADCASTMAT 				= 58;
const int MATTENSORMUL 					= 59;
const int TENSORMATMUL 					= 60;
const int TENSORMUL 					= 61;
const int TENSOR4DTENSOR3DMUL 			= 62;
const int MATRIXCONCAT 					= 63;
const int TENSOR3DCONCAT 				= 64;
const int TENSOR4DCONCAT 				= 65;
const int STACKTENSOR3D 				= 66;
const int STACKTENSOR3D_THREAD 			= 67;
const int CUSTOMMATMULTENSOR 			= 68;
const int RISICONTRACTION_4 			= 69;
const int RISICONTRACTION_4_THREAD 		= 70;
const int RISICONTRACTION_10 			= 71;
const int RISICONTRACTION_50 			= 72;
const int RISICONTRACTION_18 			= 73;
const int RISICONTRACTION_18_THREAD 	= 74;
const int RISICONTRACTION_18_DROPOUT 	= 75;
const int CONCATVECTORS 				= 76;

class GraphFlow {
public:
	GraphFlow() {
		topology.clear();
	}

	void add(Entity *entity, int type) {
		topology.push_back(make_pair(entity, type));
	}

	void clear() {
		topology.clear();
	}

	void forward() {
		for (int i = 0; i < topology.size(); ++i) {
			if (ENTITY == topology[i].second) {
			}

			if (VECTOR == topology[i].second) {
				Vector *obj = (Vector *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (MATRIX == topology[i].second) {
				Matrix *obj = (Matrix *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (TENSOR3D == topology[i].second) {
				Tensor3D *obj = (Tensor3D *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (TENSOR4D == topology[i].second) {
				Tensor4D *obj = (Tensor4D *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (RESHAPE2D == topology[i].second) {
				Reshape2D *obj = (Reshape2D *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (RESHAPE3D == topology[i].second) {
				Reshape3D *obj = (Reshape3D *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (RESHAPE4D == topology[i].second) {
				Reshape4D *obj = (Reshape4D *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (IDENTITY == topology[i].second) {
				Identity *obj = (Identity *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (SIGMOID == topology[i].second) {
				Sigmoid *obj = (Sigmoid *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (TANH == topology[i].second) {
				Tanh *obj = (Tanh *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (RELU == topology[i].second) {
				ReLU *obj = (ReLU *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (LEAKYRELU == topology[i].second) {
				LeakyReLU *obj = (LeakyReLU *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (LEAKYRELU2D == topology[i].second) {
				LeakyReLU2D *obj = (LeakyReLU2D *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (LEAKYRELU3D == topology[i].second) {
				LeakyReLU3D *obj = (LeakyReLU3D *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (INNERPRODUCT == topology[i].second) {
				InnerProduct *obj = (InnerProduct *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (OUTERPRODUCT == topology[i].second) {
				OuterProduct *obj = (OuterProduct *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (ADD == topology[i].second) {
				Add *obj = (Add *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (VECTORADDMATRIX == topology[i].second) {
				VectorAddMatrix *obj = (VectorAddMatrix *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (VECTORADDTENSOR == topology[i].second) {
				VectorAddTensor *obj = (VectorAddTensor *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (SUBTRACT == topology[i].second) {
				Subtract *obj = (Subtract *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (MULTIPLY == topology[i].second) {
				Multiply *obj = (Multiply *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (TRANSPOSE == topology[i].second) {
				Transpose *obj = (Transpose *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (SCALARMATMUL == topology[i].second) {
				ScalarMatMul *obj = (ScalarMatMul *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (MATVECMUL == topology[i].second) {
				MatVecMul *obj = (MatVecMul *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (MATMUL == topology[i].second) {
				MatMul *obj = (MatMul *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (SUMCOMPONENTS == topology[i].second) {
				SumComponents *obj = (SumComponents *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (SUMVECTORS == topology[i].second) {
				SumVectors *obj = (SumVectors *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (SUMMATRICES == topology[i].second) {
				SumMatrices *obj = (SumMatrices *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (SUMTENSOR3D == topology[i].second) {
				SumTensor3D *obj = (SumTensor3D *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (AVERAGEVECTORS == topology[i].second) {
				AverageVectors *obj = (AverageVectors *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (SUMROWS == topology[i].second) {
				SumRows *obj = (SumRows *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (VERTEXREPRESENTATION == topology[i].second) {
				VertexRepresentation *obj = (VertexRepresentation *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (RISILAYER1D == topology[i].second) {
				RisiLayer1D *obj = (RisiLayer1D *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (RISILAYER2D == topology[i].second) {
				RisiLayer2D *obj = (RisiLayer2D *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (RISILAYER3D == topology[i].second) {
				RisiLayer3D *obj = (RisiLayer3D *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (CONV1D == topology[i].second) {
				Conv1D *obj = (Conv1D *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (CONV2D == topology[i].second) {
				Conv2D *obj = (Conv2D *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (MAXPOOL2D == topology[i].second) {
				MaxPool2D *obj = (MaxPool2D *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (AVERAGEPOOL2D == topology[i].second) {
				AveragePool2D *obj = (AveragePool2D *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (MASKING == topology[i].second) {
				Masking *obj = (Masking *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (DROPOUT == topology[i].second) {
				DropOut *obj = (DropOut *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (NORM3D == topology[i].second) {
				Norm3D *obj = (Norm3D *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (KMAX == topology[i].second) {
				KMax *obj = (KMax *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (SORT == topology[i].second) {
				Sort *obj = (Sort *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (SOFTMAX == topology[i].second) {
				Softmax *obj = (Softmax *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (SOFTMAX2D == topology[i].second) {
				Softmax2D *obj = (Softmax2D *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (SOFTMAX3D == topology[i].second) {
				Softmax3D *obj = (Softmax3D *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (CONCAT == topology[i].second) {
				ConCat *obj = (ConCat *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (SHUFFLEMATRIX == topology[i].second) {
				ShuffleMatrix *obj = (ShuffleMatrix *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (LINEARGRAM == topology[i].second) {
				LinearGram *obj = (LinearGram *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (SQUAREDLOSS == topology[i].second) {
				SquaredLoss *obj = (SquaredLoss *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (LOGLOSS == topology[i].second) {
				LogLoss *obj = (LogLoss *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (L1REGULARIZATION == topology[i].second) {
				L1Regularization *obj = (L1Regularization *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (L2REGULARIZATION == topology[i].second) {
				L2Regularization *obj = (L2Regularization *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (SHRINKMATRIX == topology[i].second) {
				ShrinkMatrix *obj = (ShrinkMatrix *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (SHRINKTENSOR == topology[i].second) {
				ShrinkTensor *obj = (ShrinkTensor *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (VECTORBROADCASTMAT == topology[i].second) {
				VectorBroadcastMat *obj = (VectorBroadcastMat *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (MATBROADCASTMAT == topology[i].second) {
				MatBroadcastMat *obj = (MatBroadcastMat *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (MATTENSORMUL == topology[i].second) {
				MatTensorMul *obj = (MatTensorMul *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (TENSORMATMUL == topology[i].second) {
				TensorMatMul *obj = (TensorMatMul *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (TENSORMUL == topology[i].second) {
				TensorMul *obj = (TensorMul *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (TENSOR4DTENSOR3DMUL == topology[i].second) {
				Tensor4DTensor3DMul *obj = (Tensor4DTensor3DMul *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (MATRIXCONCAT == topology[i].second) {
				MatrixConcat *obj = (MatrixConcat *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (TENSOR3DCONCAT == topology[i].second) {
				Tensor3DConcat *obj = (Tensor3DConcat *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (TENSOR4DCONCAT == topology[i].second) {
				Tensor4DConcat *obj = (Tensor4DConcat *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (STACKTENSOR3D == topology[i].second) {
				StackTensor3D *obj = (StackTensor3D *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (STACKTENSOR3D_THREAD == topology[i].second) {
				StackTensor3D_thread *obj = (StackTensor3D_thread *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (CUSTOMMATMULTENSOR == topology[i].second) {
				CustomMatMulTensor *obj = (CustomMatMulTensor *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (RISICONTRACTION_4 == topology[i].second) {
				RisiContraction_4 *obj = (RisiContraction_4 *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (RISICONTRACTION_4_THREAD == topology[i].second) {
				RisiContraction_4_thread *obj = (RisiContraction_4_thread *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (RISICONTRACTION_10 == topology[i].second) {
				RisiContraction_10 *obj = (RisiContraction_10 *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (RISICONTRACTION_50 == topology[i].second) {
				RisiContraction_50 *obj = (RisiContraction_50 *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (RISICONTRACTION_18 == topology[i].second) {
				RisiContraction_18 *obj = (RisiContraction_18 *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (RISICONTRACTION_18_THREAD == topology[i].second) {
				RisiContraction_18_thread *obj = (RisiContraction_18_thread *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (RISICONTRACTION_18_DROPOUT == topology[i].second) {
				RisiContraction_18_dropout *obj = (RisiContraction_18_dropout *)(topology[i].first);
				obj -> forward();

				continue;
			}

			if (CONCATVECTORS == topology[i].second) {
				ConcatVectors *obj = (ConcatVectors *)(topology[i].first);
				obj -> forward();

				continue;
			}
		}
	}

	void backward() {
		for (int i = topology.size() - 1; i >= 0; --i) {
			if (ENTITY == topology[i].second) {
			}

			if (VECTOR == topology[i].second) {
				Vector *obj = (Vector *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (MATRIX == topology[i].second) {
				Matrix *obj = (Matrix *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (TENSOR3D == topology[i].second) {
				Tensor3D *obj = (Tensor3D *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (TENSOR4D == topology[i].second) {
				Tensor4D *obj = (Tensor4D *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (RESHAPE2D == topology[i].second) {
				Reshape2D *obj = (Reshape2D *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (RESHAPE3D == topology[i].second) {
				Reshape3D *obj = (Reshape3D *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (RESHAPE4D == topology[i].second) {
				Reshape4D *obj = (Reshape4D *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (IDENTITY == topology[i].second) {
				Identity *obj = (Identity *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (SIGMOID == topology[i].second) {
				Sigmoid *obj = (Sigmoid *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (TANH == topology[i].second) {
				Tanh *obj = (Tanh *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (RELU == topology[i].second) {
				ReLU *obj = (ReLU *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (LEAKYRELU == topology[i].second) {
				LeakyReLU *obj = (LeakyReLU *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (LEAKYRELU2D == topology[i].second) {
				LeakyReLU2D *obj = (LeakyReLU2D *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (LEAKYRELU3D == topology[i].second) {
				LeakyReLU3D *obj = (LeakyReLU3D *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (INNERPRODUCT == topology[i].second) {
				InnerProduct *obj = (InnerProduct *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (OUTERPRODUCT == topology[i].second) {
				OuterProduct *obj = (OuterProduct *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (ADD == topology[i].second) {
				Add *obj = (Add *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (VECTORADDMATRIX == topology[i].second) {
				VectorAddMatrix *obj = (VectorAddMatrix *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (VECTORADDTENSOR == topology[i].second) {
				VectorAddTensor *obj = (VectorAddTensor *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (SUBTRACT == topology[i].second) {
				Subtract *obj = (Subtract *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (MULTIPLY == topology[i].second) {
				Multiply *obj = (Multiply *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (TRANSPOSE == topology[i].second) {
				Transpose *obj = (Transpose *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (SCALARMATMUL == topology[i].second) {
				ScalarMatMul *obj = (ScalarMatMul *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (MATVECMUL == topology[i].second) {
				MatVecMul *obj = (MatVecMul *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (MATMUL == topology[i].second) {
				MatMul *obj = (MatMul *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (SUMCOMPONENTS == topology[i].second) {
				SumComponents *obj = (SumComponents *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (SUMVECTORS == topology[i].second) {
				SumVectors *obj = (SumVectors *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (SUMMATRICES == topology[i].second) {
				SumMatrices *obj = (SumMatrices *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (SUMTENSOR3D == topology[i].second) {
				SumTensor3D *obj = (SumTensor3D *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (AVERAGEVECTORS == topology[i].second) {
				AverageVectors *obj = (AverageVectors *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (SUMROWS == topology[i].second) {
				SumRows *obj = (SumRows *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (VERTEXREPRESENTATION == topology[i].second) {
				VertexRepresentation *obj = (VertexRepresentation *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (RISILAYER1D == topology[i].second) {
				RisiLayer1D *obj = (RisiLayer1D *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (RISILAYER2D == topology[i].second) {
				RisiLayer2D *obj = (RisiLayer2D *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (RISILAYER3D == topology[i].second) {
				RisiLayer3D *obj = (RisiLayer3D *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (CONV1D == topology[i].second) {
				Conv1D *obj = (Conv1D *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (CONV2D == topology[i].second) {
				Conv2D *obj = (Conv2D *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (MAXPOOL2D == topology[i].second) {
				MaxPool2D *obj = (MaxPool2D *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (AVERAGEPOOL2D == topology[i].second) {
				AveragePool2D *obj = (AveragePool2D *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (MASKING == topology[i].second) {
				Masking *obj = (Masking *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (DROPOUT == topology[i].second) {
				DropOut *obj = (DropOut *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (NORM3D == topology[i].second) {
				Norm3D *obj = (Norm3D *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (KMAX == topology[i].second) {
				KMax *obj = (KMax *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (SORT == topology[i].second) {
				Sort *obj = (Sort *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (SOFTMAX == topology[i].second) {
				Softmax *obj = (Softmax *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (SOFTMAX2D == topology[i].second) {
				Softmax2D *obj = (Softmax2D *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (SOFTMAX3D == topology[i].second) {
				Softmax3D *obj = (Softmax3D *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (CONCAT == topology[i].second) {
				ConCat *obj = (ConCat *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (SHUFFLEMATRIX == topology[i].second) {
				ShuffleMatrix *obj = (ShuffleMatrix *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (LINEARGRAM == topology[i].second) {
				LinearGram *obj = (LinearGram *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (SQUAREDLOSS == topology[i].second) {
				SquaredLoss *obj = (SquaredLoss *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (LOGLOSS == topology[i].second) {
				LogLoss *obj = (LogLoss *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (L1REGULARIZATION == topology[i].second) {
				L1Regularization *obj = (L1Regularization *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (L2REGULARIZATION == topology[i].second) {
				L2Regularization *obj = (L2Regularization *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (SHRINKMATRIX == topology[i].second) {
				ShrinkMatrix *obj = (ShrinkMatrix *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (SHRINKTENSOR == topology[i].second) {
				ShrinkTensor *obj = (ShrinkTensor *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (VECTORBROADCASTMAT == topology[i].second) {
				VectorBroadcastMat *obj = (VectorBroadcastMat *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (MATBROADCASTMAT == topology[i].second) {
				MatBroadcastMat *obj = (MatBroadcastMat *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (MATTENSORMUL == topology[i].second) {
				MatTensorMul *obj = (MatTensorMul *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (TENSORMATMUL == topology[i].second) {
				TensorMatMul *obj = (TensorMatMul *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (TENSORMUL == topology[i].second) {
				TensorMul *obj = (TensorMul *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (TENSOR4DTENSOR3DMUL == topology[i].second) {
				Tensor4DTensor3DMul *obj = (Tensor4DTensor3DMul *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (MATRIXCONCAT == topology[i].second) {
				MatrixConcat *obj = (MatrixConcat *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (TENSOR3DCONCAT == topology[i].second) {
				Tensor3DConcat *obj = (Tensor3DConcat *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (TENSOR4DCONCAT == topology[i].second) {
				Tensor4DConcat *obj = (Tensor4DConcat *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (STACKTENSOR3D == topology[i].second) {
				StackTensor3D *obj = (StackTensor3D *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (STACKTENSOR3D_THREAD == topology[i].second) {
				StackTensor3D_thread *obj = (StackTensor3D_thread *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (CUSTOMMATMULTENSOR == topology[i].second) {
				CustomMatMulTensor *obj = (CustomMatMulTensor *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (RISICONTRACTION_4 == topology[i].second) {
				RisiContraction_4 *obj = (RisiContraction_4 *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (RISICONTRACTION_4_THREAD == topology[i].second) {
				RisiContraction_4_thread *obj = (RisiContraction_4_thread *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (RISICONTRACTION_10 == topology[i].second) {
				RisiContraction_10 *obj = (RisiContraction_10 *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (RISICONTRACTION_50 == topology[i].second) {
				RisiContraction_50 *obj = (RisiContraction_50 *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (RISICONTRACTION_18 == topology[i].second) {
				RisiContraction_18 *obj = (RisiContraction_18 *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (RISICONTRACTION_18_THREAD == topology[i].second) {
				RisiContraction_18_thread *obj = (RisiContraction_18_thread *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (RISICONTRACTION_18_DROPOUT == topology[i].second) {
				RisiContraction_18_dropout *obj = (RisiContraction_18_dropout *)(topology[i].first);
				obj -> backward();

				continue;
			}

			if (CONCATVECTORS == topology[i].second) {
				ConcatVectors *obj = (ConcatVectors *)(topology[i].first);
				obj -> backward();

				continue;
			}
		}
	}

	void assign_init(Vector *V, float value) {
		for (int i = 0; i < V -> size; ++i) {
			V -> value[i] = value;
		}
	}

	void assign_init(Vector *V, float *value) {
		for (int i = 0; i < V -> size; ++i) {
			V -> value[i] = value[i];
		}
	}

	void uniform_init(Matrix *M) {
		int sign;
		int index;
		
		for (int row = 0; row < M -> nRows; ++row) {
			for (int column = 0; column < M -> nColumns; ++column) {
				index = row * (M -> nColumns) + column;
				M -> value[index] = (float)(rand() % 10) / (10.0 * (M -> nRows));

				sign = rand() % 2;
				if (sign == 1) {
					M -> value[index] = - M -> value[index];
				}
			}
		}
	}

	void uniform_init(Vector *V) {
		int sign;
		for (int i = 0; i < V -> size; ++i) {
			V -> value[i] = (float)(rand() % 10) / (10.0 * V -> size);
			sign = rand() % 2;
			if (sign == 1) {
				V -> value[i] = - V -> value[i];
			}
		}
	}

	float rand_uniform() {
		int RANGE = 1e4;
		int number = abs(rand() % RANGE);
		return float(number) / float(RANGE);
	}

	float rand_uniform(float left, float right) {
		float p = rand_uniform();
		return p * left + (1.0 - p) * right;
	}

	float rand_uniform(float radius) {
		return rand_uniform(-radius, radius);
	}

	void Xavier_init(Vector *V) {
		float radius = sqrt(3.0 / V -> size);
		for (int i = 0; i < V -> size; ++i) {
			V -> value[i] = rand_uniform(radius);
		}
	}

	vector < pair < Entity *, int > > topology;

	~GraphFlow() {
		for (int i = 0; i < topology.size(); ++i) {
			delete topology[i].first;
		}
	}
};

#endif
