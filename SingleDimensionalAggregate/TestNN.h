//#pragma once
//#include "pch.h"
//#include "mlpack/core.hpp"
//#include <mlpack/methods/ann/ffn_impl.hpp>
//#include <mlpack/methods/ann/ffn.hpp>
//
//using namespace arma;
//using namespace mlpack;
//using namespace std;
//
//void TestFNN() {
//	// Load the training set.
//	arma::mat dataset;
//	mlpack::data::Load("D:/mlpack3.0.4/mlpack-3.0.4/build/thyroid_train.csv", dataset, true);
//	// Split the labels from the training set.
//	arma::mat trainData = dataset.submat(0, 0, dataset.n_rows - 4, dataset.n_cols - 1);
//	// Split the data from the training set.
//	arma::mat trainLabels = dataset.submat(dataset.n_rows - 3, 0, dataset.n_rows - 1, dataset.n_cols - 1);
//
//	//mat dataset;
//	////bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset);
//	//bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/testNN.csv", dataset);
//	//arma::mat trainData = dataset.row(0);
//	//arma::mat trainLabels = dataset.row(dataset.n_rows - 1);
//
//	// Initialize the network.
//	mlpack::ann::FFN<> model;
//	model.Add<mlpack::ann::Linear<> >(trainData.n_rows, 8);
//	model.Add<mlpack::ann::SigmoidLayer<> >();
//	model.Add<mlpack::ann::Linear<> >(8, 3);
//	model.Add<mlpack::ann::LogSoftMax<> >();
//
//	// Train the model.
//	model.Train(trainData, trainLabels);
//	//	// Use the Predict method to get the assignments.
//	//	arma::mat assignments;
//	//	model.Predict(trainData, assignments);
//}