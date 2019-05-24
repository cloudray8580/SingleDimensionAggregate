#pragma once
#include "pch.h"
#include "mlpack/core.hpp"
//#include "mlpack/methods/random_forest/random_forest.hpp"
//#include "mlpack/methods/decision_tree/random_dimension_select.hpp"
//#include "mlpack/core/cv/k_fold_cv.hpp"
//#include "mlpack/core/cv/metrics/accuracy.hpp"
//#include "mlpack/core/cv/metrics/precision.hpp"
//#include "mlpack/core/cv/metrics/recall.hpp"
//#include "mlpack/core/cv/metrics/F1.hpp"
#include <mlpack/methods/linear_regression/linear_regression.hpp>
#include <mlpack/methods/lars/lars.hpp>
#include <chrono>

using namespace arma;
using namespace mlpack;
//using namespace mlpack::tree;
//using namespace mlpack::cv;
using namespace mlpack::regression;

using namespace std;

void TestLearnedMethodAggregate(string filename_query = "C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQuery2.csv", string filename_data = "C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", string filename_result = "C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimResults.csv") {
	// read and load dataset
	mat dataset;
	bool loaded = mlpack::data::Load(filename_data, dataset);
	if (loaded) {
		std::cout << "dataset loaded!\n";
	}
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);
	//dataset.shed_row(dataset.n_rows - 1);
	dataset.shed_rows(1, 2);
	//cout << "number of rows before transpose: " << dataset.n_rows << endl; // 2
	//cout << "number of columns before transpose: " << dataset.n_cols << endl; // 1000
	//dataset.print();
	
	LinearRegression lr(dataset, responses);
	arma::vec parameters = lr.Parameters();
	cout << "number of parameters: " << parameters.size() << endl;

	//LARS lars(); // no parameter constructor do not need to mention the (), or it will be regared as a function declarition
	//LARS lars;
	//lars.Train(dataset, responses);


	// read and load queryset
	mat queryset;
	bool loaded2 = mlpack::data::Load(filename_query, queryset);
	//cout << "number of rows before transpose: " << queryset.n_rows << endl; // 2
	//cout << "number of columns before transpose: " << queryset.n_cols << endl; // 1000
	mat queryset_low = queryset.row(0);
	mat queryset_up = queryset.row(1);


	// execute query
	arma::rowvec predictions_low, predictions_up, predicted_range, relative_error;

	auto t0 = chrono::steady_clock::now();

	lr.Predict(queryset_low, predictions_low);
	lr.Predict(queryset_up, predictions_up);
	predicted_range = predictions_up - predictions_low;
	//predictions_low.print();
	//predictions_up.print();
	//predicted_range.print();

	auto t1 = chrono::steady_clock::now();
	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset.size()/queryset.n_rows) << " ns" << endl;
	cout << "query set size: " << (queryset.size() / queryset.n_rows) << endl;

	// measure accuracy
	mat real_result;
	bool loaded3 = mlpack::data::Load(filename_result, real_result);
	arma::rowvec real_range = real_result.row(0);

	relative_error = abs(predicted_range - real_range);
	relative_error /= real_range;
	double total_error = arma::accu(relative_error);
	double average_relative_error = total_error / relative_error.size();
	cout << "average error: " << average_relative_error << endl;
}

#include <mlpack/methods/ann/ffn_impl.hpp>
#include <mlpack/methods/ann/ffn.hpp>
//#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>

// solve link error:
// I added libopenblas.dll.a file to additional dependencies and it worked !
// https://github.com/mlpack/mlpack/issues/881

void TestFNN() {
	// Load the training set.
	arma::mat dataset;
	mlpack::data::Load("D:/mlpack3.0.4/mlpack-3.0.4/build/thyroid_train.csv", dataset, true);
	// Split the labels from the training set.
	arma::mat trainData = dataset.submat(0, 0, dataset.n_rows - 4, dataset.n_cols - 1);
	// Split the data from the training set.
	arma::mat trainLabels = dataset.submat(dataset.n_rows - 3, 0, dataset.n_rows - 1, dataset.n_cols - 1);

	// Initialize the network.
	mlpack::ann::FFN<> model; // notice the difference of objective function (lost func)
	model.Add<mlpack::ann::Linear<> >(trainData.n_rows, 8);
	model.Add<mlpack::ann::SigmoidLayer<> >();
	model.Add<mlpack::ann::Linear<> >(8, 3);
	model.Add<mlpack::ann::LogSoftMax<> >();

	// Train the model.
	model.Train(trainData, trainLabels);
	//	// Use the Predict method to get the assignments.
	arma::mat assignments;
	model.Predict(trainData, assignments);
	assignments.print();
}


void inline CalculateNN(double &input, vector<int> &architecture, vector<double> &NNParms) {

	int current_parm_index = 0;
	vector<double> input_value, output_value;

	double node_value;
	int input_layer_width = 1;

	input_value.push_back(input);
	input_layer_width = input_value.size();

	// for each layer
	for (int i = 0; i < architecture.size(); i++) {

		// for each node
		input_layer_width = input_value.size();
		for (int j = 0; j < architecture[i]; j++) {

			// for each input
			node_value = 0;
			for (int k = 0; k < input_layer_width; k++) {
				node_value += NNParms[current_parm_index] * input_value[k];
				current_parm_index++;
			}
			node_value += NNParms[current_parm_index];
			current_parm_index++;

			//activation, leakyrelu with 0.03
			if (node_value < 0) {
				node_value *= 0.03;
			}
			output_value.push_back(node_value);
		}

		input_value = output_value;
		output_value.clear();
	}

	//cout << "input: " << input << "  output: " << input_value[0] << endl;
}

void inline CalculateNN2(vector<double> &input, vector<int> &architecture, vector<double> &NNParms, vector<double> &results) {

	int current_parm_index = 0;
	vector<double> input_value, output_value;

	double node_value;
	int input_layer_width = 1;

	for (int p = 0; p < input.size(); p++) {
		input_value.push_back(input[p]);
		input_layer_width = input_value.size();
		current_parm_index = 0;

		// for each layer
		for (int i = 0; i < architecture.size(); i++) {

			// for each node
			input_layer_width = input_value.size();
			for (int j = 0; j < architecture[i]; j++) {

				// for each input
				node_value = 0;
				for (int k = 0; k < input_layer_width; k++) {
					node_value += NNParms[current_parm_index] * input_value[k];
					current_parm_index++;
				}
				node_value += NNParms[current_parm_index];
				current_parm_index++;

				//activation, leakyrelu with 0.03
				if (node_value < 0) {
					node_value *= 0.03;
				}
				output_value.push_back(node_value);
			}

			input_value = output_value;
			output_value.clear();
		}
		results.push_back(input_value[0]);

	}
}

#include "Utils.h"

void TestFNN2() {

	mat dataset;
	//bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset);
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/testNN.csv", dataset); 
	
	arma::mat trainData = dataset.row(0);
	//arma::mat trainData = dataset.submat(0, 0, dataset.n_rows - 2, dataset.n_cols - 1);
	arma::mat trainLabels = dataset.row(dataset.n_rows - 1);  //dataset.submat(dataset.n_rows - 2, 0, dataset.n_rows - 1, dataset.n_cols - 1);

	// Initialize the network.
	mlpack::ann::FFN<mlpack::ann::MeanSquaredError<>> model;

	model.Add<mlpack::ann::Linear<>>(trainData.n_rows, 4);
	model.Add<mlpack::ann::LeakyReLU<>>();
	model.Add<mlpack::ann::Linear<>>(4, 1);
	model.Add<mlpack::ann::LeakyReLU<>>();

	// Train the model.
	for (int i = 0; i < 10; i++) {
		model.Train(trainData, trainLabels);
	}
	//model.Train(trainData, trainLabels);

	//// queryset
	//mat queryset;
	//bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQuery2.csv", queryset);
	//arma::rowvec query_x_low = queryset.row(0);
	//arma::rowvec query_x_up = queryset.row(1);

	//// Use the Predict method to get the assignments.
	//arma::mat pred_result_up, pred_result_low;

	//auto t0 = chrono::steady_clock::now();

	//model.Predict(query_x_low, pred_result_low);
	//model.Predict(query_x_up, pred_result_up);
	////pred_result_low.print();

	//auto t1 = chrono::steady_clock::now();
	//cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	//cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset.size() / queryset.n_rows) << " ns" << endl;

	arma::mat assignments;
	model.Predict(trainData, assignments);
	assignments.print();

	cout << "================" << endl;

	cout << model.Parameters() << endl; // a,b a,b
	vector<double> NNParms;
	arma::mat parms = model.Parameters();
	for (int i = 0; i < parms.n_rows; i++) {
		NNParms.push_back(parms[i]);
	}
	//RowvecToVector(parms, NNParms);

	cout << "================" << endl;

	vector<int> arch;
	arch.push_back(1);
	arch.push_back(1);

	//CalculateNN(1, arch, NNParms);
}

void DumpNNParameters(mlpack::ann::FFN<mlpack::ann::MeanSquaredError<>> &model, vector<int> &architecture) {
	
	vector<vector<vector<double>>> NNParms; // layer; node; a,a,...,a,b
	vector<vector<double>> NNParms_layer;
	vector<double> NNParms_node;

	for (int i = 0; i < architecture.size(); i++) {
		NNParms_layer.clear();
		for (int j = 0; j < architecture[i]; j++) {
			NNParms_layer.push_back(NNParms_node);
		}
		NNParms.push_back(NNParms_layer);
	}

	arma::vec parms = model.Parameters();
	for (int i = 0; i < parms.size(); i++) {

	}
}

void TestFNN3() {
	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset);
	//bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/testNN.csv", dataset);

	arma::mat trainData = dataset.row(0);
	arma::mat trainLabels = dataset.row(dataset.n_rows - 1);

	// Initialize the network.
	mlpack::ann::FFN<mlpack::ann::MeanSquaredError<>> model;

	model.Add<mlpack::ann::Linear<>>(trainData.n_rows, 4);
	model.Add<mlpack::ann::LeakyReLU<>>();
	//model.Add<mlpack::ann::Linear<>>(16, 16);
	//model.Add<mlpack::ann::LeakyReLU<>>();
	model.Add<mlpack::ann::Linear<>>(4, 1);
	model.Add<mlpack::ann::LeakyReLU<>>();

	// Train the model.
	for (int i = 0; i < 100; i++) {
		model.Train(trainData, trainLabels);
	}
	vector<double> NNParms;
	arma::mat parms = model.Parameters();
	for (int i = 0; i < parms.n_rows; i++) {
		NNParms.push_back(parms[i]);
	}

	mat queryset;
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQuery2.csv", queryset);
	mat queryset_low = queryset.row(0);
	mat queryset_up = queryset.row(1);
	vector<double> query_low_v, query_up_v;
	RowvecToVector(queryset_low, query_low_v);
	RowvecToVector(queryset_up, query_up_v);
	
	vector<int> arch;
	arch.push_back(4);
	//arch.push_back(16);
	arch.push_back(1);

	/*vector<double> results1, results2;
	auto t0 = chrono::steady_clock::now();

	CalculateNN2(query_low_v, arch, NNParms, results1);
	CalculateNN2(query_up_v, arch, NNParms, results2);

	auto t1 = chrono::steady_clock::now();
	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset.size() / queryset.n_rows) << " ns" << endl;*/

	//arma::mat assignments;
	//model.Predict(trainData, assignments);
	//assignments.print();

	//arma::rowvec absolute_error = abs(assignments - trainLabels);
	//absolute_error.elem(find_nonfinite(absolute_error)).zeros(); // 'remove' ¡ÀInf or NaN
	//cout << arma::mean(absolute_error) << endl;

	arma::mat prediction_up, prediction_low, prediction;
	model.Predict(queryset_up, prediction_up);
	model.Predict(queryset_low, prediction_low);

	prediction = prediction_up - prediction_low;

	vector<int> real_results;
	CalculateRealCountWithScan1D(queryset, real_results);
	arma::mat response;
	bool loaded3 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedDimResults_REAL.csv", response);

	arma::rowvec relative_error = abs(prediction - response);
	cout << arma::mean(relative_error) << endl; // mean abs.err
	relative_error /= response;
	relative_error.elem(find_nonfinite(relative_error)).zeros(); // 'remove' ¡ÀInf or NaN
	cout << arma::mean(relative_error) << endl; // mean rel.err
}

void TestFNN4() {
	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset);

	arma::mat trainData = dataset.row(0);
	arma::mat trainLabels = dataset.row(dataset.n_rows - 1);

	// Initialize the network.
	mlpack::ann::FFN<mlpack::ann::MeanSquaredError<>> model;

	model.Add<mlpack::ann::Linear<>>(trainData.n_rows, 1);
	model.Add<mlpack::ann::LeakyReLU<>>();

	auto t0 = chrono::steady_clock::now();

	// Train the model.
	for (int i = 0; i < 1500; i++) {
		model.Train(trainData, trainLabels);
	}
	
	auto t1 = chrono::steady_clock::now();
	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
}