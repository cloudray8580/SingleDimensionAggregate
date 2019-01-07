#pragma once
#include "pch.h"
#include "mlpack/core.hpp"
#include "mlpack/methods/random_forest/random_forest.hpp"
#include "mlpack/methods/decision_tree/random_dimension_select.hpp"
#include "mlpack/core/cv/k_fold_cv.hpp"
#include "mlpack/core/cv/metrics/accuracy.hpp"
#include "mlpack/core/cv/metrics/precision.hpp"
#include "mlpack/core/cv/metrics/recall.hpp"
#include "mlpack/core/cv/metrics/F1.hpp"
#include <mlpack/methods/linear_regression/linear_regression.hpp>
#include <mlpack/methods/lars/lars.hpp>
#include <chrono>

using namespace arma;
using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::cv;
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

