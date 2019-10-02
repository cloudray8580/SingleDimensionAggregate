#pragma once
//#include "LearnIndexAggregate.h"
#include "mlpack/core.hpp"
#include <string>
#include "StageModel.h"
#include <windows.h>
using namespace std;

void GenerateRandomQuerySet(string filename_query) {
	arma::mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);
	dataset.shed_rows(1, 2);

	//double max = dataset.max(); // 84.9807
	//double min = dataset.min(); // -85.0511
	//cout << max << " " << min << endl;
	double max = 84.9807;
	double min = -85.0511;
	double range = max - min;

	ofstream outfile;
	outfile.open(filename_query);

	double upper, lower;
	srand((unsigned)time(NULL));

	vector<double> uppers;
	vector<double> lowers;
	for (int i = 0; i < 1000; i++) {
		upper = rand() / double(RAND_MAX) * range + min;
		lower = rand() / double(RAND_MAX) * range + min;
		uppers.push_back(upper);
	}
	Sleep(1000);
	for (int i = 0; i < 1000; i++) {
		upper = rand() / double(RAND_MAX) * range + min;
		lower = rand() / double(RAND_MAX) * range + min;
		lowers.push_back(lower);
	}

	for (int i = 0; i < 1000; i++) {
		if (uppers[i] >= lowers[i]) {
			outfile << lowers[i] << "," << uppers[i] << endl;
		}
		else {
			outfile << uppers[i] << "," << lowers[i] << endl;
		}
	}
	outfile.close();
}

void Statistic(string filename_query) {

}

void CalculateRealSum(vector<double> &results) {

}

void CalculateRealCountWithScan2D(arma::mat &dataset, arma::mat &queryset, vector<int> &real_results) {
	arma::rowvec x = dataset.row(0); // x
	arma::rowvec y = dataset.row(1); // y
	vector<double> x_v, y_v;
	StageModel::RowvecToVector(x, x_v);
	StageModel::RowvecToVector(y, y_v);

	mat queryset_x_low = queryset.row(0);
	mat queryset_x_up = queryset.row(1);
	mat queryset_y_low = queryset.row(2);
	mat queryset_y_up = queryset.row(3);
	vector<double> query_x_low_v, query_x_up_v, query_y_low_v, query_y_up_v;
	StageModel::RowvecToVector(queryset_x_low, query_x_low_v);
	StageModel::RowvecToVector(queryset_x_up, query_x_up_v);
	StageModel::RowvecToVector(queryset_y_low, query_y_low_v);
	StageModel::RowvecToVector(queryset_y_up, query_y_up_v);
	auto t0 = chrono::steady_clock::now();
	int count;
	for (int i = 0; i < query_x_low_v.size(); i++) {
		count = 0;
		for (int j = 0; j < x_v.size(); j++) {
			if (x_v[j]>= query_x_low_v[i] && x_v[j]<= query_x_up_v[i] && y_v[j]>= query_y_low_v[i] && y_v[j]<= query_y_up_v[i]) {
				count++;
			}
		}
		real_results.push_back(count);
		/*if (i % 10000 == 0) {
			cout << "Utils...calculating real results..." << i << endl;
		}*/
	}
	auto t1 = chrono::steady_clock::now();
	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset.size() / queryset.n_rows) << " ns" << endl;
	// save real resutls to file
	ofstream outfile;
	outfile.open("C:/Users/Cloud/Desktop/LearnIndex/data/Sorted2DimResults_REAL.csv");
	for (int i = 0; i < real_results.size(); i++) {
		outfile << real_results[i] << endl;
	}
	outfile.close();
}

void CalculateRealCountWithScan2D(mat &queryset, vector<int> &real_results) {
	mat dataset;
	mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/HilbertSortedPOIs2_22.csv", dataset);
	arma::rowvec x = dataset.row(0); // x
	arma::rowvec y = dataset.row(1); // y
	vector<double> x_v, y_v;
	StageModel::RowvecToVector(x, x_v);
	StageModel::RowvecToVector(y, y_v);

	mat queryset_x_low = queryset.row(0);
	mat queryset_x_up = queryset.row(1);
	mat queryset_y_low = queryset.row(2);
	mat queryset_y_up = queryset.row(3);
	vector<double> query_x_low_v, query_x_up_v, query_y_low_v, query_y_up_v;
	StageModel::RowvecToVector(queryset_x_low, query_x_low_v);
	StageModel::RowvecToVector(queryset_x_up, query_x_up_v);
	StageModel::RowvecToVector(queryset_y_low, query_y_low_v);
	StageModel::RowvecToVector(queryset_y_up, query_y_up_v);
	auto t0 = chrono::steady_clock::now();
	int count;
	for (int i = 0; i < query_x_low_v.size(); i++) {
		count = 0;
		for (int j = 0; j < x_v.size(); j++) {
			if (x_v[j] >= query_x_low_v[i] && x_v[j] <= query_x_up_v[i] && y_v[j] >= query_y_low_v[i] && y_v[j] <= query_y_up_v[i]) {
				count++;
			}
		}
		real_results.push_back(count);
		/*if (i % 10000 == 0) {
			cout << "Utils...calculating real results..." << i << endl;
		}*/
	}
	auto t1 = chrono::steady_clock::now();
	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset.size() / queryset.n_rows) << " ns" << endl;
	// save real resutls to file
	ofstream outfile;
	outfile.open("C:/Users/Cloud/Desktop/LearnIndex/data/Sorted2DimResults_REAL.csv");
	for (int i = 0; i < real_results.size(); i++) {
		outfile << real_results[i] << endl;
	}
	outfile.close();
}

void CalculateRealCountWithScan1D(mat &queryset, vector<int> &real_results, string file_path="C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv") {
	mat dataset;
	mlpack::data::Load(file_path, dataset);
	arma::rowvec x = dataset.row(0); // x
	arma::rowvec y = dataset.row(1); // y
	vector<double> x_v, y_v;
	StageModel::RowvecToVector(x, x_v);
	StageModel::RowvecToVector(y, y_v);

	mat queryset_x_low = queryset.row(0);
	mat queryset_x_up = queryset.row(1);
	vector<double> query_x_low_v, query_x_up_v;
	StageModel::RowvecToVector(queryset_x_low, query_x_low_v);
	StageModel::RowvecToVector(queryset_x_up, query_x_up_v);

	auto t0 = chrono::steady_clock::now();
	int count;
	for (int i = 0; i < query_x_low_v.size(); i++) {
		count = 0;
		for (int j = 0; j < x_v.size(); j++) {
			if (x_v[j] >= query_x_low_v[i] && x_v[j] <= query_x_up_v[i]) {
				count++;
			}
		}
		real_results.push_back(count);
		/*if (i % 10000 == 0) {
			cout << "Utils...calculating real results..." << i << endl;
		}*/
	}
	auto t1 = chrono::steady_clock::now();
	//cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	//cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset.size() / queryset.n_rows) << " ns" << endl;
	
	// save real resutls to file
	ofstream outfile;
	outfile.open("C:/Users/Cloud/Desktop/LearnIndex/data/SortedDimResults_REAL.csv");
	for (int i = 0; i < real_results.size(); i++) {
		outfile << real_results[i] << endl;
	}
	outfile.close();
}

void TestApproximationWithDataset(string dataset_file="C:/Users/Cloud/Desktop/LearnIndex/data/HilbertSortedPOIs2_22.csv") {
	vector<int> arch;
	arch.push_back(1);
	arch.push_back(10);
	arch.push_back(100);
	arch.push_back(1000);

	mat dataset;
	bool loaded = mlpack::data::Load(dataset_file, dataset);
	arma::rowvec trainingset = dataset.row(4); // hilbert value
	arma::rowvec responses = dataset.row(5); // order
	StageModel stage_model(trainingset, responses, arch); // need to set TOTAL_SIZE to total_sum
	stage_model.DumpParameters();

	vector<double> train_v, results_v;
	StageModel::RowvecToVector(trainingset, train_v);

	// try with generated Hilbert
	stage_model.PredictVector(train_v, results_v);

	// record into file
	ofstream outfile;
	outfile.open("C:/Users/Cloud/Desktop/LearnIndex/data/Sorted2DimResultsCOUNT_DATASET_PREDICATED_22.csv");
	for (int i = 0; i < results_v.size(); i++) {
		outfile << train_v[i] << "," << results_v[i] << endl;
	}
	outfile.close();
}

void CompareResults() {
	mat real;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/Sorted2DimResults_REAL.csv", real);
	
	mat pred;
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/Sorted2DimResults_LEARN.csv", pred);

	vector<double> real_v, pred_v;
	StageModel::RowvecToVector(real, real_v);
	StageModel::RowvecToVector(pred, pred_v);

	ofstream outfile;
	outfile.open("C:/Users/Cloud/Desktop/LearnIndex/data/Results_Compare.csv");
	for (int i = 0; i < real_v.size(); i++) {
		outfile << i << "," << real_v[i] << "," << pred_v[i] << endl;
	}
	outfile.close();
}

void RowvecToVector(arma::mat& matrix, vector<double> &vec) {
	vec.clear();
	for (int i = 0; i < matrix.n_cols; i++) {
		vec.push_back(matrix[i]);
	}
}

void RowvecToVector(const arma::rowvec& matrix, vector<double> &vec) {
	vec.clear();
	for (int i = 0; i < matrix.n_cols; i++) {
		vec.push_back(matrix[i]);
	}
}

void RowvecToVector(arma::mat& matrix, vector<int> &vec) {
	vec.clear();
	for (int i = 0; i < matrix.n_cols; i++) {
		vec.push_back(matrix[i]);
	}
}

void VectorToRowvec(arma::mat& rv, vector<double> &v) {
	rv.clear();
	rv.set_size(v.size());
	int count = 0;
	rv.imbue([&]() { return v[count++]; });
}

void VectorToRowvec(arma::mat& rv, vector<int> &v) {
	rv.clear();
	rv.set_size(v.size());
	int count = 0;
	rv.imbue([&]() { return v[count++]; });
}

double MeasureEstimatedRelativeError(double est, double err) {
	double part1 = err / est;
	double part2 = 2 * err * err / ((est - 2 * err)*est);
	double relerr = 2*(part1 + part2);
	//cout << relerr * 100 << "%" << endl;
	return relerr;
}

double MeasureEstimatedRelativeError2(double est, double err) {
	double relerr = 2*(err/(est-2*err));
	//cout << relerr * 100 << "%" << endl;
	return relerr;
}

void MeasureAccuracy(vector<int> &predicted_results, vector<int> &real_results) {
	double abs_err = 0;
	double relative_error;
	double accu = 0;
	double accu_absolute = 0;
	double est_rel_err = 0;
	int total_size = predicted_results.size();
	for (int i = 0; i < total_size; i++) {
		if (real_results[i] == 0) {
			total_size--;
			continue;
		}
		abs_err = abs(predicted_results[i] - real_results[i]);
		relative_error = abs(double(predicted_results[i] - real_results[i])) / real_results[i];
		accu += relative_error;
		accu_absolute += abs(predicted_results[i] - real_results[i]);
		est_rel_err = MeasureEstimatedRelativeError(predicted_results[i], 100);
		//if (relative_error >= est_rel_err) {
			//cout << "debug here : " << i << "  est rel err: " << est_rel_err << endl;
		//}
		//cout << "relative error " << i << ": " << relative_error*100 <<  "\t estimated relative error:" << MeasureEstimatedRelativeError(predicted_results[i], 100)*100 << "%" << "\t estimated relative error 2:" << MeasureEstimatedRelativeError2(predicted_results[i], 100) * 100 << "%" << "\t true selectivity: " << real_results[i] << endl;
		
		/*cout << i << " absolute error: " << abs(predicted_results[i] - real_results[i]) << endl;
		if (abs_err > 4000) {
			cout << "debug here!" << endl;
		}*/
	}
	cout << "average relative error: " << accu / total_size << endl;
	cout << "average absolute error: " << accu_absolute / total_size << endl;
}

