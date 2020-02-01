#pragma once
//#include "LearnIndexAggregate.h"
//#include "StageModel.h"
#include "mlpack/core.hpp"
#include <string>
#include "mlpack/core.hpp"
#include <windows.h>
using namespace std;

struct QueryResult {
	//std::chrono::duration<double> average_query_time;
	//std::chrono::duration<double> total_query_time;
	unsigned long long average_query_time;
	unsigned long long total_query_time;
	double measured_absolute_error;
	double measured_relative_error;
	int hit_count;
	int refinement_count;
	int total_query_count;
	int model_amount;
	int total_paras; // the total minimum paras
	int tree_paras; // btree or Rtree
};

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

double MeasureEstimatedRelativeError(double est, double err) {
	double part1 = err / est;
	double part2 = 2 * err * err / ((est - 2 * err)*est);
	double relerr = 2 * (part1 + part2);
	//cout << relerr * 100 << "%" << endl;
	return relerr;
}

double MeasureEstimatedRelativeError2(double est, double err) {
	double relerr = 2 * (err / (est - 2 * err));
	//cout << relerr * 100 << "%" << endl;
	return relerr;
}

void LoadTweetDataset(vector<double> &keys, vector<double> &values) {
	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset);
	arma::rowvec trainingset = dataset.row(0);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);
	RowvecToVector(trainingset, keys);
	RowvecToVector(responses, values);
}

void LoadTweetQuerySet(vector<double> &Querykey_L, vector<double> &Querykey_U) {
	mat queryset;
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQuery2.csv", queryset);
	arma::rowvec query_x_low = queryset.row(0);
	arma::rowvec query_x_up = queryset.row(1);
	vector<double> queryset_x_up_v, queryset_x_low_v;
	RowvecToVector(query_x_up, Querykey_U);
	RowvecToVector(query_x_low, Querykey_L);
}

void LoadHKIDataset(vector<double> &keys, vector<double> &values) {
	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/iCloudDrive/ProcessedFinancialData.csv", dataset);
	arma::rowvec trainingset = dataset.row(0);
	arma::rowvec responses = dataset.row(2);
	RowvecToVector(trainingset, keys);
	RowvecToVector(responses, values);
}

void LoadHKIQuerySet(vector<double> &Querykey_L, vector<double> &Querykey_U) {
	mat queryset;
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/iCloudDrive/ProcessedFinancialQuery.csv", queryset);
	arma::rowvec query_x_low = queryset.row(0);
	arma::rowvec query_x_up = queryset.row(1);
	vector<double> queryset_x_up_v, queryset_x_low_v;
	RowvecToVector(query_x_up, Querykey_U);
	RowvecToVector(query_x_low, Querykey_L);
}

void LoadTWEET20MDataset(vector<double> &keys, vector<double> &values) {
	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/iCloudDrive/Processed20MTweetPass4_Normalized.csv", dataset);
	arma::rowvec trainingset = dataset.row(0);
	arma::rowvec responses = dataset.row(1);
	RowvecToVector(trainingset, keys);
	RowvecToVector(responses, values);
}

void LoadTWEET20MQuerySet(vector<double> &Querykey_L, vector<double> &Querykey_U) {
	mat queryset;
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/iCloudDrive/Processed20MTweetQuery_Normalized.csv", queryset);
	arma::rowvec query_x_low = queryset.row(0);
	arma::rowvec query_x_up = queryset.row(1);
	vector<double> queryset_x_up_v, queryset_x_low_v;
	RowvecToVector(query_x_up, Querykey_U);
	RowvecToVector(query_x_low, Querykey_L);
}

void LoadOSM_Dataset_Queryset_1D_SUBSET(vector<double> &keys, vector<double> &values, vector<double> &q_low, vector<double> &q_up, int option=1) {
	
	arma::mat dataset;
	arma::mat queryset;

	switch (option){
	case 1:
		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_1M_Sorted_Value.csv", dataset);
		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_1M_Query_1D.csv", queryset);
		break;
	case 2:
		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_10M_Sorted_Value.csv", dataset);
		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_10M_Query_1D.csv", queryset);
		break;
	case 3:
		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_30M_Sorted_Value.csv", dataset);
		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_30M_Query_1D.csv", queryset);
		break;
	case 4:
		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_100M_Sorted_Value.csv", dataset);
		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_100M_Query_1D.csv", queryset);
		break;
	default:
		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_100M_Sorted_Value.csv", dataset);
		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_100M_Query_1D.csv", queryset);
		break;
	}
	
	arma::rowvec key_row = dataset.row(0);
	arma::rowvec value_row = dataset.row(dataset.n_rows - 1);
	RowvecToVector(key_row, keys);
	RowvecToVector(value_row, values);
	RowvecToVector(queryset.row(0), q_low);
	RowvecToVector(queryset.row(1), q_up);
}

void LoadOSMDataset(vector<double> &key1, vector<double> &key2) {
	arma::mat dataset;
	mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/Sorted2DimTrainingSet_Long_Lat_100M.csv", dataset); // first key longitude, second key latitude
	arma::rowvec key1_row = dataset.row(0);
	arma::rowvec key2_row = dataset.row(1);
	RowvecToVector(key1_row, key1);
	RowvecToVector(key2_row, key2);
}

void LoadOSMQuerySet(vector<double> &d1_low, vector<double> &d2_low, vector<double> &d1_up, vector<double> &d2_up) {
	arma::mat queryset;
	mlpack::data::Load("C:/Users/Cloud/iCloudDrive/Queries2D_mean_18_9_sigma_36_18_normalized2.csv", queryset); // d1_low, d2_low, d1_up, d2_up

	RowvecToVector(queryset.row(0), d1_low);
	RowvecToVector(queryset.row(1), d2_low);
	RowvecToVector(queryset.row(2), d1_up);
	RowvecToVector(queryset.row(3), d2_up);
}

// find COUNT POI (TWEET20M)
void FindExactResultByScan_POI() {
	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/iCloudDrive/Processed20MTweetPass4_Normalized.csv", dataset);
	arma::rowvec x = dataset.row(0); // x
	arma::rowvec y = dataset.row(1); // y
	vector<double> x_v, y_v;
	RowvecToVector(x, x_v);
	RowvecToVector(y, y_v);

	mat queryset;
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/iCloudDrive/Processed20MTweetQuery_Normalized.csv", queryset);
	arma::rowvec queryset_x_low = queryset.row(0);
	arma::rowvec queryset_x_up = queryset.row(1);
	vector<double> query_x_low_v, query_x_up_v;
	RowvecToVector(queryset_x_low, query_x_low_v);
	RowvecToVector(queryset_x_up, query_x_up_v);

	vector<double> real_results;

	auto t0 = chrono::steady_clock::now();
	int count;
	for (int i = 0; i < query_x_low_v.size(); i++) {
		count = 0; // no nefgative records
		for (int j = 0; j < x_v.size(); j++) {
			if (x_v[j] >= query_x_low_v[i] && x_v[j] <= query_x_up_v[i]) {
				count += 1;
			}
		}
		real_results.push_back(count);
	}
	auto t1 = chrono::steady_clock::now();

	// save real resutls to file
	ofstream outfile;
	outfile.open("C:/Users/Cloud/iCloudDrive/LearnedAggregate/VLDB_Final_Experiments/RealQueryResults/POI_COUNT.csv");
	for (int i = 0; i < real_results.size(); i++) {
		outfile << real_results[i] << endl;
	}
	outfile.close();
}

// find MAX
void FindExactResultByScan() {
	mat dataset;
	mlpack::data::Load("C:/Users/Cloud/iCloudDrive/ProcessedFinancialData.csv", dataset);
	arma::rowvec x = dataset.row(0); // x
	arma::rowvec y = dataset.row(2); // y
	vector<double> x_v, y_v;
	RowvecToVector(x, x_v);
	RowvecToVector(y, y_v);
	
	mat queryset;
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/iCloudDrive/ProcessedFinancialQuery.csv", queryset);
	arma::rowvec queryset_x_low = queryset.row(0);
	arma::rowvec queryset_x_up = queryset.row(1);
	vector<double> query_x_low_v, query_x_up_v;
	RowvecToVector(queryset_x_low, query_x_low_v);
	RowvecToVector(queryset_x_up, query_x_up_v);
	
	vector<double> real_results;
	double max = 0;

	auto t0 = chrono::steady_clock::now();
	int count;
	for (int i = 0; i < query_x_low_v.size(); i++) {
		max = 0; // no nefgative records
		for (int j = 0; j < x_v.size(); j++) {
			if (x_v[j] >= query_x_low_v[i] && x_v[j] <= query_x_up_v[i]) {
				if (y_v[j] > max) {
					max = y_v[j];
				}
			}
		}
		real_results.push_back(max);
	}
	auto t1 = chrono::steady_clock::now();
	
	// save real resutls to file
	ofstream outfile;
	outfile.open("C:/Users/Cloud/iCloudDrive/LearnedAggregate/VLDB_Final_Experiments/RealQueryResults/HKI_MAX.csv");
	for (int i = 0; i < real_results.size(); i++) {
		outfile << real_results[i] << endl;
	}
	outfile.close();
}

// find 2D COUNT
void FindExactResultByScan2() {
	arma::mat dataset;
	mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/Sorted2DimTrainingSet_Long_Lat_100M.csv", dataset); // first key longitude, second key latitude
	arma::rowvec key1_row = dataset.row(0);
	arma::rowvec key2_row = dataset.row(1);
	vector<double> x1_v, x2_v;
	RowvecToVector(key1_row, x1_v);
	RowvecToVector(key2_row, x2_v);

	arma::mat queryset;
	mlpack::data::Load("C:/Users/Cloud/iCloudDrive/Queries2D_mean_18_9_sigma_36_18_normalized2.csv", queryset); // d1_low, d2_low, d1_up, d2_up
	vector<double> d1_low, d2_low, d1_up, d2_up;
	RowvecToVector(queryset.row(0), d1_low);
	RowvecToVector(queryset.row(1), d2_low);
	RowvecToVector(queryset.row(2), d1_up);
	RowvecToVector(queryset.row(3), d2_up);

	vector<double> real_results;

	auto t0 = chrono::steady_clock::now();
	int count;
	for (int i = 0; i < d1_low.size(); i++) {
		count = 0;
		for (int j = 0; j < x1_v.size(); j++) {
			if (x1_v[j] >= d1_low[i] && x1_v[j] <= d1_up[i] && x2_v[j] >= d2_low[i] && x2_v[j] <= d2_up[i]) {
				count++;
			}
		}
		real_results.push_back(count);
	}
	auto t1 = chrono::steady_clock::now();

	// save real resutls to file
	ofstream outfile;
	outfile.open("C:/Users/Cloud/iCloudDrive/LearnedAggregate/VLDB_Final_Experiments/RealQueryResults/OSM_2D.csv");
	for (int i = 0; i < real_results.size(); i++) {
		outfile << real_results[i] << endl;
	}
	outfile.close();
}

// the python version is incorrect
void Generate2DSampling(int sample_k1, int sample_k2) {
	arma::mat dataset;
	mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/Sorted2DimTrainingSet_Long_Lat_100M.csv", dataset);
	arma::rowvec key1_row = dataset.row(0);
	arma::rowvec key2_row = dataset.row(1);
	vector<double> x1_v, x2_v;
	RowvecToVector(key1_row, x1_v); // lon
	RowvecToVector(key2_row, x2_v); // lat

	vector<vector<int>> cumulative_count; // should be 1K*1K
	// initialization
	for (int i = 0; i < sample_k1; i++) {
		vector<int> inner;
		for (int j = 0; j < sample_k2; j++) {
			inner.push_back(0);
		}
		cumulative_count.push_back(inner);
	}

	double step_size_1 = 360.0 / sample_k1;
	double step_size_2 = 180.0 / sample_k2;

	double upper_k1;
	double upper_k2;

	int bucket_1, bucket_2;

	// assign the entire dataset to each bucket
	for (int i = 0; i < x1_v.size(); i++) {
		// calculate which bucket it belongs to
		bucket_1 = (x1_v[i] - (-180.0)) / step_size_1;
		bucket_2 = (x2_v[i] - (-90.0)) / step_size_2;
		if (bucket_1 == sample_k1) {
			bucket_1 = sample_k1 - 1;
		}
		if (bucket_2 == sample_k2) {
			bucket_2 = sample_k2 - 1;
		}
		cumulative_count[bucket_1][bucket_2] += 1;
	}
	
	// process vertical cumulative. i.e., accu count on key2
	for (int i = 0; i < sample_k1; i++) {
		for (int j = 1; j < sample_k2; j++) {
			cumulative_count[i][j] += cumulative_count[i][j-1];
		}
	}

	// process horizontal cumulative, i.e., accu on key1
	for (int j = 0; j < sample_k2; j++) {
		for (int i = 1; i < sample_k1; i++) {
			cumulative_count[i][j] += cumulative_count[i-1][j];
		}
	}

	// save the cumulative count to file
	ofstream outfile_result;
	outfile_result.open("C:/Users/Cloud/iCloudDrive/LearnedAggregate/COUNT2D_SAMPLING_OSM_1K_1K.csv");
	for (int i = 0; i < sample_k1; i++) {
		upper_k1 = -180 + step_size_1 * (i + 1);
		for (int j = 0; j < sample_k2; j++) {
			upper_k2 = -90 + step_size_2 * (j + 1);
			outfile_result << upper_k1 << "," << upper_k2 << "," << cumulative_count[i][j] << endl;
		}
	}
}

void SpecificScan() {
	arma::mat dataset;
	mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/Sorted2DimTrainingSet_Long_Lat_100M.csv", dataset); // first key longitude, second key latitude
	arma::rowvec key1_row = dataset.row(0);
	arma::rowvec key2_row = dataset.row(1);
	vector<double> x1_v, x2_v;
	RowvecToVector(key1_row, x1_v);
	RowvecToVector(key2_row, x2_v);

	//arma::mat queryset;
	//mlpack::data::Load("C:/Users/Cloud/iCloudDrive/Queries2D_mean_18_9_sigma_36_18_normalized2.csv", queryset); // d1_low, d2_low, d1_up, d2_up
	//vector<double> d1_low, d2_low, d1_up, d2_up;
	//RowvecToVector(queryset.row(0), d1_low);
	//RowvecToVector(queryset.row(1), d2_low);
	//RowvecToVector(queryset.row(2), d1_up);
	//RowvecToVector(queryset.row(3), d2_up);

	//vector<double> real_results;

	int count;
	count = 0;
	for (int j = 0; j < x1_v.size(); j++) {
		if (x1_v[j] >= -180 && x1_v[j] <= -136.44 && x2_v[j] >= -90 && x2_v[j] <= 29.7) {
			count++;
		}
	}
	cout << "cout1: " << count << endl;

	count = 0;
	for (int j = 0; j < x1_v.size(); j++) {
		if (x1_v[j] >= -180 && x1_v[j] <= -34.92 && x2_v[j] >= -90 && x2_v[j] <= 49.32) {
			count++;
		}
	}
	cout << "cout2: " << count << endl;

	count = 0;
	for (int j = 0; j < x1_v.size(); j++) {
		if (x1_v[j] >= -180 && x1_v[j] <= -136.44 && x2_v[j] >= -90 && x2_v[j] <= 49.32) {
			count++;
		}
	}
	cout << "cout3: " << count << endl;

	count = 0;
	for (int j = 0; j < x1_v.size(); j++) {
		if (x1_v[j] >= -180 && x1_v[j] <= -34.92 && x2_v[j] >= -90 && x2_v[j] <= 29.7) {
			count++;
		}
	}
	cout << "cout4: " << count << endl;
}

// ====================================================================================================

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


void CalculateRealCountWithScan2D(arma::mat &dataset, arma::mat &queryset, vector<int> &real_results) {}

//void CalculateRealCountWithScan2D(arma::mat &dataset, arma::mat &queryset, vector<int> &real_results) {
//	arma::rowvec x = dataset.row(0); // x
//	arma::rowvec y = dataset.row(1); // y
//	vector<double> x_v, y_v;
//	StageModel::RowvecToVector(x, x_v);
//	StageModel::RowvecToVector(y, y_v);
//
//	mat queryset_x_low = queryset.row(0);
//	mat queryset_x_up = queryset.row(1);
//	mat queryset_y_low = queryset.row(2);
//	mat queryset_y_up = queryset.row(3);
//	vector<double> query_x_low_v, query_x_up_v, query_y_low_v, query_y_up_v;
//	StageModel::RowvecToVector(queryset_x_low, query_x_low_v);
//	StageModel::RowvecToVector(queryset_x_up, query_x_up_v);
//	StageModel::RowvecToVector(queryset_y_low, query_y_low_v);
//	StageModel::RowvecToVector(queryset_y_up, query_y_up_v);
//	auto t0 = chrono::steady_clock::now();
//	int count;
//	for (int i = 0; i < query_x_low_v.size(); i++) {
//		count = 0;
//		for (int j = 0; j < x_v.size(); j++) {
//			if (x_v[j]>= query_x_low_v[i] && x_v[j]<= query_x_up_v[i] && y_v[j]>= query_y_low_v[i] && y_v[j]<= query_y_up_v[i]) {
//				count++;
//			}
//		}
//		real_results.push_back(count);
//		/*if (i % 10000 == 0) {
//			cout << "Utils...calculating real results..." << i << endl;
//		}*/
//	}
//	auto t1 = chrono::steady_clock::now();
//	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
//	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset.size() / queryset.n_rows) << " ns" << endl;
//	// save real resutls to file
//	ofstream outfile;
//	outfile.open("C:/Users/Cloud/Desktop/LearnIndex/data/Sorted2DimResults_REAL.csv");
//	for (int i = 0; i < real_results.size(); i++) {
//		outfile << real_results[i] << endl;
//	}
//	outfile.close();
//}
//
void CalculateRealCountWithScan2D(mat &queryset, vector<int> &real_results) {}
//void CalculateRealCountWithScan2D(mat &queryset, vector<int> &real_results) {
//	mat dataset;
//	mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/HilbertSortedPOIs2_22.csv", dataset);
//	arma::rowvec x = dataset.row(0); // x
//	arma::rowvec y = dataset.row(1); // y
//	vector<double> x_v, y_v;
//	StageModel::RowvecToVector(x, x_v);
//	StageModel::RowvecToVector(y, y_v);
//
//	mat queryset_x_low = queryset.row(0);
//	mat queryset_x_up = queryset.row(1);
//	mat queryset_y_low = queryset.row(2);
//	mat queryset_y_up = queryset.row(3);
//	vector<double> query_x_low_v, query_x_up_v, query_y_low_v, query_y_up_v;
//	StageModel::RowvecToVector(queryset_x_low, query_x_low_v);
//	StageModel::RowvecToVector(queryset_x_up, query_x_up_v);
//	StageModel::RowvecToVector(queryset_y_low, query_y_low_v);
//	StageModel::RowvecToVector(queryset_y_up, query_y_up_v);
//	auto t0 = chrono::steady_clock::now();
//	int count;
//	for (int i = 0; i < query_x_low_v.size(); i++) {
//		count = 0;
//		for (int j = 0; j < x_v.size(); j++) {
//			if (x_v[j] >= query_x_low_v[i] && x_v[j] <= query_x_up_v[i] && y_v[j] >= query_y_low_v[i] && y_v[j] <= query_y_up_v[i]) {
//				count++;
//			}
//		}
//		real_results.push_back(count);
//		/*if (i % 10000 == 0) {
//			cout << "Utils...calculating real results..." << i << endl;
//		}*/
//	}
//	auto t1 = chrono::steady_clock::now();
//	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
//	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset.size() / queryset.n_rows) << " ns" << endl;
//	// save real resutls to file
//	ofstream outfile;
//	outfile.open("C:/Users/Cloud/Desktop/LearnIndex/data/Sorted2DimResults_REAL.csv");
//	for (int i = 0; i < real_results.size(); i++) {
//		outfile << real_results[i] << endl;
//	}
//	outfile.close();
//}
//
void CalculateRealCountWithScan1D(mat &queryset, vector<int> &real_results, string file_path = "C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv") {}

//void CalculateRealCountWithScan1D(mat &queryset, vector<int> &real_results, string file_path="C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv") {
//	mat dataset;
//	mlpack::data::Load(file_path, dataset);
//	arma::rowvec x = dataset.row(0); // x
//	arma::rowvec y = dataset.row(1); // y
//	vector<double> x_v, y_v;
//	StageModel::RowvecToVector(x, x_v);
//	StageModel::RowvecToVector(y, y_v);
//
//	mat queryset_x_low = queryset.row(0);
//	mat queryset_x_up = queryset.row(1);
//	vector<double> query_x_low_v, query_x_up_v;
//	StageModel::RowvecToVector(queryset_x_low, query_x_low_v);
//	StageModel::RowvecToVector(queryset_x_up, query_x_up_v);
//
//	auto t0 = chrono::steady_clock::now();
//	int count;
//	for (int i = 0; i < query_x_low_v.size(); i++) {
//		count = 0;
//		for (int j = 0; j < x_v.size(); j++) {
//			if (x_v[j] >= query_x_low_v[i] && x_v[j] <= query_x_up_v[i]) {
//				count++;
//			}
//		}
//		real_results.push_back(count);
//		/*if (i % 10000 == 0) {
//			cout << "Utils...calculating real results..." << i << endl;
//		}*/
//	}
//	auto t1 = chrono::steady_clock::now();
//	//cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
//	//cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset.size() / queryset.n_rows) << " ns" << endl;
//	
//	// save real resutls to file
//	ofstream outfile;
//	//outfile.open("C:/Users/Cloud/Desktop/LearnIndex/data/SortedDimResults_REAL.csv");
//	outfile.open("C:/Users/Cloud/iCloudDrive/LearnedAggregate/VLDB_Final_Experiments/RealQueryResults/TWEET_1D.csv");
//	for (int i = 0; i < real_results.size(); i++) {
//		outfile << real_results[i] << endl;
//	}
//	outfile.close();
//}
//
//void TestApproximationWithDataset(string dataset_file="C:/Users/Cloud/Desktop/LearnIndex/data/HilbertSortedPOIs2_22.csv") {
//	vector<int> arch;
//	arch.push_back(1);
//	arch.push_back(10);
//	arch.push_back(100);
//	arch.push_back(1000);
//
//	mat dataset;
//	bool loaded = mlpack::data::Load(dataset_file, dataset);
//	arma::rowvec trainingset = dataset.row(4); // hilbert value
//	arma::rowvec responses = dataset.row(5); // order
//	StageModel stage_model(trainingset, responses, arch); // need to set TOTAL_SIZE to total_sum
//	stage_model.DumpParameters();
//
//	vector<double> train_v, results_v;
//	StageModel::RowvecToVector(trainingset, train_v);
//
//	// try with generated Hilbert
//	stage_model.PredictVector(train_v, results_v);
//
//	// record into file
//	ofstream outfile;
//	outfile.open("C:/Users/Cloud/Desktop/LearnIndex/data/Sorted2DimResultsCOUNT_DATASET_PREDICATED_22.csv");
//	for (int i = 0; i < results_v.size(); i++) {
//		outfile << train_v[i] << "," << results_v[i] << endl;
//	}
//	outfile.close();
//}
//
//void CompareResults() {
//	mat real;
//	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/Sorted2DimResults_REAL.csv", real);
//	
//	mat pred;
//	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/Sorted2DimResults_LEARN.csv", pred);
//
//	vector<double> real_v, pred_v;
//	StageModel::RowvecToVector(real, real_v);
//	StageModel::RowvecToVector(pred, pred_v);
//
//	ofstream outfile;
//	outfile.open("C:/Users/Cloud/Desktop/LearnIndex/data/Results_Compare.csv");
//	for (int i = 0; i < real_v.size(); i++) {
//		outfile << i << "," << real_v[i] << "," << pred_v[i] << endl;
//	}
//	outfile.close();
//}

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

// used to measure query accuracy
void MeasureAccuracy(vector<int> &predicted_results, string filepath, double &MEabs, double &MErel) {
	
	mat results;
	bool loaded = mlpack::data::Load(filepath, results);
	arma::rowvec real_results_row = results.row(0);
	vector<double> real_results;
	RowvecToVector(real_results_row, real_results);

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
	}

	MEabs = accu_absolute / total_size;
	MErel =  accu / total_size;
	//cout << "measured average relative error: " << accu / total_size << endl;
	//cout << "measured average absolute error: " << accu_absolute / total_size << endl;
}

// for max
void MeasureAccuracy(vector<double> &predicted_results, string filepath, double &MEabs, double &MErel) {

	mat results;
	bool loaded = mlpack::data::Load(filepath, results);
	arma::rowvec real_results_row = results.row(0);
	vector<double> real_results;
	RowvecToVector(real_results_row, real_results);

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
	}

	MEabs = accu_absolute / total_size; 
	MErel = accu / total_size;

	cout << "measured average absolute error: " << MEabs << endl;
	cout << "measured average relative error: " << MErel << endl;
}

