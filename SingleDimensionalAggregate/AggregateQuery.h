#pragma once
#include "LearnIndexAggregate.h"
#include "STXBtreeAggregate.h"
#include "StageModel.h"
#include "Hilbert.h"

using namespace std;

void Count2DLearnedIndex(unsigned max_bits, unsigned using_bits = 0) {

	if (using_bits == 0) {
		using_bits = max_bits;
	}

	vector<int> arch;
	arch.push_back(1);
	arch.push_back(10);
	arch.push_back(100);
	arch.push_back(1000);

	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/HilbertSortedPOIs2_22.csv", dataset);
	arma::rowvec trainingset = dataset.row(4); // hilbert value
	arma::rowvec responses = dataset.row(5); // order
	StageModel stage_model(trainingset, responses, arch); // need to set TOTAL_SIZE to total_sum
	stage_model.DumpParameters();

	mat queryset;
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/Sorted2DimQuery2.csv", queryset);
	mat queryset_x_low = queryset.row(0);
	mat queryset_x_up = queryset.row(1);
	mat queryset_y_low = queryset.row(2);
	mat queryset_y_up = queryset.row(3);
	vector<double> query_x_low_v, query_x_up_v, query_y_low_v, query_y_up_v;
	StageModel::RowvecToVector(queryset_x_low, query_x_low_v);
	StageModel::RowvecToVector(queryset_x_up, query_x_up_v);
	StageModel::RowvecToVector(queryset_y_low, query_y_low_v);
	StageModel::RowvecToVector(queryset_y_up, query_y_up_v);

	int FULL_BITS = 22; // that's for our dataset
	int differ_bits = FULL_BITS - max_bits;
	int lower_x, upper_x, lower_y, upper_y;
	vector<interval> intervals;
	int count;
	vector<int> counts;
	long intervals_count = 0;
	vector<double> lower_hilbert, upper_hilbert;
	vector<double> results_low, results_up;

	// query
	auto t0 = chrono::steady_clock::now();
	for (int i = 0; i < query_x_low_v.size(); i++) {
		// generate Hilbert value
		lower_x = (int)((2 * query_x_low_v[i] + 180) * 10000) >> differ_bits;
		upper_x = (int)((2 * query_x_up_v[i] + 180) * 10000) >> differ_bits;
		lower_y = (int)((query_y_low_v[i] + 180) * 10000) >> differ_bits;
		upper_y = (int)((query_y_up_v[i] + 180) * 10000) >> differ_bits;
		intervals.clear();
		GetIntervals(max_bits, using_bits, lower_x, upper_x, lower_y, upper_y, intervals);

		// extract intervals lower and upper hilbert value
		lower_hilbert.clear();
		upper_hilbert.clear();
		for (int j = 0; j < intervals.size(); j++) {
			lower_hilbert.push_back(intervals[j].lower_hilbert_value);
			upper_hilbert.push_back(intervals[j].upper_hilbert_value);
		}

		stage_model.PredictVector(lower_hilbert, results_low);
		stage_model.PredictVector(upper_hilbert, results_up);

		// calculate the count
		count = 0;
		double temp;
		for (int j = 0; j < results_low.size(); j++) {
			temp = results_up[j] - results_low[j];
			if(temp > 0)
				count += temp;
		}
		
		// save results
		counts.push_back(count);
	}
	auto t1 = chrono::steady_clock::now();
	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset.size() / queryset.n_rows) << " ns" << endl;
	cout << "finish stage model prediction.." << endl;

	// save results to file
	ofstream outfile;
	outfile.open("C:/Users/Cloud/Desktop/LearnIndex/data/Sorted2DimResults_LEARN.csv");
	for (int i = 0; i < counts.size(); i++) {
		outfile << counts[i] << endl;
	}
	outfile.close();

	// measure accuracy
	vector<int> real_results;
	CalculateRealCountWithScan2D(dataset, queryset, real_results);
	double relative_error;
	double accu = 0;
	double accu_absolute = 0;
	int total_size = counts.size();
	for (int i = 0; i < counts.size(); i++) {
		if (real_results[i] == 0) {
			total_size--;
			continue;
		}
		relative_error = abs(double(counts[i] - real_results[i]) / real_results[i]);
		accu += relative_error;
		accu_absolute += abs(counts[i] - real_results[i]);
		//cout << "relative error " << i << ": " << relative_error << endl;
	}
	double avg_rel_err = accu / total_size;
	cout << "average relative error: " << avg_rel_err << endl;
	cout << "average absolute error: " << accu_absolute / total_size << endl;
}

void SumShiftLearnedIndex() {
	vector<int> arch;
	arch.push_back(1);
	arch.push_back(10);
	arch.push_back(100);
	arch.push_back(1000);

	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimSUM_SHIFT2.csv", dataset);
	arma::rowvec trainingset = dataset.row(0); // x
	arma::rowvec responses = dataset.row(4); // sum_shift
	arma::rowvec responses_order = dataset.row(5); // order
	//double total_sum = responses[responses.n_cols - 1];
	//cout << "total_sum" << total_sum << endl;
	StageModel stage_model(trainingset, responses, arch, 1); // need to set TOTAL_SIZE to total_sum
	StageModel stage_model_order(trainingset, responses_order, arch, 0); // need to set TOTAL_SIZE to total_size
	stage_model.DumpParameters();
	stage_model_order.DumpParameters();

	mat queryset;
	//bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimSUM2.csv", queryset);
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQuery2.csv", queryset);
	mat queryset_low = queryset.row(0);
	mat queryset_up = queryset.row(1);

	arma::rowvec predictions_low, predictions_up;
	vector<double> queryset_low_v, queryset_up_v, results_low, results_up, results_low_order, results_up_order;
	vector<double> da;
	vector<int> re;
	StageModel::RowvecToVector(dataset, da);
	StageModel::RowvecToVector(queryset_low, queryset_low_v);
	StageModel::RowvecToVector(queryset_up, queryset_up_v);

	auto t0 = chrono::steady_clock::now();
	stage_model.PredictVector(queryset_low_v, results_low);
	stage_model.PredictVector(queryset_up_v, results_up);
	stage_model_order.PredictVector(queryset_low_v, results_low_order);
	stage_model_order.PredictVector(queryset_up_v, results_up_order);
	for (int i = 0; i < results_up.size(); i++) {
		results_up[i] = results_up[i] - results_low[i] - 180*(results_up_order[i]- results_low_order[i]);
	}
	auto t1 = chrono::steady_clock::now();
	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset.size() / queryset.n_rows) << " ns" << endl;
	cout << "finish stage model prediction.." << endl;

	//// save the predicted result
	//ofstream outfile;
	//outfile.open("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimResultsSUM_SHIFT_DATASET_PREDICATED.csv");
	//for (int i = 0; i < results_low.size(); i++) {
	//	outfile << queryset_low_v[i] << "," << results_low[i] << endl;
	//}
	//outfile.close();

	mat real_results;
	bool loaded3 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimResultsSUM.csv", real_results);
	arma::rowvec real = real_results.row(0);
	vector<double> real_v;
	StageModel::RowvecToVector(real, real_v);
	stage_model.MeasureAccuracyWithVector(results_up, real_v);
}

void SumLearnedIndex() {
	vector<int> arch;
	arch.push_back(1);
	arch.push_back(10);
	arch.push_back(100);
	arch.push_back(1000);

	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimSUM2.csv", dataset);
	arma::rowvec trainingset = dataset.row(0); // x
	arma::rowvec responses = dataset.row(2); // sum
	//double total_sum = responses[responses.n_cols - 1];
	//cout << "total_sum" << total_sum << endl;
	StageModel stage_model(trainingset, responses, arch, 1); // need to set TOTAL_SIZE to total_sum
	stage_model.DumpParameters();

	mat queryset;
	//bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimSUM2.csv", queryset);
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQuery2.csv", queryset);
	mat queryset_low = queryset.row(0);
	mat queryset_up = queryset.row(1);

	arma::rowvec predictions_low, predictions_up;
	vector<double> queryset_low_v, queryset_up_v, results_low, results_up;
	vector<double> da;
	vector<int> re;
	StageModel::RowvecToVector(dataset, da);
	StageModel::RowvecToVector(queryset_low, queryset_low_v);
	StageModel::RowvecToVector(queryset_up, queryset_up_v);

	auto t0 = chrono::steady_clock::now();
	stage_model.PredictVector(queryset_low_v, results_low);
	stage_model.PredictVector(queryset_up_v, results_up);
	for (int i = 0; i < results_up.size(); i++) {
		results_up[i] -= results_low[i];
	}
	auto t1 = chrono::steady_clock::now();
	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset.size() / queryset.n_rows) << " ns" << endl;
	cout << "finish stage model prediction.." << endl;

	//// save the predicted result
	//ofstream outfile;
	//outfile.open("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimResultsSUM_DATASET_PREDICATED.csv");
	//for (int i = 0; i < results_low.size(); i++) {
	//	outfile << queryset_low_v[i] << "," << results_low[i] << endl;
	//}
	//outfile.close();
	
	mat real_results;
	bool loaded3 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimResultsSUM.csv", real_results);
	arma::rowvec real = real_results.row(0);
	vector<double> real_v;
	StageModel::RowvecToVector(real, real_v);
	stage_model.MeasureAccuracyWithVector(results_up, real_v);
}

// a relative naive method, do not reconstruct the structure
void SumStxBtree() {
	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimSUM2.csv", dataset);
	arma::rowvec trainingset = dataset.row(0); // x
	arma::rowvec responses = dataset.row(2); // sum
	vector<double> train_v, response_v;
	StageModel::RowvecToVector(trainingset, train_v);
	StageModel::RowvecToVector(responses, response_v);

	stx::btree<double, double> btree;
	for (int i = 0; i < train_v.size(); i++) {
		btree.insert(pair<double, double>(train_v[i], response_v[i]));
	}

	mat queryset;
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQuery2.csv", queryset);
	mat queryset_low = queryset.row(0);
	mat queryset_up = queryset.row(1);
	vector<double> query_low_v, query_up_v;
	StageModel::RowvecToVector(queryset_low, query_low_v);
	StageModel::RowvecToVector(queryset_up, query_up_v);

	stx::btree<double, double>::iterator iter_low, iter_up;
	double sum = 0;
	vector<double> results;
	auto t0 = chrono::steady_clock::now(); // am I doing the correct thing???????????????
	for (int i = 0; i < query_low_v.size(); i++) {
		iter_low = btree.lower_bound(query_low_v[i]); // include the key
		iter_up = btree.lower_bound(query_up_v[i]); // include the key
		//cout << iter_low->first << " " << iter_up->first << endl;
		if (iter_low!= btree.begin()) {
			iter_low--;
		}
		//cout << iter_low->first << endl;
		sum = iter_up->second - iter_low->second;
		results.push_back(sum);
	}
	auto t1 = chrono::steady_clock::now();
	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset.size() / queryset.n_rows) << " ns" << endl;

	ofstream outfile;
	outfile.open("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimResultsSUM.csv");
	for (int i = 0; i < results.size(); i++) {
		outfile << results[i] << endl;
	}
	outfile.close();
}

void CountLearnedIndex() {
	TestBtreeAggregate("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQuery2.csv");

	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);
	dataset.shed_rows(1, 2);

	vector<int> arch;
	arch.push_back(1);
	arch.push_back(10);
	arch.push_back(100);
	arch.push_back(1000);
	StageModel stage_model(dataset, responses, arch);

	mat queryset;
	//bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", queryset);
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQuery2.csv", queryset);
	//bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQuery2.csv", queryset);
	mat queryset_low = queryset.row(0);
	mat queryset_up = queryset.row(1);

	arma::rowvec predictions_low, predictions_up;
	vector<double> queryset_low_v, queryset_up_v, results_low, results_up;
	stage_model.DumpParameters();
	vector<double> da;
	vector<int> re;
	StageModel::RowvecToVector(dataset, da);
	stage_model.RecordBucket(da, re, "C:/Users/Cloud/Desktop/LearnIndex/data/BucketRecord.csv");

	StageModel::RowvecToVector(queryset_low, queryset_low_v);
	StageModel::RowvecToVector(queryset_up, queryset_up_v);

	stage_model.TrainBucketAssigner("C:/Users/Cloud/Desktop/LearnIndex/data/BucketRecord.csv");

	auto t0 = chrono::steady_clock::now();

	//stage_model.PredictVectorWithBucketMap(queryset_low_v, results_low);
	//stage_model.PredictVectorWithBucketMap(queryset_up_v, results_up);
	//stage_model.PredictVectorWithBucketAssigner(queryset_low_v, results_low);
	//stage_model.PredictVectorWithBucketAssigner(queryset_up_v, results_up);

	stage_model.PredictVector(queryset_low_v, results_low);
	stage_model.PredictVector(queryset_up_v, results_up);

	for (int i = 0; i < results_up.size(); i++) {
		results_up[i] -= results_low[i];
	}

	/*StageModel::PredictNaiveSingleLR2(dataset, responses, queryset_low, predictions_low);
	StageModel::PredictNaiveSingleLR2(dataset, responses, queryset_up, predictions_up);*/

	//StageModel::PredictNaiveSingleLR(dataset, responses, queryset_low, predictions_low);
	//StageModel::PredictNaiveSingleLR(dataset, responses, queryset_up, predictions_up);

	//stage_model.InitQuerySet(queryset_low);
	//stage_model.Predict(queryset_low, predictions_low);
	//stage_model.InitQuerySet(queryset_up);   
	//stage_model.Predict(queryset_up, predictions_up);
	//predictions_up -= predictions_low;

	auto t1 = chrono::steady_clock::now();
	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset.size() / queryset.n_rows) << " ns" << endl;
	cout << "finish stage model prediction.." << endl;

	//// save the predicted result
	//ofstream outfile;
	//outfile.open("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimResultsCOUNT_DATASET_PREDICATED.csv");
	//for (int i = 0; i < results_low.size(); i++) {
	//	outfile << queryset_low_v[i] << "," << results_low[i] << endl;
	//}
	//outfile.close();

	StageModel::MeasureAccuracyWithVector(results_up);
	//StageModel::VectorToRowvec(predictions_up, results_up);
	//tage_model.MeasureAccuracy(predictions_up);
}