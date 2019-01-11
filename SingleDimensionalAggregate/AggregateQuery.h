#pragma once
#include "LearnIndexAggregate.h"
#include "STXBtreeAggregate.h"
#include "StageModel.h"

using namespace std;

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

	mat real_results;
	bool loaded3 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimResultsSUM.csv", real_results);
	arma::rowvec real = real_results.row(0);
	vector<double> real_v;
	StageModel::RowvecToVector(real, real_v);
	StageModel::MeasureAccuracyWithVector(results_up, real_v);
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
		iter_low--;
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

	StageModel::MeasureAccuracyWithVector(results_up);
	//StageModel::VectorToRowvec(predictions_up, results_up);
	//tage_model.MeasureAccuracy(predictions_up);
}